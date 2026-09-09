"""
PyTorch modeling implementation for MiniCPM5-2B-Hadamard-GSQ.
Features:
- HadamardLinear4bit: Group-scale INT4 with Walsh-Hadamard spin and Low-Rank SVD (RCO)
- ZeroCompressionShield: Pure BF16 RMSNorms, Biases, and Embeddings
- KVBSSAttentionHook: Key-Value Binding Softmax Sharpening for 128K context
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_minicpm_hadamard import MiniCPMHadamardConfig
    from .kv_bss import KVBSSAttentionHook
except (ImportError, ValueError):
    from configuration_minicpm_hadamard import MiniCPMHadamardConfig
    from kv_bss import KVBSSAttentionHook

_H_CACHE = {}

def get_hadamard_matrix(n: int, dtype=torch.float32, device=None):
    key = (n, dtype, str(device))
    if key in _H_CACHE:
        return _H_CACHE[key]
    if n == 1:
        h = torch.tensor([[1.0]], dtype=dtype, device=device)
    else:
        h_half = get_hadamard_matrix(n // 2, dtype=dtype, device=device)
        top = torch.cat([h_half, h_half], dim=1)
        bottom = torch.cat([h_half, -h_half], dim=1)
        h = torch.cat([top, bottom], dim=0) / math.sqrt(2.0)
    _H_CACHE[key] = h
    return h

def apply_hadamard_rot(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    orig_shape = x.shape
    d = orig_shape[-1]
    assert d % block_size == 0
    h = get_hadamard_matrix(block_size, dtype=x.dtype, device=x.device)
    reshaped = x.view(-1, d // block_size, block_size)
    rotated = torch.matmul(reshaped, h)
    return rotated.view(orig_shape)

class HadamardLinear4bit(nn.Module):
    """
    4-bit Group-Scale Quantized Linear layer with:
    1. Walsh-Hadamard Input Coordinate Spin (Outlier suppression)
    2. Group-Scale INT4 quantization (group size 64)
    3. Low-Rank Residual SVD Compensation: Y = X' W_quant^T + (X' B^T) A^T
    """
    def __init__(self, in_features: int, out_features: int, group_size: int = 64, rank: int = 16, block_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.rank = rank
        self.block_size = block_size

        # Packed uint8 (2 values per byte)
        self.register_buffer("qweight_packed", torch.zeros((out_features, in_features // 2), dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros((out_features, in_features // group_size), dtype=torch.bfloat16))
        self.register_buffer("svd_a", torch.zeros((out_features, rank), dtype=torch.bfloat16))
        self.register_buffer("svd_b", torch.zeros((rank, in_features), dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Rotate input activation using Walsh-Hadamard spin: X' = X · H
        if self.in_features % self.block_size == 0:
            x_rot = apply_hadamard_rot(x, block_size=self.block_size)
        else:
            x_rot = x

        # 2. Dequantize INT4 weight on-the-fly or fast matrix vector
        # Unpack uint8 -> int8 (-8 to 7)
        low = (self.qweight_packed & 0x0F).to(torch.int8)
        low = torch.where(low >= 8, low - 16, low)
        high = ((self.qweight_packed >> 4) & 0x0F).to(torch.int8)
        high = torch.where(high >= 8, high - 16, high)

        m, half_n = self.qweight_packed.shape
        w_int8 = torch.empty((m, half_n * 2), dtype=torch.int8, device=x.device)
        w_int8[:, 0::2] = low
        w_int8[:, 1::2] = high

        # Apply group scales
        w_float = (w_int8.float().view(m, self.in_features // self.group_size, self.group_size) * self.scales.unsqueeze(-1)).view(m, self.in_features).to(x.dtype)

        # Baseline linear projection
        out_base = F.linear(x_rot, w_float)

        # 3. Residual SVD Compensation: (X' · B^T) · A^T
        # B is [rank, N], A is [M, rank]
        x_svd = F.linear(x_rot, self.svd_b.to(x.dtype))  # [..., rank]
        out_res = F.linear(x_svd, self.svd_a.to(x.dtype))  # [..., M]

        return out_base + out_res

class MiniCPMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)

class MiniCPMAttention(nn.Module):
    def __init__(self, config: MiniCPMHadamardConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads

        rank = config.k_proj_rank
        # Key projection sensitivity defense (r = 32)
        self.k_proj = HadamardLinear4bit(self.hidden_size, self.num_key_value_heads * self.head_dim, group_size=config.group_size, rank=rank)
        self.q_proj = HadamardLinear4bit(self.hidden_size, self.num_heads * self.head_dim, group_size=config.group_size, rank=config.residual_rank)
        self.v_proj = HadamardLinear4bit(self.hidden_size, self.num_key_value_heads * self.head_dim, group_size=config.group_size, rank=config.residual_rank)
        self.o_proj = HadamardLinear4bit(self.num_heads * self.head_dim, self.hidden_size, group_size=config.group_size, rank=config.residual_rank)

        self.kv_bss = KVBSSAttentionHook(tau_focus=config.tau_focus, haze_floor_margin=config.haze_floor_margin)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None) -> torch.Tensor:
        b, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        attn_out = self.kv_bss(q, k, v, attention_mask=attention_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, seq_len, -1)
        out = self.o_proj(attn_out)
        return out

class MiniCPMMLP(nn.Module):
    def __init__(self, config: MiniCPMHadamardConfig):
        super().__init__()
        self.gate_proj = HadamardLinear4bit(config.hidden_size, config.intermediate_size, group_size=config.group_size, rank=config.residual_rank)
        self.up_proj = HadamardLinear4bit(config.hidden_size, config.intermediate_size, group_size=config.group_size, rank=config.residual_rank)
        self.down_proj = HadamardLinear4bit(config.intermediate_size, config.hidden_size, group_size=config.group_size, rank=config.residual_rank)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class MiniCPMDecoderLayer(nn.Module):
    def __init__(self, config: MiniCPMHadamardConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MiniCPMAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MiniCPMMLP(config)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class MiniCPMHadamardForCausalLM(PreTrainedModel):
    config_class = MiniCPMHadamardConfig

    def __init__(self, config: MiniCPMHadamardConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MiniCPMDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ])
        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return CausalLMOutputWithPast(logits=logits)
