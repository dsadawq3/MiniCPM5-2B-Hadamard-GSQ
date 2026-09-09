"""
PyTorch modeling implementation for MiniCPM5-2B-Hadamard-GSQ.
Engineered at F-Labs.

Features:
- HadamardLinear4bit: Group-wise INT4 with Walsh-Hadamard spin and Low-Rank SVD (SRC)
- ZeroCompressionShield: Pure BF16 RMSNorms, Biases, and Embeddings
- Dynamic Bifurcation Layer Rank Allocation (Layers 14-27 with r=bifurcation_rank)
- Key-Projection Sensitivity Defense (r=k_proj_rank on k_proj)
- KVBSSAttentionHook: Key-Value Binding Softmax Sharpening for 128K context
- RoPE (GPT-NeoX style, theta=rope_theta, head_dim=128) + causal mask + KV-cache
- Full compliance with Hugging Face PreTrainedModel standards.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

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
    if d % block_size != 0:
        return x
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

        self.register_buffer("qweight_packed", torch.zeros((out_features, in_features // 2), dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros((out_features, in_features // group_size), dtype=torch.bfloat16))
        self.register_buffer("svd_a", torch.zeros((out_features, rank), dtype=torch.bfloat16))
        self.register_buffer("svd_b", torch.zeros((rank, in_features), dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_features % self.block_size == 0:
            x_rot = apply_hadamard_rot(x, block_size=self.block_size)
        else:
            x_rot = x

        low = (self.qweight_packed & 0x0F).to(torch.int8)
        low = torch.where(low >= 8, low - 16, low)
        high = ((self.qweight_packed >> 4) & 0x0F).to(torch.int8)
        high = torch.where(high >= 8, high - 16, high)

        m, half_n = self.qweight_packed.shape
        w_int8 = torch.empty((m, half_n * 2), dtype=torch.int8, device=x.device)
        w_int8[:, 0::2] = low
        w_int8[:, 1::2] = high

        w_float = (w_int8.float().view(m, self.in_features // self.group_size, self.group_size) * self.scales.unsqueeze(-1)).view(m, self.in_features).to(x.dtype)
        out_base = F.linear(x_rot, w_float)

        x_svd = F.linear(x_rot, self.svd_b.to(x.dtype))
        out_res = F.linear(x_svd, self.svd_a.to(x.dtype))

        return out_base + out_res


class MiniCPMRMSNorm(nn.Module):
    # ZeroCompressionShield: stays BF16, never quantized.
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


class MiniCPMHadamardRotaryEmbedding(nn.Module):
    # GPT-NeoX style RoPE. inv_freq built from rope_theta; cached cos/sin up to max_pos.
    def __init__(self, head_dim: int = 128, rope_theta: float = 5000000.0, max_position_embeddings: int = 131072):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings
        self._cached_cos = None
        self._cached_sin = None
        self._cached_len = 0

    def _build_cache(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        self._cached_cos = emb.cos().to(dtype)
        self._cached_sin = emb.sin().to(dtype)
        self._cached_len = seq_len

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # x: (b, heads, seq, head_dim) — only used for device/dtype.
        seq_len = int(position_ids.max().item()) + 1
        if self._cached_cos is None or seq_len > self._cached_len or self._cached_cos.device != x.device:
            self._build_cache(seq_len, x.device, torch.float32)
        cos = self._cached_cos[position_ids].to(x.dtype)  # (b, seq, head_dim)
        sin = self._cached_sin[position_ids].to(x.dtype)
        return cos.unsqueeze(1), sin.unsqueeze(1)  # (b, 1, seq, head_dim) — broadcast over heads


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # cos/sin: (b, 1, seq, head_dim), broadcast over heads.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


def _make_causal_mask_4d(attention_mask_2d, q_len: int, kv_len: int, device, dtype) -> Optional[torch.Tensor]:
    # Returns additive (b, 1, q_len, kv_len) mask: 0.0 keep / -inf block.
    past_len = kv_len - q_len
    causal = torch.full((q_len, kv_len), torch.finfo(dtype).min, device=device)
    # allow j <= past_len + i
    causal = torch.where(
        torch.ones(q_len, kv_len, device=device, dtype=torch.bool).tril(diagonal=past_len),
        torch.zeros((), device=device, dtype=dtype),
        causal.to(dtype),
    )  # (q, kv)
    mask_4d = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, q, kv)
    if attention_mask_2d is not None:
        am = attention_mask_2d.to(dtype)  # (b, kv_len)
        additive = (1.0 - am) * torch.finfo(dtype).min  # pad -> -inf
        mask_4d = mask_4d + additive[:, None, None, :]
    return mask_4d


class MiniCPMAttention(nn.Module):
    def __init__(self, config: MiniCPMHadamardConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads

        is_bifurcation = 14 <= layer_idx <= 27
        res_rank = getattr(config, "bifurcation_rank", 24) if is_bifurcation else getattr(config, "residual_rank", 16)
        k_rank = getattr(config, "k_proj_rank", 32)

        self.k_proj = HadamardLinear4bit(self.hidden_size, self.num_key_value_heads * self.head_dim, group_size=config.group_size, rank=k_rank)
        self.q_proj = HadamardLinear4bit(self.hidden_size, self.num_heads * self.head_dim, group_size=config.group_size, rank=res_rank)
        self.v_proj = HadamardLinear4bit(self.hidden_size, self.num_key_value_heads * self.head_dim, group_size=config.group_size, rank=res_rank)
        self.o_proj = HadamardLinear4bit(self.num_heads * self.head_dim, self.hidden_size, group_size=config.group_size, rank=res_rank)

        self.kv_bss = KVBSSAttentionHook(tau_focus=config.tau_focus, haze_floor_margin=config.haze_floor_margin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # 4D additive, built by model
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,  # DynamicCache or tuple[(k,v)]
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        rotary_emb: Optional[MiniCPMHadamardRotaryEmbedding] = None,
    ) -> Tuple[torch.Tensor, object]:
        b, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # RoPE on fresh q/k only (past already rotated).
        if rotary_emb is not None and position_ids is not None:
            cos, sin = rotary_emb(q, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Append KV cache.
        if past_key_values is not None:
            if hasattr(past_key_values, "update"):  # transformers DynamicCache / Cache
                k, v = past_key_values.update(k, v, self.layer_idx)
            else:  # legacy tuple / list
                pk, pv = past_key_values[self.layer_idx] if past_key_values[self.layer_idx] is not None else (None, None)
                if pk is not None:
                    k = torch.cat([pk, k], dim=2)
                    v = torch.cat([pv, v], dim=2)
                    past_key_values[self.layer_idx] = (k, v)
                elif use_cache:
                    past_key_values[self.layer_idx] = (k, v)
        attn_out = self.kv_bss(q, k, v, attention_mask=attention_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, q_len, -1)
        out = self.o_proj(attn_out)
        return out, past_key_values


class MiniCPMMLP(nn.Module):
    def __init__(self, config: MiniCPMHadamardConfig, layer_idx: int = 0):
        super().__init__()
        is_bifurcation = 14 <= layer_idx <= 27
        res_rank = getattr(config, "bifurcation_rank", 24) if is_bifurcation else getattr(config, "residual_rank", 16)

        self.gate_proj = HadamardLinear4bit(config.hidden_size, config.intermediate_size, group_size=config.group_size, rank=res_rank)
        self.up_proj = HadamardLinear4bit(config.hidden_size, config.intermediate_size, group_size=config.group_size, rank=res_rank)
        self.down_proj = HadamardLinear4bit(config.intermediate_size, config.hidden_size, group_size=config.group_size, rank=res_rank)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniCPMDecoderLayer(nn.Module):
    def __init__(self, config: MiniCPMHadamardConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MiniCPMAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MiniCPMMLP(config, layer_idx=layer_idx)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, cache_position=None, rotary_emb=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_values = self.self_attn(
            hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache,
            cache_position=cache_position, rotary_emb=rotary_emb,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, past_key_values


class MiniCPMHadamardPreTrainedModel(PreTrainedModel):
    config_class = MiniCPMHadamardConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["MiniCPMDecoderLayer"]

    def _init_weights(self, module):
        pass


class MiniCPMHadamardModel(MiniCPMHadamardPreTrainedModel):
    def __init__(self, config: MiniCPMHadamardConfig):
        super().__init__(config)
        self.padding_idx = getattr(config, "pad_token_id", 1)
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            MiniCPMDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ])
        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MiniCPMHadamardRotaryEmbedding(
            head_dim=config.head_dim, rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=None, cache_position=None,
                inputs_embeds=None, **kwargs):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if inputs_embeds is not None:
            x = inputs_embeds
            b, q_len = x.shape[:2]
            device = x.device
        else:
            b, q_len = input_ids.shape
            device = input_ids.device
            x = self.embed_tokens(input_ids)

        # Past length for position ids / mask.
        if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
            past_len = past_key_values.get_seq_length()
        elif past_key_values is not None and isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[2]
        else:
            past_len = 0

        # NOTE (transformers>=5 compat): generate() passes trimmed input_ids with
        # FULL-length position_ids (and sometimes a stale cache_position). Trusting
        # them blindly lets RoPE broadcast-expand the query length, which breaks
        # KV-BSS masking (scores kv != mask kv). Rebuild whenever shapes disagree.
        if cache_position is None or cache_position.shape[0] != q_len:
            cache_position = torch.arange(past_len, past_len + q_len, device=device)
        if position_ids is None or position_ids.shape[-1] != q_len:
            position_ids = cache_position.unsqueeze(0).expand(b, -1)

        kv_len = past_len + q_len
        mask_4d = None
        if q_len > 1 or attention_mask is not None:
            # Full 2D mask over kv window if user passed one, else causal only.
            am_2d = None
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    mask_4d = attention_mask
                elif attention_mask.dim() == 2:
                    am_2d = attention_mask
            if mask_4d is None:
                mask_4d = _make_causal_mask_4d(am_2d, q_len, kv_len, device, x.dtype)

        # Legacy tuple cache init on first use_cache call.
        if use_cache and past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers

        for layer in self.layers:
            x, past_key_values = layer(
                x, attention_mask=mask_4d, position_ids=position_ids,
                past_key_values=past_key_values, use_cache=use_cache,
                cache_position=cache_position, rotary_emb=self.rotary_emb,
            )
        x = self.norm(x)
        return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=past_key_values)


class MiniCPMHadamardForCausalLM(MiniCPMHadamardPreTrainedModel, GenerationMixin):
    def __init__(self, config: MiniCPMHadamardConfig):
        super().__init__(config)
        self.model = MiniCPMHadamardModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=None, cache_position=None,
                inputs_embeds=None, labels=None, **kwargs):
        hidden = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache,
            cache_position=cache_position, inputs_embeds=inputs_embeds, **kwargs,
        )
        logits = self.lm_head(hidden.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=hidden.past_key_values,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # Trim to last token when cache is active.
        if past_key_values is not None:
            if isinstance(past_key_values, (list, tuple)):
                past_len = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
            elif hasattr(past_key_values, "get_seq_length"):
                past_len = past_key_values.get_seq_length()
            else:
                past_len = 0
            if input_ids.shape[1] > past_len:
                input_ids = input_ids[:, past_len:]
        cache_position = kwargs.get("cache_position", None)
        return {"input_ids": input_ids, "past_key_values": past_key_values,
                "attention_mask": attention_mask, "cache_position": cache_position,
                "position_ids": kwargs.get("position_ids", None), "use_cache": True}
