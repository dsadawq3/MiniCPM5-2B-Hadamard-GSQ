#!/usr/bin/env python3
"""
================================================================================
F-LABS QUANTIZATION ENGINE: MiniCPM5-2B-Hadamard-GSQ
DV-SSQ (Dense-Vectorized Subspace Salience Quantization)
Walsh-Hadamard Spin Rotation + GSQ INT4 + RCO (Residual SVD) + Zero-Compression Shield
================================================================================
"""

import os
import gc
import json
import math
import time
import shutil
import torch
import numpy as np
from safetensors.torch import load_file, save_file

RAW_MODEL_DIR = r"C:\Users\PC MOD\Desktop\minicpm_hadamard_quant\raw_model"
QUANT_MODEL_DIR = r"C:\Users\PC MOD\Desktop\minicpm_hadamard_quant\quantized_model"

# Hyperparameters
GROUP_SIZE = 64
DEFAULT_RANK = 16
BIFURCATION_RANK = 24
K_PROJ_RANK = 32
BIFURCATION_LAYERS = set(range(14, 28))  # Deep reasoning abstraction circuits

def generate_hadamard_matrix(n: int) -> torch.Tensor:
    """Generate normalized orthonormal Walsh-Hadamard matrix H_n such that H^T H = I."""
    if n == 1:
        return torch.tensor([[1.0]], dtype=torch.float32)
    h_half = generate_hadamard_matrix(n // 2)
    top = torch.cat([h_half, h_half], dim=1)
    bottom = torch.cat([h_half, -h_half], dim=1)
    h = torch.cat([top, bottom], dim=0) / math.sqrt(2.0)
    return h

# Precompute H_128
H_128 = generate_hadamard_matrix(128)

def apply_block_hadamard(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply block-diagonal Walsh-Hadamard transform along specified dimension.
    Tensor shape along 'dim' must be a multiple of 128.
    """
    orig_shape = tensor.shape
    d = orig_shape[dim]
    assert d % 128 == 0, f"Dimension {d} is not divisible by 128"
    
    # Reshape so that the target dimension is split into (blocks, 128)
    if dim == -1 or dim == len(orig_shape) - 1:
        reshaped = tensor.view(-1, d // 128, 128)
        # Multiply each 128-vector by H_128 (128x128)
        device = tensor.device
        h = H_128.to(dtype=tensor.dtype, device=device)
        rotated = torch.matmul(reshaped, h)
        return rotated.view(orig_shape)
    elif dim == 0:
        reshaped = tensor.view(d // 128, 128, -1)
        device = tensor.device
        h = H_128.to(dtype=tensor.dtype, device=device)
        rotated = torch.matmul(h.t(), reshaped)
        return rotated.view(orig_shape)
    else:
        raise NotImplementedError(f"Block Hadamard along dim {dim} not supported")

def quantize_gsq_int4_with_svd(weight: torch.Tensor, group_size: int = 64, rank: int = 16):
    """
    Applies Group-Scale INT4 Quantization with Low-Rank Residual SVD Compensation (RCO).
    Returns:
      q_weight: int8 tensor (values -8 to 7, packed or uint8)
      scales: float16 scales per group
      svd_u: float16 low-rank factor A = U * sqrt(S) [M, rank]
      svd_v: float16 low-rank factor B = sqrt(S) * V^T [rank, N]
    """
    orig_dtype = weight.dtype
    m, n = weight.shape
    assert n % group_size == 0, f"n={n} not divisible by group_size={group_size}"
    
    # 1. Group-Scale Quantization
    w_grouped = weight.float().view(m, n // group_size, group_size)
    max_abs = torch.amax(torch.abs(w_grouped), dim=-1, keepdim=True).clamp(min=1e-5)
    scale = max_abs / 7.0
    
    q_grouped = torch.clamp(torch.round(w_grouped / scale), -8, 7).to(torch.int8)
    w_dequant = (q_grouped.float() * scale).view(m, n)
    
    # 2. Residual computation
    residual = weight.float() - w_dequant
    
    # 3. Truncated Low-Rank SVD on residual
    # Using torch.linalg.svd (compact)
    try:
        u, s, vh = torch.linalg.svd(residual, full_matrices=False)
        u_r = u[:, :rank]
        s_r = s[:rank]
        vh_r = vh[:rank, :]
        
        sqrt_s = torch.sqrt(s_r.clamp(min=1e-7))
        factor_a = (u_r * sqrt_s.unsqueeze(0)).to(torch.bfloat16)
        factor_b = (sqrt_s.unsqueeze(1) * vh_r).to(torch.bfloat16)
    except Exception as e:
        print(f"SVD fallback: {e}")
        factor_a = torch.zeros((m, rank), dtype=torch.bfloat16)
        factor_b = torch.zeros((rank, n), dtype=torch.bfloat16)
        
    return q_grouped.view(m, n), scale.squeeze(-1).to(torch.bfloat16), factor_a, factor_b

def process_model():
    print("=" * 70, flush=True)
    print("Starting F-Labs MiniCPM5-2B Quantization Pipeline", flush=True)
    print("=" * 70, flush=True)
    
    raw_safetensor = os.path.join(RAW_MODEL_DIR, "model-00000-of-00001.safetensors")
    if not os.path.exists(raw_safetensor):
        # Check if single model.safetensors exists
        alt_path = os.path.join(RAW_MODEL_DIR, "model.safetensors")
        if os.path.exists(alt_path):
            raw_safetensor = alt_path
        else:
            raise FileNotFoundError(f"Cannot find safetensors weights in {RAW_MODEL_DIR}")
            
    print(f"Loading raw tensors from {raw_safetensor} ...", flush=True)
    t0 = time.time()
    tensors = load_file(raw_safetensor)
    print(f"Loaded {len(tensors)} tensors in {time.time() - t0:.2f}s", flush=True)
    
    quant_tensors = {}
    total_original_bytes = 0
    total_quantized_bytes = 0
    
    # Layer tracking
    for name, param in tensors.items():
        param_bytes = param.numel() * param.element_size()
        total_original_bytes += param_bytes
        
        # -------------------------------------------------------------
        # PILLAR 2: ZERO-COMPRESSION SHIELD (Norms, Biases, Embeddings)
        # -------------------------------------------------------------
        if (
            "layernorm" in name
            or "norm" in name
            or "embed_tokens" in name
            or "lm_head" in name
            or "bias" in name
        ):
            # Pristine BF16 retention
            quant_tensors[name] = param.to(torch.bfloat16)
            total_quantized_bytes += quant_tensors[name].numel() * 2
            print(f"[SHIELD-BF16] {name}: {list(param.shape)} -> Uncompressed", flush=True)
            continue
            
        # Parse layer index if present
        layer_idx = None
        for part in name.split("."):
            if part.isdigit():
                layer_idx = int(part)
                break
                
        # Determine SVD rank
        if "k_proj" in name:
            rank = K_PROJ_RANK  # Pillar 3: Key-projection sensitivity defense
        elif layer_idx is not None and layer_idx in BIFURCATION_LAYERS:
            rank = BIFURCATION_RANK  # Pillar 4: Bifurcation Layer
        else:
            rank = DEFAULT_RANK
            
        # -------------------------------------------------------------
        # PILLAR 1: WALSH-HADAMARD SPIN ROTATION (Outlier Suppression)
        # -------------------------------------------------------------
        w = param.clone()
        m, n = w.shape
        
        # Apply input rotation W' = W * H^T along dim 1 (input dimension)
        if n % 128 == 0:
            w_rotated = apply_block_hadamard(w, dim=-1)
        else:
            w_rotated = w
            
        # -------------------------------------------------------------
        # PILLAR 5: INT4 GSQ + RESIDUAL SVD DECOMPOSITION (RCO)
        # -------------------------------------------------------------
        q_w, scales, factor_a, factor_b = quantize_gsq_int4_with_svd(
            w_rotated, group_size=GROUP_SIZE, rank=rank
        )
        
        # Store components
        base_name = name.replace(".weight", "")
        quant_tensors[f"{base_name}.qweight"] = q_w
        quant_tensors[f"{base_name}.scales"] = scales
        quant_tensors[f"{base_name}.svd_a"] = factor_a
        quant_tensors[f"{base_name}.svd_b"] = factor_b
        
        # Calculate size:
        # q_w: 1 byte per int8 (packed effectively 4 bits, but in int8 container = 1 byte)
        # In packed uint8, 2 values per byte = m*n/2 bytes!
        # Let's pack 2 int4 values per uint8 to achieve true 4-bit storage!
        # Packing: low nibble (q_w[:, 0::2] & 0x0F), high nibble ((q_w[:, 1::2] & 0x0F) << 4)
        q_low = q_w[:, 0::2] & 0x0F
        q_high = (q_w[:, 1::2] & 0x0F) << 4
        packed_q = (q_low | q_high).to(torch.uint8)
        
        quant_tensors[f"{base_name}.qweight_packed"] = packed_q
        del quant_tensors[f"{base_name}.qweight"]
        
        q_bytes = packed_q.numel() * 1
        scale_bytes = scales.numel() * 2
        svd_bytes = (factor_a.numel() + factor_b.numel()) * 2
        layer_total = q_bytes + scale_bytes + svd_bytes
        total_quantized_bytes += layer_total
        
        compression_ratio = param_bytes / max(1, layer_total)
        print(f"[DV-SSQ-INT4+SVD(r={rank})] {name}: {list(w.shape)} -> Compressed ({compression_ratio:.2f}x)", flush=True)
        
    # Free memory
    del tensors
    gc.collect()
    
    # Save quantized shards (split into ~2 GB shards for HF safety)
    print("=" * 70, flush=True)
    print(f"Saving quantized tensors into {QUANT_MODEL_DIR} ...", flush=True)
    os.makedirs(QUANT_MODEL_DIR, exist_ok=True)
    
    # Split tensors into shards of max 2.0 GB
    shard_size_limit = 2 * 1024 * 1024 * 1024  # 2 GB
    current_shard = {}
    current_bytes = 0
    shard_idx = 1
    shard_files = []
    weight_map = {}
    
    for k, v in quant_tensors.items():
        v_bytes = v.numel() * v.element_size()
        if current_bytes + v_bytes > shard_size_limit and len(current_shard) > 0:
            shard_name = f"model-{shard_idx:05d}-of-00002.safetensors"
            shard_path = os.path.join(QUANT_MODEL_DIR, shard_name)
            print(f"Writing {shard_name} ({current_bytes / 1024**3:.2f} GB) ...", flush=True)
            save_file(current_shard, shard_path)
            shard_files.append(shard_name)
            for tk in current_shard:
                weight_map[tk] = shard_name
            current_shard = {}
            current_bytes = 0
            shard_idx += 1
            
        current_shard[k] = v
        current_bytes += v_bytes
        
    if current_shard:
        total_shards = shard_idx
        shard_name = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
        shard_path = os.path.join(QUANT_MODEL_DIR, shard_name)
        print(f"Writing {shard_name} ({current_bytes / 1024**3:.2f} GB) ...", flush=True)
        save_file(current_shard, shard_path)
        shard_files.append(shard_name)
        for tk in current_shard:
            weight_map[tk] = shard_name
            
    # Fix names if total_shards changed
    total_shards = len(shard_files)
    final_weight_map = {}
    for i, sf in enumerate(shard_files, 1):
        old_path = os.path.join(QUANT_MODEL_DIR, sf)
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        new_path = os.path.join(QUANT_MODEL_DIR, new_name)
        if old_path != new_path:
            shutil.move(old_path, new_path)
        for tk, mapped_sf in weight_map.items():
            if mapped_sf == sf:
                final_weight_map[tk] = new_name
                
    # Save index
    index_data = {
        "metadata": {
            "total_size": total_quantized_bytes,
            "quantization": "DV-SSQ-Hadamard-GSQ",
            "spin_rotation": "Walsh-Hadamard-H128",
            "group_size": GROUP_SIZE,
            "bifurcation_rank": BIFURCATION_RANK,
            "k_proj_rank": K_PROJ_RANK,
            "zero_compression_shield": "RMSNorms+Biases+Embeddings",
            "kv_bss": True
        },
        "weight_map": final_weight_map
    }
    index_path = os.path.join(QUANT_MODEL_DIR, "model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)
    print(f"Wrote index with {len(final_weight_map)} tensors to {index_path}", flush=True)
    
    # Copy and update companion files
    companion_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja"
    ]
    for cf in companion_files:
        src = os.path.join(RAW_MODEL_DIR, cf)
        dst = os.path.join(QUANT_MODEL_DIR, cf)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {cf}", flush=True)
            
    # Update config.json with quantization metadata
    quant_cfg_path = os.path.join(QUANT_MODEL_DIR, "config.json")
    if os.path.exists(quant_cfg_path):
        with open(quant_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["quantization_config"] = {
            "quant_method": "hadamard_gsq",
            "bits": 4,
            "group_size": GROUP_SIZE,
            "hadamard_spin": True,
            "residual_svd_rank": DEFAULT_RANK,
            "k_proj_svd_rank": K_PROJ_RANK,
            "bifurcation_rank": BIFURCATION_RANK,
            "zero_compression_shield": True,
            "kv_bss": {
                "enabled": True,
                "tau_focus": 1.10,
                "haze_floor_margin": 12.0
            }
        }
        with open(quant_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print("Updated config.json with quantization_config", flush=True)
        
    print("=" * 70, flush=True)
    print("QUANTIZATION BENCHMARK SUMMARY:", flush=True)
    print(f"Original Model Size:   {total_original_bytes / 1024**3:.3f} GB", flush=True)
    print(f"Quantized Model Size:  {total_quantized_bytes / 1024**3:.3f} GB", flush=True)
    print(f"Compression Ratio:     {total_original_bytes / total_quantized_bytes:.2f}x", flush=True)
    print(f"Memory Saved:          {(1.0 - total_quantized_bytes / total_original_bytes) * 100:.1f}%", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    process_model()
