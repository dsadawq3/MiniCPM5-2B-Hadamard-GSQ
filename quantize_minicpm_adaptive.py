#!/usr/bin/env python3
"""Build the calibration-aware MiniCPM5-2B FQuant research release.

The command intentionally writes a new directory.  It never overwrites the
original Hadamard-GSQ release, because the bifurcation detector and the
diagonal-Hessian approximation are research hypotheses that need a direct
comparison against the established artifact.

The pipeline has two passes over the source safetensors:

1. run deterministic multilingual/code calibration through the BF16 base model
   and estimate H ~= diag(E[x^2]) for each linear input;
2. select weighted INT4 scales, measure residual/spectral risk, allocate L0-L3
   ranks, then quantize and write the final shards.

The algebra is paired with the loader's input transform:
    x' = x D H, W' = W D H, D^2 = I, H H^T = I.
Thus the unquantized linear map is unchanged.  The approximation objective is
the diagonal-Hessian proxy tr((W-W_hat) H (W-W_hat)^T).
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
FQUANT_ROOT = HERE.parent / "FQuant"
if str(FQUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(FQUANT_ROOT))

from fquant.adaptive import (  # noqa: E402
    DEFAULT_RANK_BY_LEVEL,
    DEFAULT_SCALE_FACTORS,
    detect_bifurcation_levels,
    residual_spectral_spike,
    rotate_input_activation,
    rotate_input_weight,
    weighted_groupwise_int4,
    weighted_groupwise_int8,
    weighted_randomized_svd,
)
from fquant.gsq import pack_int4_to_uint8  # noqa: E402


CALIBRATION_PROMPTS = (
    "Explain why a diagonal Hessian approximation can be useful for post-training quantization, and derive the local linear error objective.",
    "请用中文解释长上下文 Transformer 中的注意力、旋转量化以及 KV cache 之间的关系。",
    "Разбери по шагам, как проверить причинную маску и KV-кэш в авторегрессионной языковой модели.",
    "Write a small Python function that parses a safetensors index and verifies every mapped key exists in exactly one shard.",
    "Consider a matrix W with groupwise INT4 error R. Compare Frobenius low-rank approximation with an activation-weighted approximation.",
    "Дай строгий план эксперимента для сравнения BF16, обычного INT4 и адаптивного INT4 с одинаковыми токенами и метриками.",
    "Provide a concise code review of a transformer quantizer that uses Hadamard rotations, per-group scales, and low-rank residual factors.",
    "请给出一个关于谱尖ка、鲁棒 z 分数、一阶跳变和二阶曲率的可复现实验定义。",
)

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
SHARD_LIMIT = 2 * 1024**3


def _dtype_load_kwargs() -> Dict[str, object]:
    """Transformers 4/5 compatibility for the renamed dtype argument."""
    try:
        import transformers

        major = int(str(transformers.__version__).split(".", 1)[0])
    except Exception:
        major = 4
    return {"dtype": torch.bfloat16} if major >= 5 else {"torch_dtype": torch.bfloat16}


def _layer_idx(name: str) -> Optional[int]:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _is_shielded(name: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim != 2:
        return True
    lowered = name.lower()
    return any(
        marker in lowered
        for marker in ("norm", "embed_tokens", "lm_head", "bias")
    )


def _moment_for(name: str, moments: Mapping[str, torch.Tensor]):
    """Map a safetensors ``.weight`` key to its module-hook key."""
    value = moments.get(name)
    return value if value is not None else moments.get(name.removesuffix(".weight"))


def _stable_seed(name: str, base_seed: int) -> int:
    # Python's hash() is process-randomized; this checksum is stable in reports.
    checksum = sum((index + 1) * ord(char) for index, char in enumerate(name))
    return int(base_seed) + checksum % 100_000


def _load_base_model(raw_dir: Path):
    kwargs = dict(_dtype_load_kwargs())
    kwargs.update(low_cpu_mem_usage=True, trust_remote_code=True)
    try:
        return AutoModelForCausalLM.from_pretrained(str(raw_dir), **kwargs)
    except TypeError:
        # A narrow fallback for older Transformers builds that reject dtype.
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return AutoModelForCausalLM.from_pretrained(str(raw_dir), **kwargs)


def collect_activation_second_moments(
    raw_dir: Path,
    *,
    seed: int,
    max_tokens: int,
    prompts: Iterable[str] = CALIBRATION_PROMPTS,
) -> Dict[str, torch.Tensor]:
    """Collect rotated activation second moments using deterministic hooks."""
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(str(raw_dir), trust_remote_code=True)
    model = _load_base_model(raw_dir)
    model.eval()

    totals: MutableMapping[str, torch.Tensor] = {}
    counts: MutableMapping[str, int] = defaultdict(int)
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            values = inputs[0].detach()
            if values.ndim == 0 or values.shape[-1] == 0:
                return
            values = rotate_input_activation(
                values,
                block_size=128,
                seed=seed,
            ).float().reshape(-1, values.shape[-1])
            totals[name] = totals.get(name, torch.zeros(values.shape[-1])) + values.square().sum(0).cpu()
            counts[name] += int(values.shape[0])

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    used_prompts = []
    backbone = getattr(model, "model", model)
    try:
        with torch.no_grad():
            for prompt in prompts:
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_tokens,
                )
                # The lm_head produces a 130k-way logit tensor that is not
                # needed for calibration.  Running the backbone directly
                # preserves every internal activation hook and avoids that
                # avoidable memory and matmul cost.
                backbone(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                    use_cache=False,
                )
                used_prompts.append(prompt)
    finally:
        for handle in handles:
            handle.remove()
        del backbone, model, tokenizer
        gc.collect()

    moments = {
        name: (total / max(counts[name], 1)).to(torch.float32)
        for name, total in totals.items()
    }
    if not moments:
        raise RuntimeError("Calibration collected no linear-layer activations")
    return moments


def _open_weight_files(raw_dir: Path):
    index_path = raw_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        filenames = sorted(set(index["weight_map"].values()))
    else:
        filenames = ["model.safetensors"]
    paths = [raw_dir / name for name in filenames]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source shards: " + ", ".join(missing))
    return [safe_open(str(path), framework="pt", device="cpu") for path in paths]


def _iter_source_tensors(raw_dir: Path):
    handles = _open_weight_files(raw_dir)
    try:
        for handle in handles:
            for name in handle.keys():
                yield name, handle.get_tensor(name)
    finally:
        for handle in handles:
            handle.__exit__(None, None, None)


def _metric_pass(
    raw_dir: Path,
    moments: Mapping[str, torch.Tensor],
    *,
    block_size: int,
    group_size: int,
    seed: int,
) -> Dict[str, object]:
    """Measure quantization risk and allocate ranks without retaining weights."""
    by_layer = defaultdict(list)
    tensor_metrics = {}
    original_bytes = 0
    for name, tensor in _iter_source_tensors(raw_dir):
        original_bytes += tensor.numel() * tensor.element_size()
        if _is_shielded(name, tensor):
            continue
        h = _moment_for(name, moments)
        rotated = rotate_input_weight(tensor.float(), block_size=block_size, seed=seed)
        _, _, reconstruction, metrics = weighted_groupwise_int4(
            rotated,
            h,
            group_size=group_size,
            scale_factors=DEFAULT_SCALE_FACTORS,
        )
        residual = rotated - reconstruction
        h_for_metrics = h if h is not None else torch.ones(rotated.shape[1])
        h_for_metrics = h_for_metrics.float().clamp_min(1e-12)
        activation_peak = float((h_for_metrics.max() / h_for_metrics.mean()).item())
        rms = torch.sqrt(tensor.float().square().mean()).clamp_min(1e-12)
        weight_outlier = float((tensor.float().abs().max() / rms).item())
        row = {
            "tensor": name,
            "layer_idx": _layer_idx(name),
            **metrics,
            "spectral_spike": residual_spectral_spike(
                residual, h, iterations=3, seed=_stable_seed(name, seed)
            ),
            "activation_peak": activation_peak,
            "weight_outlier": weight_outlier,
        }
        tensor_metrics[name] = row
        if row["layer_idx"] is not None:
            by_layer[row["layer_idx"]].append(row)
        del rotated, reconstruction, residual

    layer_metrics = []
    for layer_idx in sorted(by_layer):
        rows = by_layer[layer_idx]
        numeric_keys = (
            "weighted_relative_error",
            "unweighted_relative_error",
            "spectral_spike",
            "activation_peak",
            "weight_outlier",
            "mean_scale_factor",
            "clipped_group_fraction",
        )
        summary = {key: sum(float(row[key]) for row in rows) / len(rows) for key in numeric_keys}
        summary["layer_idx"] = layer_idx
        summary["tensor_count"] = len(rows)
        layer_metrics.append(summary)

    bifurcation = detect_bifurcation_levels(
        layer_metrics,
        rank_by_level=DEFAULT_RANK_BY_LEVEL,
    )
    layer_levels = {
        int(row["layer_idx"]): int(row["bifurcation_level"])
        for row in bifurcation["layers"]
    }
    # Attention errors are amplified by the softmax and the GQA fan-out.  The
    # MLP down projection is the residual write-back point, so it stays dense
    # for every layer. Gate/up projections at L2/L3 are also dense because the
    # gated product makes their independent quantization errors interact.
    # Remaining gate/up matrices stay in the adaptive INT4 path.
    dense_names = []
    int8_names = []
    for name, row in tensor_metrics.items():
        layer_idx = row["layer_idx"]
        base_name = name.removesuffix(".weight")
        if ".self_attn." in base_name:
            dense_names.append(base_name)
        elif ".mlp.down_proj" in base_name:
            dense_names.append(base_name)
        elif (".mlp.gate_proj" in base_name or ".mlp.up_proj" in base_name) and layer_levels.get(layer_idx, 0) >= 2:
            dense_names.append(base_name)
        elif ".mlp.gate_proj" in base_name or ".mlp.up_proj" in base_name:
            int8_names.append(base_name)
    return {
        "original_bytes": original_bytes,
        "layer_metrics": layer_metrics,
        "bifurcation": bifurcation,
        "tensor_metrics": tensor_metrics,
        "dense_tensor_names": sorted(dense_names),
        "int8_tensor_names": sorted(int8_names),
    }


def _updated_config(raw_dir: Path, report: Mapping[str, object], *, seed: int, group_size: int, block_size: int):
    cfg = json.loads((raw_dir / "config.json").read_text(encoding="utf-8"))
    bifurcation = report["bifurcation"]
    rank_map = bifurcation["rank_by_layer"]
    cfg.update(
        {
            "architectures": ["MiniCPMHadamardForCausalLM"],
            "model_type": "minicpm_hadamard",
            "group_size": group_size,
            "hadamard_block_size": block_size,
            "residual_rank": 16,
            "bifurcation_rank": 24,
            "k_proj_rank": 32,
            "layer_rank_map": rank_map,
            "dense_tensor_names": list(report["dense_tensor_names"]),
            "int8_tensor_names": list(report["int8_tensor_names"]),
            "rotation_mode": "rademacher_hadamard",
            "rotation_seed": seed,
            "auto_map": {
                "AutoConfig": "configuration_minicpm_hadamard.MiniCPMHadamardConfig",
                "AutoModelForCausalLM": "modeling_minicpm_hadamard.MiniCPMHadamardForCausalLM",
            },
            "fquant_quantization_config": {
                "quant_method": "adaptive_diag_hessian_mse_svd",
                "bits": 4,
                "group_size": group_size,
                "hadamard_block_size": block_size,
                "rotation_mode": "rademacher_hadamard",
                "rotation_seed": seed,
                "scale_factors": list(DEFAULT_SCALE_FACTORS),
                "rank_by_level": {str(k): int(v) for k, v in DEFAULT_RANK_BY_LEVEL.items()},
                "bifurcation_levels": bifurcation,
                "dense_tensor_names": list(report["dense_tensor_names"]),
                "dense_policy": "all_attention_plus_all_mlp_down_plus_L2_L3_mlp_gate_up",
                "int8_tensor_names": list(report["int8_tensor_names"]),
                "int8_policy": "remaining_mlp_gate_up_proj",
                "zero_compression_shield": True,
                "kv_bss": {
                    "enabled": True,
                    "tau_focus": 1.10,
                    "haze_floor_margin": 12.0,
                },
            },
        }
    )
    return cfg


def _copy_companions(raw_dir: Path, out_dir: Path):
    for path in raw_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() in {".json", ".jinja", ".model", ".txt", ".md"}:
            shutil.copy2(path, out_dir / path.name)
    for name in ("configuration_minicpm_hadamard.py", "modeling_minicpm_hadamard.py", "kv_bss.py", "LICENSE"):
        shutil.copy2(HERE / name, out_dir / name)
    shutil.copy2(HERE / "quantize_minicpm_adaptive.py", out_dir / "quantize_minicpm_adaptive.py")
    release_readme = HERE / "ADAPTIVE_RELEASE.md"
    if release_readme.exists():
        shutil.copy2(release_readme, out_dir / "README.md")
        shutil.copy2(release_readme, out_dir / "ADAPTIVE_RELEASE.md")


def _write_shard(shard: Mapping[str, torch.Tensor], path: Path):
    save_file(dict(shard), str(path), metadata={"format": "pt"})


def _quantize_and_write(
    raw_dir: Path,
    out_dir: Path,
    moments: Mapping[str, torch.Tensor],
    report: MutableMapping[str, object],
    *,
    seed: int,
    group_size: int,
    block_size: int,
    shard_limit: int,
):
    rank_by_layer = report["bifurcation"]["rank_by_layer"]
    dense_names = set(report.get("dense_tensor_names", ()))
    int8_names = set(report.get("int8_tensor_names", ()))
    quantized_bytes = 0
    shard = {}
    shard_bytes = 0
    temp_shards = []

    def flush():
        nonlocal shard, shard_bytes
        if not shard:
            return
        path = out_dir / f".model-{len(temp_shards) + 1:05d}-of-temp.safetensors"
        _write_shard(shard, path)
        temp_shards.append(path)
        shard = {}
        shard_bytes = 0

    def add(name: str, value: torch.Tensor):
        nonlocal shard_bytes, quantized_bytes
        value = value.contiguous().cpu()
        size = value.numel() * value.element_size()
        if shard and shard_bytes + size > shard_limit:
            flush()
        shard[name] = value
        shard_bytes += size
        quantized_bytes += size

    for name, tensor in _iter_source_tensors(raw_dir):
        if _is_shielded(name, tensor):
            add(name, tensor.to(torch.bfloat16))
            continue

        base_name = name.removesuffix(".weight")
        if base_name in dense_names:
            dense_weight = rotate_input_weight(
                tensor.float(), block_size=block_size, seed=seed
            ).to(torch.bfloat16)
            add(f"{base_name}.weight_bf16", dense_weight)
            del dense_weight
            continue

        layer_idx = _layer_idx(name)
        if "k_proj" in name:
            rank = 32
        elif layer_idx is not None:
            rank = int(rank_by_layer.get(str(layer_idx), 16))
        else:
            rank = 16

        rotated = rotate_input_weight(tensor.float(), block_size=block_size, seed=seed)
        h = _moment_for(name, moments)
        if base_name in int8_names:
            q, scales, reconstruction, quant_metrics = weighted_groupwise_int8(
                rotated, h, group_size=group_size
            )
            q_key = f"{base_name}.qweight_int8"
            scale_key = f"{base_name}.scales_int8"
        else:
            q, scales, reconstruction, quant_metrics = weighted_groupwise_int4(
                rotated,
                h,
                group_size=group_size,
                scale_factors=DEFAULT_SCALE_FACTORS,
            )
            q_key = f"{base_name}.qweight_packed"
            scale_key = f"{base_name}.scales"
        residual = rotated - reconstruction
        factor_a, factor_b, svd_metrics = weighted_randomized_svd(
            residual,
            h,
            rank,
            oversample=4,
            niter=1,
            seed=_stable_seed(name, seed),
        )
        base_name = name.removesuffix(".weight")
        add(q_key, q if base_name in int8_names else pack_int4_to_uint8(q))
        add(scale_key, scales)
        add(f"{base_name}.svd_a", factor_a)
        add(f"{base_name}.svd_b", factor_b)
        report.setdefault("quantized_tensor_metrics", {})[name] = {
            "layer_idx": layer_idx,
            "rank": rank,
            "precision": "int8" if base_name in int8_names else "int4",
            **quant_metrics,
            "weighted_capture": svd_metrics["weighted_capture"],
        }
        del rotated, reconstruction, residual, q, scales, factor_a, factor_b
    flush()

    total_shards = len(temp_shards)
    weight_map = {}
    for index, temp_path in enumerate(temp_shards, 1):
        final_name = f"model-{index:05d}-of-{total_shards:05d}.safetensors"
        final_path = out_dir / final_name
        temp_path.replace(final_path)
        with safe_open(str(final_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                weight_map[key] = final_name

    report["quantized_bytes"] = quantized_bytes
    report["compression_ratio"] = report["original_bytes"] / max(quantized_bytes, 1)
    index = {
        "metadata": {
            "total_size": quantized_bytes,
            "quantization": "adaptive-diag-hessian-mse-svd",
            "shards": total_shards,
        },
        "weight_map": weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )


def build_release(
    raw_dir: Path,
    out_dir: Path,
    *,
    seed: int = 1729,
    group_size: int = 64,
    block_size: int = 128,
    max_calibration_tokens: int = 96,
    calibration_prompt_count: int = len(CALIBRATION_PROMPTS),
    moments_path: Optional[Path] = None,
) -> Dict[str, object]:
    if raw_dir.resolve() == out_dir.resolve():
        raise ValueError("raw_dir and out_dir must be different")
    if not (raw_dir / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in {raw_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print("[1/4] Collecting activation moments", flush=True)
    if moments_path is None:
        moments = collect_activation_second_moments(
            raw_dir,
            seed=seed,
            max_tokens=max_calibration_tokens,
            prompts=CALIBRATION_PROMPTS[:max(1, min(calibration_prompt_count, len(CALIBRATION_PROMPTS)))],
        )
    else:
        moments = torch.load(moments_path, map_location="cpu", weights_only=True)
        if not isinstance(moments, dict) or not moments:
            raise ValueError(f"Invalid activation moments artifact: {moments_path}")
    torch.save(moments, out_dir / "adaptive_activation_second_moments.pt")
    report: MutableMapping[str, object] = {
        "algorithm": "adaptive_diag_hessian_mse_svd",
        "seed": seed,
        "group_size": group_size,
        "hadamard_block_size": block_size,
        "calibration_prompts": list(CALIBRATION_PROMPTS[:max(1, min(calibration_prompt_count, len(CALIBRATION_PROMPTS)))]),
        "calibration_tokens": max_calibration_tokens,
        "calibration_layer_count": len(moments),
    }

    print("[2/4] Measuring residual risk and detecting bifurcations", flush=True)
    report.update(
        _metric_pass(
            raw_dir,
            moments,
            block_size=block_size,
            group_size=group_size,
            seed=seed,
        )
    )
    print("[3/4] Writing adaptive INT4 shards", flush=True)
    _copy_companions(raw_dir, out_dir)
    (out_dir / "config.json").write_text(
        json.dumps(
            _updated_config(
                raw_dir,
                report,
                seed=seed,
                group_size=group_size,
                block_size=block_size,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    _quantize_and_write(
        raw_dir,
        out_dir,
        moments,
        report,
        seed=seed,
        group_size=group_size,
        block_size=block_size,
        shard_limit=SHARD_LIMIT,
    )
    report["elapsed_seconds"] = time.time() - started
    (out_dir / "adaptive_bifurcation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"[4/4] Done: {report['compression_ratio']:.2f}x, "
        f"{report['quantized_bytes'] / 1024**3:.3f} GiB, "
        f"{report['elapsed_seconds'] / 60:.1f} min",
        flush=True,
    )
    return dict(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max_calibration_tokens", type=int, default=96)
    parser.add_argument("--calibration_prompt_count", type=int, default=len(CALIBRATION_PROMPTS))
    parser.add_argument("--moments_path", type=Path, default=None)
    args = parser.parse_args()
    build_release(
        args.raw_dir,
        args.out_dir,
        seed=args.seed,
        max_calibration_tokens=args.max_calibration_tokens,
        calibration_prompt_count=args.calibration_prompt_count,
        moments_path=args.moments_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
