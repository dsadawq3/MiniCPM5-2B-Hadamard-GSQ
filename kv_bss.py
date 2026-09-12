"""
KV-BSS: Key-Value Binding Softmax Sharpening.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KVBSSAttentionHook(nn.Module):
    def __init__(self, tau_focus: float = 1.10, haze_floor_margin: float = 12.0):
        super().__init__()
        if not math.isfinite(float(tau_focus)) or float(tau_focus) <= 0:
            raise ValueError("tau_focus must be finite and greater than zero")
        if not math.isfinite(float(haze_floor_margin)) or float(haze_floor_margin) < 0:
            raise ValueError("haze_floor_margin must be finite and non-negative")
        self.tau_focus = float(tau_focus)
        self.haze_floor_margin = float(haze_floor_margin)

    @staticmethod
    def _prepare_mask(attention_mask, scores: torch.Tensor):
        """Normalize 2D/4D masks and return additive mask plus hard-valid mask."""
        b, h, q_len, kv_len = scores.shape
        min_value = torch.finfo(scores.dtype).min

        if attention_mask is None:
            shape = (1, 1, 1, kv_len)
            additive = torch.zeros(shape, dtype=scores.dtype, device=scores.device)
            valid = torch.ones(shape, dtype=torch.bool, device=scores.device)
            return additive, valid

        mask = attention_mask.to(device=scores.device)
        if mask.dim() == 2:
            if tuple(mask.shape) != (b, kv_len):
                raise ValueError(
                    "2D attention_mask must have shape "
                    f"({b}, {kv_len}), got {tuple(mask.shape)}"
                )
            valid = mask if mask.dtype == torch.bool else mask > 0
            valid = valid[:, None, None, :]
            additive = torch.where(
                valid,
                torch.zeros((), dtype=scores.dtype, device=scores.device),
                torch.full((), min_value, dtype=scores.dtype, device=scores.device),
            )
            return additive, valid

        if mask.dim() != 4:
            raise ValueError("attention_mask must be None, 2D, or 4D")
        try:
            mask = torch.broadcast_to(mask, (b, h, q_len, kv_len))
        except RuntimeError as exc:
            raise ValueError(
                "4D attention_mask is not broadcastable to "
                f"{tuple(scores.shape)}, got {tuple(mask.shape)}"
            ) from exc

        if mask.dtype == torch.bool:
            valid = mask
            additive = torch.where(
                valid,
                torch.zeros((), dtype=scores.dtype, device=scores.device),
                torch.full((), min_value, dtype=scores.dtype, device=scores.device),
            )
        else:
            additive = mask.to(dtype=scores.dtype)
            if torch.isnan(additive).any() or torch.isposinf(additive).any():
                raise ValueError("attention_mask contains NaN or positive infinity")
            # Hugging Face additive causal masks use finfo.min or -inf for
            # blocked positions. Finite negative biases remain valid scores.
            valid = torch.isfinite(additive) & (additive > min_value / 2)
        return additive, valid

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask=None, scaling=None):
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ValueError("query, key, and value must all be 4D tensors")
        b, h, q_len, d = query.shape
        kb, kv_h, kv_len, kd = key.shape
        vb, value_kv_h, value_kv_len, vd = value.shape
        if (kb, value_kv_h, value_kv_len, vd) != (b, kv_h, kv_len, kd):
            raise ValueError("key and value shapes must match in batch, heads, length, and depth")
        if h == 0 or kv_h == 0 or h % kv_h != 0:
            raise ValueError(f"query heads ({h}) must be a positive multiple of KV heads ({kv_h})")

        if h != kv_h:
            num_repeat = h // kv_h
            key = key.repeat_interleave(num_repeat, dim=1)
            value = value.repeat_interleave(num_repeat, dim=1)

        if scaling is None:
            scaling = 1.0 / math.sqrt(d)
        if not math.isfinite(float(scaling)) or float(scaling) <= 0:
            raise ValueError("scaling must be finite and greater than zero")

        # Attention score accumulation in BF16 can overflow even when most
        # individual activations are finite. Compute the score path in FP32 and
        # isolate invalid query/key positions instead of taking down generation.
        query_valid = torch.isfinite(query).all(dim=-1, keepdim=True)
        key_valid = torch.isfinite(key).all(dim=-1).unsqueeze(-2)
        query_fp32 = torch.nan_to_num(query.float(), nan=0.0, posinf=0.0, neginf=0.0)
        key_fp32 = torch.nan_to_num(key.float(), nan=0.0, posinf=0.0, neginf=0.0)
        scores = torch.matmul(query_fp32, key_fp32.transpose(-1, -2)) * float(scaling)
        scores = scores * self.tau_focus
        score_valid = torch.isfinite(scores)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        additive_mask, hard_valid = self._prepare_mask(attention_mask, scores)
        scores = scores + additive_mask
        hard_valid = hard_valid.expand_as(scores)
        hard_valid = hard_valid & score_valid & query_valid & key_valid

        row_has_valid = hard_valid.any(dim=-1, keepdim=True)
        neg_inf = torch.tensor(float("-inf"), dtype=scores.dtype, device=scores.device)
        max_scores = scores.masked_fill(~hard_valid, neg_inf).amax(dim=-1, keepdim=True)
        keep = hard_valid & (scores >= (max_scores - self.haze_floor_margin))
        # Use the dtype minimum instead of a finite magic number so masked
        # positions cannot receive probability mass at any supported dtype.
        scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)

        probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        # A fully masked row is invalid input for ordinary softmax. Returning
        # a finite zero vector keeps the failure contained and avoids NaN
        # propagation through a whole decoder stack.
        probs = probs * row_has_valid.to(dtype=probs.dtype)
        value_fp32 = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
        out = torch.matmul(probs, value_fp32)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).to(query.dtype)
