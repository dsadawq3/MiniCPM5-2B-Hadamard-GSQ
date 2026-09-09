"""
================================================================================
KV-BSS: Key-Value Binding Softmax Sharpening for MiniCPM5-2B
Hardens hallucination threshold and enhances associative recall in long contexts.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class KVBSSAttentionHook(nn.Module):
    """
    KV-BSS Attention Hook:
    1. Focus Temperature Scaling (tau_focus = 1.10):
       Scores = (Q @ K.T / sqrt(d)) * tau_focus
       Counters entropy dispersion over long context windows (up to 128k).
    2. Dynamic Attention Haze Suppression:
       Drops attention scores falling below (max_score - haze_floor_margin)
       to -inf before softmax, eliminating low-probability associative distraction.
    """
    def __init__(self, tau_focus: float = 1.10, haze_floor_margin: float = 12.0):
        super().__init__()
        self.tau_focus = tau_focus
        self.haze_floor_margin = haze_floor_margin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor = None,
        scaling: float = None
    ) -> torch.Tensor:
        """
        query: [batch, heads, q_len, head_dim]
        key:   [batch, kv_heads, kv_len, head_dim]
        value: [batch, kv_heads, kv_len, head_dim]
        """
        b, h, q_len, d = query.shape
        b, kv_h, kv_len, _ = key.shape
        
        # Handle GQA repeat if needed
        if h != kv_h:
            num_repeat = h // kv_h
            key = key.repeat_interleave(num_repeat, dim=1)
            value = value.repeat_interleave(num_repeat, dim=1)
            
        if scaling is None:
            scaling = 1.0 / math.sqrt(d)
            
        # 1. Compute raw dot-product attention scores
        scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
        
        # 2. KV-BSS Focus Sharpening
        scores = scores * self.tau_focus
        
        # 3. Attention Mask
        if attention_mask is not None:
            scores = scores + attention_mask
            
        # 4. Attention Haze Suppression (Floor Filter)
        max_scores = torch.amax(scores, dim=-1, keepdim=True)
        haze_mask = scores < (max_scores - self.haze_floor_margin)
        scores = scores.masked_fill(haze_mask, -1e4)
        
        # 5. High-Precision Softmax
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # 6. Context projection
        out = torch.matmul(probs, value)
        return out
