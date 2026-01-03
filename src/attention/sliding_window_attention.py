"""
Sliding Window Attention
=========================

Attention mechanism with a fixed-size sliding window.
Only attends to nearby tokens within a window, reducing complexity.

Reference: "Longformer: The Long-Document Transformer"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention.

    Each token attends only to tokens within a fixed window around it.
    This reduces complexity from O(N²) to O(N*W) where W is window size.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        window_size: Size of attention window (one-sided)
        dropout: Dropout probability
        attention_dropout: Dropout for attention weights
        qkv_bias: Whether to use bias in QKV projections
        out_bias: Whether to use bias in output projection
        global_attention: Whether to allow global attention on special tokens
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int = 256,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        global_attention: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        self.global_attention = global_attention
        self.scale = 1.0 / math.sqrt(self.d_k)

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=out_bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def _create_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device,
        global_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Create sliding window attention mask."""
        # Create a matrix of positions
        positions = torch.arange(seq_len, device=device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)

        # Create window mask: tokens can attend within window_size
        window_mask = distance.abs() <= self.window_size

        if global_mask is not None:
            # Allow certain positions to attend globally
            window_mask = window_mask | global_mask.unsqueeze(1) | global_mask.unsqueeze(0)

        return window_mask

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor
            value: Value tensor
            attention_mask: Additional attention mask
            global_attention_mask: Mask indicating which tokens attend globally
            need_weights: Whether to return attention weights

        Returns:
            output: Attention output
            attn_weights: Attention weights (optional)
        """
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size, seq_len, _ = query.shape

        # Project to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Create sliding window mask
        window_mask = self._create_sliding_window_mask(
            seq_len,
            query.device,
            global_attention_mask if self.global_attention else None
        )

        # Apply window mask
        attn_scores = attn_scores.masked_fill(
            window_mask.unsqueeze(0).unsqueeze(0) == 0,
            float('-inf')
        )

        # Apply additional attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        output = self.out_proj(attn_output)
        output = self.dropout(output)

        if need_weights:
            return output, attn_weights.mean(dim=1)

        return output, None

    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, num_heads={self.num_heads}, '
                f'window_size={self.window_size}')
