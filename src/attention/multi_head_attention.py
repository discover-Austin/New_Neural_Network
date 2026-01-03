"""
Multi-Head Attention (MHA)
===========================

Standard multi-head attention mechanism as described in "Attention is All You Need".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in linear projections
        attention_dropout: Separate dropout for attention weights
        qkv_bias: Whether to use bias in QKV projections
        out_bias: Whether to use bias in output projection
        scale: Custom attention scale factor (default: 1/sqrt(d_k))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = scale or (1.0 / math.sqrt(self.d_k))

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=out_bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model] (defaults to query)
            value: Value tensor [batch_size, seq_len, d_model] (defaults to key)
            attention_mask: Attention mask [seq_len, seq_len] or [batch_size, seq_len, seq_len]
            key_padding_mask: Key padding mask [batch_size, seq_len]
            need_weights: Whether to return attention weights
            average_attn_weights: Whether to average attention weights across heads

        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attn_weights: Attention weights (if need_weights=True)
        """
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size, seq_len, _ = query.shape

        # Project to Q, K, V
        Q = self.q_proj(query)  # [batch, seq_len, d_model]
        K = self.k_proj(key)    # [batch, key_len, d_model]
        V = self.v_proj(value)  # [batch, key_len, d_model]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch, heads, seq_len, d_k]
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)       # [batch, heads, key_len, d_k]
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)       # [batch, heads, key_len, d_k]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, heads, seq_len, key_len]

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        # Apply key padding mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch, heads, seq_len, d_k]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        output = self.dropout(output)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)  # Average across heads
            return output, attn_weights

        return output, None

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, num_heads={self.num_heads}, d_k={self.d_k}'
