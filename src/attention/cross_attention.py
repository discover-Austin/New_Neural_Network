"""
Cross Attention
===============

Cross-attention mechanism for attending from one sequence to another.
Used in encoder-decoder architectures and multimodal models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism.

    Queries come from one sequence, keys and values from another.
    Essential for encoder-decoder models and multimodal fusion.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_kv: Key-Value dimension (defaults to d_model)
        dropout: Dropout probability
        attention_dropout: Dropout for attention weights
        qkv_bias: Whether to use bias in QKV projections
        out_bias: Whether to use bias in output projection
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_kv: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        super().__init__()
        d_kv = d_kv or d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert d_kv % num_heads == 0, "d_kv must be divisible by num_heads"

        self.d_model = d_model
        self.d_kv = d_kv
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_kv // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Query projection (from decoder/target sequence)
        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)

        # Key and Value projections (from encoder/source sequence)
        self.k_proj = nn.Linear(d_kv, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_kv, d_model, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=out_bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            query: Query tensor from decoder [batch_size, tgt_len, d_model]
            key: Key tensor from encoder [batch_size, src_len, d_kv]
            value: Value tensor from encoder [batch_size, src_len, d_kv]
            attention_mask: Attention mask [tgt_len, src_len]
            key_padding_mask: Padding mask for encoder [batch_size, src_len]
            need_weights: Whether to return attention weights
            average_attn_weights: Whether to average weights across heads

        Returns:
            output: Cross-attention output [batch_size, tgt_len, d_model]
            attn_weights: Attention weights (optional)
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # Project queries, keys, values
        Q = self.q_proj(query)  # [batch, tgt_len, d_model]
        K = self.k_proj(key)    # [batch, src_len, d_model]
        V = self.v_proj(value)  # [batch, src_len, d_model]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, src_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, src_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

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
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, tgt_len, self.d_model)

        output = self.out_proj(attn_output)
        output = self.dropout(output)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return output, attn_weights

        return output, None

    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, d_kv={self.d_kv}, '
                f'num_heads={self.num_heads} (Cross)')
