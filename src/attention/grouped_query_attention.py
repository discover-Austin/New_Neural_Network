"""
Grouped-Query Attention (GQA)
==============================

GQA is a generalization of MQA that uses intermediate number of key-value heads.
This provides a balance between MHA (expensive) and MQA (limited expressiveness).

Reference: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention mechanism.

    GQA divides query heads into groups, with each group sharing a single KV head.

    Args:
        d_model: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (must divide num_heads)
        dropout: Dropout probability
        attention_dropout: Separate dropout for attention weights
        qkv_bias: Whether to use bias in QKV projections
        out_bias: Whether to use bias in output projection
        scale: Custom attention scale factor
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        self.scale = scale or (1.0 / math.sqrt(self.d_k))

        # Query projection (all heads)
        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)

        # Key and Value projections (kv_heads)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.d_k, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.d_k, bias=qkv_bias)

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
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask
            key_padding_mask: Key padding mask
            need_weights: Whether to return attention weights
            average_attn_weights: Whether to average attention weights

        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attn_weights: Attention weights (optional)
        """
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size, seq_len, _ = query.shape
        key_len = key.shape[1]

        # Project queries (all heads)
        Q = self.q_proj(query)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Project keys and values (kv_heads)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape K and V for grouped attention
        K = K.view(batch_size, key_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, key_len, self.num_kv_heads, self.d_k).transpose(1, 2)

        # Expand K and V to match query heads
        # Each KV head is repeated for num_queries_per_kv query heads
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply masks
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

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
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        output = self.dropout(output)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return output, attn_weights

        return output, None

    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, num_heads={self.num_heads}, '
                f'num_kv_heads={self.num_kv_heads}, d_k={self.d_k} (GQA)')
