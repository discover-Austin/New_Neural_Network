"""
Sparse Attention
================

Sparse attention patterns for reduced computational complexity.
Implements various sparsity patterns: strided, fixed, and block-sparse.

Reference: "Generating Long Sequences with Sparse Transformers"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
import math


class SparseAttention(nn.Module):
    """
    Sparse Attention with configurable sparsity patterns.

    Supports:
    - Strided attention: Attend to every k-th token
    - Fixed attention: Attend to fixed positions
    - Block sparse: Attend within blocks

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        sparsity_pattern: Type of sparsity ("strided", "fixed", "block")
        stride: Stride for strided attention
        block_size: Block size for block-sparse attention
        num_fixed: Number of fixed positions to attend to
        dropout: Dropout probability
        attention_dropout: Dropout for attention weights
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        sparsity_pattern: Literal["strided", "fixed", "block"] = "strided",
        stride: int = 64,
        block_size: int = 32,
        num_fixed: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.sparsity_pattern = sparsity_pattern
        self.stride = stride
        self.block_size = block_size
        self.num_fixed = num_fixed

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=out_bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def _create_strided_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create strided attention mask."""
        positions = torch.arange(seq_len, device=device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)

        # Local attention (diagonal band)
        local_mask = distance.abs() <= self.block_size

        # Strided attention
        strided_mask = (positions.unsqueeze(1) % self.stride) == 0

        # Combine local and strided
        mask = local_mask | strided_mask

        return mask

    def _create_fixed_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create fixed attention mask (attend to first num_fixed tokens)."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Local attention
        for i in range(seq_len):
            start = max(0, i - self.block_size)
            end = min(seq_len, i + self.block_size + 1)
            mask[i, start:end] = True

        # Fixed positions (first num_fixed tokens)
        mask[:, :self.num_fixed] = True

        return mask

    def _create_block_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create block-sparse attention mask."""
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min((block_idx + 1) * self.block_size, seq_len)

            # Attend within block
            mask[start:end, start:end] = True

            # Attend to previous block
            if block_idx > 0:
                prev_start = (block_idx - 1) * self.block_size
                prev_end = block_idx * self.block_size
                mask[start:end, prev_start:prev_end] = True

        return mask

    def _create_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sparse attention mask based on pattern."""
        if self.sparsity_pattern == "strided":
            return self._create_strided_mask(seq_len, device)
        elif self.sparsity_pattern == "fixed":
            return self._create_fixed_mask(seq_len, device)
        elif self.sparsity_pattern == "block":
            return self._create_block_sparse_mask(seq_len, device)
        else:
            raise ValueError(f"Unknown sparsity pattern: {self.sparsity_pattern}")

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor
            value: Value tensor
            attention_mask: Additional attention mask
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

        # Create and apply sparse mask
        sparse_mask = self._create_sparse_mask(seq_len, query.device)
        attn_scores = attn_scores.masked_fill(
            sparse_mask.unsqueeze(0).unsqueeze(0) == 0,
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
                f'pattern={self.sparsity_pattern}')
