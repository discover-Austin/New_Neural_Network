"""
Linear Attention
================

Linear complexity attention mechanisms using kernel methods.
Reduces attention complexity from O(N²) to O(N).

Reference: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
import math


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU + 1 feature map for linear attention."""
    return F.elu(x) + 1


def relu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ReLU feature map for linear attention."""
    return F.relu(x)


def softmax_feature_map(x: torch.Tensor) -> torch.Tensor:
    """Softmax-based feature map."""
    return torch.exp(x - x.max(dim=-1, keepdim=True)[0])


class LinearAttention(nn.Module):
    """
    Linear Attention using kernel-based approximation.

    Instead of computing full attention matrix, uses feature maps to
    compute attention in linear time.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        qkv_bias: Whether to use bias in QKV projections
        out_bias: Whether to use bias in output projection
        feature_map: Feature map function to use
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        qkv_bias: bool = True,
        out_bias: bool = True,
        feature_map: str = "elu",
        eps: float = 1e-6,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.eps = eps

        # Select feature map
        self.feature_map = {
            "elu": elu_feature_map,
            "relu": relu_feature_map,
            "softmax": softmax_feature_map,
        }[feature_map]

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=out_bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass with linear complexity.

        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor
            value: Value tensor
            attention_mask: Not used in linear attention

        Returns:
            output: Attention output
            None: Linear attention doesn't return weights
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

        # Apply feature map
        Q = self.feature_map(Q)
        K = self.feature_map(K)

        # Linear attention computation
        # Instead of softmax(QK^T)V, we compute Q(K^TV) in linear time
        # Q: [batch, heads, seq_len, d_k]
        # K: [batch, heads, key_len, d_k]
        # V: [batch, heads, key_len, d_k]

        # Compute K^T V: [batch, heads, d_k, d_k]
        KV = torch.matmul(K.transpose(-2, -1), V)

        # Compute normalization: [batch, heads, d_k]
        Z = torch.matmul(K.sum(dim=-2, keepdim=True).transpose(-2, -1),
                         torch.ones_like(V[..., :1]))

        # Compute output: [batch, heads, seq_len, d_k]
        attn_output = torch.matmul(Q, KV)
        attn_output = attn_output / (torch.matmul(Q, Z) + self.eps)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        output = self.out_proj(attn_output)
        output = self.dropout(output)

        return output, None

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, num_heads={self.num_heads}, d_k={self.d_k} (Linear)'
