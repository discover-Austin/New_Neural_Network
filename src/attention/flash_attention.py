"""
Flash Attention
===============

Memory-efficient attention with O(N) memory complexity instead of O(N²).
Uses tiling and recomputation to avoid materializing the full attention matrix.

Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class FlashAttention(nn.Module):
    """
    Flash Attention implementation with fallback to standard attention.

    Flash Attention is significantly faster and more memory-efficient for long sequences.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        attention_dropout: Dropout for attention weights
        qkv_bias: Whether to use bias in QKV projections
        out_bias: Whether to use bias in output projection
        causal: Whether to use causal (autoregressive) attention
        softmax_scale: Scale factor for attention scores
        use_flash_attn: Whether to use flash attention (if available)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal
        self.softmax_scale = softmax_scale or (1.0 / math.sqrt(self.d_k))
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE

        # QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=out_bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = attention_dropout

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask
            key_padding_mask: Key padding mask
            need_weights: Whether to return attention weights

        Returns:
            output: Attention output
            attn_weights: Attention weights (None for flash attention)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.d_k)

        if self.use_flash_attn and not need_weights:
            # Use flash attention
            q, k, v = qkv.unbind(dim=2)  # Each: [batch, seq_len, num_heads, d_k]

            # Flash attention expects [batch, seq_len, num_heads, d_k]
            output = flash_attn_func(
                q, k, v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )

            output = output.reshape(batch_size, seq_len, self.d_model)
            output = self.out_proj(output)
            output = self.dropout(output)

            return output, None
        else:
            # Fallback to standard attention
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)  # [batch, heads, seq_len, d_k]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Compute attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

            # Apply causal mask
            if self.causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

            # Apply attention mask
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
                attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

            # Apply key padding mask
            if key_padding_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf')
                )

            # Compute attention weights
            attn_weights = F.softmax(attn_scores, dim=-1)

            if self.training:
                attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)

            output = self.out_proj(attn_output)
            output = self.dropout(output)

            if need_weights:
                return output, attn_weights.mean(dim=1)

            return output, None

    def extra_repr(self) -> str:
        flash_status = "enabled" if self.use_flash_attn else "disabled"
        return (f'd_model={self.d_model}, num_heads={self.num_heads}, '
                f'causal={self.causal}, flash_attn={flash_status}')
