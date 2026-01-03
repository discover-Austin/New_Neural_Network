"""
Positional Encodings
====================

Various positional encoding schemes for transformers.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention is All You Need".

    Uses sine and cosine functions of different frequencies to encode positions.

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        scale: Whether to scale embeddings
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        scale: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model) if scale else 1.0

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        x = x * self.scale + self.pe[:, :seq_len]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding using embedding layer.

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.position_embeddings = nn.Embedding(max_len, d_model)

        # Initialize
        nn.init.normal_(self.position_embeddings.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            x with positional encoding added
        """
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(positions)
        return self.dropout(x + position_embeddings)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes position information by rotating the query and key vectors.
    More effective for long sequences than absolute positional encodings.

    Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"

    Args:
        d_model: Model dimension (per head)
        max_len: Maximum sequence length
        base: Base for inverse frequency computation
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        # Cache for cos and sin
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos and sin values."""
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(dtype))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embedding to queries and keys.

        Args:
            q: Query tensor [..., seq_len, d_model]
            k: Key tensor [..., seq_len, d_model]
            seq_len: Sequence length (defaults to q.shape[-2])

        Returns:
            q_rotated: Query with RoPE applied
            k_rotated: Key with RoPE applied
        """
        if seq_len is None:
            seq_len = q.shape[-2]

        self._update_cos_sin_cache(seq_len, device=q.device, dtype=q.dtype)

        # Apply rotation
        q_rotated = (q * self._cos_cached[:seq_len]) + (self._rotate_half(q) * self._sin_cached[:seq_len])
        k_rotated = (k * self._cos_cached[:seq_len]) + (self._rotate_half(k) * self._sin_cached[:seq_len])

        return q_rotated, k_rotated


class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi).

    Instead of adding positional information to embeddings, adds bias to
    attention scores based on distance between tokens.

    Reference: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"

    Args:
        num_heads: Number of attention heads
        max_len: Maximum sequence length
    """

    def __init__(
        self,
        num_heads: int,
        max_len: int = 2048,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len

        # Compute slopes for each head
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)

        # Cache for bias matrix
        self._bias_cached = None
        self._seq_len_cached = 0

    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """Compute slopes for ALiBi."""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return torch.tensor([start * (ratio ** i) for i in range(n)])

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            # Handle non-power-of-2 heads
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = self._get_slopes(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]
            return torch.cat([slopes_a, slopes_b])

    def _get_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or compute bias matrix."""
        if seq_len > self._seq_len_cached or self._bias_cached is None:
            # Create distance matrix
            positions = torch.arange(seq_len, device=device)
            distance = positions.unsqueeze(0) - positions.unsqueeze(1)

            # Apply slopes to get bias
            bias = distance.unsqueeze(0) * self.slopes.unsqueeze(1).unsqueeze(1).to(device)
            self._bias_cached = bias
            self._seq_len_cached = seq_len

        return self._bias_cached[:, :seq_len, :seq_len]

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Get ALiBi bias for attention scores.

        Args:
            seq_len: Sequence length
            device: Device to create bias on

        Returns:
            bias: ALiBi bias [num_heads, seq_len, seq_len]
        """
        return self._get_bias(seq_len, device)


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding.

    Encodes relative distances between tokens rather than absolute positions.

    Reference: "Self-Attention with Relative Position Representations"

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        max_relative_position: Maximum relative position to encode
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_relative_position: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position
        self.d_k = d_model // num_heads

        # Embeddings for relative positions
        vocab_size = 2 * max_relative_position + 1
        self.relative_position_embeddings = nn.Embedding(vocab_size, self.d_k)

        nn.init.normal_(self.relative_position_embeddings.weight, std=0.02)

    def _relative_position_bucket(
        self,
        relative_position: torch.Tensor,
    ) -> torch.Tensor:
        """Map relative positions to buckets."""
        num_buckets = 2 * self.max_relative_position + 1
        relative_buckets = torch.clamp(
            relative_position + self.max_relative_position,
            0,
            num_buckets - 1
        )
        return relative_buckets

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute relative positional encoding.

        Args:
            seq_len: Sequence length
            device: Device

        Returns:
            relative_encoding: Relative position encoding [seq_len, seq_len, d_k]
        """
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_buckets = self._relative_position_bucket(relative_positions)
        relative_encoding = self.relative_position_embeddings(relative_buckets)
        return relative_encoding
