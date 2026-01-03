"""
Prefix Tuning
==============

Learns continuous task-specific prefix vectors for each layer.

Reference: "Prefix-Tuning: Optimizing Continuous Prompts for Generation"
"""

import torch
import torch.nn as nn
from typing import Optional


class PrefixTuning(nn.Module):
    """
    Prefix Tuning for transformer models.

    Prepends trainable prefix vectors to each attention layer.

    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_k: Dimension per head
        prefix_length: Length of prefix sequence
        prefix_dropout: Dropout for prefix
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_k: int,
        prefix_length: int = 20,
        prefix_dropout: float = 0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_k = d_k
        self.prefix_length = prefix_length

        # Prefix parameters for each layer (key and value)
        self.prefix_params = nn.ParameterList([
            nn.Parameter(torch.randn(2, prefix_length, num_heads, d_k))
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(prefix_dropout)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize prefix parameters."""
        for param in self.prefix_params:
            nn.init.xavier_uniform_(param)

    def get_prefix(self, layer_idx: int, batch_size: int) -> tuple:
        """
        Get prefix for a specific layer.

        Args:
            layer_idx: Layer index
            batch_size: Batch size

        Returns:
            Tuple of (prefix_key, prefix_value)
        """
        prefix = self.prefix_params[layer_idx]
        prefix = self.dropout(prefix)

        # Split into key and value
        prefix_key, prefix_value = prefix[0], prefix[1]

        # Expand for batch
        prefix_key = prefix_key.unsqueeze(0).expand(batch_size, -1, -1, -1)
        prefix_value = prefix_value.unsqueeze(0).expand(batch_size, -1, -1, -1)

        return prefix_key, prefix_value

    def forward(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> tuple:
        """
        Prepend prefix to key and value.

        Args:
            layer_idx: Layer index
            key: Key tensor [batch, num_heads, seq_len, d_k]
            value: Value tensor [batch, num_heads, seq_len, d_k]

        Returns:
            Tuple of (augmented_key, augmented_value)
        """
        batch_size = key.size(0)

        # Get prefix for this layer
        prefix_key, prefix_value = self.get_prefix(layer_idx, batch_size)

        # Transpose prefix to match key/value shape
        prefix_key = prefix_key.transpose(1, 2)  # [batch, num_heads, prefix_len, d_k]
        prefix_value = prefix_value.transpose(1, 2)

        # Prepend prefix
        augmented_key = torch.cat([prefix_key, key], dim=2)
        augmented_value = torch.cat([prefix_value, value], dim=2)

        return augmented_key, augmented_value
