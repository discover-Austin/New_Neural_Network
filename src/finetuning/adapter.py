"""
Adapter Layers
==============

Small bottleneck layers inserted into pretrained models.

Reference: "Parameter-Efficient Transfer Learning for NLP"
"""

import torch
import torch.nn as nn


class AdapterLayer(nn.Module):
    """
    Adapter layer with bottleneck architecture.

    Projects down to smaller dimension, applies activation, then projects back.

    Args:
        d_model: Model dimension
        bottleneck_dim: Bottleneck dimension (typically d_model // 16)
        activation: Activation function
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)

        # Initialize near-identity
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.01)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor [..., d_model]

        Returns:
            Output tensor [..., d_model]
        """
        residual = x

        # Bottleneck
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.dropout(x)

        # Residual connection
        return residual + x
