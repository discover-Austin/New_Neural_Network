"""
Activation Functions
====================

Advanced activation functions including GLU variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation function.

    Supports both exact and approximate (tanh) variants.

    Args:
        approximate: Whether to use tanh approximation
    """

    def __init__(self, approximate: bool = False):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU activation."""
        if self.approximate:
            return F.gelu(x, approximate='tanh')
        return F.gelu(x)


class Swish(nn.Module):
    """
    Swish (SiLU) activation function.

    Swish(x) = x * sigmoid(x)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Swish activation."""
        return F.silu(x)


class GLU(nn.Module):
    """
    Gated Linear Unit.

    Splits input in half and applies gating mechanism.

    Args:
        dim: Dimension to split on
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GLU."""
        a, b = x.chunk(2, dim=self.dim)
        return a * torch.sigmoid(b)


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.

    Variant of GLU using Swish activation instead of sigmoid.
    Used in models like PaLM and LLaMA.

    Reference: "GLU Variants Improve Transformer"

    Args:
        dim_in: Input dimension
        dim_out: Output dimension
        bias: Whether to use bias
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = False,
    ):
        super().__init__()
        # Project to 2*dim_out for gating
        self.proj = nn.Linear(dim_in, 2 * dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.

        Args:
            x: Input tensor [..., dim_in]

        Returns:
            Output tensor [..., dim_out]
        """
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return F.silu(x1) * x2


class GeGLU(nn.Module):
    """
    GELU-Gated Linear Unit.

    Variant of GLU using GELU activation.

    Reference: "GLU Variants Improve Transformer"

    Args:
        dim_in: Input dimension
        dim_out: Output dimension
        bias: Whether to use bias
        approximate: Whether to use approximate GELU
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = False,
        approximate: bool = False,
    ):
        super().__init__()
        self.proj = nn.Linear(dim_in, 2 * dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GeGLU activation.

        Args:
            x: Input tensor [..., dim_in]

        Returns:
            Output tensor [..., dim_out]
        """
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)

        if self.approximate:
            gelu_x1 = F.gelu(x1, approximate='tanh')
        else:
            gelu_x1 = F.gelu(x1)

        return gelu_x1 * x2


class ReGLU(nn.Module):
    """
    ReLU-Gated Linear Unit.

    Variant of GLU using ReLU activation.

    Args:
        dim_in: Input dimension
        dim_out: Output dimension
        bias: Whether to use bias
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = False,
    ):
        super().__init__()
        self.proj = nn.Linear(dim_in, 2 * dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ReGLU activation.

        Args:
            x: Input tensor [..., dim_in]

        Returns:
            Output tensor [..., dim_out]
        """
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return F.relu(x1) * x2
