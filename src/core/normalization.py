"""
Normalization Layers
====================

Various normalization techniques for neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List


class LayerNorm(nn.Module):
    """
    Layer Normalization.

    Standard layer normalization from "Layer Normalization" paper.

    Args:
        normalized_shape: Input shape (int or list)
        eps: Small constant for numerical stability
        elementwise_affine: Whether to learn affine parameters
        bias: Whether to use bias
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            if bias:
                self.bias = nn.Parameter(torch.zeros(normalized_shape))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization."""
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps
        )

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler and faster variant of LayerNorm that doesn't subtract mean.
    Used in models like LLaMA and GPT-NeoX.

    Reference: "Root Mean Square Layer Normalization"

    Args:
        d_model: Model dimension
        eps: Small constant for numerical stability
        elementwise_affine: Whether to learn affine parameters
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(d_model))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor [..., d_model]

        Returns:
            Normalized tensor
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize
        x_normed = x / rms

        # Apply learned scale
        if self.elementwise_affine:
            x_normed = x_normed * self.weight

        return x_normed

    def extra_repr(self) -> str:
        return f'eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class GroupNorm(nn.Module):
    """
    Group Normalization.

    Divides channels into groups and normalizes within each group.

    Reference: "Group Normalization"

    Args:
        num_groups: Number of groups
        num_channels: Number of channels
        eps: Small constant for numerical stability
        affine: Whether to learn affine parameters
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        assert num_channels % num_groups == 0, \
            f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply group normalization."""
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f'num_groups={self.num_groups}, num_channels={self.num_channels}, eps={self.eps}'
