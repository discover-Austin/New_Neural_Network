"""
LoRA: Low-Rank Adaptation
==========================

Parameter-efficient fine-tuning by learning low-rank decomposition matrices.

Reference: "LoRA: Low-Rank Adaptation of Large Language Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class LoRALayer(nn.Module):
    """
    LoRA layer implementation.

    Adds trainable low-rank matrices A and B to a frozen pretrained weight W.
    During forward: h = W@x + (B@A)@x * (alpha/r)

    Args:
        r: Rank of the low-rank matrices
        lora_alpha: Scaling factor
        lora_dropout: Dropout probability for LoRA
    """

    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: Dropout probability
        merge_weights: Whether to merge LoRA weights with base weights
        bias: Whether to use bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merge_weights = merge_weights
        self.merged = False

        # Pretrained weight (frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x

        self.reset_parameters()

        # Freeze pretrained weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        """Override train to handle weight merging."""
        super().train(mode)

        if mode:
            # Training mode: unmerge weights
            if self.merge_weights and self.merged:
                self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            # Eval mode: merge weights for efficiency
            if self.merge_weights and not self.merged:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        if self.merged:
            # Weights are merged, just use standard linear
            return F.linear(x, self.weight, self.bias)

        # Standard linear + LoRA adaptation
        result = F.linear(x, self.weight, self.bias)

        # Add LoRA contribution
        lora_result = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        result = result + lora_result

        return result

    def extra_repr(self) -> str:
        return f'in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, r={self.r}'


def apply_lora(
    model: nn.Module,
    target_modules: Optional[list] = None,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.

    Args:
        model: Model to apply LoRA to
        target_modules: List of module names to apply LoRA (e.g., ["q_proj", "v_proj"])
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout

    Returns:
        Model with LoRA layers applied
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

    for name, module in model.named_modules():
        # Check if this module should have LoRA applied
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA linear
                parent_name, child_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name) if parent_name else model

                lora_linear = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=module.bias is not None,
                )

                # Copy pretrained weights
                lora_linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_linear.bias.data = module.bias.data.clone()

                setattr(parent, child_name, lora_linear)

    return model


def get_lora_parameters(model: nn.Module) -> list:
    """
    Get only the LoRA parameters for training.

    Args:
        model: Model with LoRA layers

    Returns:
        List of LoRA parameters
    """
    lora_params = []

    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)

    return lora_params


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into base model weights.

    Args:
        model: Model with LoRA layers

    Returns:
        Model with merged weights
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            if not module.merged:
                module.weight.data += (module.lora_B @ module.lora_A) * module.scaling
                module.merged = True

    return model
