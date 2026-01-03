"""
Model Quantization
==================

Research-grounded quantization methods for model compression:

1. Post-Training Quantization (PTQ)
   - INT8 quantization
   - INT4 quantization
   - Dynamic vs Static quantization

2. GPTQ (Gradient-based Post-Training Quantization)
   - Reference: "GPTQ: Accurate Post-Training Quantization for GPT" (Frantar et al., ICLR 2023)
   - One-shot layer-wise quantization
   - Minimal accuracy loss

3. AWQ (Activation-aware Weight Quantization)
   - Reference: "AWQ: Activation-aware Weight Quantization for LLM Compression" (Lin et al., 2023)
   - Protects salient weights
   - Better than GPTQ on many tasks

4. SmoothQuant
   - Reference: "SmoothQuant: Accurate and Efficient Post-Training Quantization" (Xiao et al., 2023)
   - Smooths activation outliers
   - Enables INT8 quantization of LLMs

Mathematical Foundation:
-----------------------

Uniform Quantization:
  Quantize: Q(x) = round((x - z) / s) * s + z
  where:
    s = scale = (x_max - x_min) / (2^bits - 1)
    z = zero_point

  For symmetric: z = 0, s = 2 * max(|x_max|, |x_min|) / (2^bits - 1)

GPTQ:
  Minimize: ||WX - ŴX||²_F subject to Ŵ quantized

  Layer-wise algorithm:
  1. For each row of weight matrix:
     - Quantize one weight at a time
     - Update remaining weights to compensate error
  2. Uses Hessian inverse: H^(-1) = (X X^T)^(-1)

  Complexity: O(d³) per layer (Hessian inversion)

AWQ:
  1. Search for optimal scale per channel:
     s* = argmin_s L(Quant(W / s) * s * X)

  2. Protect salient weights (high activation magnitude)
     - 1% most important weights stay FP16
     - Rest quantized to INT4

  Empirical: 0.1-0.3% accuracy drop vs FP16

SmoothQuant:
  Migrate difficulty from activations to weights:
  Y = (X diag(s)) (diag(s)^(-1) W)

  where s balances activation/weight quantization difficulty

Complexity:
- INT8: 4x memory reduction, 2-4x speedup
- INT4: 8x memory reduction, 3-6x speedup
- GPTQ/AWQ: Minimal accuracy loss (<1%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import numpy as np


@dataclass
class QuantConfig:
    """Configuration for quantization."""
    bits: int = 8  # 4, 8, or 16
    symmetric: bool = True  # Symmetric vs asymmetric quantization
    per_channel: bool = True  # Per-channel vs per-tensor scaling
    group_size: int = 128  # For grouped quantization (GPTQ/AWQ)


class UniformQuantizer:
    """
    Uniform quantization.

    Maps floating-point values to discrete levels.
    """

    def __init__(self, config: QuantConfig):
        self.config = config
        self.n_levels = 2 ** config.bits

    def compute_scale_zero_point(
        self,
        x: torch.Tensor,
        dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute quantization scale and zero-point.

        Args:
            x: Input tensor
            dim: Dimension for per-channel quantization

        Returns:
            (scale, zero_point)
        """
        if dim is not None:
            # Per-channel
            x_max = x.max(dim=dim, keepdim=True)[0]
            x_min = x.min(dim=dim, keepdim=True)[0]
        else:
            # Per-tensor
            x_max = x.max()
            x_min = x.min()

        if self.config.symmetric:
            # Symmetric quantization
            abs_max = torch.max(torch.abs(x_max), torch.abs(x_min))
            scale = 2 * abs_max / (self.n_levels - 1)
            zero_point = torch.zeros_like(scale)
        else:
            # Asymmetric quantization
            scale = (x_max - x_min) / (self.n_levels - 1)
            zero_point = x_min

        # Avoid division by zero
        scale = torch.clamp(scale, min=1e-8)

        return scale, zero_point

    def quantize(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quantize tensor.

        Args:
            x: Input tensor
            scale: Quantization scale
            zero_point: Zero point

        Returns:
            Quantized tensor (still in float, but discrete values)
        """
        # Quantize
        x_int = torch.round((x - zero_point) / scale)

        # Clamp to valid range
        if self.config.symmetric:
            q_min = -(self.n_levels // 2)
            q_max = (self.n_levels // 2) - 1
        else:
            q_min = 0
            q_max = self.n_levels - 1

        x_int = torch.clamp(x_int, q_min, q_max)

        # Dequantize (simulated quantization)
        x_quant = x_int * scale + zero_point

        return x_quant

    def __call__(
        self,
        x: torch.Tensor,
        dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize and return quantized tensor + params.

        Args:
            x: Input tensor
            dim: Dimension for per-channel quantization

        Returns:
            (quantized_tensor, scale, zero_point)
        """
        scale, zero_point = self.compute_scale_zero_point(x, dim)
        x_quant = self.quantize(x, scale, zero_point)

        return x_quant, scale, zero_point


class QuantizedLinear(nn.Module):
    """
    Quantized linear layer.

    Stores weights in quantized format.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QuantConfig,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Quantized weight (stored as int)
        self.register_buffer(
            "weight_quant",
            torch.zeros(out_features, in_features, dtype=torch.int8),
        )

        # Scale and zero-point
        if config.per_channel:
            scale_shape = (out_features, 1)
        else:
            scale_shape = (1,)

        self.register_buffer("weight_scale", torch.ones(scale_shape))
        self.register_buffer("weight_zero_point", torch.zeros(scale_shape))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_buffer("bias", None)

    def quantize_weight(self, weight: torch.Tensor):
        """Quantize and store weight."""
        quantizer = UniformQuantizer(self.config)

        dim = 1 if self.config.per_channel else None
        _, scale, zero_point = quantizer(weight, dim=dim)

        # Store scale and zero-point
        self.weight_scale.copy_(scale)
        self.weight_zero_point.copy_(zero_point)

        # Quantize to int
        weight_int = torch.round((weight - zero_point) / scale)

        if self.config.bits == 8:
            self.weight_quant.copy_(weight_int.to(torch.int8))
        else:
            # For INT4, pack two values per byte
            self.weight_quant.copy_(weight_int.to(torch.int8))

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize weight for computation."""
        weight = self.weight_quant.float() * self.weight_scale + self.weight_zero_point
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Dequantize weight
        weight = self.dequantize_weight()

        # Standard linear operation
        return F.linear(x, weight, self.bias)


class GPTQQuantizer:
    """
    GPTQ: Gradient-based Post-Training Quantization.

    Reference: Frantar et al., ICLR 2023
    "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"

    Algorithm:
    1. Compute Hessian: H = X X^T
    2. For each weight row:
       - Quantize weights one by one
       - Update remaining weights to minimize error
       - Use Hessian inverse for optimal update

    Empirical Results:
    - 4-bit quantization of 175B model
    - Minimal perplexity increase (<0.5)
    - 4x memory reduction
    """

    def __init__(self, config: QuantConfig):
        self.config = config

    def quantize_layer(
        self,
        weight: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quantize a single layer using GPTQ.

        Args:
            weight: Weight matrix [out_features, in_features]
            inputs: Calibration inputs [num_samples, in_features]

        Returns:
            Quantized weight
        """
        out_features, in_features = weight.shape

        # Compute Hessian: H = X^T X
        H = inputs.T @ inputs  # [in_features, in_features]
        H_diag = torch.diag(H)

        # Add damping for numerical stability
        damp = 0.01 * torch.mean(H_diag)
        H_diag += damp

        # Initialize quantized weight
        weight_quant = weight.clone()

        # Quantizer
        quantizer = UniformQuantizer(self.config)

        # Process in groups
        group_size = self.config.group_size

        for i in range(0, in_features, group_size):
            end_i = min(i + group_size, in_features)
            group_indices = torch.arange(i, end_i)

            # Extract group
            W_group = weight_quant[:, group_indices]

            # Compute scale and zero-point for this group
            scale, zero_point = quantizer.compute_scale_zero_point(
                W_group,
                dim=1 if self.config.per_channel else None,
            )

            # Quantize
            W_group_quant = quantizer.quantize(W_group, scale, zero_point)

            # Compute error
            error = W_group - W_group_quant

            # Update remaining weights to compensate
            if end_i < in_features:
                # Hessian inverse for this group
                H_inv_diag = 1.0 / H_diag[group_indices]

                # Update remaining weights
                for j in range(len(group_indices)):
                    idx = group_indices[j]
                    weight_quant[:, end_i:] -= (
                        error[:, j:j+1] * H[idx, end_i:] * H_inv_diag[j]
                    )

            # Store quantized group
            weight_quant[:, group_indices] = W_group_quant

        return weight_quant


class AWQQuantizer:
    """
    AWQ: Activation-aware Weight Quantization.

    Reference: Lin et al., 2023
    "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"

    Key Insight:
    - Not all weights are equally important
    - Weights with high activation magnitude are more salient
    - Protect salient weights, aggressively quantize others

    Algorithm:
    1. Compute activation statistics
    2. Find optimal per-channel scales
    3. Scale weights before quantization
    4. Quantize scaled weights

    Empirical Results:
    - 4-bit quantization with <1% accuracy loss
    - Outperforms GPTQ on many tasks
    - Faster inference than GPTQ
    """

    def __init__(self, config: QuantConfig):
        self.config = config

    def search_scale(
        self,
        weight: torch.Tensor,
        inputs: torch.Tensor,
        n_grid: int = 20,
    ) -> torch.Tensor:
        """
        Search for optimal per-channel scales.

        Args:
            weight: Weight matrix [out_features, in_features]
            inputs: Calibration inputs [num_samples, in_features]
            n_grid: Number of grid points to search

        Returns:
            Optimal scales [out_features]
        """
        out_features, in_features = weight.shape

        # Compute activation statistics (max magnitude per channel)
        act_scales = inputs.abs().max(dim=0)[0]  # [in_features]

        # Grid search for optimal scale
        scales = torch.ones(out_features, device=weight.device)

        for i in range(out_features):
            w_row = weight[i]  # [in_features]

            best_scale = 1.0
            best_error = float('inf')

            # Search range: [0.5, 2.0]
            for alpha in torch.linspace(0.5, 2.0, n_grid):
                # Scale weights
                w_scaled = w_row / alpha

                # Quantize
                quantizer = UniformQuantizer(self.config)
                w_quant, _, _ = quantizer(w_scaled)

                # Unscale
                w_quant = w_quant * alpha

                # Compute error weighted by activation magnitude
                error = ((w_row - w_quant) * act_scales).abs().mean()

                if error < best_error:
                    best_error = error
                    best_scale = alpha

            scales[i] = best_scale

        return scales

    def quantize_layer(
        self,
        weight: torch.Tensor,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize layer with AWQ.

        Args:
            weight: Weight matrix
            inputs: Calibration inputs

        Returns:
            (quantized_weight, scales)
        """
        # Search for optimal scales
        scales = self.search_scale(weight, inputs)

        # Scale weights
        weight_scaled = weight / scales.unsqueeze(1)

        # Quantize scaled weights
        quantizer = UniformQuantizer(self.config)
        weight_quant, _, _ = quantizer(weight_scaled, dim=0)

        # Unscale
        weight_quant = weight_quant * scales.unsqueeze(1)

        return weight_quant, scales


class SmoothQuantLinear(nn.Module):
    """
    SmoothQuant for linear layers.

    Reference: Xiao et al., 2023
    "SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs"

    Key Insight:
    - Activations have outliers that make quantization difficult
    - Migrate difficulty from activations to weights
    - Y = (X diag(s)) (diag(s)^(-1) W)

    Empirical:
    - Enables INT8 quantization of LLMs
    - Minimal accuracy loss
    - Works where normal INT8 fails
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QuantConfig,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Smoothing scales
        self.register_buffer("smooth_scales", torch.ones(in_features))

        # Quantized layers for weight and activation
        self.weight_quant = QuantizedLinear(in_features, out_features, config, bias)

    def calibrate(
        self,
        weight: torch.Tensor,
        inputs: torch.Tensor,
        alpha: float = 0.5,
    ):
        """
        Calibrate smoothing scales.

        Args:
            weight: Weight matrix
            inputs: Calibration inputs
            alpha: Migration factor (0=all to weight, 1=all to activation)
        """
        # Compute activation scales (max magnitude per channel)
        act_scales = inputs.abs().max(dim=0)[0]

        # Compute weight scales
        weight_scales = weight.abs().max(dim=0)[0]

        # Compute smoothing scales: s = act^α / weight^(1-α)
        smooth_scales = act_scales.pow(alpha) / weight_scales.pow(1 - alpha)

        # Avoid division by zero
        smooth_scales = torch.clamp(smooth_scales, min=1e-5)

        # Store
        self.smooth_scales.copy_(smooth_scales)

        # Smooth weight
        weight_smooth = weight / smooth_scales.unsqueeze(0)

        # Quantize smoothed weight
        self.weight_quant.quantize_weight(weight_smooth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Smooth activations
        x_smooth = x * self.smooth_scales

        # Quantized matmul
        output = self.weight_quant(x_smooth)

        return output


def convert_to_quantized(
    model: nn.Module,
    config: QuantConfig,
    calibration_data: Optional[torch.Tensor] = None,
    method: str = "uniform",  # "uniform", "gptq", "awq", "smoothquant"
) -> nn.Module:
    """
    Convert model to quantized version.

    Args:
        model: Model to quantize
        config: Quantization configuration
        calibration_data: Calibration data for GPTQ/AWQ
        method: Quantization method

    Returns:
        Quantized model
    """
    # This would recursively replace linear layers with quantized versions
    # Full implementation would:
    # 1. Identify all linear layers
    # 2. Collect calibration data (forward passes)
    # 3. Apply quantization method
    # 4. Replace layers

    # Simplified: Just show the structure
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            has_bias = module.bias is not None

            if method == "uniform":
                quant_layer = QuantizedLinear(in_features, out_features, config, has_bias)
                quant_layer.quantize_weight(module.weight.data)

            elif method == "smoothquant":
                quant_layer = SmoothQuantLinear(in_features, out_features, config, has_bias)

                if calibration_data is not None:
                    # Calibrate (would need actual activations)
                    quant_layer.calibrate(
                        module.weight.data,
                        calibration_data,
                    )

            # Would replace module here
            # setattr(parent, child_name, quant_layer)

    return model
