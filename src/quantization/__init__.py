"""
Model Quantization
==================

Research-grounded quantization methods for efficient deployment:

1. Uniform Quantization (INT8, INT4)
   - Post-training quantization
   - Per-channel and per-tensor scaling

2. GPTQ (Gradient-based PTQ)
   - Layer-wise quantization
   - Uses Hessian for optimal weight updates

3. AWQ (Activation-aware Weight Quantization)
   - Protects salient weights
   - Optimal per-channel scaling

4. SmoothQuant
   - Smooths activation outliers
   - Enables INT8 for LLMs

Citations:
- GPTQ: Frantar et al., ICLR 2023
- AWQ: Lin et al., 2023
- SmoothQuant: Xiao et al., 2023
"""

from .quantization import (
    QuantConfig,
    UniformQuantizer,
    QuantizedLinear,
    GPTQQuantizer,
    AWQQuantizer,
    SmoothQuantLinear,
    convert_to_quantized,
)

__all__ = [
    "QuantConfig",
    "UniformQuantizer",
    "QuantizedLinear",
    "GPTQQuantizer",
    "AWQQuantizer",
    "SmoothQuantLinear",
    "convert_to_quantized",
]
