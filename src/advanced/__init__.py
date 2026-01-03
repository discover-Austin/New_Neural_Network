"""
Research-Verified Advanced Transformer Components
==================================================

This module contains only components with peer-reviewed validation and
mathematical verification. Each component includes:

1. Theoretical foundation and complexity analysis
2. Citations to peer-reviewed papers
3. Mathematical proofs where applicable
4. Empirical validation references

All implementations prioritize:
- Mathematical correctness
- Numerical stability
- Computational efficiency
- Empirical validation
"""

from .multi_scale_transformer import (
    MultiScaleTransformer,
    MultiScaleConfig,
    HierarchicalPooling,
    CrossScaleFusion,
)

from .memory_augmented import (
    MemoryAugmentedTransformer,
    kNNMemory,
    CompressiveMemory,
)

from .calibrated_exit import (
    CalibratedEarlyExit,
    ConfidenceCalibrator,
    ExitDecisionModule,
)

__all__ = [
    "MultiScaleTransformer",
    "MultiScaleConfig",
    "HierarchicalPooling",
    "CrossScaleFusion",
    "MemoryAugmentedTransformer",
    "kNNMemory",
    "CompressiveMemory",
    "CalibratedEarlyExit",
    "ConfidenceCalibrator",
    "ExitDecisionModule",
]
