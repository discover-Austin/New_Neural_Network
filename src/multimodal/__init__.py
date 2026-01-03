"""
Multimodal Components
=====================

Research-grounded implementations for vision-language understanding:

1. Vision Transformer (ViT)
   - Patch-based image encoding
   - Transformer architecture for vision
   - DeiT (data-efficient variant)
   - Hybrid ViT with CNN backbone

2. Cross-Modal Fusion
   - CLIP (Contrastive Language-Image Pre-training)
   - Cross-attention between vision and text
   - Flamingo-style gated fusion
   - Vision-Language models for VQA

Citations:
- ViT: Dosovitskiy et al., ICLR 2021
- DeiT: Touvron et al., ICML 2021
- CLIP: Radford et al., ICML 2021
- BLIP: Li et al., ICML 2022
- Flamingo: Alayrac et al., NeurIPS 2022
"""

from .vision_transformer import (
    ViTConfig,
    PatchEmbedding,
    VisionTransformer,
    HybridViT,
    DeiT,
)

from .cross_modal_fusion import (
    CLIPConfig,
    CLIP,
    CrossModalConfig,
    CrossModalAttention,
    FlamingoGatedCrossAttention,
    MultimodalFusionTransformer,
    VisionLanguageModel,
)

__all__ = [
    # Vision Transformer
    "ViTConfig",
    "PatchEmbedding",
    "VisionTransformer",
    "HybridViT",
    "DeiT",
    # Cross-Modal
    "CLIPConfig",
    "CLIP",
    "CrossModalConfig",
    "CrossModalAttention",
    "FlamingoGatedCrossAttention",
    "MultimodalFusionTransformer",
    "VisionLanguageModel",
]
