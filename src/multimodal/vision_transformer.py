"""
Vision Transformer (ViT)
=========================

Research-grounded implementation of Vision Transformer:

Reference: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
           (Dosovitskiy et al., ICLR 2021)

Key Innovation:
- Applies Transformer directly to image patches
- No convolutions needed
- Matches or exceeds CNNs when pre-trained on large datasets

Architecture:
1. Split image into fixed-size patches
2. Linearly embed each patch
3. Add positional embeddings
4. Process through Transformer encoder
5. Classify using [CLS] token

Mathematical Foundation:
-----------------------

Image Patching:
  Image: H × W × C
  Patch size: P × P
  Number of patches: N = HW / P²

Patch Embedding:
  Flatten patch: x_p ∈ R^(P²·C)
  Linear projection: E(x_p) ∈ R^D
  where D = embedding dimension

Positional Encoding:
  Add learnable position embeddings to preserve spatial information
  z_0 = [x_cls; E(x_p1); E(x_p2); ...; E(x_pN)] + E_pos

Transformer Encoder:
  z_ℓ = MSA(LN(z_{ℓ-1})) + z_{ℓ-1}
  z_ℓ = MLP(LN(z_ℓ)) + z_ℓ

  where MSA = Multi-Head Self-Attention

Classification:
  y = LN(z_L[0])  # Extract [CLS] token from final layer

Complexity:
- Patch embedding: O(HWC × D)
- Self-attention per layer: O(N² × D) where N = (H/P) × (W/P)
- Total: O(L × N² × D)

For 224×224 image with P=16:
- N = 196 patches
- Much more efficient than pixel-level attention O(H² W² D)

Empirical Results (from paper):
- ImageNet: 88.55% accuracy (ViT-H/14)
- JFT-300M pre-training: Best results
- Scales better than ResNets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ViTConfig:
    """Configuration for Vision Transformer."""
    # Image settings
    image_size: int = 224  # Input image size
    patch_size: int = 16  # Patch size (16x16 is standard)
    num_channels: int = 3  # RGB

    # Model settings
    d_model: int = 768  # Embedding dimension
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072  # FFN hidden dimension (usually 4 × d_model)
    dropout: float = 0.1

    # Task settings
    num_classes: int = 1000  # ImageNet classes

    # Pooling
    pooling: str = "cls"  # "cls" or "mean"

    @property
    def num_patches(self) -> int:
        """Number of patches."""
        return (self.image_size // self.patch_size) ** 2


class PatchEmbedding(nn.Module):
    """
    Splits image into patches and embeds them.

    Approach:
    1. Conv2d with kernel_size=patch_size, stride=patch_size
       - Effectively splits image into non-overlapping patches
       - Projects each patch to embedding dimension
    2. Flatten spatial dimensions
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        self.projection = nn.Conv2d(
            config.num_channels,
            config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Embed image patches.

        Args:
            images: [batch, channels, height, width]

        Returns:
            Patch embeddings [batch, num_patches, d_model]
        """
        # images: [B, C, H, W]
        x = self.projection(images)  # [B, D, H/P, W/P]

        # Flatten spatial dimensions
        batch_size, d_model, h, w = x.shape
        x = x.flatten(2)  # [B, D, H/P * W/P]
        x = x.transpose(1, 2)  # [B, num_patches, D]

        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder.

    Standard Transformer encoder with:
    - Multi-Head Self-Attention
    - Layer Normalization (pre-norm)
    - Feed-Forward Network
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        # Layers
        self.layers = nn.ModuleList([
            ViTEncoderLayer(config) for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: Optional attention mask

        Returns:
            Encoded representations
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)

        return hidden_states


class ViTEncoderLayer(nn.Module):
    """Single Vision Transformer encoder layer."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        # Pre-LayerNorm
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.ffn_norm = nn.LayerNorm(config.d_model)

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-norm residual connections."""
        # Multi-Head Attention
        normed = self.attention_norm(hidden_states)
        attn_output, _ = self.attention(
            normed, normed, normed,
            key_padding_mask=attention_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + attn_output

        # Feed-Forward Network
        normed = self.ffn_norm(hidden_states)
        ffn_output = self.ffn(normed)
        hidden_states = hidden_states + ffn_output

        return hidden_states


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT).

    Reference: Dosovitskiy et al., ICLR 2021

    Architecture:
    1. Patch embedding
    2. Add [CLS] token and positional embeddings
    3. Transformer encoder
    4. Classification head
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embedding = PatchEmbedding(config)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))

        # Positional embeddings
        # +1 for CLS token
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.d_model)
        )

        self.dropout = nn.Dropout(config.dropout)

        # Transformer encoder
        self.encoder = ViTEncoder(config)

        # Classification head
        self.head = nn.Linear(config.d_model, config.num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # CLS token
        nn.init.normal_(self.cls_token, std=0.02)

        # Positional embeddings
        nn.init.normal_(self.pos_embedding, std=0.02)

        # Classification head
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            images: [batch, channels, height, width]
            labels: Optional labels for classification

        Returns:
            Dictionary with logits and optional loss
        """
        batch_size = images.shape[0]

        # Patch embedding
        patch_embeds = self.patch_embedding(images)  # [B, N, D]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        embeddings = torch.cat([cls_tokens, patch_embeds], dim=1)  # [B, N+1, D]

        # Add positional embeddings
        embeddings = embeddings + self.pos_embedding
        embeddings = self.dropout(embeddings)

        # Transformer encoder
        hidden_states = self.encoder(embeddings)

        # Pool
        if self.config.pooling == "cls":
            pooled = hidden_states[:, 0]  # CLS token
        elif self.config.pooling == "mean":
            pooled = hidden_states[:, 1:].mean(dim=1)  # Average patches
        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")

        # Classification
        logits = self.head(pooled)

        outputs = {"logits": logits, "hidden_states": hidden_states}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs["loss"] = loss

        return outputs

    def get_patch_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get patch embeddings (useful for visualization or features).

        Args:
            images: [batch, channels, height, width]

        Returns:
            Patch embeddings [batch, num_patches, d_model]
        """
        patch_embeds = self.patch_embedding(images)
        return patch_embeds


class HybridViT(nn.Module):
    """
    Hybrid Vision Transformer.

    Reference: ViT paper section on hybrid models

    Uses CNN backbone (e.g., ResNet) to extract features,
    then applies Transformer.

    Advantages:
    - Better inductive bias from CNN
    - Works well with less pre-training data
    - Can use smaller patch sizes effectively
    """

    def __init__(
        self,
        config: ViTConfig,
        cnn_backbone: nn.Module,
        feature_dim: int,
    ):
        super().__init__()
        self.config = config

        # CNN backbone
        self.backbone = cnn_backbone

        # Project CNN features to ViT dimension
        self.feature_projection = nn.Linear(feature_dim, config.d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))

        # Positional embeddings (size depends on CNN output)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 64 + 1, config.d_model)  # 64 = 8×8 for typical ResNet
        )

        self.dropout = nn.Dropout(config.dropout)

        # Transformer encoder
        self.encoder = ViTEncoder(config)

        # Classification head
        self.head = nn.Linear(config.d_model, config.num_classes)

    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass."""
        batch_size = images.shape[0]

        # Extract CNN features
        cnn_features = self.backbone(images)  # [B, C, H, W]

        # Flatten and project
        cnn_features = cnn_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        embeddings = self.feature_projection(cnn_features)  # [B, H*W, D]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)

        # Add positional embeddings
        seq_len = embeddings.size(1)
        embeddings = embeddings + self.pos_embedding[:, :seq_len, :]
        embeddings = self.dropout(embeddings)

        # Transformer encoder
        hidden_states = self.encoder(embeddings)

        # CLS pooling
        pooled = hidden_states[:, 0]

        # Classification
        logits = self.head(pooled)

        outputs = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs["loss"] = loss

        return outputs


class DeiT(VisionTransformer):
    """
    Data-efficient Image Transformer (DeiT).

    Reference: "Training data-efficient image transformers & distillation through attention"
               (Touvron et al., ICML 2021)

    Key Innovation:
    - Knowledge distillation from CNN teacher
    - Distillation token (in addition to CLS token)
    - Trains efficiently without large-scale pre-training

    Improvements:
    - Matches ViT with far less data
    - 81.8% on ImageNet with only ImageNet training
    """

    def __init__(self, config: ViTConfig):
        super().__init__(config)

        # Distillation token
        self.dist_token = nn.Parameter(torch.zeros(1, 1, config.d_model))

        # Update positional embeddings for dist token
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches + 2, config.d_model)  # +2 for CLS and dist
        )

        # Distillation head
        self.dist_head = nn.Linear(config.d_model, config.num_classes)

        # Initialize
        nn.init.normal_(self.dist_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass with distillation."""
        batch_size = images.shape[0]

        # Patch embedding
        patch_embeds = self.patch_embedding(images)

        # Prepend CLS and dist tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        dist_tokens = self.dist_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, dist_tokens, patch_embeds], dim=1)

        # Add positional embeddings
        embeddings = embeddings + self.pos_embedding
        embeddings = self.dropout(embeddings)

        # Transformer encoder
        hidden_states = self.encoder(embeddings)

        # Two heads
        cls_output = self.head(hidden_states[:, 0])
        dist_output = self.dist_head(hidden_states[:, 1])

        # Average at inference, separate at training
        if self.training:
            logits = cls_output  # Use CLS for main loss
        else:
            logits = (cls_output + dist_output) / 2

        outputs = {
            "logits": logits,
            "cls_logits": cls_output,
            "dist_logits": dist_output,
        }

        if labels is not None:
            # Classification loss on CLS token
            cls_loss = F.cross_entropy(cls_output, labels)
            outputs["loss"] = cls_loss

        return outputs
