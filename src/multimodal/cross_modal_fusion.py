"""
Cross-Modal Fusion
==================

Research-grounded implementations for text-image understanding:

1. CLIP (Contrastive Language-Image Pre-training)
   - Reference: "Learning Transferable Visual Models From Natural Language Supervision"
                (Radford et al., ICML 2021)
   - Contrastive learning on image-text pairs

2. BLIP (Bootstrapping Language-Image Pre-training)
   - Reference: "BLIP: Bootstrapping Language-Image Pre-training" (Li et al., ICML 2022)
   - Unified vision-language understanding and generation

3. Flamingo
   - Reference: "Flamingo: a Visual Language Model for Few-Shot Learning"
                (Alayrac et al., NeurIPS 2022)
   - Interleaves vision and language with cross-attention

Mathematical Foundation:
-----------------------

CLIP Contrastive Loss:
  Image encoder: f(image) → v ∈ R^d
  Text encoder: g(text) → t ∈ R^d

  Similarity: s_ij = (v_i · t_j) / (||v_i|| ||t_j||)

  Loss: -1/N Σ_i log(exp(s_ii / τ) / Σ_j exp(s_ij / τ))

  where τ = temperature parameter

  Symmetric: Loss on both image→text and text→image directions

Cross-Attention Fusion:
  Vision features: V ∈ R^(N_v × d_v)
  Text features: T ∈ R^(N_t × d_t)

  Attention: A = softmax((T W_Q)(V W_K)^T / √d)
  Output: O = A(V W_V)

  where W_Q, W_K, W_V are projection matrices

Complexity:
- CLIP encoding: O(N_v × d_v) + O(N_t × d_t) (parallel)
- Contrastive loss: O(B^2 × d) where B = batch size
- Cross-attention: O(N_v × N_t × d)

Empirical Results (from papers):
- CLIP: 76.2% zero-shot ImageNet (without seeing ImageNet)
- BLIP: 82.3% on VQA
- Flamingo: 82.0% on VQAv2 with 32 shots
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CLIPConfig:
    """Configuration for CLIP model."""
    # Vision encoder
    vision_model: str = "vit"  # "vit" or "resnet"
    image_size: int = 224
    vision_d_model: int = 768
    vision_num_layers: int = 12

    # Text encoder
    text_d_model: int = 512
    text_num_layers: int = 12
    vocab_size: int = 49408

    # Shared
    projection_dim: int = 512  # Dimension for contrastive learning

    # Training
    temperature: float = 0.07  # Temperature for contrastive loss


class CLIPVisionEncoder(nn.Module):
    """
    CLIP Vision Encoder.

    Uses Vision Transformer to encode images.
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config

        from .vision_transformer import ViTConfig, VisionTransformer

        # ViT configuration
        vit_config = ViTConfig(
            image_size=config.image_size,
            d_model=config.vision_d_model,
            num_layers=config.vision_num_layers,
            num_classes=config.projection_dim,  # Project to shared space
        )

        self.vit = VisionTransformer(vit_config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images.

        Args:
            images: [batch, channels, height, width]

        Returns:
            Image embeddings [batch, projection_dim]
        """
        outputs = self.vit(images)
        return outputs["logits"]  # Actually embeddings, not logits


class CLIPTextEncoder(nn.Module):
    """
    CLIP Text Encoder.

    Uses Transformer to encode text.
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.text_d_model)

        # Positional embedding
        self.positional_embedding = nn.Parameter(
            torch.zeros(77, config.text_d_model)  # Max sequence length = 77
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.text_d_model,
            nhead=8,
            dim_feedforward=config.text_d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.text_num_layers,
        )

        # Projection to shared space
        self.projection = nn.Linear(config.text_d_model, config.projection_dim)

        self.ln_final = nn.LayerNorm(config.text_d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            Text embeddings [batch, projection_dim]
        """
        seq_len = input_ids.shape[1]

        # Embed tokens
        x = self.token_embedding(input_ids)

        # Add positional embeddings
        x = x + self.positional_embedding[:seq_len, :]

        # Create causal mask for Transformer
        if attention_mask is not None:
            # Convert to float and invert (Transformer expects 0 for valid, -inf for masked)
            mask = (1 - attention_mask).bool()
        else:
            mask = None

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=mask)

        # Layer norm
        x = self.ln_final(x)

        # Pool: Take features from EOT token (end of text)
        # For simplicity, use last token
        pooled = x[:, -1, :]

        # Project to shared space
        embeddings = self.projection(pooled)

        return embeddings


class CLIP(nn.Module):
    """
    CLIP: Contrastive Language-Image Pre-training.

    Reference: Radford et al., ICML 2021

    Key Innovation:
    - Learns aligned image-text representations
    - Zero-shot transfer to downstream tasks
    - Trained with contrastive learning

    Training:
    - Given batch of (image, text) pairs
    - Maximize similarity of matching pairs
    - Minimize similarity of non-matching pairs
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config

        # Encoders
        self.vision_encoder = CLIPVisionEncoder(config)
        self.text_encoder = CLIPTextEncoder(config)

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings."""
        embeddings = self.vision_encoder(images)
        # L2 normalize
        return F.normalize(embeddings, dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text to embeddings."""
        embeddings = self.text_encoder(input_ids, attention_mask)
        # L2 normalize
        return F.normalize(embeddings, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with contrastive loss.

        Args:
            images: [batch, channels, height, width]
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            Dictionary with loss and logits
        """
        # Encode
        image_embeds = self.encode_image(images)  # [B, D]
        text_embeds = self.encode_text(input_ids, attention_mask)  # [B, D]

        # Cosine similarity (already normalized)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.T  # [B, B]
        logits_per_text = logits_per_image.T  # [B, B]

        # Contrastive loss
        batch_size = images.shape[0]
        labels = torch.arange(batch_size, device=images.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        return {
            "loss": loss,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
        }

    def get_similarity(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute image-text similarity.

        Args:
            images: [batch_images, C, H, W]
            input_ids: [batch_texts, seq_len]
            attention_mask: [batch_texts, seq_len]

        Returns:
            Similarity matrix [batch_images, batch_texts]
        """
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(input_ids, attention_mask)

        # Cosine similarity
        similarity = image_embeds @ text_embeds.T

        return similarity


@dataclass
class CrossModalConfig:
    """Configuration for cross-modal fusion."""
    vision_d_model: int = 768
    text_d_model: int = 768
    num_cross_attention_layers: int = 6
    num_heads: int = 12
    dropout: float = 0.1


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention layer.

    Allows text to attend over visual features.
    """

    def __init__(self, config: CrossModalConfig):
        super().__init__()
        self.config = config

        # Project vision and text to same dimension
        hidden_size = config.text_d_model

        self.vision_proj = nn.Linear(config.vision_d_model, hidden_size)
        self.text_proj = nn.Linear(config.text_d_model, hidden_size)

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Layer norm
        self.norm = nn.LayerNorm(hidden_size)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(config.dropout),
        )

        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-modal attention.

        Args:
            text_features: [batch, text_len, text_d_model]
            vision_features: [batch, vision_len, vision_d_model]
            text_mask: [batch, text_len]
            vision_mask: [batch, vision_len]

        Returns:
            Fused features [batch, text_len, text_d_model]
        """
        # Project to same dimension
        text_proj = self.text_proj(text_features)
        vision_proj = self.vision_proj(vision_features)

        # Cross-attention: text queries attend over vision keys/values
        attn_output, _ = self.cross_attention(
            query=text_proj,
            key=vision_proj,
            value=vision_proj,
            key_padding_mask=vision_mask,
            need_weights=False,
        )

        # Residual connection
        text_features = text_features + attn_output
        text_features = self.norm(text_features)

        # FFN
        ffn_output = self.ffn(text_features)
        text_features = text_features + ffn_output
        text_features = self.ffn_norm(text_features)

        return text_features


class FlamingoGatedCrossAttention(nn.Module):
    """
    Flamingo-style gated cross-attention.

    Reference: Alayrac et al., NeurIPS 2022

    Key Innovation:
    - Tanh gating to control vision-text fusion
    - Initialized to zero (doesn't affect pre-trained LM initially)
    - Gradually learns to incorporate vision
    """

    def __init__(self, config: CrossModalConfig):
        super().__init__()
        self.config = config

        hidden_size = config.text_d_model

        # Cross-attention
        self.cross_attention = CrossModalAttention(config)

        # Tanh gating
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Gated cross-attention.

        Args:
            text_features: [batch, text_len, text_d_model]
            vision_features: [batch, vision_len, vision_d_model]
            text_mask: [batch, text_len]
            vision_mask: [batch, vision_len]

        Returns:
            Gated fused features
        """
        # Cross-attention
        fused = self.cross_attention(
            text_features,
            vision_features,
            text_mask,
            vision_mask,
        )

        # Gate
        gate_value = torch.tanh(self.gate)
        output = text_features + gate_value * (fused - text_features)

        return output


class MultimodalFusionTransformer(nn.Module):
    """
    Multimodal fusion transformer.

    Combines vision and language representations
    for tasks like VQA, image captioning, etc.
    """

    def __init__(
        self,
        config: CrossModalConfig,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
    ):
        super().__init__()
        self.config = config

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        # Cross-modal fusion layers
        self.fusion_layers = nn.ModuleList([
            CrossModalAttention(config)
            for _ in range(config.num_cross_attention_layers)
        ])

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with multimodal fusion.

        Args:
            images: [batch, C, H, W]
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            Fused multimodal features
        """
        # Encode vision
        vision_outputs = self.vision_encoder(images)
        vision_features = vision_outputs["hidden_states"][:, 1:]  # Remove CLS

        # Encode text
        text_outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_features = text_outputs  # Assume returns hidden states

        # Fuse modalities
        fused_features = text_features

        for layer in self.fusion_layers:
            fused_features = layer(
                text_features=fused_features,
                vision_features=vision_features,
                text_mask=(1 - attention_mask).bool() if attention_mask is not None else None,
            )

        return fused_features


class VisionLanguageModel(nn.Module):
    """
    Vision-Language Model for tasks like VQA.

    Combines CLIP-style encoders with cross-modal fusion
    for answering questions about images.
    """

    def __init__(
        self,
        clip_config: CLIPConfig,
        cross_modal_config: CrossModalConfig,
        num_answers: int = 3000,  # VQA answer space
    ):
        super().__init__()

        # CLIP encoders
        self.clip = CLIP(clip_config)

        # Cross-modal fusion
        self.fusion = CrossModalAttention(cross_modal_config)

        # Answer classification head
        self.answer_head = nn.Linear(
            cross_modal_config.text_d_model,
            num_answers,
        )

    def forward(
        self,
        images: torch.Tensor,
        questions: torch.Tensor,
        question_mask: Optional[torch.Tensor] = None,
        answers: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for VQA.

        Args:
            images: [batch, C, H, W]
            questions: [batch, seq_len]
            question_mask: [batch, seq_len]
            answers: [batch] Optional answer labels

        Returns:
            Dictionary with logits and loss
        """
        # Encode (without normalization for fusion)
        image_features = self.clip.vision_encoder(images).unsqueeze(1)  # [B, 1, D]
        text_features = self.clip.text_encoder(questions, question_mask).unsqueeze(1)  # [B, 1, D]

        # Cross-modal fusion
        fused = self.fusion(
            text_features=text_features,
            vision_features=image_features,
        )

        # Pool and predict
        pooled = fused.mean(dim=1)
        logits = self.answer_head(pooled)

        outputs = {"logits": logits}

        if answers is not None:
            loss = F.cross_entropy(logits, answers)
            outputs["loss"] = loss

        return outputs
