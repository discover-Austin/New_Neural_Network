"""
BERT Model
==========

Bidirectional Encoder Representations from Transformers (Encoder-only architecture).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from ..embeddings import TokenEmbedding, LearnedPositionalEncoding
from .transformer import TransformerEncoder


@dataclass
class BERTConfig:
    """Configuration for BERT model."""
    vocab_size: int = 30522
    max_seq_len: int = 512
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    activation: str = "gelu"
    norm_first: bool = False  # BERT uses Post-LN
    norm_type: str = "layer_norm"
    use_glu: bool = False
    pad_token_id: int = 0
    num_token_types: int = 2  # For segment embeddings


class BERTModel(nn.Module):
    """
    BERT: Bidirectional Encoder Representations from Transformers.

    Encoder-only transformer for masked language modeling and other NLU tasks.

    Args:
        config: Model configuration
    """

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            padding_idx=config.pad_token_id,
        )

        # Positional embeddings
        self.position_embedding = LearnedPositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Token type (segment) embeddings
        self.token_type_embedding = nn.Embedding(config.num_token_types, config.d_model)

        # Transformer encoder
        self.encoder = TransformerEncoder(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            norm_first=config.norm_first,
            norm_type=config.norm_type,
            use_glu=config.use_glu,
        )

        # Pooler for sentence-level tasks
        self.pooler = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
        )

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.vocab_size, bias=False),
        )

        # NSP head (Next Sentence Prediction)
        self.nsp_head = nn.Linear(config.d_model, 2)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            token_type_ids: Token type IDs for segment embeddings [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Masked token labels [batch_size, seq_len]
            next_sentence_label: NSP labels [batch_size]
            return_dict: Whether to return dictionary

        Returns:
            Dictionary containing:
                - last_hidden_state: Encoder output
                - pooled_output: Pooled representation
                - mlm_logits: MLM prediction logits
                - nsp_logits: NSP prediction logits
                - mlm_loss: MLM loss (if labels provided)
                - nsp_loss: NSP loss (if next_sentence_label provided)
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids)

        # Add positional embeddings
        x = self.position_embedding(x)

        # Add token type embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeds = self.token_type_embedding(token_type_ids)
        x = x + token_type_embeds

        # Apply transformer encoder
        encoder_output = self.encoder(
            x=x,
            key_padding_mask=attention_mask if attention_mask is not None else None,
        )

        # Pool for sentence-level representation (use [CLS] token)
        pooled_output = self.pooler(encoder_output[:, 0])

        # MLM predictions
        mlm_logits = self.mlm_head(encoder_output)

        # NSP predictions
        nsp_logits = self.nsp_head(pooled_output)

        # Compute losses
        mlm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(
                mlm_logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )

        nsp_loss = None
        if next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            nsp_loss = loss_fct(nsp_logits, next_sentence_label)

        if return_dict:
            return {
                "last_hidden_state": encoder_output,
                "pooled_output": pooled_output,
                "mlm_logits": mlm_logits,
                "nsp_logits": nsp_logits,
                "mlm_loss": mlm_loss,
                "nsp_loss": nsp_loss,
            }

        return encoder_output, pooled_output, mlm_logits, nsp_logits, mlm_loss, nsp_loss

    def encode(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get sentence embeddings.

        Args:
            input_ids: Input token IDs
            token_type_ids: Token type IDs
            attention_mask: Attention mask

        Returns:
            Sentence embeddings [batch_size, d_model]
        """
        outputs = self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs["pooled_output"]

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Count number of parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
