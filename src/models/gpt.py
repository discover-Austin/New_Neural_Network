"""
GPT Model
=========

Generative Pre-trained Transformer (Decoder-only architecture).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from ..embeddings import TokenEmbedding, LearnedPositionalEncoding
from .transformer import TransformerDecoder


@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    activation: str = "gelu"
    norm_first: bool = True
    norm_type: str = "layer_norm"
    use_glu: bool = False
    tie_weights: bool = True
    pad_token_id: int = 0


class GPTModel(nn.Module):
    """
    GPT: Generative Pre-trained Transformer.

    Decoder-only transformer for autoregressive language modeling.

    Args:
        config: Model configuration
    """

    def __init__(self, config: GPTConfig):
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

        # Transformer decoder (no cross-attention for GPT)
        self.decoder = TransformerDecoder(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            norm_first=config.norm_first,
            norm_type=config.norm_type,
            use_glu=config.use_glu,
            has_cross_attention=False,
        )

        # Output projection (language modeling head)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights between embedding and output projection
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.embedding.weight

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

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask for autoregressive generation."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        return ~mask  # Invert: 1 = attend, 0 = mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for language modeling [batch_size, seq_len]
            return_loss: Whether to compute and return loss

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            loss: Language modeling loss (if labels provided)
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids)

        # Add positional embeddings
        x = self.position_embedding(x)

        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, input_ids.device)

        # Apply transformer decoder
        x = self.decoder(
            x=x,
            attention_mask=causal_mask,
            key_padding_mask=attention_mask if attention_mask is not None else None,
        )

        # Project to vocabulary
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None and return_loss:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus (top-p) filtering
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                       else input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self.forward(idx_cond, return_loss=False)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Apply nucleus (top-p) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample or greedy decode
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, idx_next], dim=1)

        return input_ids

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Count number of parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
