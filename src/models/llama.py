"""
LLaMA Model
===========

Large Language Model Meta AI - Modern decoder-only architecture with:
- RMSNorm instead of LayerNorm
- SwiGLU activation
- Rotary Position Embeddings (RoPE)
- Grouped-Query Attention (GQA)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
import math
from ..embeddings import TokenEmbedding, RotaryPositionalEmbedding
from ..attention import GroupedQueryAttention
from ..core import RMSNorm, GLUFeedForward


@dataclass
class LLaMAConfig:
    """Configuration for LLaMA model."""
    vocab_size: int = 32000
    max_seq_len: int = 2048
    d_model: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8  # For GQA
    d_ff: int = 11008
    dropout: float = 0.0  # LLaMA doesn't use dropout
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_weights: bool = False
    pad_token_id: int = 0


class LLaMADecoderLayer(nn.Module):
    """
    LLaMA Decoder Layer with RMSNorm, SwiGLU, and RoPE.
    """

    def __init__(self, config: LLaMAConfig):
        super().__init__()

        # Pre-attention normalization
        self.input_layernorm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Grouped-query attention with RoPE
        self.self_attn = GroupedQueryAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout=config.dropout,
        )

        # RoPE for positional encoding
        self.rotary_emb = RotaryPositionalEmbedding(
            d_model=config.d_model // config.num_heads,
            max_len=config.max_seq_len,
            base=config.rope_theta,
        )

        # Pre-feedforward normalization
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.norm_eps)

        # SwiGLU feed-forward
        self.mlp = GLUFeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            glu_variant="swiglu",
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with RoPE
        residual = x
        x = self.input_layernorm(x)

        # Apply attention
        attn_output, _ = self.self_attn(
            query=x,
            attention_mask=attention_mask,
        )

        x = residual + attn_output

        # Feed-forward
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)

        return x


class LLaMAModel(nn.Module):
    """
    LLaMA: Large Language Model Meta AI.

    Modern decoder-only transformer with advanced architectural improvements.

    Args:
        config: Model configuration
    """

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            padding_idx=config.pad_token_id,
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            LLaMADecoderLayer(config)
            for _ in range(config.num_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Output projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights if specified
        if config.tie_weights:
            self.lm_head.weight = self.embed_tokens.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        return ~mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            position_ids: Position IDs
            labels: Labels for language modeling
            return_dict: Whether to return dictionary

        Returns:
            logits and optional loss
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.embed_tokens(input_ids)

        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, input_ids.device)

        # Combine with attention mask if provided
        if attention_mask is not None:
            causal_mask = causal_mask & attention_mask.unsqueeze(1)

        # Apply decoder layers
        for layer in self.layers:
            x = layer(
                x=x,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )

        # Final normalization
        x = self.norm(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        # Compute loss
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        if return_dict:
            return {"logits": logits, "loss": loss}

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus filtering
            do_sample: Whether to sample
            repetition_penalty: Penalty for repeating tokens

        Returns:
            Generated token IDs
        """
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                       else input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            outputs = self.forward(idx_cond, return_dict=True)
            logits = outputs["logits"]

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(input_ids.size(0)):
                    for token_id in set(input_ids[i].tolist()):
                        logits[i, token_id] /= repetition_penalty

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample or greedy
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
