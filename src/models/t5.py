"""
T5 Model
========

Text-to-Text Transfer Transformer (Encoder-Decoder architecture).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from ..embeddings import TokenEmbedding, RelativePositionalEncoding
from .transformer import TransformerEncoder, TransformerDecoder


@dataclass
class T5Config:
    """Configuration for T5 model."""
    vocab_size: int = 32128
    max_seq_len: int = 512
    d_model: int = 512
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    d_kv: int = 64  # Key/Value dimension per head
    dropout: float = 0.1
    activation: str = "relu"
    norm_first: bool = True  # T5 uses Pre-LN
    norm_type: str = "rms_norm"  # T5 uses RMSNorm
    use_glu: bool = True  # T5 uses GeGLU
    tie_weights: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int = 0


class T5Model(nn.Module):
    """
    T5: Text-to-Text Transfer Transformer.

    Encoder-decoder transformer that treats all NLP tasks as text-to-text.

    Args:
        config: Model configuration
    """

    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config

        # Shared token embeddings
        self.shared_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            padding_idx=config.pad_token_id,
        )

        # Transformer encoder
        self.encoder = TransformerEncoder(
            num_layers=config.num_encoder_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            norm_first=config.norm_first,
            norm_type=config.norm_type,
            use_glu=config.use_glu,
        )

        # Transformer decoder
        self.decoder = TransformerDecoder(
            num_layers=config.num_decoder_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            norm_first=config.norm_first,
            norm_type=config.norm_type,
            use_glu=config.use_glu,
            has_cross_attention=True,
        )

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights
        if config.tie_weights:
            self.lm_head.weight = self.shared_embedding.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for decoder."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        return ~mask

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Encoder input token IDs [batch_size, src_len]
            decoder_input_ids: Decoder input token IDs [batch_size, tgt_len]
            attention_mask: Encoder attention mask [batch_size, src_len]
            decoder_attention_mask: Decoder attention mask [batch_size, tgt_len]
            labels: Target labels [batch_size, tgt_len]
            return_dict: Whether to return dictionary

        Returns:
            Dictionary containing encoder/decoder outputs and loss
        """
        # Encode
        encoder_input_embeds = self.shared_embedding(input_ids)
        encoder_output = self.encoder(
            x=encoder_input_embeds,
            key_padding_mask=attention_mask if attention_mask is not None else None,
        )

        # Prepare decoder inputs
        if decoder_input_ids is None and labels is not None:
            # Shift labels for teacher forcing
            decoder_input_ids = self._shift_right(labels)

        # Decode
        if decoder_input_ids is not None:
            decoder_input_embeds = self.shared_embedding(decoder_input_ids)
            seq_len = decoder_input_ids.size(1)
            causal_mask = self._create_causal_mask(seq_len, decoder_input_ids.device)

            decoder_output = self.decoder(
                x=decoder_input_embeds,
                encoder_output=encoder_output,
                attention_mask=causal_mask,
                key_padding_mask=decoder_attention_mask if decoder_attention_mask is not None else None,
                encoder_key_padding_mask=attention_mask if attention_mask is not None else None,
            )

            # Project to vocabulary
            logits = self.lm_head(decoder_output)
        else:
            logits = None
            decoder_output = None

        # Compute loss
        loss = None
        if labels is not None and logits is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )

        if return_dict:
            return {
                "encoder_output": encoder_output,
                "decoder_output": decoder_output,
                "logits": logits,
                "loss": loss,
            }

        return encoder_output, decoder_output, logits, loss

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Shift input ids to the right for teacher forcing."""
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = self.config.decoder_start_token_id
        return shifted_input_ids

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """
        Generate sequences.

        Args:
            input_ids: Encoder input IDs [batch_size, src_len]
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus filtering
            do_sample: Whether to sample

        Returns:
            Generated token IDs [batch_size, max_length]
        """
        batch_size = input_ids.size(0)

        # Encode once
        encoder_input_embeds = self.shared_embedding(input_ids)
        encoder_output = self.encoder(x=encoder_input_embeds)

        # Start with decoder start token
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=input_ids.device
        )

        # Generate autoregressively
        for _ in range(max_length - 1):
            # Forward decoder
            decoder_input_embeds = self.shared_embedding(decoder_input_ids)
            seq_len = decoder_input_ids.size(1)
            causal_mask = self._create_causal_mask(seq_len, decoder_input_ids.device)

            decoder_output = self.decoder(
                x=decoder_input_embeds,
                encoder_output=encoder_output,
                attention_mask=causal_mask,
            )

            # Get logits for last token
            logits = self.lm_head(decoder_output[:, -1, :])
            logits = logits / temperature

            # Sample next token
            if do_sample:
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append to sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

            # Check if all sequences have generated EOS
            if (next_token == self.config.eos_token_id).all():
                break

        return decoder_input_ids

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Count number of parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
