"""
Token Embeddings
================

Token embedding layers with various initialization strategies.
"""

import torch
import torch.nn as nn
from typing import Optional


class TokenEmbedding(nn.Module):
    """
    Token embedding layer with optional scaling and freezing.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        padding_idx: Index for padding token
        max_norm: Maximum norm for embedding vectors
        norm_type: Type of norm (2.0 for L2 norm)
        scale_grad_by_freq: Scale gradients by token frequency
        sparse: Whether to use sparse gradients
        freeze: Whether to freeze embeddings (no training)
        init_std: Standard deviation for initialization
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        freeze: bool = False,
        init_std: float = 0.02,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )

        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
        if padding_idx is not None:
            nn.init.constant_(self.embedding.weight[padding_idx], 0.0)

        # Freeze if requested
        if freeze:
            self.embedding.weight.requires_grad = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            embeddings: Token embeddings [batch_size, seq_len, d_model]
        """
        return self.embedding(input_ids)

    def from_pretrained(
        self,
        embeddings: torch.Tensor,
        freeze: bool = True,
        padding_idx: Optional[int] = None,
    ) -> None:
        """
        Load pretrained embeddings.

        Args:
            embeddings: Pretrained embedding tensor
            freeze: Whether to freeze the embeddings
            padding_idx: Padding index to zero out
        """
        assert embeddings.shape == self.embedding.weight.shape, \
            f"Shape mismatch: {embeddings.shape} vs {self.embedding.weight.shape}"

        self.embedding.weight.data.copy_(embeddings)

        if padding_idx is not None:
            self.embedding.weight.data[padding_idx].fill_(0)

        if freeze:
            self.embedding.weight.requires_grad = False

    def extra_repr(self) -> str:
        return f'vocab_size={self.vocab_size}, d_model={self.d_model}'
