"""
Transformer Building Blocks
============================

Core transformer encoder and decoder layers.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from ..attention import MultiHeadAttention, CrossAttention
from ..core import FeedForward, GLUFeedForward, LayerNorm, RMSNorm


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer.

    Consists of:
    1. Multi-head self-attention
    2. Feed-forward network
    With residual connections and layer normalization.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        activation: Activation function
        norm_first: Whether to apply norm before attention (Pre-LN)
        norm_type: Type of normalization ("layer_norm" or "rms_norm")
        use_glu: Whether to use GLU feed-forward
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        norm_type: str = "layer_norm",
        use_glu: bool = False,
    ):
        super().__init__()

        self.norm_first = norm_first

        # Self-attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Feed-forward
        if use_glu:
            self.feed_forward = GLUFeedForward(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
            )
        else:
            self.feed_forward = FeedForward(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
            )

        # Normalization
        if norm_type == "rms_norm":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask
            key_padding_mask: Padding mask

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        if self.norm_first:
            # Pre-LN (more stable for deep networks)
            # Self-attention block
            x_norm = self.norm1(x)
            attn_output, _ = self.self_attn(
                query=x_norm,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
            )
            x = x + self.dropout(attn_output)

            # Feed-forward block
            x_norm = self.norm2(x)
            ff_output = self.feed_forward(x_norm)
            x = x + ff_output
        else:
            # Post-LN (original transformer)
            # Self-attention block
            attn_output, _ = self.self_attn(
                query=x,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
            )
            x = self.norm1(x + self.dropout(attn_output))

            # Feed-forward block
            ff_output = self.feed_forward(x)
            x = self.norm2(x + ff_output)

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer.

    Consists of:
    1. Masked multi-head self-attention
    2. Cross-attention (optional, for encoder-decoder)
    3. Feed-forward network
    With residual connections and layer normalization.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        activation: Activation function
        norm_first: Whether to apply norm before attention (Pre-LN)
        norm_type: Type of normalization
        use_glu: Whether to use GLU feed-forward
        has_cross_attention: Whether to include cross-attention
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        norm_type: str = "layer_norm",
        use_glu: bool = False,
        has_cross_attention: bool = True,
    ):
        super().__init__()

        self.norm_first = norm_first
        self.has_cross_attention = has_cross_attention

        # Self-attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Cross-attention (for encoder-decoder)
        if has_cross_attention:
            self.cross_attn = CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )

        # Feed-forward
        if use_glu:
            self.feed_forward = GLUFeedForward(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
            )
        else:
            self.feed_forward = FeedForward(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
            )

        # Normalization
        if norm_type == "rms_norm":
            self.norm1 = RMSNorm(d_model)
            if has_cross_attention:
                self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        else:
            self.norm1 = LayerNorm(d_model)
            if has_cross_attention:
                self.norm2 = LayerNorm(d_model)
            self.norm3 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, tgt_len, d_model]
            encoder_output: Encoder output for cross-attention
            attention_mask: Self-attention mask (causal)
            key_padding_mask: Padding mask for decoder
            encoder_attention_mask: Cross-attention mask
            encoder_key_padding_mask: Padding mask for encoder

        Returns:
            Output tensor [batch_size, tgt_len, d_model]
        """
        if self.norm_first:
            # Pre-LN
            # Self-attention block
            x_norm = self.norm1(x)
            attn_output, _ = self.self_attn(
                query=x_norm,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
            )
            x = x + self.dropout(attn_output)

            # Cross-attention block
            if self.has_cross_attention and encoder_output is not None:
                x_norm = self.norm2(x)
                cross_attn_output, _ = self.cross_attn(
                    query=x_norm,
                    key=encoder_output,
                    value=encoder_output,
                    attention_mask=encoder_attention_mask,
                    key_padding_mask=encoder_key_padding_mask,
                )
                x = x + self.dropout(cross_attn_output)

            # Feed-forward block
            x_norm = self.norm3(x)
            ff_output = self.feed_forward(x_norm)
            x = x + ff_output
        else:
            # Post-LN
            # Self-attention block
            attn_output, _ = self.self_attn(
                query=x,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
            )
            x = self.norm1(x + self.dropout(attn_output))

            # Cross-attention block
            if self.has_cross_attention and encoder_output is not None:
                cross_attn_output, _ = self.cross_attn(
                    query=x,
                    key=encoder_output,
                    value=encoder_output,
                    attention_mask=encoder_attention_mask,
                    key_padding_mask=encoder_key_padding_mask,
                )
                x = self.norm2(x + self.dropout(cross_attn_output))

            # Feed-forward block
            ff_output = self.feed_forward(x)
            x = self.norm3(x + ff_output)

        return x


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer Encoder Layers.

    Args:
        num_layers: Number of encoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        activation: Activation function
        norm_first: Whether to use Pre-LN
        norm_type: Type of normalization
        use_glu: Whether to use GLU feed-forward
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        norm_type: str = "layer_norm",
        use_glu: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                norm_type=norm_type,
                use_glu=use_glu,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        if norm_type == "rms_norm":
            self.norm = RMSNorm(d_model)
        else:
            self.norm = LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through all encoder layers.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask
            key_padding_mask: Padding mask

        Returns:
            Encoder output [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, attention_mask, key_padding_mask)

        x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    """
    Stack of Transformer Decoder Layers.

    Args:
        num_layers: Number of decoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        activation: Activation function
        norm_first: Whether to use Pre-LN
        norm_type: Type of normalization
        use_glu: Whether to use GLU feed-forward
        has_cross_attention: Whether to include cross-attention
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        norm_type: str = "layer_norm",
        use_glu: bool = False,
        has_cross_attention: bool = True,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                norm_type=norm_type,
                use_glu=use_glu,
                has_cross_attention=has_cross_attention,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        if norm_type == "rms_norm":
            self.norm = RMSNorm(d_model)
        else:
            self.norm = LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through all decoder layers.

        Args:
            x: Input tensor [batch_size, tgt_len, d_model]
            encoder_output: Encoder output for cross-attention
            attention_mask: Self-attention mask (causal)
            key_padding_mask: Padding mask for decoder
            encoder_attention_mask: Cross-attention mask
            encoder_key_padding_mask: Padding mask for encoder

        Returns:
            Decoder output [batch_size, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(
                x=x,
                encoder_output=encoder_output,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask,
                encoder_attention_mask=encoder_attention_mask,
                encoder_key_padding_mask=encoder_key_padding_mask,
            )

        x = self.norm(x)
        return x
