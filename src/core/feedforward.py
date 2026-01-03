"""
Feed-Forward Networks
=====================

Various feed-forward network architectures for transformers.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
from .activations import SwiGLU, GeGLU, ReGLU, GELU, Swish


class FeedForward(nn.Module):
    """
    Standard feed-forward network with configurable activation.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension (typically 4 * d_model)
        dropout: Dropout probability
        activation: Activation function to use
        bias: Whether to use bias
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu", "swish", "silu"] = "gelu",
        bias: bool = True,
    ):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # Select activation
        if activation == "gelu":
            self.activation = GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation in ["swish", "silu"]:
            self.activation = Swish()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [..., d_model]

        Returns:
            Output tensor [..., d_model]
        """
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x


class GLUFeedForward(nn.Module):
    """
    Feed-forward network using GLU variants (SwiGLU, GeGLU, ReGLU).

    GLU variants have been shown to improve transformer performance.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        glu_variant: Which GLU variant to use
        bias: Whether to use bias
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        glu_variant: Literal["swiglu", "geglu", "reglu"] = "swiglu",
        bias: bool = False,
    ):
        super().__init__()

        # GLU variants expand then gate, so we use d_ff as the gated dimension
        if glu_variant == "swiglu":
            self.glu = SwiGLU(d_model, d_ff, bias=bias)
        elif glu_variant == "geglu":
            self.glu = GeGLU(d_model, d_ff, bias=bias)
        elif glu_variant == "reglu":
            self.glu = ReGLU(d_model, d_ff, bias=bias)
        else:
            raise ValueError(f"Unknown GLU variant: {glu_variant}")

        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [..., d_model]

        Returns:
            Output tensor [..., d_model]
        """
        x = self.glu(x)
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x


class MoEFeedForward(nn.Module):
    """
    Mixture of Experts (MoE) Feed-Forward Network.

    Routes inputs to a subset of expert networks based on a learned gating function.
    Enables scaling model capacity without proportional compute increase.

    Reference: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension per expert
        num_experts: Number of expert networks
        num_experts_per_token: Number of experts to route each token to
        dropout: Dropout probability
        expert_capacity_factor: Capacity factor for load balancing
        jitter_noise: Noise for load balancing
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        dropout: float = 0.1,
        expert_capacity_factor: float = 1.25,
        jitter_noise: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.expert_capacity_factor = expert_capacity_factor
        self.jitter_noise = jitter_noise

        # Gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # Expert networks
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_ff, dropout=dropout)
            for _ in range(num_experts)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with expert routing.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Flatten for routing
        x_flat = x.view(-1, d_model)  # [batch*seq, d_model]

        # Compute gating scores
        gate_logits = self.gate(x_flat)  # [batch*seq, num_experts]

        # Add jitter noise during training for load balancing
        if self.training and self.jitter_noise > 0:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.jitter_noise

        # Get top-k experts per token
        gate_scores = torch.softmax(gate_logits, dim=-1)
        top_k_scores, top_k_indices = torch.topk(
            gate_scores,
            self.num_experts_per_token,
            dim=-1
        )

        # Normalize top-k scores
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)

        # Route to experts
        output = torch.zeros_like(x_flat)

        for i in range(self.num_experts_per_token):
            expert_indices = top_k_indices[:, i]
            expert_scores = top_k_scores[:, i].unsqueeze(-1)

            for expert_id in range(self.num_experts):
                # Get tokens routed to this expert
                expert_mask = (expert_indices == expert_id)

                if expert_mask.any():
                    expert_input = x_flat[expert_mask]
                    expert_output = self.experts[expert_id](expert_input)

                    # Weight by gating score
                    weighted_output = expert_output * expert_scores[expert_mask]
                    output[expert_mask] += weighted_output

        # Reshape back
        output = output.view(batch_size, seq_len, d_model)
        return self.dropout(output)

    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, d_ff={self.d_ff}, '
                f'num_experts={self.num_experts}, num_experts_per_token={self.num_experts_per_token}')
