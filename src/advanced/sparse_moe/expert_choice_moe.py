"""
Expert Choice Routing for Mixture-of-Experts
=============================================

Revolutionary MoE where EXPERTS choose tokens (not tokens choose experts).

Reference: "Mixture-of-Experts with Expert Choice Routing" (Zhou et al., Google 2022)
         https://arxiv.org/abs/2202.09368

Key Innovation:
- Traditional MoE: Each token chooses top-k experts → Load imbalance
- Expert Choice: Each expert chooses top-k tokens → Perfect load balance

Advantages:
1. Perfect load balancing (no auxiliary loss needed)
2. Better training stability
3. Higher quality (experts see diverse tokens)
4. Scales to 1000s of experts
5. No token dropping

Mathematical Foundation:
-----------------------

Traditional Token Choice:
  For each token t, select experts: E_t = TopK(Router(t), k)
  Problem: Popular experts get overloaded

Expert Choice:
  For each expert e, select tokens: T_e = TopK(Router(t)[:, e], capacity)
  Solution: Each expert processes exactly 'capacity' tokens

Router:
  S = softmax(x W_r)  # [batch*seq, num_experts]
  For each expert e:
    scores_e = S[:, e]
    top_indices = TopK(scores_e, capacity)
    Process tokens[top_indices]

Complexity:
- Routing: O(N × E) where N=tokens, E=experts
- Compute: O(C × E × D²) where C=capacity per expert
- Communication: O(C × E × D)

vs Token Choice:
- Token Choice: O(N × K × D²) where K=experts per token
- Load imbalance can make K vary wildly
- Expert Choice: Always exactly C × E compute

Empirical Results (from paper):
- 2x better load balancing
- 1.5x faster training
- 5% better quality at same compute
- Scales to 4096 experts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class ExpertChoiceMoEConfig:
    """Configuration for Expert Choice MoE."""
    d_model: int = 1024
    num_experts: int = 128  # Can scale to 1000s
    expert_capacity_factor: float = 1.0  # Capacity = factor * (tokens / experts)
    d_ff: int = 4096
    dropout: float = 0.1
    activation: str = "gelu"
    use_bias: bool = True
    # Advanced options
    use_soft_routing: bool = False  # Soft vs hard routing
    router_jitter: float = 0.0  # Add noise to routing for exploration
    normalize_scores: bool = True  # Normalize expert scores


class ExpertChoiceRouter(nn.Module):
    """
    Expert Choice Routing: Experts choose which tokens to process.

    Each expert selects exactly 'capacity' tokens, ensuring perfect load balance.
    """

    def __init__(self, config: ExpertChoiceMoEConfig):
        super().__init__()
        self.config = config

        # Router projects tokens to expert scores
        self.router = nn.Linear(config.d_model, config.num_experts, bias=False)

        # Router z-loss for stability (from ST-MoE)
        self.use_router_z_loss = True
        self.router_z_loss_coef = 0.001

    def router_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Router z-loss: Encourages router logits to stay small.

        Prevents one expert from dominating by keeping logits bounded.
        L_z = (1/n) * sum(log(sum(exp(logits))))²
        """
        num_groups, tokens_per_group = router_logits.shape[0], router_logits.shape[1]
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = torch.square(log_z).sum() / (num_groups * tokens_per_group)
        return z_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Expert Choice Routing.

        Args:
            hidden_states: [batch_size, seq_len, d_model]
            training: Whether in training mode

        Returns:
            dispatch_mask: [num_experts, capacity, num_tokens] - sparse routing matrix
            combine_weights: [num_experts, capacity, num_tokens] - combining weights
            aux_loss_dict: Auxiliary losses for training
        """
        batch_size, seq_len, d_model = hidden_states.shape
        num_tokens = batch_size * seq_len

        # Reshape to [num_tokens, d_model]
        hidden_states_flat = hidden_states.reshape(-1, d_model)

        # Compute router logits: [num_tokens, num_experts]
        router_logits = self.router(hidden_states_flat)

        # Add jitter for exploration during training
        if training and self.config.router_jitter > 0:
            router_logits += torch.randn_like(router_logits) * self.config.router_jitter

        # Compute routing scores
        router_probs = F.softmax(router_logits, dim=-1)  # [num_tokens, num_experts]

        # Expert capacity: each expert processes this many tokens
        capacity_per_expert = int(
            self.config.expert_capacity_factor * num_tokens / self.config.num_experts
        )
        capacity_per_expert = max(4, capacity_per_expert)  # Minimum capacity

        # EXPERT CHOICE ROUTING
        # Each expert selects top-k tokens based on routing scores
        dispatch_mask = torch.zeros(
            self.config.num_experts,
            capacity_per_expert,
            num_tokens,
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        combine_weights = torch.zeros_like(dispatch_mask)

        # For each expert, select top-capacity tokens
        for expert_idx in range(self.config.num_experts):
            # Get routing scores for this expert: [num_tokens]
            expert_scores = router_probs[:, expert_idx]

            # Select top-k tokens for this expert
            top_scores, top_indices = torch.topk(
                expert_scores,
                k=min(capacity_per_expert, num_tokens),
                dim=0,
                sorted=False
            )

            # Fill dispatch mask
            for i, token_idx in enumerate(top_indices):
                if i < capacity_per_expert:
                    dispatch_mask[expert_idx, i, token_idx] = 1.0
                    combine_weights[expert_idx, i, token_idx] = top_scores[i]

        # Normalize combine weights per token (tokens can be selected by multiple experts)
        # Sum over experts: [capacity, num_tokens]
        total_weight_per_token = combine_weights.sum(dim=(0, 1))  # [num_tokens]
        total_weight_per_token = total_weight_per_token.clamp(min=1e-6)

        if self.config.normalize_scores:
            # Normalize so each token's weights sum to 1
            combine_weights = combine_weights / total_weight_per_token.unsqueeze(0).unsqueeze(0)

        # Compute auxiliary losses
        aux_loss_dict = {}

        if self.use_router_z_loss and training:
            z_loss = self.router_z_loss(router_logits)
            aux_loss_dict['router_z_loss'] = z_loss * self.router_z_loss_coef

        # Load balancing: Should be perfect with expert choice, but track for monitoring
        tokens_per_expert = dispatch_mask.sum(dim=(1, 2))  # [num_experts]
        aux_loss_dict['load_balance'] = tokens_per_expert.std() / (tokens_per_expert.mean() + 1e-6)

        # Coverage: What fraction of tokens were processed by at least one expert
        tokens_covered = (dispatch_mask.sum(dim=(0, 1)) > 0).float().mean()
        aux_loss_dict['coverage'] = tokens_covered

        return dispatch_mask, combine_weights, aux_loss_dict


class Expert(nn.Module):
    """Single expert: FFN with configurable activation."""

    def __init__(self, config: ExpertChoiceMoEConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = self._get_activation(config.activation)

    def _get_activation(self, name: str):
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "glu": nn.GLU(dim=-1),
        }
        return activations.get(name, nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expert forward pass.

        Args:
            x: [capacity, d_model]

        Returns:
            output: [capacity, d_model]
        """
        h = self.fc1(x)
        h = self.activation(h)
        h = self.dropout(h)
        out = self.fc2(h)
        return out


class ExpertChoiceMoELayer(nn.Module):
    """
    Expert Choice Mixture-of-Experts Layer.

    Revolutionary approach where experts choose tokens instead of tokens choosing experts.
    Achieves perfect load balancing and scales to 1000s of experts.
    """

    def __init__(self, config: ExpertChoiceMoEConfig):
        super().__init__()
        self.config = config

        # Router
        self.router = ExpertChoiceRouter(config)

        # Expert network
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(config.num_experts)
        ])

        # Optional: Shared expert (always activated)
        self.use_shared_expert = True
        if self.use_shared_expert:
            self.shared_expert = Expert(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with expert choice routing.

        Args:
            hidden_states: [batch_size, seq_len, d_model]
            training: Whether in training mode

        Returns:
            output: [batch_size, seq_len, d_model]
            aux_loss_dict: Dictionary of auxiliary losses
        """
        batch_size, seq_len, d_model = hidden_states.shape
        num_tokens = batch_size * seq_len

        # Flatten for routing
        hidden_states_flat = hidden_states.reshape(-1, d_model)  # [num_tokens, d_model]

        # Expert choice routing
        dispatch_mask, combine_weights, aux_loss_dict = self.router(
            hidden_states, training=training
        )

        # Initialize output
        expert_outputs = torch.zeros_like(hidden_states_flat)  # [num_tokens, d_model]

        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Get tokens for this expert
            # dispatch_mask[expert_idx]: [capacity, num_tokens]
            expert_dispatch = dispatch_mask[expert_idx]  # [capacity, num_tokens]
            expert_combine = combine_weights[expert_idx]  # [capacity, num_tokens]

            # Find which tokens this expert processes
            # Get indices where dispatch > 0
            token_indices = (expert_dispatch.sum(dim=0) > 0).nonzero(as_tuple=True)[0]

            if len(token_indices) == 0:
                continue

            # Extract tokens: [num_selected, d_model]
            expert_input = hidden_states_flat[token_indices]

            # Process through expert
            expert_output = expert(expert_input)  # [num_selected, d_model]

            # Get combination weights for these tokens
            # This is tricky: we need to get the weights from the dispatch mask
            # For simplicity, use the routing scores
            weights = expert_combine[:, token_indices].sum(dim=0, keepdim=True).t()  # [num_selected, 1]

            # Accumulate weighted expert outputs
            expert_outputs[token_indices] += expert_output * weights

        # Add shared expert (always active)
        if self.use_shared_expert:
            shared_output = self.shared_expert(hidden_states_flat)
            expert_outputs += shared_output * 0.1  # Small weight for shared expert

        # Reshape back
        output = expert_outputs.reshape(batch_size, seq_len, d_model)

        return output, aux_loss_dict

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Count number of parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class SoftMoE(nn.Module):
    """
    Soft Mixture-of-Experts using Slot Attention.

    Reference: "From Sparse to Soft Mixtures of Experts" (Puigcerver et al., Google 2024)
              https://arxiv.org/abs/2308.00951

    Key Innovation:
    - No discrete routing (continuous soft assignment)
    - Uses slot attention mechanism
    - Each expert becomes a "slot" that attends to all tokens
    - Fully differentiable, no auxiliary losses needed
    - Better performance than hard routing

    Mathematical Foundation:
    ----------------------

    Traditional MoE: Hard assignment
      expert_i processes only selected tokens

    Soft MoE: Soft assignment via attention
      For each expert slot:
        1. Slot attends to all input tokens
        2. Compute weighted average of inputs
        3. Process through expert FFN
        4. Outputs are combined

    Slot Attention:
      Q = slots  # [num_experts, d_model]
      K = V = tokens  # [num_tokens, d_model]

      attention = softmax(QK^T / √d)  # [num_experts, num_tokens]
      slot_inputs = attention @ V      # [num_experts, d_model]
      slot_outputs = FFN(slot_inputs)  # [num_experts, d_model]

      final = attention^T @ slot_outputs  # [num_tokens, d_model]

    Advantages:
    - Fully differentiable (no straight-through estimators)
    - No load balancing issues
    - No auxiliary losses
    - Better gradient flow
    - Can specialize without discrete routing
    """

    def __init__(self, config: ExpertChoiceMoEConfig):
        super().__init__()
        self.config = config

        # Learnable expert slots
        self.expert_slots = nn.Parameter(
            torch.randn(config.num_experts, config.d_model) * 0.02
        )

        # Slot attention components
        self.slot_query = nn.Linear(config.d_model, config.d_model)
        self.token_key = nn.Linear(config.d_model, config.d_model)
        self.token_value = nn.Linear(config.d_model, config.d_model)

        # Expert networks
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(config.num_experts)
        ])

        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Soft MoE forward pass.

        Args:
            hidden_states: [batch_size, seq_len, d_model]

        Returns:
            output: [batch_size, seq_len, d_model]
            aux_loss_dict: Empty dict (no auxiliary losses needed!)
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Reshape
        tokens = hidden_states.reshape(batch_size * seq_len, d_model)  # [N, d]

        # Compute slot attention
        slots = self.expert_slots  # [E, d]
        Q = self.slot_query(slots)  # [E, d]
        K = self.token_key(tokens)   # [N, d]
        V = self.token_value(tokens) # [N, d]

        # Attention: [E, N]
        attention_scores = torch.matmul(Q, K.t()) / math.sqrt(d_model)
        attention = F.softmax(attention_scores, dim=-1)  # Each slot attends to all tokens
        attention = self.dropout(attention)

        # Aggregate inputs for each slot: [E, d]
        slot_inputs = torch.matmul(attention, V)

        # Process each slot through its expert
        slot_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(slot_inputs[i:i+1])  # [1, d]
            slot_outputs.append(expert_out)
        slot_outputs = torch.cat(slot_outputs, dim=0)  # [E, d]

        # Distribute expert outputs back to tokens: [N, d]
        # attention^T: [N, E], slot_outputs: [E, d] → [N, d]
        output = torch.matmul(attention.t(), slot_outputs)

        # Output projection
        output = self.output_proj(output)

        # Reshape back
        output = output.reshape(batch_size, seq_len, d_model)

        return output, {}  # No auxiliary losses!


if __name__ == "__main__":
    # Test Expert Choice MoE
    config = ExpertChoiceMoEConfig(
        d_model=512,
        num_experts=64,
        expert_capacity_factor=1.25,
        d_ff=2048
    )

    moe_layer = ExpertChoiceMoELayer(config)

    # Test input
    x = torch.randn(2, 128, 512)  # [batch, seq, dim]

    output, aux_losses = moe_layer(x, training=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary losses: {aux_losses}")
    print(f"Parameters: {moe_layer.num_parameters():,}")
    print(f"Load balance: {aux_losses['load_balance']:.4f}")
    print(f"Coverage: {aux_losses['coverage']:.4f}")

    # Test Soft MoE
    soft_moe = SoftMoE(config)
    output_soft, aux_soft = soft_moe(x)
    print(f"\nSoft MoE output shape: {output_soft.shape}")
    print(f"Soft MoE auxiliary losses: {aux_soft}")
