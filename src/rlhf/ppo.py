"""
PPO: Proximal Policy Optimization
==================================

PPO trainer for RLHF.

Reference: "Proximal Policy Optimization Algorithms"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # PPO hyperparameters
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    ppo_epochs: int = 4
    num_rollouts: int = 128
    batch_size: int = 32

    # Coefficients
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.1

    # Optimization
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0

    # Generation
    max_length: int = 512
    temperature: float = 1.0
    top_p: float = 0.9


class PPOTrainer:
    """
    PPO Trainer for RLHF.

    Args:
        policy_model: Policy model (language model)
        ref_model: Reference model (frozen copy of initial policy)
        reward_model: Reward model for scoring generations
        value_model: Value function model
        config: PPO configuration
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        value_model: Optional[nn.Module] = None,
        config: Optional[PPOConfig] = None,
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.value_model = value_model or self._create_value_model(policy_model)
        self.config = config or PPOConfig()

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_model.parameters()),
            lr=self.config.learning_rate,
        )

    def _create_value_model(self, policy_model: nn.Module) -> nn.Module:
        """Create value model from policy model."""
        # Simple value head on top of policy
        value_model = nn.Sequential(
            policy_model,
            nn.Linear(policy_model.config.d_model, 1),
        )
        return value_model

    @torch.no_grad()
    def generate_rollouts(
        self,
        prompts: torch.Tensor,
        num_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate rollouts from prompts.

        Args:
            prompts: Prompt token IDs [batch_size, prompt_len]
            num_samples: Number of samples per prompt

        Returns:
            Dictionary with rollout data
        """
        self.policy.eval()

        # Generate responses
        # Placeholder: would use actual generation
        responses = self.policy.generate(
            prompts,
            max_new_tokens=self.config.max_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # Get log probabilities from policy
        policy_outputs = self.policy(responses)
        policy_logprobs = F.log_softmax(policy_outputs.logits if hasattr(policy_outputs, 'logits') else policy_outputs[0], dim=-1)

        # Get log probabilities from reference model
        ref_outputs = self.ref_model(responses)
        ref_logprobs = F.log_softmax(ref_outputs.logits if hasattr(ref_outputs, 'logits') else ref_outputs[0], dim=-1)

        # Get rewards
        rewards = self.reward_model(responses)

        # Get values
        values = self.value_model(responses)

        rollouts = {
            "prompts": prompts,
            "responses": responses,
            "policy_logprobs": policy_logprobs,
            "ref_logprobs": ref_logprobs,
            "rewards": rewards,
            "values": values,
        }

        return rollouts

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE (Generalized Advantage Estimation).

        Args:
            rewards: Rewards [batch_size, seq_len]
            values: Value estimates [batch_size, seq_len]
            gamma: Discount factor
            lam: GAE lambda

        Returns:
            advantages and returns
        """
        batch_size, seq_len = rewards.shape

        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + gamma * next_value - values[:, t]
            advantages[:, t] = last_gae = delta + gamma * lam * last_gae

        returns = advantages + values

        return advantages, returns

    def ppo_loss(
        self,
        policy_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        ref_logprobs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss.

        Args:
            policy_logprobs: Current policy log probs
            old_logprobs: Old policy log probs
            advantages: Advantage estimates
            values: Value estimates
            returns: Target returns
            ref_logprobs: Reference model log probs (for KL penalty)

        Returns:
            Dictionary with loss components
        """
        # Policy loss (clipped)
        ratio = torch.exp(policy_logprobs - old_logprobs)
        clip_ratio = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)

        policy_loss = -torch.min(
            ratio * advantages,
            clip_ratio * advantages,
        ).mean()

        # Value loss (clipped)
        value_pred_clipped = values + torch.clamp(
            values - returns,
            -self.config.value_clip_range,
            self.config.value_clip_range,
        )

        value_loss = torch.max(
            F.mse_loss(values, returns, reduction='none'),
            F.mse_loss(value_pred_clipped, returns, reduction='none'),
        ).mean()

        # Entropy bonus (encourage exploration)
        entropy = -(policy_logprobs * torch.exp(policy_logprobs)).sum(dim=-1).mean()

        # KL penalty (keep policy close to reference)
        kl_penalty = 0
        if ref_logprobs is not None:
            kl_div = (torch.exp(policy_logprobs) * (policy_logprobs - ref_logprobs)).sum(dim=-1).mean()
            kl_penalty = self.config.kl_coef * kl_div

        # Total loss
        total_loss = (
            policy_loss +
            self.config.value_coef * value_loss -
            self.config.entropy_coef * entropy +
            kl_penalty
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "kl_penalty": kl_penalty,
        }

    def train_step(self, prompts: torch.Tensor) -> Dict[str, float]:
        """
        Single PPO training step.

        Args:
            prompts: Batch of prompts

        Returns:
            Training metrics
        """
        # Generate rollouts
        rollouts = self.generate_rollouts(prompts)

        # Compute advantages
        advantages, returns = self.compute_advantages(
            rollouts["rewards"],
            rollouts["values"],
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        metrics = {}

        for epoch in range(self.config.ppo_epochs):
            # Forward pass
            policy_outputs = self.policy(rollouts["responses"])
            policy_logprobs = F.log_softmax(
                policy_outputs.logits if hasattr(policy_outputs, 'logits') else policy_outputs[0],
                dim=-1
            )

            value_outputs = self.value_model(rollouts["responses"])

            # Compute loss
            losses = self.ppo_loss(
                policy_logprobs=policy_logprobs,
                old_logprobs=rollouts["policy_logprobs"].detach(),
                advantages=advantages,
                values=value_outputs,
                returns=returns,
                ref_logprobs=rollouts["ref_logprobs"],
            )

            # Backward
            self.optimizer.zero_grad()
            losses["total_loss"].backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_model.parameters()),
                self.config.max_grad_norm,
            )

            self.optimizer.step()

            # Update metrics
            for k, v in losses.items():
                metrics[k] = v.item()

        # Add reward stats
        metrics["mean_reward"] = rollouts["rewards"].mean().item()
        metrics["std_reward"] = rollouts["rewards"].std().item()

        return metrics
