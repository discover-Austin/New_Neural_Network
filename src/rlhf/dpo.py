"""
DPO: Direct Preference Optimization
====================================

Simpler alternative to RLHF that doesn't require separate reward model or RL.

Reference: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    beta: float = 0.1  # Temperature parameter
    learning_rate: float = 1e-6
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0


class DPOTrainer:
    """
    Direct Preference Optimization Trainer.

    Directly optimizes language model using preference pairs,
    without explicit reward model or reinforcement learning.

    Args:
        policy_model: Policy model to train
        ref_model: Reference model (frozen initial policy)
        config: DPO configuration
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        config: Optional[DPOConfig] = None,
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        self.config = config or DPOConfig()

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
        )

    def get_logprobs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get log probabilities of labels under model.

        Args:
            model: Language model
            input_ids: Input token IDs
            labels: Label token IDs
            attention_mask: Attention mask

        Returns:
            Log probabilities [batch_size]
        """
        outputs = model(input_ids, attention_mask=attention_mask)

        if isinstance(outputs, dict):
            logits = outputs["logits"]
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for labels
        labels_log_probs = torch.gather(
            log_probs[:, :-1, :],
            dim=2,
            index=labels[:, 1:].unsqueeze(-1),
        ).squeeze(-1)

        # Mask padding tokens
        if attention_mask is not None:
            labels_log_probs = labels_log_probs * attention_mask[:, 1:]

        # Sum over sequence
        return labels_log_probs.sum(dim=1)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss.

        The loss encourages the policy to assign higher probability to chosen
        responses relative to rejected responses, compared to the reference model.

        Args:
            policy_chosen_logps: Policy log probs for chosen responses
            policy_rejected_logps: Policy log probs for rejected responses
            ref_chosen_logps: Reference log probs for chosen responses
            ref_rejected_logps: Reference log probs for rejected responses

        Returns:
            Loss and metrics dictionary
        """
        # Compute log ratios
        policy_log_ratios = policy_chosen_logps - policy_rejected_logps
        ref_log_ratios = ref_chosen_logps - ref_rejected_logps

        # DPO loss: -log sigmoid(beta * (log_ratio_policy - log_ratio_ref))
        logits = self.config.beta * (policy_log_ratios - ref_log_ratios)

        if self.config.label_smoothing > 0:
            loss = -F.logsigmoid(logits) * (1 - self.config.label_smoothing) - \
                   F.logsigmoid(-logits) * self.config.label_smoothing
        else:
            loss = -F.logsigmoid(logits)

        loss = loss.mean()

        # Compute metrics
        with torch.no_grad():
            # Implicit reward
            chosen_rewards = self.config.beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = self.config.beta * (policy_rejected_logps - ref_rejected_logps)

            # Accuracy: how often chosen > rejected
            accuracy = (chosen_rewards > rejected_rewards).float().mean()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        }

        return loss, metrics

    def train_step(
        self,
        chosen_input_ids: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single DPO training step.

        Args:
            chosen_input_ids: Input IDs for chosen responses
            rejected_input_ids: Input IDs for rejected responses
            chosen_labels: Labels for chosen responses
            rejected_labels: Labels for rejected responses
            chosen_attention_mask: Attention mask for chosen
            rejected_attention_mask: Attention mask for rejected

        Returns:
            Training metrics
        """
        self.policy.train()

        # Get log probabilities from policy
        policy_chosen_logps = self.get_logprobs(
            self.policy,
            chosen_input_ids,
            chosen_labels,
            chosen_attention_mask,
        )

        policy_rejected_logps = self.get_logprobs(
            self.policy,
            rejected_input_ids,
            rejected_labels,
            rejected_attention_mask,
        )

        # Get log probabilities from reference model
        with torch.no_grad():
            ref_chosen_logps = self.get_logprobs(
                self.ref_model,
                chosen_input_ids,
                chosen_labels,
                chosen_attention_mask,
            )

            ref_rejected_logps = self.get_logprobs(
                self.ref_model,
                rejected_input_ids,
                rejected_labels,
                rejected_attention_mask,
            )

        # Compute loss
        loss, metrics = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.config.max_grad_norm,
        )

        self.optimizer.step()

        return metrics
