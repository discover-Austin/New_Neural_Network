"""
Reward Model
============

Model for predicting human preferences.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RewardModel(nn.Module):
    """
    Reward model for RLHF.

    Takes a language model backbone and adds a reward head.

    Args:
        backbone: Pretrained language model
        d_model: Model hidden dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone = backbone
        self.d_model = d_model

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Initialize reward head
        for module in self.reward_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward for input sequence.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask

        Returns:
            Reward scalar [batch_size]
        """
        # Get hidden states from backbone
        outputs = self.backbone(input_ids, attention_mask=attention_mask)

        if isinstance(outputs, dict):
            hidden_states = outputs["last_hidden_state"]
        elif isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        # Get last non-padding token
        if attention_mask is not None:
            # Get index of last non-padding token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            last_hidden = hidden_states[torch.arange(batch_size), sequence_lengths]
        else:
            # Use last token
            last_hidden = hidden_states[:, -1, :]

        # Compute reward
        reward = self.reward_head(last_hidden).squeeze(-1)

        return reward

    def compute_pairwise_loss(
        self,
        chosen_input_ids: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute pairwise ranking loss.

        The chosen response should have higher reward than rejected.

        Args:
            chosen_input_ids: Chosen sequence token IDs
            rejected_input_ids: Rejected sequence token IDs
            chosen_attention_mask: Attention mask for chosen
            rejected_attention_mask: Attention mask for rejected

        Returns:
            Loss and metrics dict
        """
        # Compute rewards
        chosen_reward = self.forward(chosen_input_ids, chosen_attention_mask)
        rejected_reward = self.forward(rejected_input_ids, rejected_attention_mask)

        # Ranking loss: chosen should have higher reward
        # Using log-sigmoid for numerical stability
        loss = -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward).mean()

        # Compute metrics
        metrics = {
            "loss": loss.item(),
            "chosen_reward": chosen_reward.mean().item(),
            "rejected_reward": rejected_reward.mean().item(),
            "reward_diff": (chosen_reward - rejected_reward).mean().item(),
            "accuracy": (chosen_reward > rejected_reward).float().mean().item(),
        }

        return loss, metrics
