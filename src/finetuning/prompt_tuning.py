"""
Prompt Tuning
=============

Learns continuous prompt embeddings prepended to input.

Reference: "The Power of Scale for Parameter-Efficient Prompt Tuning"
"""

import torch
import torch.nn as nn


class PromptTuning(nn.Module):
    """
    Prompt tuning by learning soft prompt embeddings.

    Args:
        num_tokens: Number of prompt tokens
        d_model: Model embedding dimension
        init_method: Initialization method ("random" or "vocab")
        init_text: Optional text for vocabulary initialization
    """

    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        init_method: str = "random",
        init_text: Optional[str] = None,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.d_model = d_model

        # Learnable soft prompts
        self.soft_prompts = nn.Parameter(torch.randn(num_tokens, d_model))

        # Initialize
        if init_method == "random":
            nn.init.xavier_uniform_(self.soft_prompts)
        elif init_method == "vocab" and init_text is not None:
            # Would initialize from vocabulary embeddings
            pass

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Prepend soft prompts to input embeddings.

        Args:
            embeddings: Input embeddings [batch_size, seq_len, d_model]

        Returns:
            Augmented embeddings [batch_size, num_tokens + seq_len, d_model]
        """
        batch_size = embeddings.size(0)

        # Expand prompts for batch
        prompts = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)

        # Prepend to embeddings
        augmented_embeddings = torch.cat([prompts, embeddings], dim=1)

        return augmented_embeddings
