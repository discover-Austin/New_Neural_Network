"""
Embedding Models
================

Text embedding models for semantic search.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Union


class SentenceEmbedder:
    """
    Sentence embedding model.

    Args:
        model: Embedding model (can be BERT, sentence-transformers, etc.)
        device: Device to run model on
        batch_size: Batch size for encoding
        normalize: Whether to normalize embeddings
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize

        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embed texts into vectors.

        Args:
            texts: Single text or list of texts

        Returns:
            Embedding vectors [num_texts, dim] or [dim]
        """
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # TODO: Tokenize texts (requires tokenizer)
            # input_ids = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            # input_ids = input_ids.to(self.device)

            # Forward pass
            # outputs = self.model(**input_ids)

            # Get embeddings (mean pooling or [CLS] token)
            # embeddings = self._mean_pooling(outputs, attention_mask)

            # Placeholder: random embeddings
            embeddings = torch.randn(len(batch_texts), 768)

            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all batches
        all_embeddings = np.concatenate(all_embeddings, axis=0)

        if return_single:
            return all_embeddings[0]

        return all_embeddings

    def _mean_pooling(
        self,
        model_output: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean pooling of token embeddings.

        Args:
            model_output: Model output
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Alias for embed()."""
        return self.embed(texts)
