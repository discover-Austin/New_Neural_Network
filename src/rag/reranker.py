"""
Rerankers
=========

Reranking models to improve retrieval quality.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from .vector_store import Document


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval results.

    Uses a model that jointly encodes query and document.

    Args:
        model: Cross-encoder model
        device: Device to run on
        batch_size: Batch size for reranking
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size

        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: Query text
            documents: List of documents to rerank
            top_k: Number of top documents to return

        Returns:
            Reranked list of (document, score) tuples
        """
        scores = []

        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]

            # Create query-document pairs
            pairs = [(query, doc.text) for doc in batch_docs]

            # TODO: Tokenize pairs (requires tokenizer)
            # inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
            # inputs = inputs.to(self.device)

            # Forward pass
            # outputs = self.model(**inputs)
            # batch_scores = outputs.logits.squeeze(-1)

            # Placeholder: random scores
            batch_scores = torch.randn(len(batch_docs))

            scores.extend(batch_scores.cpu().numpy().tolist())

        # Create (document, score) pairs and sort
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return doc_score_pairs[:top_k]


class MonoT5Reranker:
    """
    MonoT5 reranker using T5 model for pointwise ranking.

    Reference: "Document Ranking with a Pretrained Sequence-to-Sequence Model"
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size

        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """Rerank using MonoT5."""
        scores = []

        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]

            # Create prompts
            prompts = [
                f"Query: {query} Document: {doc.text} Relevant:"
                for doc in batch_docs
            ]

            # TODO: Tokenize and generate
            # inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            # outputs = model.generate(**inputs, max_length=2)
            # Decode "true"/"false" tokens to scores

            # Placeholder
            batch_scores = torch.randn(len(batch_docs))
            scores.extend(batch_scores.tolist())

        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return doc_score_pairs[:top_k]
