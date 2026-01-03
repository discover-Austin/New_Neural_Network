"""
Retrievers
==========

Different retrieval strategies.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .vector_store import VectorStore, Document
from .embedder import SentenceEmbedder


class DenseRetriever:
    """
    Dense retrieval using learned embeddings.

    Args:
        vector_store: Vector store for similarity search
        embedder: Embedding model for queries
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: SentenceEmbedder,
    ):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents for query.

        Args:
            query: Query text
            k: Number of documents to retrieve
            filter_metadata: Optional metadata filters

        Returns:
            List of (document, score) tuples
        """
        # Embed query
        query_embedding = self.embedder.embed(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            k=k,
            filter_metadata=filter_metadata,
        )

        return results


class SparseRetriever:
    """
    Sparse retrieval using keyword matching (BM25).

    Args:
        documents: List of documents
        k1: BM25 k1 parameter
        b: BM25 b parameter
    """

    def __init__(
        self,
        documents: List[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = documents
        self.k1 = k1
        self.b = b

        # Build inverted index
        self._build_index()

    def _build_index(self) -> None:
        """Build inverted index for BM25."""
        from collections import defaultdict, Counter
        import re

        self.inverted_index = defaultdict(list)
        self.doc_lengths = []
        self.avg_doc_length = 0

        for doc_id, doc in enumerate(self.documents):
            # Tokenize
            tokens = re.findall(r'\w+', doc.text.lower())
            self.doc_lengths.append(len(tokens))

            # Build inverted index
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                self.inverted_index[token].append((doc_id, count))

        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0

    def _bm25_score(self, query_tokens: List[str], doc_id: int) -> float:
        """Compute BM25 score for document."""
        import math

        score = 0.0
        doc_length = self.doc_lengths[doc_id]

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            # Document frequency
            df = len(self.inverted_index[token])

            # Inverse document frequency
            idf = math.log((len(self.documents) - df + 0.5) / (df + 0.5) + 1.0)

            # Term frequency in document
            tf = 0
            for did, count in self.inverted_index[token]:
                if did == doc_id:
                    tf = count
                    break

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

            score += idf * (numerator / denominator)

        return score

    def retrieve(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents using BM25."""
        import re

        # Tokenize query
        query_tokens = re.findall(r'\w+', query.lower())

        # Score all documents
        scores = []
        for doc_id in range(len(self.documents)):
            doc = self.documents[doc_id]

            # Apply metadata filter
            if filter_metadata is not None:
                if not all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue

            score = self._bm25_score(query_tokens, doc_id)
            scores.append((doc, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:k]


class HybridRetriever:
    """
    Hybrid retrieval combining dense and sparse methods.

    Args:
        dense_retriever: Dense retriever
        sparse_retriever: Sparse retriever
        alpha: Weight for dense retrieval (1-alpha for sparse)
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        alpha: float = 0.5,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve using hybrid approach."""
        # Get results from both retrievers
        dense_results = self.dense_retriever.retrieve(query, k=k * 2, filter_metadata=filter_metadata)
        sparse_results = self.sparse_retriever.retrieve(query, k=k * 2, filter_metadata=filter_metadata)

        # Normalize scores
        dense_scores = {id(doc): score for doc, score in dense_results}
        sparse_scores = {id(doc): score for doc, score in sparse_results}

        # Combine scores
        all_docs = {}
        for doc, _ in dense_results + sparse_results:
            doc_id = id(doc)
            if doc_id not in all_docs:
                all_docs[doc_id] = doc

        combined_scores = []
        for doc_id, doc in all_docs.items():
            dense_score = dense_scores.get(doc_id, 0)
            sparse_score = sparse_scores.get(doc_id, 0)

            # Weighted combination
            combined_score = self.alpha * dense_score + (1 - self.alpha) * sparse_score
            combined_scores.append((doc, combined_score))

        # Sort and return top k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores[:k]
