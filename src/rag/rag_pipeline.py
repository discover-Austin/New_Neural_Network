"""
RAG Pipeline
============

Complete retrieval-augmented generation pipeline.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from .vector_store import VectorStore, Document
from .embedder import SentenceEmbedder
from .retriever import DenseRetriever
from .reranker import CrossEncoderReranker


class RAGPipeline:
    """
    Complete RAG pipeline combining retrieval and generation.

    Production-ready implementation with proper tokenization support.

    Args:
        language_model: Generation model
        tokenizer: Tokenizer for encoding/decoding text
        vector_store: Vector store for retrieval
        embedder: Embedding model for queries
        retriever: Retriever for finding relevant documents
        reranker: Optional reranker for improving results
        top_k: Number of documents to retrieve
        rerank_top_k: Number of documents after reranking
        device: Device for inference (cuda/cpu)
    """

    def __init__(
        self,
        language_model: nn.Module,
        tokenizer: Any,  # Can be any tokenizer with encode/decode methods
        vector_store: VectorStore,
        embedder: SentenceEmbedder,
        retriever: Optional[DenseRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        top_k: int = 10,
        rerank_top_k: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.language_model = language_model.to(device)
        self.language_model.eval()
        self.tokenizer = tokenizer
        self.vector_store = vector_store
        self.embedder = embedder
        self.retriever = retriever or DenseRetriever(vector_store, embedder)
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.device = device

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents for query.

        Args:
            query: Query text
            k: Number of documents to retrieve
            filter_metadata: Optional metadata filters

        Returns:
            List of relevant documents
        """
        k = k or self.top_k

        # Retrieve initial candidates
        results = self.retriever.retrieve(query, k=k, filter_metadata=filter_metadata)

        # Rerank if reranker is provided
        if self.reranker is not None:
            documents = [doc for doc, _ in results]
            reranked_results = self.reranker.rerank(query, documents, top_k=self.rerank_top_k)
            return [doc for doc, _ in reranked_results]

        return [doc for doc, _ in results[:self.rerank_top_k]]

    def create_context(self, documents: List[Document]) -> str:
        """
        Create context string from retrieved documents.

        Args:
            documents: Retrieved documents

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i}:\n{doc.text}")

        return "\n\n".join(context_parts)

    def create_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Create prompt for language model.

        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Use the following context to answer the question. "
                "If you cannot answer the question based on the context, say so."
            )

        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        return prompt

    @torch.no_grad()
    def generate(
        self,
        query: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        include_sources: bool = False,
        filter_metadata: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer using RAG.

        Production implementation with full tokenization and generation.

        Args:
            query: User query
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            include_sources: Whether to include source documents
            filter_metadata: Optional metadata filters
            system_prompt: Optional system prompt

        Returns:
            Dictionary with answer and optional sources
        """
        # Retrieve relevant documents
        documents = self.retrieve(query, filter_metadata=filter_metadata)

        # Create context from documents
        context = self.create_context(documents)

        # Create prompt
        prompt = self.create_prompt(query, context, system_prompt)

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)

        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
        elif len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(self.device)

        # Generate answer
        output_ids = self.language_model.generate(
            input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        # Decode answer
        answer = self.tokenizer.decode(output_ids[0].cpu().tolist())

        # Post-process: remove prompt from answer
        if isinstance(answer, str):
            # Try to extract only the generated part
            prompt_decoded = self.tokenizer.decode(input_ids[0].cpu().tolist())
            if answer.startswith(prompt_decoded):
                answer = answer[len(prompt_decoded):].strip()

        result = {
            "query": query,
            "answer": answer,
            "context": context,
            "num_documents": len(documents),
        }

        if include_sources:
            result["sources"] = [
                {
                    "text": doc.text,
                    "metadata": doc.metadata,
                }
                for doc in documents
            ]

        return result

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents to add
        """
        # Embed documents
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self.embedder.embed(doc.text)

        # Add to vector store
        self.vector_store.add_documents(documents)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add texts to the knowledge base.

        Args:
            texts: List of text strings
            metadatas: Optional metadata for each text
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        documents = [
            Document(text=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]

        self.add_documents(documents)
