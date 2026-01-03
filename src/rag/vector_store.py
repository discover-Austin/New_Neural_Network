"""
Vector Stores
=============

Vector database implementations for semantic search.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import pickle


@dataclass
class Document:
    """Document with text and metadata."""
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class VectorStore:
    """Base class for vector stores."""

    def add_documents(self, documents: List[Document]) -> None:
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        raise NotImplementedError

    def delete(self, ids: List[str]) -> None:
        raise NotImplementedError


class FAISSVectorStore(VectorStore):
    """
    Vector store using FAISS for efficient similarity search.

    Args:
        dimension: Embedding dimension
        index_type: Type of FAISS index ("flat", "ivf", "hnsw")
        metric: Distance metric ("l2" or "ip" for inner product)
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        metric: str = "l2",
        nlist: int = 100,
    ):
        try:
            import faiss
        except ImportError:
            raise ImportError("Please install faiss: pip install faiss-cpu or faiss-gpu")

        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.faiss = faiss

        # Create index
        if index_type == "flat":
            if metric == "l2":
                self.index = faiss.IndexFlatL2(dimension)
            else:
                self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension) if metric == "l2" else faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        embeddings = []

        for doc in documents:
            if doc.embedding is None:
                raise ValueError("Document must have embedding")
            embeddings.append(doc.embedding)

        embeddings = np.array(embeddings).astype('float32')

        # Train index if needed
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings)

        # Add to index
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of (document, score) tuples
        """
        query_embedding = query_embedding.astype('float32').reshape(1, -1)

        # Search
        distances, indices = self.index.search(query_embedding, k)

        # Get results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                doc = self.documents[idx]

                # Apply metadata filter
                if filter_metadata is not None:
                    if not all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                        continue

                results.append((doc, float(dist)))

        return results

    def delete(self, ids: List[int]) -> None:
        """Delete documents by IDs."""
        # FAISS doesn't support deletion natively
        # Need to rebuild index
        remaining_embeddings = []
        remaining_docs = []

        for i, doc in enumerate(self.documents):
            if i not in ids:
                remaining_embeddings.append(doc.embedding)
                remaining_docs.append(doc)

        # Recreate index
        self.index.reset()
        if remaining_embeddings:
            embeddings = np.array(remaining_embeddings).astype('float32')
            if self.index_type == "ivf":
                self.index.train(embeddings)
            self.index.add(embeddings)

        self.documents = remaining_docs

    def save(self, path: str) -> None:
        """Save index and documents to disk."""
        self.faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.docs", 'wb') as f:
            pickle.dump(self.documents, f)

    def load(self, path: str) -> None:
        """Load index and documents from disk."""
        self.index = self.faiss.read_index(f"{path}.index")
        with open(f"{path}.docs", 'rb') as f:
            self.documents = pickle.load(f)


class ChromaVectorStore(VectorStore):
    """
    Vector store using ChromaDB.

    Args:
        collection_name: Name of the collection
        persist_directory: Directory to persist data
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError("Please install chromadb: pip install chromadb")

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to ChromaDB."""
        ids = [f"doc_{i}" for i in range(len(documents))]
        embeddings = [doc.embedding.tolist() for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        query_embeddings = [query_embedding.tolist()]

        where = filter_metadata if filter_metadata else None

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=k,
            where=where,
        )

        # Convert to Document objects
        documents = []
        for i in range(len(results['ids'][0])):
            doc = Document(
                text=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                embedding=np.array(results['embeddings'][0][i]) if results.get('embeddings') else None,
            )
            distance = results['distances'][0][i]
            documents.append((doc, distance))

        return documents

    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        self.collection.delete(ids=ids)
