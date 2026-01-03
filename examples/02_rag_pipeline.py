"""
Example 2: Retrieval-Augmented Generation (RAG) Pipeline
=========================================================

This example demonstrates how to:
1. Create a vector store and add documents
2. Set up embeddings and retrievers
3. Build a complete RAG pipeline
4. Query the pipeline with questions

Note: This example uses simple embeddings. In production, use actual
sentence transformers or other embedding models.
"""

import torch
import torch.nn as nn
import numpy as np
from src.rag import (
    FAISSVectorStore,
    Document,
    SentenceEmbedder,
    DenseRetriever,
    CrossEncoderReranker,
    RAGPipeline,
)
from src.models import GPTModel, GPTConfig


class SimpleEmbeddingModel(nn.Module):
    """
    Simple embedding model for demonstration.

    In production, use actual pre-trained models like:
    - sentence-transformers/all-MiniLM-L6-v2
    - sentence-transformers/all-mpnet-base-v2
    """

    def __init__(self, vocab_size=10000, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input_ids):
        # Simple averaging of embeddings
        embeds = self.embedding(input_ids)
        return self.projection(embeds.mean(dim=1))


def create_knowledge_base():
    """
    Create a sample knowledge base of documents.

    In production, this would be your actual document corpus.
    """
    documents = [
        Document(
            text="The Eiffel Tower is located in Paris, France. It was completed in 1889.",
            metadata={"source": "encyclopedia", "topic": "landmarks"},
        ),
        Document(
            text="Machine learning is a subset of artificial intelligence focused on learning from data.",
            metadata={"source": "textbook", "topic": "AI"},
        ),
        Document(
            text="Python is a high-level programming language known for its simplicity and readability.",
            metadata={"source": "programming_guide", "topic": "programming"},
        ),
        Document(
            text="The Great Wall of China is an ancient series of walls built to protect Chinese states.",
            metadata={"source": "encyclopedia", "topic": "landmarks"},
        ),
        Document(
            text="Deep learning uses neural networks with multiple layers to learn hierarchical representations.",
            metadata={"source": "textbook", "topic": "AI"},
        ),
        Document(
            text="The Colosseum in Rome is an ancient amphitheater built in 70-80 AD.",
            metadata={"source": "encyclopedia", "topic": "landmarks"},
        ),
        Document(
            text="Natural language processing enables computers to understand and generate human language.",
            metadata={"source": "textbook", "topic": "AI"},
        ),
        Document(
            text="JavaScript is a programming language primarily used for web development.",
            metadata={"source": "programming_guide", "topic": "programming"},
        ),
        Document(
            text="The Statue of Liberty was a gift from France to the United States in 1886.",
            metadata={"source": "encyclopedia", "topic": "landmarks"},
        ),
        Document(
            text="Transformers revolutionized NLP by using self-attention mechanisms.",
            metadata={"source": "textbook", "topic": "AI"},
        ),
    ]

    return documents


def simple_tokenize(text, vocab_size=10000):
    """
    Simple tokenization for demonstration.

    In production, use proper tokenizers from the tokenization module.
    """
    # Very basic: hash characters to vocab
    tokens = [hash(char) % vocab_size for char in text.lower()]
    return torch.tensor(tokens[:50], dtype=torch.long)  # Truncate to 50


def embed_documents(documents, embedding_model):
    """
    Embed documents using the embedding model.
    """
    embedded_docs = []

    for doc in documents:
        # Tokenize and embed
        tokens = simple_tokenize(doc.text).unsqueeze(0)
        with torch.no_grad():
            embedding = embedding_model(tokens).squeeze(0).cpu().numpy()

        # Add embedding to document
        doc.embedding = embedding
        embedded_docs.append(doc)

    return embedded_docs


def main():
    print("=" * 70)
    print("Example 2: Retrieval-Augmented Generation (RAG) Pipeline")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Create Knowledge Base
    # -------------------------------------------------------------------------
    print("\n[Step 1] Creating knowledge base...")

    documents = create_knowledge_base()
    print(f"Created knowledge base with {len(documents)} documents")

    # -------------------------------------------------------------------------
    # Step 2: Setup Embedding Model
    # -------------------------------------------------------------------------
    print("\n[Step 2] Setting up embedding model...")

    embedding_model = SimpleEmbeddingModel(vocab_size=10000, embedding_dim=128)
    embedding_model.eval()

    print("Embedding model ready")

    # -------------------------------------------------------------------------
    # Step 3: Embed Documents and Create Vector Store
    # -------------------------------------------------------------------------
    print("\n[Step 3] Embedding documents and creating vector store...")

    # Embed all documents
    embedded_docs = embed_documents(documents, embedding_model)
    print(f"Embedded {len(embedded_docs)} documents")

    # Create FAISS vector store
    vector_store = FAISSVectorStore(
        dimension=128,
        index_type="flat",  # Use "ivf" or "hnsw" for large-scale
        metric="l2",
    )

    # Add documents to vector store
    vector_store.add_documents(embedded_docs)
    print(f"Added {len(embedded_docs)} documents to vector store")

    # -------------------------------------------------------------------------
    # Step 4: Create Embedder and Retriever
    # -------------------------------------------------------------------------
    print("\n[Step 4] Creating embedder and retriever...")

    # Wrap embedding model in SentenceEmbedder
    class SimpleEmbedder:
        """Wrapper for embedding model."""

        def __init__(self, model):
            self.model = model

        def embed(self, texts):
            if isinstance(texts, str):
                texts = [texts]

            embeddings = []
            for text in texts:
                tokens = simple_tokenize(text).unsqueeze(0)
                with torch.no_grad():
                    emb = self.model(tokens).squeeze(0).cpu().numpy()
                embeddings.append(emb)

            return np.array(embeddings)

    embedder = SimpleEmbedder(embedding_model)

    # Create retriever
    retriever = DenseRetriever(vector_store, embedder)
    print("Retriever ready")

    # -------------------------------------------------------------------------
    # Step 5: Create Language Model for Generation
    # -------------------------------------------------------------------------
    print("\n[Step 5] Creating language model...")

    # Create a simple GPT model (would be pre-trained in production)
    gpt_config = GPTConfig(
        vocab_size=1000,
        d_model=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=512,
    )

    language_model = GPTModel(gpt_config)
    language_model.eval()
    print("Language model ready")

    # -------------------------------------------------------------------------
    # Step 6: Create RAG Pipeline
    # -------------------------------------------------------------------------
    print("\n[Step 6] Creating RAG pipeline...")

    rag = RAGPipeline(
        language_model=language_model,
        vector_store=vector_store,
        embedder=embedder,
        retriever=retriever,
        top_k=3,  # Retrieve top 3 documents
        rerank_top_k=2,  # Use top 2 after reranking
    )

    print("RAG pipeline ready")

    # -------------------------------------------------------------------------
    # Step 7: Query the RAG Pipeline
    # -------------------------------------------------------------------------
    print("\n[Step 7] Querying the RAG pipeline...")

    queries = [
        "What is machine learning?",
        "Tell me about the Eiffel Tower",
        "What programming languages are mentioned?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Question: {query}")

        # Retrieve relevant documents
        retrieved_docs = rag.retrieve(query, k=3)

        print(f"\nRetrieved {len(retrieved_docs)} documents:")
        for j, doc in enumerate(retrieved_docs, 1):
            print(f"  {j}. {doc.text[:80]}...")
            print(f"     (Source: {doc.metadata['source']}, Topic: {doc.metadata['topic']})")

        # Create context from retrieved documents
        context = rag.create_context(retrieved_docs)

        # Create prompt
        prompt = rag.create_prompt(query, context)

        print(f"\n  Generated prompt (truncated):")
        print(f"  {prompt[:200]}...")

    # -------------------------------------------------------------------------
    # Step 8: Filter by Metadata
    # -------------------------------------------------------------------------
    print("\n[Step 8] Demonstrating metadata filtering...")

    query = "What can you tell me about landmarks?"
    print(f"\nQuery: {query}")
    print("Filter: topic='landmarks'")

    # Retrieve only landmark-related documents
    filtered_docs = rag.retrieve(
        query,
        k=5,
        filter_metadata={"topic": "landmarks"},
    )

    print(f"\nRetrieved {len(filtered_docs)} landmark documents:")
    for j, doc in enumerate(filtered_docs, 1):
        print(f"  {j}. {doc.text[:80]}...")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("\nKey takeaways:")
    print("- RAG combines retrieval with generation")
    print("- Vector stores enable semantic search")
    print("- Metadata filtering allows targeted retrieval")
    print("- In production, use pre-trained embedding models and LLMs")
    print("=" * 70)


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Run example
    main()
