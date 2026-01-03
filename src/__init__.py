"""
Advanced NLP/NLU/LLM System
===========================

A comprehensive, bleeding-edge neural network system with state-of-the-art
NLP, NLU, and LLM capabilities.

Main Components:
----------------
- Core: Foundational components and base classes
- Models: Complete model architectures (Transformer, GPT, BERT, T5, etc.)
- Attention: Various attention mechanisms (MHA, MQA, GQA, Flash Attention)
- Tokenization: Advanced tokenizers (BPE, WordPiece, SentencePiece)
- Embeddings: Embedding layers and positional encodings
- Training: Training loops, optimizers, schedulers
- Inference: Generation strategies and decoding methods
- RAG: Retrieval Augmented Generation
- Memory: Short-term, long-term, and episodic memory systems
- RLHF: Reinforcement Learning from Human Feedback
- Fine-tuning: LoRA, QLoRA, Prefix Tuning, P-Tuning
- Quantization: Model compression techniques
- Multimodal: Vision and audio processing
- NLU: NER, sentiment, intent, slot filling
- Reasoning: Chain-of-thought, tree-of-thought
- Tools: Function calling and tool use
- Distributed: Multi-GPU and distributed training
- Serving: Model deployment and serving
- Safety: Content filtering and bias detection
"""

__version__ = "0.1.0"

from src.core import *
from src.models import *
from src.attention import *
from src.tokenization import *
from src.embeddings import *

__all__ = [
    "__version__",
]
