# Advanced NLP/NLU/LLM System

A bleeding-edge, comprehensive neural network system with state-of-the-art NLP, NLU, and LLM capabilities. This system includes every modern advancement in language modeling and neural networks, with no simplifications.

## Features

### Core Architectures
- **Complete Transformer Models**: GPT, BERT, T5, LLaMA
- **Multiple Attention Mechanisms**:
  - Multi-Head Attention (MHA)
  - Multi-Query Attention (MQA)
  - Grouped-Query Attention (GQA)
  - Flash Attention (memory-efficient)
  - Linear Attention (O(N) complexity)
  - Sliding Window Attention
  - Sparse Attention patterns
  - Cross Attention for multimodal

### Positional Encodings
- Sinusoidal Positional Encoding
- Learned Positional Encoding
- Rotary Position Embedding (RoPE)
- ALiBi (Attention with Linear Biases)
- Relative Positional Encoding

### Advanced Components
- **Normalization**: LayerNorm, RMSNorm, GroupNorm
- **Activations**: GELU, Swish, SwiGLU, GeGLU, ReGLU
- **Feed-Forward Networks**: Standard, GLU variants, Mixture of Experts (MoE)

### Text Generation
- Greedy Decoding
- Beam Search
- Nucleus (top-p) Sampling
- Top-k Sampling
- Contrastive Decoding
- Speculative Decoding (fast inference)
- Logits Processors (temperature, repetition penalty, etc.)
- Stopping Criteria

### Retrieval Augmented Generation (RAG)
- **Vector Stores**: FAISS, ChromaDB integration
- **Retrievers**: Dense, Sparse (BM25), Hybrid
- **Rerankers**: Cross-encoder, MonoT5
- **Complete RAG Pipeline** with query expansion and context management

### Parameter-Efficient Fine-Tuning (PEFT)
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Prefix Tuning
- P-Tuning v2
- Adapter Layers
- Prompt Tuning

### Reinforcement Learning from Human Feedback (RLHF)
- Reward Modeling with pairwise ranking
- PPO (Proximal Policy Optimization)
- DPO (Direct Preference Optimization)
- Complete alignment pipeline

## Installation

```bash
# Clone repository
git clone <repository-url>
cd New_Neural_Network

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

### Using GPT Model

```python
from src.models import GPTModel, GPTConfig

# Create model
config = GPTConfig(
    vocab_size=50257,
    d_model=768,
    num_layers=12,
    num_heads=12,
)
model = GPTModel(config)

# Generate text
import torch
input_ids = torch.randint(0, config.vocab_size, (1, 10))
output = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,
)
```

### Using LLaMA Model

```python
from src.models import LLaMAModel, LLaMAConfig

# Create LLaMA model with modern features
config = LLaMAConfig(
    d_model=4096,
    num_layers=32,
    num_heads=32,
    num_kv_heads=8,  # Grouped-query attention
)
model = LLaMAModel(config)
```

### RAG Pipeline

```python
from src.rag import RAGPipeline, FAISSVectorStore, SentenceEmbedder
from src.models import GPTModel

# Setup components
vector_store = FAISSVectorStore(dimension=768)
embedder = SentenceEmbedder(model=embedding_model)

# Create RAG pipeline
rag = RAGPipeline(
    language_model=model,
    vector_store=vector_store,
    embedder=embedder,
)

# Add knowledge
rag.add_texts([
    "The sky is blue because of Rayleigh scattering.",
    "Machine learning is a subset of artificial intelligence.",
])

# Generate with retrieval
result = rag.generate(
    "Why is the sky blue?",
    max_length=100,
)
```

### Fine-Tuning with LoRA

```python
from src.finetuning import apply_lora, get_lora_parameters

# Apply LoRA to model
model = apply_lora(
    model,
    target_modules=["q_proj", "v_proj"],
    r=8,
    lora_alpha=16,
)

# Get only LoRA parameters for training
lora_params = get_lora_parameters(model)
optimizer = torch.optim.Adam(lora_params, lr=1e-4)
```

### RLHF with DPO

```python
from src.rlhf import DPOTrainer, DPOConfig

# Create DPO trainer
dpo_config = DPOConfig(beta=0.1)
trainer = DPOTrainer(
    policy_model=model,
    ref_model=reference_model,
    config=dpo_config,
)

# Train on preference pairs
metrics = trainer.train_step(
    chosen_input_ids=chosen_ids,
    rejected_input_ids=rejected_ids,
    chosen_labels=chosen_labels,
    rejected_labels=rejected_labels,
)
```

## Architecture Details

### Model Architectures

#### GPT (Generative Pre-trained Transformer)
- Decoder-only architecture for autoregressive generation
- Causal self-attention masking
- Learned or sinusoidal positional embeddings
- Flexible configuration (layers, heads, dimensions)

#### BERT (Bidirectional Encoder Representations from Transformers)
- Encoder-only architecture for understanding tasks
- Bidirectional self-attention
- Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
- Segment embeddings for sentence pairs

#### T5 (Text-to-Text Transfer Transformer)
- Encoder-decoder architecture
- Treats all tasks as text-to-text
- Relative positional encodings
- RMSNorm and GeGLU activations

#### LLaMA
- Modern decoder-only architecture
- Rotary Position Embeddings (RoPE)
- RMSNorm instead of LayerNorm
- SwiGLU activations
- Grouped-Query Attention for efficiency

### Attention Mechanisms

**Standard Attention Complexity**: O(N²) in sequence length

**Efficient Variants**:
- **Flash Attention**: O(N) memory using tiling and recomputation
- **Linear Attention**: O(N) time using kernel tricks
- **Sliding Window**: O(N*W) where W is window size
- **Sparse Attention**: Various sparsity patterns

### Generation Strategies

- **Greedy**: Always select highest probability token
- **Beam Search**: Maintain top-k hypotheses
- **Nucleus Sampling**: Sample from smallest set with cumulative probability >= p
- **Contrastive Decoding**: Use difference between expert and amateur models
- **Speculative Decoding**: Use draft model to accelerate inference

## Project Structure

```
src/
├── attention/           # Attention mechanisms
│   ├── multi_head_attention.py
│   ├── multi_query_attention.py
│   ├── grouped_query_attention.py
│   ├── flash_attention.py
│   ├── linear_attention.py
│   ├── sliding_window_attention.py
│   ├── sparse_attention.py
│   └── cross_attention.py
├── core/               # Core components
│   ├── normalization.py
│   ├── activations.py
│   └── feedforward.py
├── embeddings/         # Embeddings and positional encodings
│   ├── token_embeddings.py
│   └── positional_encoding.py
├── models/             # Complete model architectures
│   ├── transformer.py
│   ├── gpt.py
│   ├── bert.py
│   ├── t5.py
│   └── llama.py
├── generation/         # Text generation
│   ├── strategies.py
│   ├── logits_processors.py
│   └── stopping_criteria.py
├── rag/               # Retrieval Augmented Generation
│   ├── vector_store.py
│   ├── retriever.py
│   ├── embedder.py
│   ├── reranker.py
│   └── rag_pipeline.py
├── finetuning/        # Parameter-efficient fine-tuning
│   ├── lora.py
│   ├── prefix_tuning.py
│   ├── adapter.py
│   └── prompt_tuning.py
└── rlhf/              # Reinforcement learning from human feedback
    ├── reward_model.py
    ├── ppo.py
    └── dpo.py
```

## Advanced Features

### Mixture of Experts (MoE)
Enables scaling model capacity without proportional compute increase by routing tokens to subset of expert networks.

### Flash Attention
Memory-efficient attention with O(N) memory complexity using tiling and kernel fusion.

### Grouped-Query Attention
Balances between MHA (expensive) and MQA (limited expressiveness) by sharing KV heads across query groups.

### LoRA
Freezes pretrained weights and learns low-rank decomposition matrices for efficient fine-tuning with minimal parameters.

### Direct Preference Optimization
Simpler alternative to RLHF that directly optimizes policy using preference pairs without separate reward model or RL.

## Performance Optimizations

- Mixed precision training support
- Gradient checkpointing for memory efficiency
- Efficient attention variants (Flash, Linear, Sparse)
- Parameter-efficient fine-tuning methods
- Speculative decoding for faster inference

## Research References

1. Attention Is All You Need (Vaswani et al., 2017)
2. BERT (Devlin et al., 2018)
3. GPT-2 (Radford et al., 2019)
4. T5 (Raffel et al., 2020)
5. LLaMA (Touvron et al., 2023)
6. LoRA (Hu et al., 2021)
7. FlashAttention (Dao et al., 2022)
8. DPO (Rafailov et al., 2023)
9. RoFormer (Su et al., 2021)
10. GLU Variants (Shazeer, 2020)

## License

MIT License

## Contributing

Contributions welcome! This is a comprehensive implementation of state-of-the-art NLP/LLM techniques.

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{advanced_nlp_llm_system,
  title={Advanced NLP/NLU/LLM System},
  author={Advanced AI Research},
  year={2026},
  url={https://github.com/yourusername/advanced-nlp-system}
}
```
