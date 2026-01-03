# Examples

This directory contains end-to-end examples demonstrating how to use the Advanced NLP/NLU/LLM System components.

## Available Examples

### 1. Basic GPT Training (`01_basic_gpt_training.py`)

**What it demonstrates:**
- Creating a GPT model from scratch
- Training with a simple dataset
- Using the character-level tokenizer
- Text generation with different strategies (greedy, sampling)

**Key concepts:**
- Model initialization and configuration
- Training loop basics
- Tokenization workflow
- Generation strategies

**Run it:**
```bash
python examples/01_basic_gpt_training.py
```

---

### 2. RAG Pipeline (`02_rag_pipeline.py`)

**What it demonstrates:**
- Setting up a vector store (FAISS)
- Creating document embeddings
- Building a retrieval pipeline
- Querying with metadata filtering

**Key concepts:**
- Retrieval-Augmented Generation (RAG)
- Vector similarity search
- Document retrieval and ranking
- Context creation for prompts

**Run it:**
```bash
python examples/02_rag_pipeline.py
```

---

### 3. LoRA Fine-Tuning (`03_lora_finetuning.py`)

**What it demonstrates:**
- Applying LoRA to a pre-trained model
- Fine-tuning with reduced parameters
- Saving and loading LoRA weights
- Merging LoRA back into base model

**Key concepts:**
- Parameter-Efficient Fine-Tuning (PEFT)
- Low-Rank Adaptation (LoRA)
- Memory-efficient training
- Adapter management

**Run it:**
```bash
python examples/03_lora_finetuning.py
```

---

## Requirements

These examples use only the components from this repository. To run them:

```bash
# Install the package
pip install -e .

# Or install minimal dependencies
pip install torch numpy
```

## Notes

### For Production Use

These examples are simplified for educational purposes. For production:

1. **Use Pre-trained Models**: Load weights from HuggingFace or other sources
2. **Use Proper Tokenizers**: BPE or WordPiece instead of character-level
3. **Use Real Embeddings**: Sentence transformers for RAG
4. **Scale Up**: Larger models, more data, distributed training
5. **Add Evaluation**: Metrics, validation sets, checkpointing

### Example Data

All examples use toy datasets. Replace with your actual data:
- Text corpora for language modeling
- Question-answer pairs for fine-tuning
- Domain-specific documents for RAG

### GPU Acceleration

Examples will run on CPU but are much faster with GPU:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

## Learning Path

Recommended order:

1. **Start with Example 1** - Understand basic training
2. **Then Example 3** - Learn fine-tuning techniques
3. **Finally Example 2** - Explore retrieval systems

## Extending Examples

Feel free to modify these examples:

- Change model architectures (GPT → LLaMA)
- Try different attention mechanisms
- Add more sophisticated generation strategies
- Implement custom training loops
- Integrate with your data sources

## Getting Help

- Check the main README for component documentation
- Read the source code in `src/` for implementation details
- See `tests/` for unit tests and usage patterns
- Review research papers cited in documentation

## Contributing

Have a useful example? Contributions welcome:

1. Create a new example file: `04_your_example.py`
2. Add documentation in this README
3. Keep it simple and educational
4. Include comments explaining key concepts
