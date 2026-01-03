"""
Example 1: Basic GPT Model Training
====================================

This example demonstrates how to:
1. Create a GPT model from scratch
2. Prepare training data
3. Train the model with the advanced trainer
4. Generate text with the trained model

This is a minimal end-to-end example for educational purposes.
"""

import torch
import torch.nn as nn
from src.models import GPTModel, GPTConfig
from src.training import AdvancedTrainer, TrainerConfig
from src.tokenization import CharacterTokenizer
from src.generation import GreedyDecoding, NucleusSampling


def prepare_toy_dataset():
    """
    Create a toy dataset for demonstration.

    In practice, you would load real text data here.
    """
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning has revolutionized computer vision and NLP.",
        "Transformers are the foundation of modern language models.",
    ] * 100  # Repeat for more training data

    return texts


def main():
    print("=" * 60)
    print("Example 1: Basic GPT Model Training")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Prepare Data and Tokenizer
    # -------------------------------------------------------------------------
    print("\n[Step 1] Preparing data and tokenizer...")

    texts = prepare_toy_dataset()
    print(f"Dataset size: {len(texts)} texts")

    # Create and train tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.train(texts)
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Tokenize all texts
    tokenized_data = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) > 0:
            tokenized_data.append(torch.tensor(tokens, dtype=torch.long))

    print(f"Tokenized {len(tokenized_data)} sequences")

    # -------------------------------------------------------------------------
    # Step 2: Create Model
    # -------------------------------------------------------------------------
    print("\n[Step 2] Creating GPT model...")

    config = GPTConfig(
        vocab_size=vocab_size,
        d_model=128,  # Small model for quick training
        num_layers=4,
        num_heads=4,
        d_ff=512,
        max_seq_len=256,
        dropout=0.1,
        activation="gelu",
    )

    model = GPTModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -------------------------------------------------------------------------
    # Step 3: Setup Training
    # -------------------------------------------------------------------------
    print("\n[Step 3] Setting up training...")

    trainer_config = TrainerConfig(
        learning_rate=3e-4,
        batch_size=4,
        num_epochs=3,
        warmup_steps=100,
        max_grad_norm=1.0,
        use_amp=False,  # Mixed precision (requires CUDA)
        log_interval=50,
    )

    # Create simple dataloader
    def collate_fn(batch):
        """Collate function to create batches."""
        # Pad sequences to same length
        max_len = min(max(len(x) for x in batch), config.max_seq_len)
        padded = []

        for seq in batch:
            if len(seq) > max_len:
                padded.append(seq[:max_len])
            else:
                # Pad with zeros
                padding = torch.zeros(max_len - len(seq), dtype=torch.long)
                padded.append(torch.cat([seq, padding]))

        return torch.stack(padded)

    from torch.utils.data import DataLoader, TensorDataset

    # Prepare dataset (for simplicity, using same sequence as input and target)
    dataset = tokenized_data[:400]  # Use subset for quick demo
    dataloader = DataLoader(
        dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer_config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # -------------------------------------------------------------------------
    # Step 4: Training Loop
    # -------------------------------------------------------------------------
    print("\n[Step 4] Training model...")
    print("(This is a simplified training loop for demonstration)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.train()
    total_steps = 0

    for epoch in range(trainer_config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, input_ids in enumerate(dataloader):
            input_ids = input_ids.to(device)

            # Forward pass (use same tokens as targets)
            logits, loss = model(input_ids, labels=input_ids)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                trainer_config.max_grad_norm
            )

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            total_steps += 1

            if total_steps % trainer_config.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch + 1}, Step {total_steps}, Loss: {avg_loss:.4f}")

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

    print("\n[Training completed!]")

    # -------------------------------------------------------------------------
    # Step 5: Generate Text
    # -------------------------------------------------------------------------
    print("\n[Step 5] Generating text with trained model...")

    model.eval()

    # Create a prompt
    prompt_text = "Machine learning is"
    print(f"\nPrompt: '{prompt_text}'")

    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt_text)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)

    # Generate with greedy decoding
    print("\n--- Greedy Decoding ---")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=1.0,
            do_sample=False,  # Greedy
        )

    generated_text = tokenizer.decode(output_ids[0].cpu().tolist())
    print(f"Generated: '{generated_text}'")

    # Generate with sampling
    print("\n--- Nucleus Sampling (top-p=0.9) ---")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
        )

    generated_text = tokenizer.decode(output_ids[0].cpu().tolist())
    print(f"Generated: '{generated_text}'")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)

    # Run example
    main()
