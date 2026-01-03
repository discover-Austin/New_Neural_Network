"""
Example 3: Fine-Tuning with LoRA (Low-Rank Adaptation)
======================================================

This example demonstrates how to:
1. Load a pre-trained model
2. Apply LoRA adaptation
3. Fine-tune only LoRA parameters
4. Save and load LoRA weights

LoRA enables efficient fine-tuning by freezing the base model and
only training small low-rank matrices.
"""

import torch
import torch.nn as nn
from src.models import GPTModel, GPTConfig
from src.finetuning import apply_lora, get_lora_parameters, merge_lora_weights


def create_pretrained_model():
    """
    Create a 'pre-trained' model for demonstration.

    In production, you would load actual pre-trained weights from:
    - HuggingFace Hub
    - Your own checkpoint
    - Published model weights
    """
    config = GPTConfig(
        vocab_size=1000,
        d_model=256,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        max_seq_len=128,
    )

    model = GPTModel(config)

    # Simulate pre-training by setting some specific weights
    # (In reality, this would be trained on large corpus)
    print("Creating 'pre-trained' model...")

    return model


def create_finetuning_dataset():
    """
    Create a small fine-tuning dataset.

    In production, this would be your task-specific data.
    """
    # Simulate task-specific data (e.g., specific domain, style, or task)
    data = []

    for _ in range(100):
        # Random sequences for demonstration
        seq_len = torch.randint(20, 50, (1,)).item()
        sequence = torch.randint(0, 1000, (seq_len,))
        data.append(sequence)

    return data


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def main():
    print("=" * 70)
    print("Example 3: Fine-Tuning with LoRA")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Create Pre-trained Model
    # -------------------------------------------------------------------------
    print("\n[Step 1] Creating pre-trained model...")

    model = create_pretrained_model()
    total_params, trainable_params = count_parameters(model)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # -------------------------------------------------------------------------
    # Step 2: Apply LoRA
    # -------------------------------------------------------------------------
    print("\n[Step 2] Applying LoRA adaptation...")

    # Apply LoRA to specific modules
    # Typically target attention projection layers
    model = apply_lora(
        model,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
        r=8,  # Rank of decomposition (lower = fewer params)
        lora_alpha=16,  # Scaling factor
        lora_dropout=0.1,  # Dropout for LoRA layers
    )

    # Check parameters after LoRA
    total_params_lora, trainable_params_lora = count_parameters(model)

    print(f"\nAfter applying LoRA:")
    print(f"Total parameters: {total_params_lora:,}")
    print(f"Trainable parameters: {trainable_params_lora:,}")
    print(f"Trainable percentage: {100 * trainable_params_lora / total_params_lora:.2f}%")
    print(f"Parameter reduction: {100 * (1 - trainable_params_lora / total_params):.1f}%")

    # -------------------------------------------------------------------------
    # Step 3: Freeze Base Model, Train Only LoRA
    # -------------------------------------------------------------------------
    print("\n[Step 3] Setting up training (LoRA parameters only)...")

    # Get only LoRA parameters
    lora_params = get_lora_parameters(model)

    print(f"LoRA parameter count: {sum(p.numel() for p in lora_params):,}")

    # Create optimizer for LoRA parameters only
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=3e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    print(f"Optimizer initialized for {len(list(lora_params))} parameter groups")

    # -------------------------------------------------------------------------
    # Step 4: Fine-Tuning Loop
    # -------------------------------------------------------------------------
    print("\n[Step 4] Fine-tuning with LoRA...")

    # Create fine-tuning data
    finetune_data = create_finetuning_dataset()
    print(f"Fine-tuning dataset size: {len(finetune_data)} sequences")

    # Simple training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    num_epochs = 3
    batch_size = 4

    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Create batches
        for i in range(0, len(finetune_data), batch_size):
            batch = finetune_data[i:i + batch_size]

            # Pad to same length
            max_len = max(len(seq) for seq in batch)
            padded_batch = []

            for seq in batch:
                if len(seq) < max_len:
                    padding = torch.zeros(max_len - len(seq), dtype=torch.long)
                    padded_seq = torch.cat([seq, padding])
                else:
                    padded_seq = seq

                padded_batch.append(padded_seq)

            input_ids = torch.stack(padded_batch).to(device)

            # Forward pass
            logits, loss = model(input_ids, labels=input_ids)

            # Backward pass (only updates LoRA parameters!)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    print("\nFine-tuning completed!")

    # -------------------------------------------------------------------------
    # Step 5: Verify Only LoRA Parameters Changed
    # -------------------------------------------------------------------------
    print("\n[Step 5] Verifying only LoRA parameters were updated...")

    # Check that base model parameters have no gradients
    base_params_with_grad = 0
    lora_params_with_grad = 0

    for name, param in model.named_parameters():
        if "lora_" in name:
            if param.grad is not None:
                lora_params_with_grad += 1
        else:
            if param.grad is not None:
                base_params_with_grad += 1

    print(f"Base model parameters with gradients: {base_params_with_grad}")
    print(f"LoRA parameters with gradients: {lora_params_with_grad}")

    if base_params_with_grad == 0 and lora_params_with_grad > 0:
        print("✓ Verification passed: Only LoRA parameters were trained!")
    else:
        print("⚠ Warning: Some base parameters may have been updated")

    # -------------------------------------------------------------------------
    # Step 6: Save LoRA Weights
    # -------------------------------------------------------------------------
    print("\n[Step 6] Saving LoRA weights...")

    # Extract only LoRA weights
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state_dict[name] = param.cpu()

    # Save LoRA weights (much smaller than full model!)
    lora_path = "/tmp/lora_weights.pt"
    torch.save(lora_state_dict, lora_path)

    lora_size_kb = sum(p.numel() * 4 for p in lora_state_dict.values()) / 1024
    full_size_kb = sum(p.numel() * 4 for p in model.parameters()) / 1024

    print(f"LoRA weights saved to: {lora_path}")
    print(f"LoRA weights size: {lora_size_kb:.2f} KB")
    print(f"Full model size: {full_size_kb:.2f} KB")
    print(f"Size reduction: {100 * (1 - lora_size_kb / full_size_kb):.1f}%")

    # -------------------------------------------------------------------------
    # Step 7: Merge LoRA Weights (Optional)
    # -------------------------------------------------------------------------
    print("\n[Step 7] Merging LoRA weights into base model...")

    # Merge LoRA weights back into the base model
    # This creates a standalone model without LoRA layers
    merged_model = merge_lora_weights(model)

    print("LoRA weights merged into base model")
    print("Merged model can be used without LoRA overhead")

    # -------------------------------------------------------------------------
    # Step 8: Generate with Fine-Tuned Model
    # -------------------------------------------------------------------------
    print("\n[Step 8] Generating with fine-tuned model...")

    model.eval()

    # Create a sample prompt
    prompt = torch.randint(0, 1000, (1, 10)).to(device)

    with torch.no_grad():
        output = model.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.8,
            do_sample=True,
        )

    print(f"Input length: {prompt.shape[1]}")
    print(f"Output length: {output.shape[1]}")
    print(f"Generated {output.shape[1] - prompt.shape[1]} new tokens")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("\nKey takeaways:")
    print("- LoRA reduces trainable parameters by >90%")
    print("- Only LoRA weights need to be saved (~MB vs ~GB)")
    print("- Fine-tuning is faster and requires less memory")
    print("- Base model knowledge is preserved")
    print("- Multiple LoRA adapters can be created for different tasks")
    print("=" * 70)


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)

    # Run example
    main()
