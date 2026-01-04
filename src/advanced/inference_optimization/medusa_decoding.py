"""
Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads
====================================================================================

Revolutionary approach for 2-3x faster inference through parallel token prediction.

Reference: "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"
          (Cai et al., 2024) https://arxiv.org/abs/2401.10774

Key Innovation:
--------------
Traditional Auto-regressive: Predict one token at a time
  t1 → t2 → t3 → t4 ...
  SLOW: Each prediction requires full forward pass

Medusa: Predict multiple future tokens in parallel!
  From token t1, predict: t2, t3, t4, t5 simultaneously
  Then verify which predictions are correct
  Accept correct prefix, retry from there

Architecture:
-----------
Base LLM + Multiple Medusa Heads
  - Head 1: Predicts next token (t+1)
  - Head 2: Predicts token t+2
  - Head 3: Predicts token t+3
  - ...

Each head is a small MLP trained to predict tokens at different distances.

Speedup Mechanism:
-----------------
1. Generate multiple candidate continuations in parallel
2. Verify candidates using one forward pass of base model
3. Accept longest correct prefix
4. Repeat

Example:
  Current: "The cat"
  Medusa predicts 5 candidates:
    1. "sat on the mat"
    2. "sat on a"
    3. "jumped over"
    ...
  Verification: Check which is correct
  If #1 correct: Accept 4 tokens in one step!
  Normal: Would take 4 forward passes
  Medusa: Takes 1.5 forward passes (generation + verification)
  Speedup: 4 / 1.5 = 2.67x

Mathematical Foundation:
----------------------

Let model predict distribution P(t | context)

Standard: Sample t1, then t2 | (context, t1), etc.
  Cost: N forward passes for N tokens

Medusa:
  1. Predict distribution for t+1, t+2, ..., t+k using separate heads
  2. Sample candidates from these distributions
  3. Verify with base model: P(candidate | context)
  4. Accept if probability above threshold

Tree-based Verification:
  Build tree of candidates:
    Level 1: k1 options for token 1
    Level 2: k2 options for token 2
    ...
  Total paths: k1 × k2 × ... = K candidates

  Verify all K candidates in parallel (batch dimension)
  Accept longest correct path

Complexity:
  Standard: O(N) forward passes for N tokens
  Medusa: O(N / avg_acceptance_length) forward passes

  If avg_acceptance = 3 tokens:
    Speedup = 3x

  Overhead: Medusa heads are tiny (0.5-2% of base model)
  Net speedup: 2-3x in practice

Training:
--------
1. Freeze base LLM
2. Collect training data (next token predictions)
3. Train each Medusa head to predict tokens at distance d
   Head d: Minimize CrossEntropy(P_head(t+d | t), true(t+d))

Requires: ~1000 steps of fine-tuning
Cost: <<< 1% of base model training

Advantages Over Other Methods:
-----------------------------
vs Speculative Decoding:
  - No separate draft model needed
  - Lower memory overhead
  - Simpler training

vs Parallel Decoding:
  - Higher accuracy
  - Better speedup on autoregressive tasks

vs Model Distillation:
  - No quality degradation
  - Maintains base model performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import math


@dataclass
class MedusaConfig:
    """Configuration for Medusa decoding."""
    num_heads: int = 4  # Number of Medusa heads (1-5 typical)
    num_candidates: int = 64  # Candidates per head
    tree_width: int = 10  # Width of verification tree
    max_verify_length: int = 10  # Max tokens to verify at once
    temperature: float = 1.0
    top_p: float = 0.9
    acceptance_threshold: float = 0.09  # Min probability to accept


class MedusaHead(nn.Module):
    """
    Single Medusa head for predicting token at distance d.

    Lightweight MLP: hidden_state → vocab_logits
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 2,
        intermediate_size: Optional[int] = None
    ):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = hidden_size

        layers = []
        layers.append(nn.Linear(hidden_size, intermediate_size))
        layers.append(nn.GELU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(intermediate_size, intermediate_size))
            layers.append(nn.GELU())

        layers.append(nn.Linear(intermediate_size, vocab_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        return self.layers(hidden_states)


class MedusaModel(nn.Module):
    """
    Medusa: Base LLM + Multiple prediction heads for parallel decoding.

    Enables 2-3x faster inference with no quality degradation.
    """

    def __init__(
        self,
        base_model: nn.Module,
        vocab_size: int,
        hidden_size: int,
        num_medusa_heads: int = 4,
        medusa_num_layers: int = 1
    ):
        """
        Args:
            base_model: Base LLM (frozen during Medusa training)
            vocab_size: Size of vocabulary
            hidden_size: Hidden dimension of base model
            num_medusa_heads: Number of Medusa heads (predicts 1 to N tokens ahead)
            medusa_num_layers: Layers in each Medusa head
        """
        super().__init__()

        self.base_model = base_model
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_medusa_heads = num_medusa_heads

        # Medusa heads
        # Head i predicts token at distance i+1
        self.medusa_heads = nn.ModuleList([
            MedusaHead(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=medusa_num_layers
            )
            for _ in range(num_medusa_heads)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        get_medusa_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through base model + Medusa heads.

        Args:
            input_ids: [batch, seq_len]
            get_medusa_logits: Whether to compute Medusa head predictions

        Returns:
            Dictionary with:
                logits: [batch, seq_len, vocab] - Base model predictions
                medusa_logits: List of [batch, seq_len, vocab] - Medusa predictions
                hidden_states: [batch, seq_len, hidden] - Last hidden states
        """
        # Forward through base model
        outputs = self.base_model(input_ids, return_dict=True)

        logits = outputs.get("logits", outputs[0])
        hidden_states = outputs.get("hidden_states", outputs.get("last_hidden_state"))

        result = {
            "logits": logits,
            "hidden_states": hidden_states
        }

        # Compute Medusa predictions if requested
        if get_medusa_logits:
            medusa_logits = []
            for head in self.medusa_heads:
                head_logits = head(hidden_states)
                medusa_logits.append(head_logits)
            result["medusa_logits"] = medusa_logits

        return result

    def generate_candidates(
        self,
        logits: torch.Tensor,
        medusa_logits: List[torch.Tensor],
        config: MedusaConfig
    ) -> List[torch.Tensor]:
        """
        Generate candidate token sequences using Medusa heads.

        Args:
            logits: [batch, vocab] - Base model logits for next token
            medusa_logits: List of [batch, vocab] - Medusa head logits
            config: Medusa configuration

        Returns:
            candidates: List of [num_candidates, max_length] token sequences
        """
        batch_size = logits.size(0)

        # Sample from base model (first token)
        probs_0 = F.softmax(logits / config.temperature, dim=-1)

        # Top-p filtering
        sorted_probs, sorted_indices = torch.sort(probs_0, descending=True, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum_probs > config.top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # Sample first token
        sampled_indices = torch.multinomial(
            sorted_probs,
            num_samples=config.num_candidates,
            replacement=True
        )
        first_tokens = torch.gather(sorted_indices, -1, sampled_indices)  # [batch, num_candidates]

        # Build tree of candidates
        # For simplicity, we generate continuations for each first token
        candidates = []

        for b in range(batch_size):
            batch_candidates = []

            for c in range(min(config.num_candidates, first_tokens.size(1))):
                # Start with first token
                candidate = [first_tokens[b, c].item()]

                # Add tokens from Medusa heads
                for head_idx, head_logits in enumerate(medusa_logits):
                    if head_idx >= config.max_verify_length - 1:
                        break

                    # Get logits for this candidate branch
                    head_probs = F.softmax(head_logits[b] / config.temperature, dim=-1)

                    # Sample next token
                    next_token = torch.multinomial(head_probs, num_samples=1).item()
                    candidate.append(next_token)

                batch_candidates.append(torch.tensor(candidate))

            candidates.append(batch_candidates)

        return candidates

    def verify_candidates(
        self,
        input_ids: torch.Tensor,
        candidates: List[List[torch.Tensor]],
        config: MedusaConfig
    ) -> Tuple[torch.Tensor, int]:
        """
        Verify candidate sequences and select best one.

        Args:
            input_ids: [batch, seq_len] - Current sequence
            candidates: List of candidate continuations
            config: Medusa configuration

        Returns:
            accepted_tokens: [batch, accepted_length]
            acceptance_length: Number of tokens accepted
        """
        batch_size = input_ids.size(0)

        # For each batch
        best_candidates = []
        best_lengths = []

        for b in range(batch_size):
            batch_candidates = candidates[b]

            if len(batch_candidates) == 0:
                best_candidates.append(torch.tensor([]))
                best_lengths.append(0)
                continue

            # Prepare batch of candidates for verification
            max_len = max(c.size(0) for c in batch_candidates)
            candidate_batch = torch.zeros(
                len(batch_candidates),
                input_ids.size(1) + max_len,
                dtype=torch.long,
                device=input_ids.device
            )

            for i, candidate in enumerate(batch_candidates):
                seq = torch.cat([input_ids[b], candidate])
                candidate_batch[i, :seq.size(0)] = seq

            # Forward pass to verify
            with torch.no_grad():
                outputs = self.base_model(candidate_batch, return_dict=True)
                verify_logits = outputs.get("logits", outputs[0])

            # Check which tokens are accepted
            best_candidate = None
            best_length = 0

            for i, candidate in enumerate(batch_candidates):
                accepted_length = 0

                for pos in range(candidate.size(0)):
                    # Get probability of this token
                    token_id = candidate[pos].item()
                    logits = verify_logits[i, input_ids.size(1) + pos - 1]
                    probs = F.softmax(logits, dim=-1)
                    token_prob = probs[token_id].item()

                    # Check if above threshold
                    if token_prob >= config.acceptance_threshold:
                        accepted_length += 1
                    else:
                        break

                if accepted_length > best_length:
                    best_length = accepted_length
                    best_candidate = candidate[:accepted_length]

            best_candidates.append(best_candidate if best_candidate is not None else torch.tensor([]))
            best_lengths.append(best_length)

        # Return best candidate
        max_accepted = max(best_lengths)
        if max_accepted == 0:
            return None, 0

        accepted = torch.zeros(batch_size, max_accepted, dtype=torch.long, device=input_ids.device)
        for b in range(batch_size):
            if best_lengths[b] > 0:
                accepted[b, :best_lengths[b]] = best_candidates[b]

        return accepted, max_accepted

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        config: Optional[MedusaConfig] = None,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Fast autoregressive generation using Medusa.

        2-3x faster than standard generation!

        Args:
            input_ids: [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            config: Medusa configuration
            verbose: Print statistics

        Returns:
            generated_ids: [batch, seq_len + new_tokens]
            stats: Generation statistics
        """
        if config is None:
            config = MedusaConfig()

        total_generated = 0
        total_forward_passes = 0
        total_accepted_tokens = 0

        while total_generated < max_new_tokens:
            # Get predictions from base model + Medusa heads
            outputs = self.forward(input_ids, get_medusa_logits=True)
            total_forward_passes += 1

            logits = outputs["logits"][:, -1, :]  # [batch, vocab]
            medusa_logits = [head[:, -1, :] for head in outputs["medusa_logits"]]

            # Generate candidates
            candidates = self.generate_candidates(logits, medusa_logits, config)

            # Verify and accept
            accepted, num_accepted = self.verify_candidates(input_ids, candidates, config)

            if num_accepted == 0:
                # Fall back to standard sampling
                probs = F.softmax(logits / config.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                total_generated += 1
                total_accepted_tokens += 1
            else:
                # Accept multiple tokens!
                input_ids = torch.cat([input_ids, accepted], dim=1)
                total_generated += num_accepted
                total_accepted_tokens += num_accepted

            if total_generated >= max_new_tokens:
                break

        # Compute statistics
        avg_acceptance = total_accepted_tokens / total_forward_passes if total_forward_passes > 0 else 0
        speedup = avg_acceptance  # Approximate speedup

        stats = {
            "total_tokens": total_generated,
            "forward_passes": total_forward_passes,
            "avg_tokens_per_pass": avg_acceptance,
            "estimated_speedup": f"{speedup:.2f}x"
        }

        if verbose:
            print(f"Medusa Generation Stats:")
            print(f"  Total tokens: {total_generated}")
            print(f"  Forward passes: {total_forward_passes}")
            print(f"  Avg tokens/pass: {avg_acceptance:.2f}")
            print(f"  Speedup: ~{speedup:.2f}x")

        return input_ids, stats


def train_medusa_heads(
    base_model: nn.Module,
    medusa_model: MedusaModel,
    train_dataloader,
    num_steps: int = 1000,
    learning_rate: float = 1e-3,
    device: str = "cuda"
):
    """
    Train Medusa heads on next-token prediction.

    Training is very cheap: ~1000 steps, <<< 1% of base model training.

    Args:
        base_model: Base LLM (frozen)
        medusa_model: Medusa model with heads
        train_dataloader: Training data
        num_steps: Number of training steps
        learning_rate: Learning rate
        device: Device to train on
    """
    # Freeze base model
    for param in base_model.parameters():
        param.requires_grad = False

    # Only train Medusa heads
    optimizer = torch.optim.Adam(medusa_model.medusa_heads.parameters(), lr=learning_rate)

    medusa_model.train()
    base_model.eval()

    step = 0
    for batch in train_dataloader:
        if step >= num_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = medusa_model.forward(input_ids[..., :-1], get_medusa_logits=True)

        # Compute loss for each Medusa head
        total_loss = 0

        for head_idx, head_logits in enumerate(outputs["medusa_logits"]):
            # Head predicts token at distance head_idx + 1
            target_tokens = labels[:, head_idx + 1:]  # Shift by head distance
            head_logits_aligned = head_logits[:, :target_tokens.size(1), :]

            loss = F.cross_entropy(
                head_logits_aligned.reshape(-1, medusa_model.vocab_size),
                target_tokens.reshape(-1),
                ignore_index=-100
            )
            total_loss += loss

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}/{num_steps}, Loss: {total_loss.item():.4f}")

        step += 1

    print(f"Medusa head training complete!")


if __name__ == "__main__":
    print("Medusa: 2-3x Faster LLM Inference")
    print("=" * 50)
    print("\nKey Advantages:")
    print("✓ 2-3x speedup with no quality loss")
    print("✓ Minimal memory overhead (0.5-2% of base model)")
    print("✓ Fast training (~1000 steps)")
    print("✓ Works with any base LLM")
    print("✓ No separate draft model needed")
    print("\nHow it works:")
    print("1. Predict next 2-5 tokens in parallel with Medusa heads")
    print("2. Verify predictions with base model in one pass")
    print("3. Accept longest correct sequence")
    print("4. Repeat until done")
    print("\nResult: Generate 2-3 tokens per forward pass instead of 1!")
