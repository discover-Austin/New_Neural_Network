"""
Generation Strategies
=====================

Various decoding strategies for text generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
import math


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 100
    min_length: int = 0
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    num_beams: int = 1
    num_return_sequences: int = 1
    early_stopping: bool = False
    do_sample: bool = False
    eos_token_id: int = 2
    pad_token_id: int = 0


class GreedySearch:
    """
    Greedy decoding: always select the most probable token.

    Args:
        config: Generation configuration
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate using greedy decoding.

        Args:
            model: Language model
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask

        Returns:
            Generated token IDs
        """
        batch_size = input_ids.size(0)
        cur_len = input_ids.size(1)

        # Track which sequences are finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        while cur_len < self.config.max_length:
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs.get("logits", outputs)

            # Get logits for next token
            next_token_logits = logits[:, -1, :]

            # Greedy: select most probable token
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Update sequences
            next_tokens = next_tokens * unfinished_sequences + \
                         self.config.pad_token_id * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((batch_size, 1))
                ], dim=-1)

            # Check for EOS tokens
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.ne(self.config.eos_token_id).long()
            )

            cur_len += 1

            # Stop if all sequences are finished
            if unfinished_sequences.max() == 0:
                break

        return input_ids


class BeamSearch:
    """
    Beam search decoding.

    Maintains top-k most probable sequences at each step.

    Args:
        config: Generation configuration
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.num_beams = config.num_beams

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate using beam search.

        Args:
            model: Language model
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask

        Returns:
            Generated token IDs [batch_size * num_return_sequences, seq_len]
        """
        batch_size = input_ids.size(0)
        cur_len = input_ids.size(1)

        # Expand inputs for beam search
        input_ids = input_ids.unsqueeze(1).repeat(1, self.num_beams, 1)
        input_ids = input_ids.view(batch_size * self.num_beams, cur_len)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_beams, 1)
            attention_mask = attention_mask.view(batch_size * self.num_beams, -1)

        # Initialize beam scores
        beam_scores = torch.zeros(batch_size, self.num_beams, device=input_ids.device)
        beam_scores[:, 1:] = float('-inf')
        beam_scores = beam_scores.view(-1)

        # Track finished sequences
        done = [False] * batch_size

        while cur_len < self.config.max_length:
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs.get("logits", outputs)

            # Get next token logits
            next_token_logits = logits[:, -1, :]

            # Compute scores
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            # Add beam scores
            next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)

            # Reshape for beam search
            vocab_size = next_token_scores.size(-1)
            next_token_scores = next_token_scores.view(batch_size, self.num_beams * vocab_size)

            # Get top 2*num_beams candidates
            next_token_scores, next_tokens = torch.topk(
                next_token_scores,
                2 * self.num_beams,
                dim=1,
                largest=True,
                sorted=True
            )

            # Process each batch
            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    next_batch_beam.extend([(0, self.config.pad_token_id, 0)] * self.num_beams)
                    continue

                next_sent_beam = []

                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_token_scores[batch_idx])
                ):
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * self.num_beams + beam_id

                    # Check if we have enough beams
                    if len(next_sent_beam) >= self.num_beams:
                        break

                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # Check if done
                if next_sent_beam[0][1].item() == self.config.eos_token_id:
                    if self.config.early_stopping:
                        done[batch_idx] = True

                next_batch_beam.extend(next_sent_beam)

            # Prepare next iteration
            beam_scores = torch.tensor([x[0] for x in next_batch_beam], device=input_ids.device)
            beam_tokens = torch.tensor([x[1] for x in next_batch_beam], device=input_ids.device)
            beam_idx = torch.tensor([x[2] for x in next_batch_beam], device=input_ids.device, dtype=torch.long)

            # Update input_ids
            input_ids = input_ids[beam_idx]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(-1)], dim=-1)

            if attention_mask is not None:
                attention_mask = attention_mask[beam_idx]
                attention_mask = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((batch_size * self.num_beams, 1))
                ], dim=-1)

            cur_len += 1

            if all(done):
                break

        # Return top sequences
        return input_ids.view(batch_size, self.num_beams, -1)[:, :self.config.num_return_sequences].reshape(
            batch_size * self.config.num_return_sequences, -1
        )


class NucleusSampling:
    """
    Nucleus (top-p) sampling.

    Samples from smallest set of tokens with cumulative probability >= p.

    Reference: "The Curious Case of Neural Text Degeneration"
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

    def _top_p_filtering(
        self,
        logits: torch.Tensor,
        top_p: float,
        min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        """Apply nucleus filtering."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

        # Shift right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

        return logits

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate using nucleus sampling."""
        batch_size = input_ids.size(0)
        cur_len = input_ids.size(1)

        while cur_len < self.config.max_length:
            outputs = model(input_ids, attention_mask=attention_mask)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs.get("logits", outputs)

            next_token_logits = logits[:, -1, :] / self.config.temperature

            # Apply nucleus filtering
            if self.config.top_p is not None and self.config.top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, self.config.top_p)

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((batch_size, 1))
                ], dim=-1)

            cur_len += 1

            # Check for EOS
            if (next_tokens == self.config.eos_token_id).all():
                break

        return input_ids


class TopKSampling:
    """Top-k sampling: sample from top k most probable tokens."""

    def __init__(self, config: GenerationConfig):
        self.config = config

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate using top-k sampling."""
        batch_size = input_ids.size(0)
        cur_len = input_ids.size(1)

        while cur_len < self.config.max_length:
            outputs = model(input_ids, attention_mask=attention_mask)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs.get("logits", outputs)

            next_token_logits = logits[:, -1, :] / self.config.temperature

            # Apply top-k filtering
            if self.config.top_k is not None and self.config.top_k > 0:
                top_k = min(self.config.top_k, next_token_logits.size(-1))
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((batch_size, 1))
                ], dim=-1)

            cur_len += 1

            if (next_tokens == self.config.eos_token_id).all():
                break

        return input_ids


class ContrastiveDecoding:
    """
    Contrastive Decoding.

    Uses difference between expert and amateur model logits to improve generation.

    Reference: "Contrastive Decoding: Open-ended Text Generation as Optimization"
    """

    def __init__(
        self,
        config: GenerationConfig,
        amateur_model: Optional[nn.Module] = None,
        alpha: float = 0.5,
    ):
        self.config = config
        self.amateur_model = amateur_model
        self.alpha = alpha

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate using contrastive decoding."""
        batch_size = input_ids.size(0)
        cur_len = input_ids.size(1)

        while cur_len < self.config.max_length:
            # Expert model logits
            expert_outputs = model(input_ids, attention_mask=attention_mask)
            if isinstance(expert_outputs, tuple):
                expert_logits = expert_outputs[0]
            else:
                expert_logits = expert_outputs.get("logits", expert_outputs)

            expert_logits = expert_logits[:, -1, :]

            # Amateur model logits (if available)
            if self.amateur_model is not None:
                amateur_outputs = self.amateur_model(input_ids, attention_mask=attention_mask)
                if isinstance(amateur_outputs, tuple):
                    amateur_logits = amateur_outputs[0]
                else:
                    amateur_logits = amateur_outputs.get("logits", amateur_outputs)

                amateur_logits = amateur_logits[:, -1, :]

                # Contrastive logits
                next_token_logits = (1 + self.alpha) * expert_logits - self.alpha * amateur_logits
            else:
                next_token_logits = expert_logits

            next_token_logits = next_token_logits / self.config.temperature

            # Sample
            if self.config.do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((batch_size, 1))
                ], dim=-1)

            cur_len += 1

            if (next_tokens == self.config.eos_token_id).all():
                break

        return input_ids


class SpeculativeDecoding:
    """
    Speculative Decoding.

    Uses smaller draft model to generate candidates, verified by larger target model.
    Significantly speeds up inference.

    Reference: "Fast Inference from Transformers via Speculative Decoding"
    """

    def __init__(
        self,
        config: GenerationConfig,
        draft_model: nn.Module,
        num_speculative_tokens: int = 4,
    ):
        self.config = config
        self.draft_model = draft_model
        self.num_speculative_tokens = num_speculative_tokens

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate using speculative decoding."""
        batch_size = input_ids.size(0)
        cur_len = input_ids.size(1)

        while cur_len < self.config.max_length:
            # Draft model generates candidate tokens
            draft_input_ids = input_ids.clone()

            for _ in range(self.num_speculative_tokens):
                draft_outputs = self.draft_model(draft_input_ids, attention_mask=attention_mask)
                if isinstance(draft_outputs, tuple):
                    draft_logits = draft_outputs[0]
                else:
                    draft_logits = draft_outputs.get("logits", draft_outputs)

                draft_next_token = torch.argmax(draft_logits[:, -1, :], dim=-1)
                draft_input_ids = torch.cat([draft_input_ids, draft_next_token.unsqueeze(-1)], dim=-1)

            # Target model verifies candidates
            target_outputs = model(draft_input_ids, attention_mask=attention_mask)
            if isinstance(target_outputs, tuple):
                target_logits = target_outputs[0]
            else:
                target_logits = target_outputs.get("logits", target_outputs)

            # Verify each speculative token
            accepted = 0
            for i in range(self.num_speculative_tokens):
                target_probs = F.softmax(target_logits[:, cur_len + i - 1, :], dim=-1)
                draft_token = draft_input_ids[:, cur_len + i]

                # Accept with probability min(1, p_target / p_draft)
                if torch.rand(1).item() < target_probs[0, draft_token].item():
                    accepted += 1
                else:
                    break

            # Update input_ids with accepted tokens
            if accepted > 0:
                input_ids = draft_input_ids[:, :cur_len + accepted]
                cur_len += accepted
            else:
                # Fallback: sample from target model
                next_token = torch.multinomial(F.softmax(target_logits[:, -1, :], dim=-1), 1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                cur_len += 1

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((batch_size, accepted if accepted > 0 else 1))
                ], dim=-1)

            if cur_len >= self.config.max_length:
                break

        return input_ids
