"""
Logits Processors
=================

Processors that modify logits during generation.
"""

import torch
import torch.nn.functional as F
from typing import List
import math


class LogitsProcessor:
    """Base class for logits processors."""

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TemperatureLogitsProcessor(LogitsProcessor):
    """
    Apply temperature scaling to logits.

    Lower temperature makes distribution sharper (more deterministic).
    Higher temperature makes distribution flatter (more random).
    """

    def __init__(self, temperature: float):
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


class TopKLogitsProcessor(LogitsProcessor):
    """
    Keep only top-k logits, set others to -inf.
    """

    def __init__(self, top_k: int, min_tokens_to_keep: int = 1):
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")
        self.top_k = max(top_k, min_tokens_to_keep)
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        top_k = min(self.top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits


class TopPLogitsProcessor(LogitsProcessor):
    """
    Keep smallest set of tokens with cumulative probability >= top_p.
    """

    def __init__(self, top_p: float, min_tokens_to_keep: int = 1):
        if top_p < 0 or top_p > 1:
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        self.top_p = top_p
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > self.top_p

        # Keep at least min_tokens_to_keep
        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        # Shift to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    Penalize tokens that have already appeared.

    Reference: CTRL paper
    """

    def __init__(self, penalty: float):
        if penalty <= 0:
            raise ValueError(f"penalty must be positive, got {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        score = torch.gather(logits, 1, input_ids)

        # Apply penalty
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        logits.scatter_(1, input_ids, score)

        return logits


class MinLengthLogitsProcessor(LogitsProcessor):
    """
    Enforce minimum generation length by suppressing EOS token.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        if min_length < 0:
            raise ValueError(f"min_length must be non-negative, got {min_length}")
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        cur_len = input_ids.shape[-1]

        if cur_len < self.min_length:
            logits[:, self.eos_token_id] = float('-inf')

        return logits


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    Prevent repeating n-grams during generation.
    """

    def __init__(self, ngram_size: int):
        if ngram_size < 1:
            raise ValueError(f"ngram_size must be positive, got {ngram_size}")
        self.ngram_size = ngram_size

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        batch_size, cur_len = input_ids.shape

        if cur_len + 1 < self.ngram_size:
            return logits

        for batch_idx in range(batch_size):
            # Get all n-grams that end at current position
            generated_ngrams = {}
            for ngram_idx in range(cur_len - self.ngram_size + 1):
                ngram = tuple(input_ids[batch_idx, ngram_idx:ngram_idx + self.ngram_size].tolist())
                prev_ngram = ngram[:-1]
                if prev_ngram in generated_ngrams:
                    generated_ngrams[prev_ngram].append(ngram[-1])
                else:
                    generated_ngrams[prev_ngram] = [ngram[-1]]

            # Get current n-gram prefix
            current_ngram = tuple(input_ids[batch_idx, -(self.ngram_size - 1):].tolist())

            # Ban tokens that would complete a repeated n-gram
            if current_ngram in generated_ngrams:
                for banned_token in generated_ngrams[current_ngram]:
                    logits[batch_idx, banned_token] = float('-inf')

        return logits


class LogitsProcessorList(List[LogitsProcessor]):
    """
    List of logits processors that are applied sequentially.
    """

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        for processor in self:
            logits = processor(input_ids, logits)
        return logits
