"""
Stopping Criteria
=================

Criteria for stopping text generation.
"""

import torch
from typing import List
import time


class StoppingCriteria:
    """Base class for stopping criteria."""

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> bool:
        raise NotImplementedError


class MaxLengthCriteria(StoppingCriteria):
    """Stop when maximum length is reached."""

    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> bool:
        return input_ids.shape[-1] >= self.max_length


class MaxTimeCriteria(StoppingCriteria):
    """Stop after maximum time has elapsed."""

    def __init__(self, max_time: float):
        self.max_time = max_time
        self.start_time = time.time()

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> bool:
        return (time.time() - self.start_time) >= self.max_time


class EosTokenCriteria(StoppingCriteria):
    """Stop when EOS token is generated."""

    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> bool:
        return (input_ids[:, -1] == self.eos_token_id).all()


class StoppingCriteriaList(List[StoppingCriteria]):
    """
    List of stopping criteria (stop if ANY criterion is met).
    """

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> bool:
        return any(criteria(input_ids, scores) for criteria in self)
