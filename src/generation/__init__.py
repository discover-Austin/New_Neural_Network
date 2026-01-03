"""
Text Generation
===============

Advanced text generation strategies:
- Greedy decoding
- Beam search
- Nucleus (top-p) sampling
- Top-k sampling
- Contrastive decoding
- Speculative decoding
"""

from .strategies import (
    GreedySearch,
    BeamSearch,
    NucleusSampling,
    TopKSampling,
    ContrastiveDecoding,
    SpeculativeDecoding,
)
from .logits_processors import (
    TemperatureLogitsProcessor,
    TopKLogitsProcessor,
    TopPLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    MinLengthLogitsProcessor,
)
from .stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    EosTokenCriteria,
)

__all__ = [
    "GreedySearch",
    "BeamSearch",
    "NucleusSampling",
    "TopKSampling",
    "ContrastiveDecoding",
    "SpeculativeDecoding",
    "TemperatureLogitsProcessor",
    "TopKLogitsProcessor",
    "TopPLogitsProcessor",
    "RepetitionPenaltyLogitsProcessor",
    "MinLengthLogitsProcessor",
    "MaxLengthCriteria",
    "MaxTimeCriteria",
    "EosTokenCriteria",
]
