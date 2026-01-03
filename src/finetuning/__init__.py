"""
Parameter-Efficient Fine-Tuning (PEFT)
=======================================

Methods for efficient fine-tuning of large language models:
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Prefix Tuning
- P-Tuning v2
- Adapter Layers
- Prompt Tuning
"""

from .lora import LoRALayer, LoRALinear, apply_lora
from .prefix_tuning import PrefixTuning
from .adapter import AdapterLayer
from .prompt_tuning import PromptTuning

__all__ = [
    "LoRALayer",
    "LoRALinear",
    "apply_lora",
    "PrefixTuning",
    "AdapterLayer",
    "PromptTuning",
]
