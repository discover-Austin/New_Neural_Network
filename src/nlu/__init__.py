"""
Natural Language Understanding (NLU) Components
================================================

Research-grounded NLU implementations for:

1. Named Entity Recognition (NER)
   - Token classification with CRF
   - Span-based NER
   - BIO/BIOES tagging schemes

2. Sentiment Analysis
   - Sequence classification
   - Aspect-based sentiment (ABSA)
   - Hierarchical classification
   - Multi-task sentiment

3. Intent Classification and Slot Filling
   - Intent classification
   - Slot filling (sequence labeling)
   - Joint models with slot-gating
   - Stack-propagation framework
   - Dialogue state tracking

All components are research-verified with peer-reviewed citations.
"""

from .named_entity_recognition import (
    NERConfig,
    NERTagScheme,
    CRF,
    TokenClassificationNER,
    SpanBasedNER,
    NERPostProcessor,
)

from .sentiment_analysis import (
    SentimentConfig,
    SentimentPolarity,
    SequenceSentimentClassifier,
    AspectBasedSentimentAnalysis,
    HierarchicalSentimentClassifier,
    MultiTaskSentimentClassifier,
)

from .intent_and_slot import (
    IntentSlotConfig,
    IntentClassifier,
    SlotFilling,
    SlotGatedIntentSlot,
    StackPropagationIntentSlot,
    DialogueStateTracker,
)

__all__ = [
    # NER
    "NERConfig",
    "NERTagScheme",
    "CRF",
    "TokenClassificationNER",
    "SpanBasedNER",
    "NERPostProcessor",
    # Sentiment
    "SentimentConfig",
    "SentimentPolarity",
    "SequenceSentimentClassifier",
    "AspectBasedSentimentAnalysis",
    "HierarchicalSentimentClassifier",
    "MultiTaskSentimentClassifier",
    # Intent and Slot
    "IntentSlotConfig",
    "IntentClassifier",
    "SlotFilling",
    "SlotGatedIntentSlot",
    "StackPropagationIntentSlot",
    "DialogueStateTracker",
]
