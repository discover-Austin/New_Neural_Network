"""
Sentiment Analysis
==================

Research-grounded sentiment analysis implementations:

1. Sequence Classification
   - Reference: "BERT for Sequence Classification" (Devlin et al., 2019)
   - Uses [CLS] token representation

2. Aspect-Based Sentiment Analysis (ABSA)
   - Reference: "Aspect-Based Sentiment Analysis" (Pontiki et al., SemEval 2014)
   - Reference: "ABSA with Attention" (Wang et al., EMNLP 2016)
   - Identifies sentiment towards specific aspects

3. Fine-Grained Sentiment
   - Reference: "Recursive Deep Models for Semantic Compositionality" (Socher et al., EMNLP 2013)
   - 5-class: Very Negative, Negative, Neutral, Positive, Very Positive

Mathematical Foundation:
-----------------------

Sequence Classification:
  h_[CLS] = Encoder(input)[0]  # First token representation
  logits = Linear(Dropout(h_[CLS]))
  P(class|input) = softmax(logits)

Aspect-Based:
  For each aspect:
    - Extract aspect representation
    - Attend over context with aspect query
    - Classify sentiment

  Attention: α_i = softmax(h_aspect^T W h_context_i)
  Context vector: c = Σ α_i h_context_i
  Sentiment: softmax(Linear([c; h_aspect]))

Complexity:
- Sequence: O(1) classification after encoding
- ABSA: O(A × T) where A=num aspects, T=seq length
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SentimentPolarity(Enum):
    """Sentiment polarities."""
    VERY_NEGATIVE = 0
    NEGATIVE = 1
    NEUTRAL = 2
    POSITIVE = 3
    VERY_POSITIVE = 4


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""
    d_model: int = 768
    num_classes: int = 3  # Negative, Neutral, Positive
    dropout: float = 0.1
    pooling_strategy: str = "cls"  # "cls", "mean", "max"
    use_attention: bool = False  # For ABSA


class SequenceSentimentClassifier(nn.Module):
    """
    Sequence-level sentiment classification.

    Reference: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)

    Architecture:
    Encoder → Pool → Dropout → Linear → Softmax

    Pooling strategies:
    - CLS: Use [CLS] token representation (BERT-style)
    - Mean: Average all token representations
    - Max: Max pooling over sequence
    """

    def __init__(self, config: SentimentConfig):
        super().__init__()
        self.config = config

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, config.num_classes)

    def pool_sequence(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool sequence representations.

        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len]

        Returns:
            Pooled representation [batch, d_model]
        """
        if self.config.pooling_strategy == "cls":
            # Use first token (CLS)
            return hidden_states[:, 0]

        elif self.config.pooling_strategy == "mean":
            # Mean pooling
            if attention_mask is not None:
                # Mask padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_hidden / sum_mask
            else:
                return hidden_states.mean(dim=1)

        elif self.config.pooling_strategy == "max":
            # Max pooling
            if attention_mask is not None:
                # Set padding to large negative value
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_states = hidden_states.clone()
                hidden_states[mask_expanded == 0] = -1e9
            return hidden_states.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling_strategy}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            hidden_states: Encoder outputs [batch, seq_len, d_model]
            labels: Ground truth labels [batch]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Dictionary with loss (if labels provided) and predictions
        """
        # Pool sequence
        pooled = self.pool_sequence(hidden_states, attention_mask)

        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        outputs = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs["loss"] = loss

        # Predictions
        predictions = logits.argmax(dim=-1)
        probabilities = F.softmax(logits, dim=-1)

        outputs["predictions"] = predictions
        outputs["probabilities"] = probabilities

        return outputs


class AspectBasedSentimentAnalysis(nn.Module):
    """
    Aspect-Based Sentiment Analysis (ABSA).

    Reference: "Aspect Level Sentiment Classification with Attention" (Wang et al., EMNLP 2016)
    Reference: "Target-Dependent Sentiment Classification with LSTM" (Tang et al., ACL 2016)

    Task: Given text and aspect, classify sentiment towards that aspect.

    Example:
    - Text: "The food was great but the service was terrible"
    - Aspect: "food" → Positive
    - Aspect: "service" → Negative

    Architecture:
    1. Encode text and aspect separately
    2. Attention: Use aspect as query to attend over text
    3. Combine attended context with aspect representation
    4. Classify sentiment
    """

    def __init__(self, config: SentimentConfig):
        super().__init__()
        self.config = config

        # Attention mechanism
        self.attention = nn.Linear(config.d_model, config.d_model)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_classes),
        )

    def compute_attention(
        self,
        aspect_repr: torch.Tensor,
        text_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention over text using aspect as query.

        Args:
            aspect_repr: Aspect representation [batch, d_model]
            text_hidden: Text hidden states [batch, seq_len, d_model]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            context_vector: Attended context [batch, d_model]
            attention_weights: Attention weights [batch, seq_len]
        """
        # Query: aspect representation
        query = self.attention(aspect_repr).unsqueeze(1)  # [batch, 1, d_model]

        # Scores: query^T · keys
        scores = torch.matmul(query, text_hidden.transpose(1, 2)).squeeze(1)  # [batch, seq_len]

        # Mask padding
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch, seq_len]

        # Context vector: weighted sum
        context_vector = torch.matmul(
            attention_weights.unsqueeze(1),
            text_hidden,
        ).squeeze(1)  # [batch, d_model]

        return context_vector, attention_weights

    def forward(
        self,
        text_hidden: torch.Tensor,
        aspect_hidden: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            text_hidden: Text encoder outputs [batch, seq_len, d_model]
            aspect_hidden: Aspect representation [batch, d_model]
            labels: Ground truth labels [batch]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Dictionary with loss and predictions
        """
        # Attend over text with aspect query
        context_vector, attention_weights = self.compute_attention(
            aspect_hidden,
            text_hidden,
            attention_mask,
        )

        # Combine context and aspect
        combined = torch.cat([context_vector, aspect_hidden], dim=-1)

        # Classify
        logits = self.classifier(combined)

        outputs = {
            "logits": logits,
            "attention_weights": attention_weights,
        }

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs["loss"] = loss

        # Predictions
        predictions = logits.argmax(dim=-1)
        probabilities = F.softmax(logits, dim=-1)

        outputs["predictions"] = predictions
        outputs["probabilities"] = probabilities

        return outputs


class HierarchicalSentimentClassifier(nn.Module):
    """
    Hierarchical sentiment classification for documents.

    Reference: "Hierarchical Attention Networks for Document Classification" (Yang et al., NAACL 2016)

    Architecture:
    1. Word-level attention: Attend over words in each sentence
    2. Sentence-level attention: Attend over sentences in document
    3. Document representation → Classification

    Use case: Long documents where sentence structure matters
    """

    def __init__(self, config: SentimentConfig):
        super().__init__()
        self.config = config

        # Word-level attention
        self.word_attention = nn.Linear(config.d_model, 1)

        # Sentence encoder (optional)
        self.sentence_encoder = nn.GRU(
            config.d_model,
            config.d_model // 2,
            bidirectional=True,
            batch_first=True,
        )

        # Sentence-level attention
        self.sentence_attention = nn.Linear(config.d_model, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_classes),
        )

    def attend_words(
        self,
        word_hidden: torch.Tensor,
        word_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Word-level attention.

        Args:
            word_hidden: [batch, num_sentences, num_words, d_model]
            word_mask: [batch, num_sentences, num_words]

        Returns:
            Sentence representations [batch, num_sentences, d_model]
        """
        batch_size, num_sentences, num_words, d_model = word_hidden.shape

        # Flatten for attention
        word_hidden_flat = word_hidden.view(-1, num_words, d_model)

        # Attention scores
        scores = self.word_attention(word_hidden_flat).squeeze(-1)  # [batch*num_sent, num_words]

        # Mask
        if word_mask is not None:
            word_mask_flat = word_mask.view(-1, num_words)
            scores = scores.masked_fill(word_mask_flat == 0, -1e9)

        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Sentence representations
        sentence_repr = torch.matmul(
            attention_weights.unsqueeze(1),
            word_hidden_flat,
        ).squeeze(1)  # [batch*num_sent, d_model]

        # Reshape
        sentence_repr = sentence_repr.view(batch_size, num_sentences, d_model)

        return sentence_repr

    def attend_sentences(
        self,
        sentence_hidden: torch.Tensor,
        sentence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sentence-level attention.

        Args:
            sentence_hidden: [batch, num_sentences, d_model]
            sentence_mask: [batch, num_sentences]

        Returns:
            Document representation [batch, d_model]
        """
        # Encode sentences
        sentence_encoded, _ = self.sentence_encoder(sentence_hidden)

        # Attention scores
        scores = self.sentence_attention(sentence_encoded).squeeze(-1)

        # Mask
        if sentence_mask is not None:
            scores = scores.masked_fill(sentence_mask == 0, -1e9)

        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Document representation
        doc_repr = torch.matmul(
            attention_weights.unsqueeze(1),
            sentence_encoded,
        ).squeeze(1)

        return doc_repr

    def forward(
        self,
        word_hidden: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        word_mask: Optional[torch.Tensor] = None,
        sentence_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            word_hidden: [batch, num_sentences, num_words, d_model]
            labels: [batch]
            word_mask: [batch, num_sentences, num_words]
            sentence_mask: [batch, num_sentences]

        Returns:
            Dictionary with loss and predictions
        """
        # Word-level attention
        sentence_repr = self.attend_words(word_hidden, word_mask)

        # Sentence-level attention
        doc_repr = self.attend_sentences(sentence_repr, sentence_mask)

        # Classify
        logits = self.classifier(doc_repr)

        outputs = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs["loss"] = loss

        # Predictions
        predictions = logits.argmax(dim=-1)
        probabilities = F.softmax(logits, dim=-1)

        outputs["predictions"] = predictions
        outputs["probabilities"] = probabilities

        return outputs


class MultiTaskSentimentClassifier(nn.Module):
    """
    Multi-task sentiment classifier.

    Jointly learns:
    1. Sentiment polarity (positive/negative/neutral)
    2. Sentiment intensity (weak/moderate/strong)
    3. Emotion classification (joy/anger/sadness/etc.)

    Reference: "Multi-Task Deep Neural Networks for Natural Language Understanding" (Liu et al., ACL 2019)

    Shared encoder with task-specific heads.
    """

    def __init__(
        self,
        config: SentimentConfig,
        num_intensity_classes: int = 3,
        num_emotion_classes: int = 6,
    ):
        super().__init__()
        self.config = config

        self.dropout = nn.Dropout(config.dropout)

        # Task-specific classifiers
        self.polarity_classifier = nn.Linear(config.d_model, config.num_classes)
        self.intensity_classifier = nn.Linear(config.d_model, num_intensity_classes)
        self.emotion_classifier = nn.Linear(config.d_model, num_emotion_classes)

    def forward(
        self,
        hidden_states: torch.Tensor,
        polarity_labels: Optional[torch.Tensor] = None,
        intensity_labels: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            hidden_states: Encoder outputs [batch, seq_len, d_model]
            polarity_labels: [batch]
            intensity_labels: [batch]
            emotion_labels: [batch]
            attention_mask: [batch, seq_len]

        Returns:
            Dictionary with losses and predictions for all tasks
        """
        # Pool (use CLS token)
        pooled = hidden_states[:, 0]
        pooled = self.dropout(pooled)

        # Task-specific predictions
        polarity_logits = self.polarity_classifier(pooled)
        intensity_logits = self.intensity_classifier(pooled)
        emotion_logits = self.emotion_classifier(pooled)

        outputs = {
            "polarity_logits": polarity_logits,
            "intensity_logits": intensity_logits,
            "emotion_logits": emotion_logits,
        }

        # Losses
        total_loss = 0.0

        if polarity_labels is not None:
            polarity_loss = F.cross_entropy(polarity_logits, polarity_labels)
            outputs["polarity_loss"] = polarity_loss
            total_loss += polarity_loss

        if intensity_labels is not None:
            intensity_loss = F.cross_entropy(intensity_logits, intensity_labels)
            outputs["intensity_loss"] = intensity_loss
            total_loss += intensity_loss

        if emotion_labels is not None:
            emotion_loss = F.cross_entropy(emotion_logits, emotion_labels)
            outputs["emotion_loss"] = emotion_loss
            total_loss += emotion_loss

        if total_loss > 0:
            outputs["loss"] = total_loss

        # Predictions
        outputs["polarity_predictions"] = polarity_logits.argmax(dim=-1)
        outputs["intensity_predictions"] = intensity_logits.argmax(dim=-1)
        outputs["emotion_predictions"] = emotion_logits.argmax(dim=-1)

        return outputs
