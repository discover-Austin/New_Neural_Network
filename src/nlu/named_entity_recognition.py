"""
Named Entity Recognition (NER)
==============================

Research-grounded NER implementations:

1. Token Classification (CRF-based)
   - Reference: "Conditional Random Fields" (Lafferty et al., ICML 2001)
   - Reference: "Neural Architectures for NER" (Lample et al., NAACL 2016)
   - Uses: BERT + CRF, BiLSTM + CRF

2. Span-based NER
   - Reference: "A Unified MRC Framework for NER" (Li et al., ACL 2020)
   - Reference: "SpanNER" (Zhong et al., EMNLP 2021)
   - Treats NER as span extraction

3. BIO Tagging Scheme
   - B: Beginning of entity
   - I: Inside entity
   - O: Outside entity
   - Extensions: BIOES (also E=End, S=Single)

Mathematical Foundation:
-----------------------

CRF (Conditional Random Fields):
  P(y|x) = exp(Σ_t ψ(y_t, y_{t-1}, x)) / Z(x)

  where:
  - ψ = transition score + emission score
  - Transition: score for y_{t-1} → y_t
  - Emission: score for label y_t given features x_t
  - Z(x) = normalization constant

Viterbi Decoding:
  y* = argmax_y P(y|x)

  Dynamic programming: O(T × K²) where T=sequence length, K=num labels

Span-based:
  Score(span) = MLP([h_start; h_end; h_max; width])

  Enumerate all spans, classify each independently or with beam search

Complexity:
- Token classification: O(T × K²) for CRF, O(T × K) for softmax
- Span-based: O(T² × K) to enumerate and classify all spans
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class NERTagScheme(Enum):
    """NER tagging schemes."""
    BIO = "bio"  # Begin, Inside, Outside
    BIOES = "bioes"  # Begin, Inside, Outside, End, Single
    IO = "io"  # Inside, Outside


@dataclass
class NERConfig:
    """Configuration for NER models."""
    d_model: int = 768
    num_labels: int = 9  # Example: PER, ORG, LOC, MISC (BIO = 9 labels)
    dropout: float = 0.1
    use_crf: bool = True
    tag_scheme: NERTagScheme = NERTagScheme.BIO
    max_span_length: int = 10  # For span-based NER


class CRF(nn.Module):
    """
    Conditional Random Fields for sequence labeling.

    Reference: Lafferty et al., ICML 2001
    "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data"

    Mathematical:
    P(y|x) = exp(score(x, y)) / Σ_{y'} exp(score(x, y'))

    score(x, y) = Σ_t [transition(y_{t-1}, y_t) + emission(x_t, y_t)]

    Viterbi decoding finds:
    y* = argmax_y score(x, y)
    """

    def __init__(self, num_labels: int, batch_first: bool = True):
        super().__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first

        # Transition scores: transitions[i, j] = score for tag i → tag j
        self.transitions = nn.Parameter(torch.empty(num_labels, num_labels))

        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.empty(num_labels))
        self.end_transitions = nn.Parameter(torch.empty(num_labels))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute negative log likelihood.

        Args:
            emissions: Emission scores [batch, seq_len, num_labels]
            tags: Ground truth tags [batch, seq_len]
            mask: Valid positions [batch, seq_len]

        Returns:
            Negative log likelihood
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        # Compute log partition function (normalization)
        log_partition = self._compute_log_partition(emissions, mask)

        # Compute score of gold sequence
        gold_score = self._compute_score(emissions, tags, mask)

        # Negative log likelihood
        return (log_partition - gold_score).mean()

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute score of a tag sequence.

        score = Σ_t [transition(y_{t-1}, y_t) + emission(x_t, y_t)]
        """
        batch_size, seq_length = tags.shape

        # Start transition
        score = self.start_transitions[tags[:, 0]]

        # Emission score for first token
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        # Transitions and emissions for remaining tokens
        for t in range(1, seq_length):
            # Only add if position is valid (mask)
            score_t = self.transitions[tags[:, t - 1], tags[:, t]]
            score_t += emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            score_t *= mask[:, t].float()
            score += score_t

        # End transition (use last valid tag)
        last_tag_indices = mask.sum(1) - 1
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def _compute_log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log partition function using forward algorithm.

        Z(x) = Σ_{y} exp(score(x, y))
        log Z(x) computed via dynamic programming
        """
        batch_size, seq_length, num_labels = emissions.shape

        # Initialize forward variables
        # alpha[t, k] = log Σ_{y_1:t-1} exp(score(x_1:t, y_1:t-1, y_t=k))
        alpha = self.start_transitions + emissions[:, 0]

        for t in range(1, seq_length):
            # alpha[t] = log Σ_k' exp(alpha[t-1, k'] + trans[k', k] + emit[t, k])
            # Shape: [batch, num_labels, num_labels]
            broadcast_emissions = emissions[:, t].unsqueeze(1)
            broadcast_transitions = self.transitions.unsqueeze(0)
            broadcast_alpha = alpha.unsqueeze(2)

            # Sum in log space
            scores = broadcast_alpha + broadcast_transitions + broadcast_emissions
            alpha_t = torch.logsumexp(scores, dim=1)

            # Mask out invalid positions
            alpha = torch.where(
                mask[:, t].unsqueeze(1),
                alpha_t,
                alpha,
            )

        # Add end transitions
        log_partition = torch.logsumexp(alpha + self.end_transitions, dim=1)

        return log_partition

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """
        Viterbi decoding to find most likely tag sequence.

        y* = argmax_y score(x, y)

        Complexity: O(T × K²) where T=seq_len, K=num_labels

        Args:
            emissions: Emission scores [batch, seq_len, num_labels]
            mask: Valid positions [batch, seq_len]

        Returns:
            Most likely tag sequences
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)

        batch_size, seq_length, num_labels = emissions.shape

        # Viterbi variables
        # viterbi[t, k] = max score for sequences ending in tag k at time t
        viterbi = self.start_transitions + emissions[:, 0]

        # Backpointers
        backpointers = []

        for t in range(1, seq_length):
            # viterbi[t, k] = max_k' (viterbi[t-1, k'] + trans[k', k]) + emit[t, k]
            broadcast_viterbi = viterbi.unsqueeze(2)
            broadcast_transitions = self.transitions.unsqueeze(0)

            scores = broadcast_viterbi + broadcast_transitions
            max_scores, max_indices = scores.max(dim=1)

            viterbi_t = max_scores + emissions[:, t]

            # Mask invalid positions
            viterbi = torch.where(
                mask[:, t].unsqueeze(1),
                viterbi_t,
                viterbi,
            )

            backpointers.append(max_indices)

        # Add end transitions and find best final tag
        viterbi += self.end_transitions
        best_tags_list = []

        for b in range(batch_size):
            # Find sequence length
            seq_len = mask[b].sum().item()

            # Backtrack
            best_last_tag = viterbi[b].argmax().item()
            best_tags = [best_last_tag]

            for t in range(len(backpointers) - 1, 0, -1):
                if t < seq_len - 1:
                    best_last_tag = backpointers[t][b, best_last_tag].item()
                    best_tags.insert(0, best_last_tag)

            best_tags_list.append(best_tags[:seq_len])

        return best_tags_list


class TokenClassificationNER(nn.Module):
    """
    Token classification NER with optional CRF.

    Reference: "Neural Architectures for NER" (Lample et al., NAACL 2016)

    Architecture:
    - BERT/Encoder → dropout → linear → CRF (optional)

    With CRF:
    - Enforces label consistency (e.g., I-PER must follow B-PER or I-PER)
    - Jointly models label sequence

    Without CRF:
    - Independent classification per token
    - Faster but may produce invalid sequences
    """

    def __init__(self, config: NERConfig):
        super().__init__()
        self.config = config

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        if config.use_crf:
            self.crf = CRF(config.num_labels, batch_first=True)
        else:
            self.crf = None

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
            labels: Ground truth labels [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Dictionary with loss (if labels provided) and predictions
        """
        # Classifier
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        outputs = {"logits": logits}

        if labels is not None:
            if self.crf is not None:
                # CRF loss
                loss = self.crf(logits, labels, attention_mask)
                outputs["loss"] = loss
            else:
                # Cross-entropy loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

                # Only compute loss on valid positions
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

                outputs["loss"] = loss

        # Predictions
        if self.crf is not None:
            predictions = self.crf.decode(logits, attention_mask)
        else:
            predictions = logits.argmax(dim=-1)

        outputs["predictions"] = predictions

        return outputs


class SpanBasedNER(nn.Module):
    """
    Span-based NER.

    Reference: "A Unified MRC Framework for NER" (Li et al., ACL 2020)
    Reference: "SpanNER" (Zhong et al., EMNLP 2021)

    Approach:
    1. Enumerate all possible spans up to max_span_length
    2. For each span, compute representation: [h_start; h_end; h_max; width_emb]
    3. Classify span into entity types

    Advantages:
    - Can handle nested entities
    - No tagging scheme constraints
    - Explicitly models span boundaries

    Complexity: O(T² × K) to enumerate and classify all spans
    """

    def __init__(self, config: NERConfig):
        super().__init__()
        self.config = config

        # Span width embeddings
        self.width_embeddings = nn.Embedding(
            config.max_span_length + 1,
            config.d_model // 4,
        )

        # Span representation: [start; end; max_pool; width]
        span_repr_size = config.d_model * 3 + config.d_model // 4

        self.span_classifier = nn.Sequential(
            nn.Linear(span_repr_size, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_labels + 1),  # +1 for "not an entity"
        )

    def _enumerate_spans(
        self,
        seq_length: int,
    ) -> List[Tuple[int, int]]:
        """
        Enumerate all spans up to max_span_length.

        Returns:
            List of (start, end) indices
        """
        spans = []
        for start in range(seq_length):
            for length in range(1, min(self.config.max_span_length + 1, seq_length - start + 1)):
                end = start + length - 1
                spans.append((start, end))
        return spans

    def _get_span_representation(
        self,
        hidden_states: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """
        Compute span representation.

        Args:
            hidden_states: [batch, seq_len, d_model]
            start: Start index
            end: End index (inclusive)

        Returns:
            Span representation [batch, span_repr_size]
        """
        batch_size = hidden_states.shape[0]

        # Start and end representations
        h_start = hidden_states[:, start]
        h_end = hidden_states[:, end]

        # Max pooling over span
        if start == end:
            h_max = h_start
        else:
            h_max = hidden_states[:, start:end+1].max(dim=1)[0]

        # Width embedding
        width = end - start + 1
        width_emb = self.width_embeddings(
            torch.tensor([width], device=hidden_states.device)
        ).expand(batch_size, -1)

        # Concatenate
        span_repr = torch.cat([h_start, h_end, h_max, width_emb], dim=-1)

        return span_repr

    def forward(
        self,
        hidden_states: torch.Tensor,
        span_labels: Optional[Dict[Tuple[int, int], int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            hidden_states: Encoder outputs [batch, seq_len, d_model]
            span_labels: Dictionary mapping (start, end) to label
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Dictionary with loss and predictions
        """
        batch_size, seq_length, _ = hidden_states.shape

        # Enumerate spans
        spans = self._enumerate_spans(seq_length)

        # Classify each span
        all_span_logits = []

        for start, end in spans:
            span_repr = self._get_span_representation(hidden_states, start, end)
            span_logits = self.span_classifier(span_repr)
            all_span_logits.append(span_logits)

        # Stack: [batch, num_spans, num_labels + 1]
        span_logits = torch.stack(all_span_logits, dim=1)

        outputs = {"span_logits": span_logits, "spans": spans}

        if span_labels is not None:
            # Create labels tensor
            labels = torch.zeros(
                batch_size,
                len(spans),
                dtype=torch.long,
                device=hidden_states.device,
            )

            for i, (start, end) in enumerate(spans):
                if (start, end) in span_labels:
                    labels[:, i] = span_labels[(start, end)]

            # Cross-entropy loss
            loss = F.cross_entropy(
                span_logits.view(-1, self.config.num_labels + 1),
                labels.view(-1),
            )
            outputs["loss"] = loss

        # Predictions
        predictions = span_logits.argmax(dim=-1)

        # Extract predicted entities (filter out "not an entity" class)
        predicted_entities = []
        for b in range(batch_size):
            entities_b = []
            for i, (start, end) in enumerate(spans):
                label = predictions[b, i].item()
                if label != self.config.num_labels:  # Not "O" class
                    entities_b.append({
                        "start": start,
                        "end": end,
                        "label": label,
                    })
            predicted_entities.append(entities_b)

        outputs["predictions"] = predicted_entities

        return outputs


class NERPostProcessor:
    """
    Post-processing for NER predictions.

    Converts BIO/BIOES tags to entity spans.
    """

    @staticmethod
    def bio_to_entities(
        tokens: List[str],
        tags: List[str],
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Convert BIO tags to entity spans.

        Args:
            tokens: Input tokens
            tags: BIO tags

        Returns:
            List of entities with start, end, type, text
        """
        entities = []
        current_entity = None

        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                # Start new entity
                if current_entity is not None:
                    entities.append(current_entity)

                entity_type = tag[2:]
                current_entity = {
                    "start": i,
                    "end": i,
                    "type": entity_type,
                    "text": tokens[i],
                }

            elif tag.startswith("I-"):
                # Continue entity
                if current_entity is not None:
                    entity_type = tag[2:]
                    if current_entity["type"] == entity_type:
                        current_entity["end"] = i
                        current_entity["text"] += " " + tokens[i]
                    else:
                        # Invalid sequence, start new entity
                        entities.append(current_entity)
                        current_entity = {
                            "start": i,
                            "end": i,
                            "type": entity_type,
                            "text": tokens[i],
                        }

            else:  # "O"
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None

        # Add last entity
        if current_entity is not None:
            entities.append(current_entity)

        return entities
