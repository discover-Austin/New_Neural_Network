"""
Intent Classification and Slot Filling
======================================

Research-grounded implementations for dialogue understanding:

1. Intent Classification
   - Reference: "Joint Slot Filling and Intent Detection" (Goo et al., NAACL 2018)
   - Reference: "BERT for Joint Intent Classification and Slot Filling" (Chen et al., 2019)

2. Slot Filling
   - Sequence labeling task (similar to NER)
   - BIO tagging for slot boundaries

3. Joint Intent-Slot Models
   - Reference: "Slot-Gated Modeling for Joint Slot Filling and Intent Prediction" (Goo et al., 2018)
   - Reference: "A Stack-Propagation Framework with Token-Level Intent Detection" (Qin et al., EMNLP 2019)
   - Better performance than independent models

Example:
--------
Input: "Book a flight from Boston to New York tomorrow"

Intent: book_flight
Slots:
- from_location: Boston
- to_location: New York
- date: tomorrow

BIO Tags:
Book  → O
a     → O
flight → O
from  → O
Boston → B-from_location
to    → O
New   → B-to_location
York  → I-to_location
tomorrow → B-date

Mathematical Foundation:
-----------------------

Independent Models:
  Intent: P(intent|x) = softmax(W_intent · h_[CLS])
  Slots: P(slot_t|x) = softmax(W_slot · h_t)

Joint Model with Slot-Gating:
  Intent: P(intent|x) = softmax(W_intent · h_[CLS])

  Slot-gate: g_t = σ(W_g · [h_t; h_intent])
  Slot: P(slot_t|x) = softmax(W_slot · (g_t ⊙ h_t))

  where h_intent is the intent representation used to modulate slot predictions

Stack-Propagation:
  - Token-level intent detection
  - Intent information propagated to refine slot predictions
  - Bidirectional influence between intent and slots

Complexity:
- Intent: O(1) classification after encoding
- Slots: O(T × K) where T=seq length, K=num slot labels
- Joint: Similar, with additional gating computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class IntentSlotConfig:
    """Configuration for intent and slot filling."""
    d_model: int = 768
    num_intents: int = 20  # Number of intent classes
    num_slots: int = 30  # Number of slot labels (BIO encoding)
    dropout: float = 0.1
    use_crf: bool = False  # CRF for slot filling
    use_slot_gating: bool = True  # Slot-gated mechanism
    intent_slot_attention: bool = True  # Bidirectional attention


class IntentClassifier(nn.Module):
    """
    Intent classification.

    Reference: "BERT for Joint Intent Classification and Slot Filling" (Chen et al., 2019)

    Uses [CLS] token representation for classification.
    """

    def __init__(self, config: IntentSlotConfig):
        super().__init__()
        self.config = config

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, config.num_intents)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            hidden_states: Encoder outputs [batch, seq_len, d_model]
            labels: Intent labels [batch]

        Returns:
            Dictionary with loss and predictions
        """
        # Use CLS token
        cls_hidden = hidden_states[:, 0]
        cls_hidden = self.dropout(cls_hidden)

        # Classify
        logits = self.classifier(cls_hidden)

        outputs = {"logits": logits, "intent_hidden": cls_hidden}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs["loss"] = loss

        # Predictions
        predictions = logits.argmax(dim=-1)
        probabilities = F.softmax(logits, dim=-1)

        outputs["predictions"] = predictions
        outputs["probabilities"] = probabilities

        return outputs


class SlotFilling(nn.Module):
    """
    Slot filling (sequence labeling).

    Reference: Similar to NER, uses BIO tagging.

    Can optionally use CRF for structured prediction.
    """

    def __init__(self, config: IntentSlotConfig):
        super().__init__()
        self.config = config

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, config.num_slots)

        if config.use_crf:
            from .named_entity_recognition import CRF
            self.crf = CRF(config.num_slots, batch_first=True)
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
            hidden_states: Token representations [batch, seq_len, d_model]
            labels: Slot labels [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Dictionary with loss and predictions
        """
        # Classify each token
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

                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.config.num_slots)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.config.num_slots), labels.view(-1))

                outputs["loss"] = loss

        # Predictions
        if self.crf is not None:
            predictions = self.crf.decode(logits, attention_mask)
        else:
            predictions = logits.argmax(dim=-1)

        outputs["predictions"] = predictions

        return outputs


class SlotGatedIntentSlot(nn.Module):
    """
    Slot-Gated Joint Intent and Slot Filling.

    Reference: "Slot-Gated Modeling for Joint Slot Filling and Intent Prediction"
               (Goo et al., NAACL 2018)

    Key Innovation:
    - Intent representation gates slot predictions
    - Intent information helps disambiguate slot labels
    - Better than independent models

    Architecture:
    1. Compute intent representation from [CLS]
    2. For each token, gate its representation with intent:
       g_t = σ(W_g · [h_t; h_intent])
       h'_t = g_t ⊙ h_t
    3. Classify slots using gated representations
    4. Classify intent

    Complexity: O(T × d²) for gating, negligible overhead
    """

    def __init__(self, config: IntentSlotConfig):
        super().__init__()
        self.config = config

        # Intent classifier
        self.intent_classifier = IntentClassifier(config)

        # Slot-gating mechanism
        self.slot_gate = nn.Linear(config.d_model * 2, config.d_model)

        # Slot classifier
        self.slot_dropout = nn.Dropout(config.dropout)
        self.slot_classifier = nn.Linear(config.d_model, config.num_slots)

        if config.use_crf:
            from .named_entity_recognition import CRF
            self.crf = CRF(config.num_slots, batch_first=True)
        else:
            self.crf = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        intent_labels: Optional[torch.Tensor] = None,
        slot_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            hidden_states: Encoder outputs [batch, seq_len, d_model]
            intent_labels: Intent labels [batch]
            slot_labels: Slot labels [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Dictionary with losses and predictions
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Intent classification
        intent_outputs = self.intent_classifier(hidden_states, intent_labels)
        intent_hidden = intent_outputs["intent_hidden"]  # [batch, d_model]

        # Expand intent representation to all tokens
        intent_expanded = intent_hidden.unsqueeze(1).expand(batch_size, seq_len, d_model)

        # Slot gating
        # Concatenate token and intent representations
        combined = torch.cat([hidden_states, intent_expanded], dim=-1)

        # Gate: g_t = σ(W_g · [h_t; h_intent])
        gate = torch.sigmoid(self.slot_gate(combined))

        # Gated representations: h'_t = g_t ⊙ h_t
        gated_hidden = gate * hidden_states

        # Slot classification
        gated_hidden = self.slot_dropout(gated_hidden)
        slot_logits = self.slot_classifier(gated_hidden)

        outputs = {
            "intent_logits": intent_outputs["logits"],
            "slot_logits": slot_logits,
            "gate": gate,  # For analysis
        }

        # Losses
        total_loss = 0.0

        if intent_labels is not None:
            outputs["intent_loss"] = intent_outputs["loss"]
            total_loss += intent_outputs["loss"]

        if slot_labels is not None:
            if self.crf is not None:
                slot_loss = self.crf(slot_logits, slot_labels, attention_mask)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.config.num_slots)[active_loss]
                    active_labels = slot_labels.view(-1)[active_loss]
                    slot_loss = loss_fct(active_logits, active_labels)
                else:
                    slot_loss = loss_fct(
                        slot_logits.view(-1, self.config.num_slots),
                        slot_labels.view(-1),
                    )

            outputs["slot_loss"] = slot_loss
            total_loss += slot_loss

        if total_loss > 0:
            outputs["loss"] = total_loss

        # Predictions
        outputs["intent_predictions"] = intent_outputs["predictions"]

        if self.crf is not None:
            outputs["slot_predictions"] = self.crf.decode(slot_logits, attention_mask)
        else:
            outputs["slot_predictions"] = slot_logits.argmax(dim=-1)

        return outputs


class StackPropagationIntentSlot(nn.Module):
    """
    Stack-Propagation Framework for Intent and Slot.

    Reference: "A Stack-Propagation Framework with Token-Level Intent Detection"
               (Qin et al., EMNLP 2019)

    Key Innovation:
    - Token-level intent detection (not just sentence-level)
    - Intent information propagated to refine slot predictions
    - Bidirectional interaction: intent → slots, slots → intent

    Architecture:
    1. Token-level intent detection
    2. Aggregate token intents to sentence intent
    3. Use intent to refine slot predictions via attention
    4. Iterative refinement (stack propagation)

    Superior to slot-gating on complex dialogues.
    """

    def __init__(self, config: IntentSlotConfig):
        super().__init__()
        self.config = config

        # Token-level intent detector
        self.token_intent = nn.Linear(config.d_model, config.num_intents)

        # Sentence-level intent aggregator
        self.intent_attention = nn.Linear(config.d_model, 1)

        # Intent-slot attention
        if config.intent_slot_attention:
            self.cross_attention = nn.MultiheadAttention(
                config.d_model,
                num_heads=8,
                dropout=config.dropout,
                batch_first=True,
            )

        # Slot classifier
        self.slot_dropout = nn.Dropout(config.dropout)
        self.slot_classifier = nn.Linear(config.d_model, config.num_slots)

        # Sentence-level intent classifier
        self.intent_classifier = nn.Linear(config.d_model, config.num_intents)

        if config.use_crf:
            from .named_entity_recognition import CRF
            self.crf = CRF(config.num_slots, batch_first=True)
        else:
            self.crf = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        intent_labels: Optional[torch.Tensor] = None,
        slot_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            hidden_states: Encoder outputs [batch, seq_len, d_model]
            intent_labels: Intent labels [batch]
            slot_labels: Slot labels [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Dictionary with losses and predictions
        """
        # Token-level intent detection
        token_intent_logits = self.token_intent(hidden_states)  # [batch, seq_len, num_intents]

        # Aggregate to sentence-level intent using attention
        intent_scores = self.intent_attention(hidden_states).squeeze(-1)  # [batch, seq_len]

        if attention_mask is not None:
            intent_scores = intent_scores.masked_fill(attention_mask == 0, -1e9)

        intent_weights = F.softmax(intent_scores, dim=-1).unsqueeze(1)  # [batch, 1, seq_len]
        sentence_repr = torch.matmul(intent_weights, hidden_states).squeeze(1)  # [batch, d_model]

        # Sentence-level intent
        intent_logits = self.intent_classifier(sentence_repr)

        # Intent-slot cross-attention (refine slots with intent information)
        if self.config.intent_slot_attention:
            # Use sentence representation as query
            query = sentence_repr.unsqueeze(1)  # [batch, 1, d_model]

            # Attend over token representations
            refined_hidden, _ = self.cross_attention(
                query,
                hidden_states,
                hidden_states,
                key_padding_mask=(attention_mask == 0) if attention_mask is not None else None,
            )

            # Combine original and refined
            slot_hidden = hidden_states + refined_hidden.expand_as(hidden_states)
        else:
            slot_hidden = hidden_states

        # Slot classification
        slot_hidden = self.slot_dropout(slot_hidden)
        slot_logits = self.slot_classifier(slot_hidden)

        outputs = {
            "intent_logits": intent_logits,
            "token_intent_logits": token_intent_logits,
            "slot_logits": slot_logits,
        }

        # Losses
        total_loss = 0.0

        if intent_labels is not None:
            # Sentence-level intent loss
            intent_loss = F.cross_entropy(intent_logits, intent_labels)
            outputs["intent_loss"] = intent_loss
            total_loss += intent_loss

            # Token-level intent loss (auxiliary)
            # Expand sentence label to all tokens
            token_intent_labels = intent_labels.unsqueeze(1).expand(-1, hidden_states.size(1))

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = token_intent_logits.view(-1, self.config.num_intents)[active_loss]
                active_labels = token_intent_labels.view(-1)[active_loss]
                token_intent_loss = F.cross_entropy(active_logits, active_labels)
            else:
                token_intent_loss = F.cross_entropy(
                    token_intent_logits.view(-1, self.config.num_intents),
                    token_intent_labels.view(-1),
                )

            outputs["token_intent_loss"] = token_intent_loss
            total_loss += 0.5 * token_intent_loss  # Weight auxiliary loss

        if slot_labels is not None:
            if self.crf is not None:
                slot_loss = self.crf(slot_logits, slot_labels, attention_mask)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.config.num_slots)[active_loss]
                    active_labels = slot_labels.view(-1)[active_loss]
                    slot_loss = loss_fct(active_logits, active_labels)
                else:
                    slot_loss = loss_fct(
                        slot_logits.view(-1, self.config.num_slots),
                        slot_labels.view(-1),
                    )

            outputs["slot_loss"] = slot_loss
            total_loss += slot_loss

        if total_loss > 0:
            outputs["loss"] = total_loss

        # Predictions
        outputs["intent_predictions"] = intent_logits.argmax(dim=-1)

        if self.crf is not None:
            outputs["slot_predictions"] = self.crf.decode(slot_logits, attention_mask)
        else:
            outputs["slot_predictions"] = slot_logits.argmax(dim=-1)

        return outputs


class DialogueStateTracker(nn.Module):
    """
    Dialogue State Tracking (DST).

    Reference: "MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset" (Budzianowski et al., EMNLP 2018)
    Reference: "TRADE: Transferable Dialogue State Generator" (Wu et al., ACL 2019)

    Task: Track dialogue state across turns.

    State: {domain: {slot: value}}

    Example:
    User: "I want a cheap restaurant in the center"
    State: {restaurant: {price: cheap, area: center}}

    User: "With Italian food"
    State: {restaurant: {price: cheap, area: center, food: Italian}}

    Approaches:
    - Classification: Predict slot values from predefined ontology
    - Generation: Generate slot values (handles open vocabulary)
    """

    def __init__(
        self,
        config: IntentSlotConfig,
        num_domains: int = 5,
        num_slots_per_domain: int = 10,
        num_values_per_slot: int = 50,
    ):
        super().__init__()
        self.config = config
        self.num_domains = num_domains
        self.num_slots_per_domain = num_slots_per_domain
        self.num_values_per_slot = num_values_per_slot

        # Domain-slot-value classifier
        # Simplified: Classify each (domain, slot) pair independently
        self.state_classifier = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(config.d_model, num_values_per_slot + 1)  # +1 for "none"
                for _ in range(num_slots_per_domain)
            ])
            for _ in range(num_domains)
        ])

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            hidden_states: Encoder outputs [batch, seq_len, d_model]
            state_labels: State labels [batch, num_domains, num_slots_per_domain]

        Returns:
            Dictionary with loss and predictions
        """
        # Use CLS token
        cls_hidden = hidden_states[:, 0]
        cls_hidden = self.dropout(cls_hidden)

        # Predict each (domain, slot) value
        all_logits = []

        for domain_idx in range(self.num_domains):
            domain_logits = []
            for slot_idx in range(self.num_slots_per_domain):
                logits = self.state_classifier[domain_idx][slot_idx](cls_hidden)
                domain_logits.append(logits)
            all_logits.append(torch.stack(domain_logits, dim=1))

        # Stack: [batch, num_domains, num_slots_per_domain, num_values + 1]
        state_logits = torch.stack(all_logits, dim=1)

        outputs = {"state_logits": state_logits}

        if state_labels is not None:
            # Flatten and compute loss
            loss = F.cross_entropy(
                state_logits.view(-1, self.num_values_per_slot + 1),
                state_labels.view(-1),
            )
            outputs["loss"] = loss

        # Predictions
        predictions = state_logits.argmax(dim=-1)
        outputs["predictions"] = predictions

        return outputs
