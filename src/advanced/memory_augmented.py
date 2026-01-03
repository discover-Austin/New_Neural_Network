"""
Memory-Augmented Transformer
=============================

Research Foundation:
--------------------
1. "Generalization through Memorization: Nearest Neighbor Language Models"
   Khandelwal et al., ICLR 2020
   - kNN-LM improves perplexity by 0.2-1.0 on WikiText
   - Retrieves from datastore of (context, next_token) pairs
   - No retraining required

2. "Compressive Transformers for Long-Range Sequence Modelling"
   Rae et al., ICLR 2020
   - Extends context to 100K+ tokens via compression
   - Old memories compressed, recent memories kept
   - Empirical: Improved perplexity on long documents

3. "Memorizing Transformers"
   Wu et al., ICLR 2022
   - Integrates kNN into transformer architecture
   - Learnable interpolation between LM and kNN

Mathematical Foundation:
-----------------------
kNN-LM Formulation:
  P_kNN(w|context) = Σ 1[w_i = w] · k(context, context_i)
  where k is similarity kernel (cosine or learned)

Interpolation:
  P_final(w) = λ·P_LM(w) + (1-λ)·P_kNN(w)
  where λ is learnable or fixed

Complexity:
  - Memory storage: O(N_datastore × d)
  - Retrieval: O(k·log N) with FAISS index
  - Inference overhead: ~5-15% depending on k

Verification:
  - Proven to improve perplexity (Khandelwal et al., 2020)
  - Works across domains without retraining
  - Scales to billions of tokens in datastore
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import numpy as np
from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Configuration for memory-augmented transformer."""
    # Memory parameters
    memory_size: int = 65536  # Size of memory buffer
    k_neighbors: int = 32  # Number of neighbors to retrieve
    memory_dim: int = 768  # Dimension of memory keys

    # kNN parameters
    temperature: float = 10.0  # Temperature for kNN softmax
    lambda_interpolation: float = 0.25  # Fixed interpolation weight

    # Compression parameters (for compressive memory)
    compression_rate: int = 4  # Compress by 4x
    num_compressed_memories: int = 16384


class kNNMemory(nn.Module):
    """
    k-Nearest Neighbor Memory for language modeling.

    Implementation of kNN-LM (Khandelwal et al., 2020).

    Mathematical Framework:
    -----------------------
    1. Build datastore:
       For each (context, target) pair in training:
       Store (f(context), target) where f is encoder

    2. Retrieval:
       Given query context q:
       - Encode: k_q = f(q)
       - Find k-nearest: {(k_i, v_i)} where sim(k_q, k_i) highest
       - Compute distribution: p_i ∝ exp(-d(k_q, k_i)/T)

    3. Prediction:
       P_kNN(w) = Σ p_i · 1[v_i = w]

    Verified Properties:
    - Improves perplexity: -0.2 to -1.0 (empirical)
    - No retraining needed
    - Complements neural LM
    """

    def __init__(self, config: MemoryConfig):
        super().__init__()

        self.config = config
        self.memory_size = config.memory_size
        self.k_neighbors = config.k_neighbors
        self.temperature = config.temperature

        # Memory buffers (registered as buffers, not parameters)
        self.register_buffer(
            "memory_keys",
            torch.zeros(config.memory_size, config.memory_dim),
        )
        self.register_buffer(
            "memory_values",
            torch.zeros(config.memory_size, dtype=torch.long),
        )
        self.register_buffer(
            "memory_ptr",
            torch.zeros(1, dtype=torch.long),
        )
        self.register_buffer(
            "memory_filled",
            torch.zeros(1, dtype=torch.long),
        )

        # Learnable interpolation weight
        self.lambda_param = nn.Parameter(torch.tensor(config.lambda_interpolation))

    def add_to_memory(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Add (key, value) pairs to memory.

        Args:
            keys: Encoded context [batch, seq_len, d_model]
            values: Target tokens [batch, seq_len]
        """
        batch_size, seq_len, d_model = keys.shape

        # Flatten
        keys = keys.reshape(-1, d_model)
        values = values.reshape(-1)

        # Add to circular buffer
        for i in range(len(keys)):
            ptr = int(self.memory_ptr.item())
            self.memory_keys[ptr] = keys[i]
            self.memory_values[ptr] = values[i]

            # Update pointer (circular)
            self.memory_ptr[0] = (ptr + 1) % self.memory_size

            # Track if memory is full
            if ptr + 1 >= self.memory_size:
                self.memory_filled[0] = 1

    @torch.no_grad()
    def retrieve(
        self,
        query_keys: torch.Tensor,
        k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve k-nearest neighbors from memory.

        Args:
            query_keys: Query vectors [batch, seq_len, d_model]
            k: Number of neighbors (default: self.k_neighbors)

        Returns:
            neighbor_values: Values of neighbors [batch, seq_len, k]
            neighbor_distances: Distances to neighbors [batch, seq_len, k]
            neighbor_probs: Probabilities based on distance [batch, seq_len, k]
        """
        if k is None:
            k = self.k_neighbors

        batch_size, seq_len, d_model = query_keys.shape

        # Determine how many memories are available
        if self.memory_filled.item() == 1:
            available_memories = self.memory_size
        else:
            available_memories = max(int(self.memory_ptr.item()), k)

        if available_memories < k:
            # Not enough memories yet
            return None, None, None

        # Flatten queries
        queries_flat = query_keys.reshape(-1, d_model)  # [batch*seq_len, d_model]

        # Compute distances (using negative cosine similarity)
        # Normalize for cosine similarity
        queries_norm = F.normalize(queries_flat, p=2, dim=-1)
        keys_norm = F.normalize(self.memory_keys[:available_memories], p=2, dim=-1)

        # Cosine similarity: [batch*seq_len, available_memories]
        similarities = torch.matmul(queries_norm, keys_norm.T)

        # Convert to distances (higher similarity = lower distance)
        distances = 1.0 - similarities

        # Get k-nearest neighbors
        top_k_distances, top_k_indices = torch.topk(
            distances,
            k=k,
            dim=-1,
            largest=False,  # Smallest distances
        )

        # Get corresponding values
        top_k_values = self.memory_values[:available_memories][top_k_indices]

        # Compute probabilities using softmax over distances
        # Lower distance = higher probability
        neighbor_probs = F.softmax(-top_k_distances / self.temperature, dim=-1)

        # Reshape back
        neighbor_values = top_k_values.view(batch_size, seq_len, k)
        neighbor_distances = top_k_distances.view(batch_size, seq_len, k)
        neighbor_probs = neighbor_probs.view(batch_size, seq_len, k)

        return neighbor_values, neighbor_distances, neighbor_probs

    def compute_knn_distribution(
        self,
        neighbor_values: torch.Tensor,
        neighbor_probs: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        """
        Compute kNN distribution over vocabulary.

        Args:
            neighbor_values: Neighbor tokens [batch, seq_len, k]
            neighbor_probs: Neighbor probabilities [batch, seq_len, k]
            vocab_size: Size of vocabulary

        Returns:
            knn_probs: Distribution over vocabulary [batch, seq_len, vocab_size]
        """
        batch_size, seq_len, k = neighbor_values.shape

        # Initialize distribution
        knn_probs = torch.zeros(
            batch_size, seq_len, vocab_size,
            device=neighbor_values.device,
        )

        # Scatter probabilities to vocabulary positions
        knn_probs.scatter_add_(
            dim=2,
            index=neighbor_values,
            src=neighbor_probs,
        )

        return knn_probs

    def forward(
        self,
        query_keys: torch.Tensor,
        lm_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute interpolated kNN-LM distribution.

        Args:
            query_keys: Encoded contexts [batch, seq_len, d_model]
            lm_logits: Language model logits [batch, seq_len, vocab_size]

        Returns:
            Dictionary with final distribution and components
        """
        vocab_size = lm_logits.size(-1)

        # Retrieve neighbors
        neighbor_values, neighbor_distances, neighbor_probs = self.retrieve(query_keys)

        if neighbor_values is None:
            # Not enough memories, return LM only
            return {
                "logits": lm_logits,
                "knn_probs": None,
                "lm_probs": F.softmax(lm_logits, dim=-1),
                "lambda": 0.0,
            }

        # Compute kNN distribution
        knn_probs = self.compute_knn_distribution(
            neighbor_values,
            neighbor_probs,
            vocab_size,
        )

        # LM distribution
        lm_probs = F.softmax(lm_logits, dim=-1)

        # Interpolate (constrain lambda to [0, 1])
        lambda_weight = torch.sigmoid(self.lambda_param)
        final_probs = lambda_weight * lm_probs + (1 - lambda_weight) * knn_probs

        # Convert back to logits
        final_logits = torch.log(final_probs + 1e-10)

        return {
            "logits": final_logits,
            "knn_probs": knn_probs,
            "lm_probs": lm_probs,
            "lambda": lambda_weight.item(),
        }


class CompressiveMemory(nn.Module):
    """
    Compressive Memory for long-range context.

    Implementation based on "Compressive Transformers" (Rae et al., 2020).

    Key Ideas:
    - Recent memories: Stored in full fidelity
    - Old memories: Compressed to save space
    - Compression: Learned function or simple pooling

    Mathematical Framework:
    -----------------------
    Memory structure:
      [Compressed old memories | Recent memories | Current context]
      [    M_compressed        |    M_recent     |       x        ]

    Compression function:
      c = f_compress(M_old)
      where f can be:
      - Average pooling
      - Max pooling
      - Learned compression (small transformer)

    Attention:
      Attend over: concat(M_compressed, M_recent, x)
    """

    def __init__(
        self,
        d_model: int,
        memory_size: int = 1024,
        compression_rate: int = 4,
        compression_method: str = "learned",
    ):
        super().__init__()

        self.d_model = d_model
        self.memory_size = memory_size
        self.compression_rate = compression_rate
        self.compression_method = compression_method

        # Recent memory buffer
        self.register_buffer(
            "recent_memory",
            torch.zeros(memory_size, d_model),
        )
        self.register_buffer(
            "recent_memory_ptr",
            torch.zeros(1, dtype=torch.long),
        )

        # Compressed memory buffer
        compressed_size = memory_size // compression_rate
        self.register_buffer(
            "compressed_memory",
            torch.zeros(compressed_size, d_model),
        )

        # Learned compression
        if compression_method == "learned":
            self.compressor = nn.Sequential(
                nn.Linear(d_model * compression_rate, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )

    def compress_memories(self, memories: torch.Tensor) -> torch.Tensor:
        """
        Compress memories.

        Args:
            memories: Memories to compress [num_memories, d_model]

        Returns:
            Compressed memories [num_memories // compression_rate, d_model]
        """
        num_memories = memories.size(0)

        # Ensure divisible by compression rate
        if num_memories % self.compression_rate != 0:
            padding = self.compression_rate - (num_memories % self.compression_rate)
            memories = F.pad(memories, (0, 0, 0, padding))
            num_memories = memories.size(0)

        # Reshape for compression
        memories = memories.view(
            num_memories // self.compression_rate,
            self.compression_rate,
            self.d_model,
        )

        if self.compression_method == "average":
            compressed = memories.mean(dim=1)

        elif self.compression_method == "max":
            compressed = memories.max(dim=1)[0]

        elif self.compression_method == "learned":
            # Flatten compression group
            memories_flat = memories.view(
                num_memories // self.compression_rate,
                self.compression_rate * self.d_model,
            )
            compressed = self.compressor(memories_flat)

        return compressed

    def add_memories(self, new_memories: torch.Tensor) -> None:
        """
        Add new memories, compressing old ones if buffer full.

        Args:
            new_memories: New memory vectors [num_new, d_model]
        """
        num_new = new_memories.size(0)

        for i in range(num_new):
            ptr = int(self.recent_memory_ptr.item())

            if ptr >= self.memory_size:
                # Buffer full, compress oldest memories
                to_compress = self.recent_memory[:self.compression_rate]
                compressed = self.compress_memories(to_compress)

                # Shift compressed memory
                self.compressed_memory = torch.cat([
                    self.compressed_memory[1:],
                    compressed,
                ], dim=0)

                # Shift recent memory
                self.recent_memory = torch.cat([
                    self.recent_memory[self.compression_rate:],
                    torch.zeros(self.compression_rate, self.d_model, device=self.recent_memory.device),
                ], dim=0)

                # Reset pointer
                ptr = self.memory_size - self.compression_rate

            # Add new memory
            self.recent_memory[ptr] = new_memories[i]
            self.recent_memory_ptr[0] = ptr + 1

    def get_all_memories(self) -> torch.Tensor:
        """
        Get all memories (compressed + recent).

        Returns:
            All memories concatenated
        """
        ptr = int(self.recent_memory_ptr.item())
        recent = self.recent_memory[:ptr]

        return torch.cat([self.compressed_memory, recent], dim=0)


class MemoryAugmentedTransformer(nn.Module):
    """
    Transformer with kNN memory augmentation.

    Combines neural language model with non-parametric kNN memory.

    Verified Benefits (Khandelwal et al., 2020):
    - Perplexity improvement: 0.2-1.0
    - Works across domains
    - No retraining required
    """

    def __init__(
        self,
        base_model: nn.Module,
        memory_config: MemoryConfig,
        use_compression: bool = False,
    ):
        super().__init__()

        self.base_model = base_model
        self.use_compression = use_compression

        # kNN memory
        self.knn_memory = kNNMemory(memory_config)

        # Optionally add compressive memory
        if use_compression:
            self.compressive_memory = CompressiveMemory(
                d_model=memory_config.memory_dim,
                memory_size=1024,
                compression_rate=memory_config.compression_rate,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        add_to_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with memory augmentation.

        Args:
            input_ids: Input tokens
            labels: Target labels
            add_to_memory: Whether to add to memory during this pass

        Returns:
            Dictionary with outputs
        """
        # Base model forward pass
        base_outputs = self.base_model(input_ids, labels=labels, return_dict=True)

        if isinstance(base_outputs, dict):
            lm_logits = base_outputs["logits"]
            hidden_states = base_outputs.get("last_hidden_state", None)
        else:
            lm_logits = base_outputs[0]
            hidden_states = base_outputs[1] if len(base_outputs) > 1 else None

        # Add to memory if requested
        if add_to_memory and labels is not None and hidden_states is not None:
            # Use hidden states as keys, labels as values
            self.knn_memory.add_to_memory(hidden_states[:, :-1], labels[:, 1:])

        # Augment with kNN memory
        if hidden_states is not None:
            memory_outputs = self.knn_memory(hidden_states, lm_logits)
            final_logits = memory_outputs["logits"]
        else:
            final_logits = lm_logits
            memory_outputs = {"lambda": 0.0}

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {
            "logits": final_logits,
            "loss": loss,
            "lm_logits": lm_logits,
            "knn_lambda": memory_outputs.get("lambda", 0.0),
        }
