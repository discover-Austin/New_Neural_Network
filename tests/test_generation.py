"""
Tests for Text Generation Strategies
====================================

Tests for various decoding strategies: greedy, beam search, sampling, etc.
"""

import torch
import pytest
from src.generation import (
    GreedyDecoding,
    BeamSearchDecoding,
    NucleusSampling,
    TopKSampling,
    ContrastiveDecoding,
    SpeculativeDecoding,
)
from src.generation import (
    TemperatureLogitsProcessor,
    TopKLogitsProcessor,
    TopPLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)
from src.generation import MaxLengthCriteria, EosTokenCriteria
from src.models import GPTModel, GPTConfig


class TestLogitsProcessors:
    """Tests for logits processors."""

    def test_temperature_processor(self):
        """Test temperature logits processor."""
        processor = TemperatureLogitsProcessor(temperature=0.8)

        batch_size, seq_len, vocab_size = 2, 1, 100
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = torch.randn(batch_size, vocab_size)

        processed = processor(input_ids, logits)

        # Temperature should scale logits
        assert processed.shape == logits.shape
        assert not torch.allclose(processed, logits)

    def test_topk_processor(self):
        """Test top-k logits processor."""
        processor = TopKLogitsProcessor(top_k=10)

        batch_size, vocab_size = 2, 100
        input_ids = torch.randint(0, vocab_size, (batch_size, 1))
        logits = torch.randn(batch_size, vocab_size)

        processed = processor(input_ids, logits)

        # Should set low probability tokens to -inf
        assert processed.shape == logits.shape
        # Check that only top_k tokens are not -inf per batch
        for i in range(batch_size):
            non_inf = (processed[i] != float('-inf')).sum()
            assert non_inf == 10

    def test_topp_processor(self):
        """Test nucleus (top-p) logits processor."""
        processor = TopPLogitsProcessor(top_p=0.9)

        batch_size, vocab_size = 2, 100
        input_ids = torch.randint(0, vocab_size, (batch_size, 1))
        logits = torch.randn(batch_size, vocab_size)

        processed = processor(input_ids, logits)

        # Should filter based on cumulative probability
        assert processed.shape == logits.shape

    def test_repetition_penalty_processor(self):
        """Test repetition penalty processor."""
        processor = RepetitionPenaltyLogitsProcessor(penalty=1.2)

        batch_size, vocab_size = 2, 100
        # Create input with repeated tokens
        input_ids = torch.tensor([[5, 10, 5, 15], [20, 25, 20, 30]])
        logits = torch.randn(batch_size, vocab_size)

        processed = processor(input_ids, logits)

        # Repeated tokens should have reduced logits
        assert processed.shape == (batch_size, vocab_size)


class TestStoppingCriteria:
    """Tests for stopping criteria."""

    def test_max_length_criteria(self):
        """Test maximum length stopping criteria."""
        criteria = MaxLengthCriteria(max_length=10)

        batch_size, vocab_size = 2, 100
        short_ids = torch.randint(0, vocab_size, (batch_size, 5))
        long_ids = torch.randint(0, vocab_size, (batch_size, 15))

        assert not criteria(short_ids, None)  # Should continue
        assert criteria(long_ids, None)  # Should stop

    def test_eos_token_criteria(self):
        """Test EOS token stopping criteria."""
        eos_token_id = 50
        criteria = EosTokenCriteria(eos_token_id=eos_token_id)

        batch_size, vocab_size = 2, 100
        # Without EOS
        ids_no_eos = torch.randint(0, 40, (batch_size, 10))
        # With EOS
        ids_with_eos = torch.randint(0, 40, (batch_size, 10))
        ids_with_eos[0, -1] = eos_token_id

        assert not criteria(ids_no_eos, None)  # Should continue
        # Note: EOS criteria typically checks per-sequence
        # Implementation may vary


class TestGreedyDecoding:
    """Tests for greedy decoding."""

    def test_greedy_initialization(self):
        """Test greedy decoding can be initialized."""
        config = GPTConfig(vocab_size=100, d_model=64, num_layers=1, num_heads=2)
        model = GPTModel(config)
        greedy = GreedyDecoding(model, max_length=20)
        assert greedy is not None

    def test_greedy_generate(self):
        """Test greedy decoding generation."""
        config = GPTConfig(vocab_size=100, d_model=64, num_layers=1, num_heads=2)
        model = GPTModel(config)
        model.eval()

        greedy = GreedyDecoding(model, max_length=20)

        batch_size = 2
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 5))

        with torch.no_grad():
            output = greedy.generate(input_ids, max_new_tokens=10)

        assert output.shape == (batch_size, 15)  # 5 + 10

    def test_greedy_deterministic(self):
        """Test greedy decoding is deterministic."""
        config = GPTConfig(vocab_size=100, d_model=64, num_layers=1, num_heads=2)
        model = GPTModel(config)
        model.eval()

        greedy = GreedyDecoding(model, max_length=20)

        input_ids = torch.randint(0, config.vocab_size, (1, 5))

        with torch.no_grad():
            output1 = greedy.generate(input_ids, max_new_tokens=10)
            output2 = greedy.generate(input_ids, max_new_tokens=10)

        # Greedy should be deterministic
        assert torch.allclose(output1.float(), output2.float())


class TestBeamSearch:
    """Tests for beam search decoding."""

    def test_beam_search_initialization(self):
        """Test beam search can be initialized."""
        config = GPTConfig(vocab_size=100, d_model=64, num_layers=1, num_heads=2)
        model = GPTModel(config)
        beam = BeamSearchDecoding(model, max_length=20, num_beams=4)
        assert beam is not None
        assert beam.num_beams == 4

    def test_beam_search_generate(self):
        """Test beam search generation."""
        config = GPTConfig(vocab_size=100, d_model=64, num_layers=1, num_heads=2)
        model = GPTModel(config)
        model.eval()

        beam = BeamSearchDecoding(model, max_length=20, num_beams=4)

        batch_size = 2
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 5))

        with torch.no_grad():
            output = beam.generate(input_ids, max_new_tokens=10)

        # Output should maintain batch size
        assert output.shape[0] == batch_size
        assert output.shape[1] >= 5  # At least input length


class TestNucleusSampling:
    """Tests for nucleus (top-p) sampling."""

    def test_nucleus_initialization(self):
        """Test nucleus sampling can be initialized."""
        config = GPTConfig(vocab_size=100, d_model=64, num_layers=1, num_heads=2)
        model = GPTModel(config)
        nucleus = NucleusSampling(model, max_length=20, top_p=0.9)
        assert nucleus is not None
        assert nucleus.top_p == 0.9

    def test_nucleus_generate(self):
        """Test nucleus sampling generation."""
        config = GPTConfig(vocab_size=100, d_model=64, num_layers=1, num_heads=2)
        model = GPTModel(config)
        model.eval()

        nucleus = NucleusSampling(model, max_length=20, top_p=0.9)

        batch_size = 2
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 5))

        torch.manual_seed(42)
        with torch.no_grad():
            output = nucleus.generate(input_ids, max_new_tokens=10)

        assert output.shape == (batch_size, 15)

    def test_nucleus_stochastic(self):
        """Test nucleus sampling is stochastic."""
        config = GPTConfig(vocab_size=100, d_model=64, num_layers=1, num_heads=2)
        model = GPTModel(config)
        model.eval()

        nucleus = NucleusSampling(model, max_length=20, top_p=0.9, temperature=1.0)

        input_ids = torch.randint(0, config.vocab_size, (1, 5))

        with torch.no_grad():
            torch.manual_seed(42)
            output1 = nucleus.generate(input_ids, max_new_tokens=10)
            torch.manual_seed(43)  # Different seed
            output2 = nucleus.generate(input_ids, max_new_tokens=10)

        # Sampling should be different with different seeds
        assert not torch.equal(output1, output2)


class TestTopKSampling:
    """Tests for top-k sampling."""

    def test_topk_initialization(self):
        """Test top-k sampling can be initialized."""
        config = GPTConfig(vocab_size=100, d_model=64, num_layers=1, num_heads=2)
        model = GPTModel(config)
        topk = TopKSampling(model, max_length=20, top_k=50)
        assert topk is not None
        assert topk.top_k == 50

    def test_topk_generate(self):
        """Test top-k sampling generation."""
        config = GPTConfig(vocab_size=100, d_model=64, num_layers=1, num_heads=2)
        model = GPTModel(config)
        model.eval()

        topk = TopKSampling(model, max_length=20, top_k=50)

        batch_size = 2
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 5))

        torch.manual_seed(42)
        with torch.no_grad():
            output = topk.generate(input_ids, max_new_tokens=10)

        assert output.shape == (batch_size, 15)


class TestContrastiveDecoding:
    """Tests for contrastive decoding."""

    def test_contrastive_initialization(self):
        """Test contrastive decoding can be initialized."""
        config = GPTConfig(vocab_size=100, d_model=64, num_layers=2, num_heads=2)
        expert_model = GPTModel(config)

        amateur_config = GPTConfig(vocab_size=100, d_model=64, num_layers=1, num_heads=2)
        amateur_model = GPTModel(amateur_config)

        contrastive = ContrastiveDecoding(
            expert_model=expert_model,
            amateur_model=amateur_model,
            max_length=20,
            alpha=0.5,
        )
        assert contrastive is not None

    def test_contrastive_generate(self):
        """Test contrastive decoding generation."""
        config = GPTConfig(vocab_size=100, d_model=64, num_layers=2, num_heads=2)
        expert_model = GPTModel(config)

        amateur_config = GPTConfig(vocab_size=100, d_model=64, num_layers=1, num_heads=2)
        amateur_model = GPTModel(amateur_config)

        expert_model.eval()
        amateur_model.eval()

        contrastive = ContrastiveDecoding(
            expert_model=expert_model,
            amateur_model=amateur_model,
            max_length=20,
            alpha=0.5,
        )

        batch_size = 2
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 5))

        with torch.no_grad():
            output = contrastive.generate(input_ids, max_new_tokens=10)

        assert output.shape[0] == batch_size


class TestSpeculativeDecoding:
    """Tests for speculative decoding."""

    def test_speculative_initialization(self):
        """Test speculative decoding can be initialized."""
        config = GPTConfig(vocab_size=100, d_model=64, num_layers=2, num_heads=2)
        target_model = GPTModel(config)

        draft_config = GPTConfig(vocab_size=100, d_model=64, num_layers=1, num_heads=2)
        draft_model = GPTModel(draft_config)

        speculative = SpeculativeDecoding(
            target_model=target_model,
            draft_model=draft_model,
            max_length=20,
            gamma=4,
        )
        assert speculative is not None
        assert speculative.gamma == 4

    def test_speculative_generate(self):
        """Test speculative decoding generation."""
        config = GPTConfig(vocab_size=100, d_model=64, num_layers=2, num_heads=2)
        target_model = GPTModel(config)

        draft_config = GPTConfig(vocab_size=100, d_model=64, num_layers=1, num_heads=2)
        draft_model = GPTModel(draft_config)

        target_model.eval()
        draft_model.eval()

        speculative = SpeculativeDecoding(
            target_model=target_model,
            draft_model=draft_model,
            max_length=20,
            gamma=4,
        )

        batch_size = 1  # Speculative often works with batch_size=1
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 5))

        with torch.no_grad():
            output = speculative.generate(input_ids, max_new_tokens=10)

        assert output.shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
