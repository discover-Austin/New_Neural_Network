"""
Tests for Core Model Architectures
===================================

Tests for GPT, BERT, T5, and LLaMA models.
"""

import torch
import pytest
from src.models import GPTModel, GPTConfig, BERTModel, BERTConfig, T5Model, T5Config, LLaMAModel, LLaMAConfig


class TestGPTModel:
    """Tests for GPT model."""

    def test_gpt_initialization(self):
        """Test GPT model can be initialized."""
        config = GPTConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
        )
        model = GPTModel(config)
        assert model is not None
        assert model.config == config

    def test_gpt_forward_pass(self):
        """Test GPT forward pass with correct shapes."""
        config = GPTConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
        )
        model = GPTModel(config)

        # Create sample input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits, loss = model(input_ids, return_loss=False)

        # Check output shape
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is None

    def test_gpt_forward_with_loss(self):
        """Test GPT forward pass with loss computation."""
        config = GPTConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
        )
        model = GPTModel(config)

        # Create sample input and labels
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass with loss
        logits, loss = model(input_ids, labels=labels, return_loss=True)

        # Check loss is computed
        assert loss is not None
        assert loss.item() > 0

    def test_gpt_generate(self):
        """Test GPT text generation."""
        config = GPTConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
        )
        model = GPTModel(config)
        model.eval()

        # Create sample input
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Generate
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=1.0,
                do_sample=False,  # Greedy
            )

        # Check output shape
        assert output.shape == (batch_size, seq_len + 10)

    def test_gpt_gradient_flow(self):
        """Test gradients flow through GPT model."""
        config = GPTConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
        )
        model = GPTModel(config)

        # Create sample input and labels
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward and backward
        logits, loss = model(input_ids, labels=labels)
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestBERTModel:
    """Tests for BERT model."""

    def test_bert_initialization(self):
        """Test BERT model can be initialized."""
        config = BERTConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
        )
        model = BERTModel(config)
        assert model is not None
        assert model.config == config

    def test_bert_forward_pass(self):
        """Test BERT forward pass with correct shapes."""
        config = BERTConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
        )
        model = BERTModel(config)

        # Create sample input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        output = model(input_ids)

        # Check output shape
        assert output["last_hidden_state"].shape == (batch_size, seq_len, config.d_model)
        assert output["pooled_output"].shape == (batch_size, config.d_model)

    def test_bert_mlm_task(self):
        """Test BERT masked language modeling."""
        config = BERTConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
        )
        model = BERTModel(config)

        # Create sample input with masked tokens
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        # Mask some tokens
        mask_positions = torch.rand(batch_size, seq_len) < 0.15
        input_ids[mask_positions] = config.mask_token_id

        # Forward with MLM
        output = model(input_ids, labels=labels, task="mlm")

        # Check loss exists
        assert "loss" in output
        assert output["loss"] is not None


class TestT5Model:
    """Tests for T5 model."""

    def test_t5_initialization(self):
        """Test T5 model can be initialized."""
        config = T5Config(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
        )
        model = T5Model(config)
        assert model is not None
        assert model.config == config

    def test_t5_forward_pass(self):
        """Test T5 forward pass with encoder-decoder."""
        config = T5Config(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
        )
        model = T5Model(config)

        # Create sample input
        batch_size, src_len, tgt_len = 2, 10, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, src_len))
        decoder_input_ids = torch.randint(0, config.vocab_size, (batch_size, tgt_len))

        # Forward pass
        output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

        # Check output shape
        assert output["logits"].shape == (batch_size, tgt_len, config.vocab_size)

    def test_t5_seq2seq_task(self):
        """Test T5 sequence-to-sequence task."""
        config = T5Config(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            max_seq_len=64,
        )
        model = T5Model(config)

        # Create sample input and labels
        batch_size, src_len, tgt_len = 2, 10, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, src_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, tgt_len))

        # Forward with loss
        output = model(input_ids=input_ids, labels=labels)

        # Check loss is computed
        assert "loss" in output
        assert output["loss"] is not None


class TestLLaMAModel:
    """Tests for LLaMA model."""

    def test_llama_initialization(self):
        """Test LLaMA model can be initialized."""
        config = LLaMAConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,  # GQA
            max_seq_len=64,
        )
        model = LLaMAModel(config)
        assert model is not None
        assert model.config == config

    def test_llama_forward_pass(self):
        """Test LLaMA forward pass with RoPE and GQA."""
        config = LLaMAConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,  # GQA
            max_seq_len=64,
        )
        model = LLaMAModel(config)

        # Create sample input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits, loss = model(input_ids, return_loss=False)

        # Check output shape
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_llama_uses_rmsnorm(self):
        """Test LLaMA uses RMSNorm instead of LayerNorm."""
        config = LLaMAConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=64,
        )
        model = LLaMAModel(config)

        # Check that RMSNorm is used
        from src.core import RMSNorm
        has_rmsnorm = False
        for module in model.modules():
            if isinstance(module, RMSNorm):
                has_rmsnorm = True
                break

        assert has_rmsnorm, "LLaMA should use RMSNorm"

    def test_llama_generate(self):
        """Test LLaMA text generation."""
        config = LLaMAConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=64,
        )
        model = LLaMAModel(config)
        model.eval()

        # Create sample input
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Generate
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=1.0,
            )

        # Check output shape
        assert output.shape[1] == seq_len + 10


class TestModelComparison:
    """Cross-model comparison tests."""

    def test_all_models_trainable(self):
        """Test all models can perform a training step."""
        configs_and_models = [
            (GPTConfig(vocab_size=500, d_model=64, num_layers=1, num_heads=2, max_seq_len=32), GPTModel),
            (BERTConfig(vocab_size=500, d_model=64, num_layers=1, num_heads=2, max_seq_len=32), BERTModel),
            (LLaMAConfig(vocab_size=500, d_model=64, num_layers=1, num_heads=2, num_kv_heads=1, max_seq_len=32), LLaMAModel),
        ]

        for config, ModelClass in configs_and_models:
            model = ModelClass(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Create sample batch
            batch_size, seq_len = 2, 8
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

            # Training step
            optimizer.zero_grad()

            if isinstance(model, (GPTModel, LLaMAModel)):
                logits, loss = model(input_ids, labels=input_ids)
            elif isinstance(model, BERTModel):
                output = model(input_ids, labels=input_ids, task="mlm")
                loss = output["loss"]

            loss.backward()
            optimizer.step()

            # Check loss decreased or at least computed
            assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
