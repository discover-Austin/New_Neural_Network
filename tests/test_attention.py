"""
Tests for Attention Mechanisms
===============================

Tests for all attention variants: MHA, MQA, GQA, Flash, Linear, etc.
"""

import torch
import pytest
from src.attention import (
    MultiHeadAttention,
    MultiQueryAttention,
    GroupedQueryAttention,
    FlashAttention,
    LinearAttention,
    SlidingWindowAttention,
    SparseAttention,
    CrossAttention,
)


class TestMultiHeadAttention:
    """Tests for Multi-Head Attention (MHA)."""

    def test_mha_initialization(self):
        """Test MHA can be initialized."""
        mha = MultiHeadAttention(d_model=128, num_heads=8)
        assert mha is not None
        assert mha.num_heads == 8

    def test_mha_forward_self_attention(self):
        """Test MHA self-attention forward pass."""
        d_model, num_heads = 128, 8
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model)

        # Self-attention
        output = mha(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_mha_with_mask(self):
        """Test MHA with attention mask."""
        d_model, num_heads = 128, 8
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        mask = ~mask  # Invert for PyTorch convention

        output = mha(x, x, x, attention_mask=mask)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_mha_gradient_flow(self):
        """Test gradients flow through MHA."""
        d_model, num_heads = 128, 8
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output = mha(x, x, x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestMultiQueryAttention:
    """Tests for Multi-Query Attention (MQA)."""

    def test_mqa_initialization(self):
        """Test MQA can be initialized."""
        mqa = MultiQueryAttention(d_model=128, num_heads=8)
        assert mqa is not None

    def test_mqa_forward(self):
        """Test MQA forward pass."""
        d_model, num_heads = 128, 8
        mqa = MultiQueryAttention(d_model=d_model, num_heads=num_heads)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model)

        output = mqa(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_mqa_kv_cache_smaller(self):
        """Test MQA has smaller KV cache than MHA."""
        d_model, num_heads = 128, 8
        mqa = MultiQueryAttention(d_model=d_model, num_heads=num_heads)

        # Check that k_proj and v_proj output dimension is d_head (not d_model)
        # This means 1 KV head instead of num_heads
        head_dim = d_model // num_heads
        assert mqa.k_proj.out_features == head_dim
        assert mqa.v_proj.out_features == head_dim


class TestGroupedQueryAttention:
    """Tests for Grouped-Query Attention (GQA)."""

    def test_gqa_initialization(self):
        """Test GQA can be initialized."""
        gqa = GroupedQueryAttention(d_model=128, num_heads=8, num_kv_heads=2)
        assert gqa is not None
        assert gqa.num_kv_heads == 2

    def test_gqa_forward(self):
        """Test GQA forward pass."""
        d_model, num_heads, num_kv_heads = 128, 8, 2
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model)

        output = gqa(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_gqa_balances_mha_mqa(self):
        """Test GQA is between MHA and MQA in KV size."""
        d_model, num_heads, num_kv_heads = 128, 8, 2
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        head_dim = d_model // num_heads
        expected_kv_dim = head_dim * num_kv_heads

        # Check KV projections
        assert gqa.k_proj.out_features == expected_kv_dim
        assert gqa.v_proj.out_features == expected_kv_dim


class TestFlashAttention:
    """Tests for Flash Attention."""

    def test_flash_attention_initialization(self):
        """Test Flash Attention can be initialized."""
        flash = FlashAttention(d_model=128, num_heads=8)
        assert flash is not None

    def test_flash_attention_forward(self):
        """Test Flash Attention forward pass."""
        d_model, num_heads = 128, 8
        flash = FlashAttention(d_model=d_model, num_heads=num_heads)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model)

        output = flash(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_flash_attention_memory_efficient(self):
        """Test Flash Attention is more memory efficient."""
        # Note: This test verifies the implementation exists
        # Actual memory efficiency would require profiling
        d_model, num_heads = 128, 8
        flash = FlashAttention(d_model=d_model, num_heads=num_heads)

        batch_size, seq_len = 2, 64  # Longer sequence
        x = torch.randn(batch_size, seq_len, d_model)

        # Should not create full attention matrix
        output = flash(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)


class TestLinearAttention:
    """Tests for Linear Attention."""

    def test_linear_attention_initialization(self):
        """Test Linear Attention can be initialized."""
        linear_attn = LinearAttention(d_model=128, num_heads=8)
        assert linear_attn is not None

    def test_linear_attention_forward(self):
        """Test Linear Attention forward pass."""
        d_model, num_heads = 128, 8
        linear_attn = LinearAttention(d_model=d_model, num_heads=num_heads)

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model)

        output = linear_attn(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_linear_attention_long_sequence(self):
        """Test Linear Attention on long sequences (O(N) complexity)."""
        d_model, num_heads = 128, 8
        linear_attn = LinearAttention(d_model=d_model, num_heads=num_heads)

        batch_size, seq_len = 2, 512  # Long sequence
        x = torch.randn(batch_size, seq_len, d_model)

        output = linear_attn(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)


class TestSlidingWindowAttention:
    """Tests for Sliding Window Attention."""

    def test_sliding_window_initialization(self):
        """Test Sliding Window Attention can be initialized."""
        sliding = SlidingWindowAttention(d_model=128, num_heads=8, window_size=32)
        assert sliding is not None
        assert sliding.window_size == 32

    def test_sliding_window_forward(self):
        """Test Sliding Window Attention forward pass."""
        d_model, num_heads, window_size = 128, 8, 16
        sliding = SlidingWindowAttention(
            d_model=d_model,
            num_heads=num_heads,
            window_size=window_size,
        )

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, d_model)

        output = sliding(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)


class TestSparseAttention:
    """Tests for Sparse Attention."""

    def test_sparse_attention_initialization(self):
        """Test Sparse Attention can be initialized."""
        sparse = SparseAttention(d_model=128, num_heads=8)
        assert sparse is not None

    def test_sparse_attention_forward(self):
        """Test Sparse Attention forward pass."""
        d_model, num_heads = 128, 8
        sparse = SparseAttention(d_model=d_model, num_heads=num_heads)

        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, d_model)

        output = sparse(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)


class TestCrossAttention:
    """Tests for Cross Attention."""

    def test_cross_attention_initialization(self):
        """Test Cross Attention can be initialized."""
        cross_attn = CrossAttention(d_model=128, num_heads=8)
        assert cross_attn is not None

    def test_cross_attention_forward(self):
        """Test Cross Attention forward pass."""
        d_model, num_heads = 128, 8
        cross_attn = CrossAttention(d_model=d_model, num_heads=num_heads)

        batch_size = 2
        query_len, context_len = 10, 20

        query = torch.randn(batch_size, query_len, d_model)
        context = torch.randn(batch_size, context_len, d_model)

        # Cross attention: query attends to context
        output = cross_attn(query, context, context)

        assert output.shape == (batch_size, query_len, d_model)

    def test_cross_attention_different_lengths(self):
        """Test Cross Attention with different sequence lengths."""
        d_model, num_heads = 128, 8
        cross_attn = CrossAttention(d_model=d_model, num_heads=num_heads)

        batch_size = 2
        query_len, context_len = 5, 50  # Very different lengths

        query = torch.randn(batch_size, query_len, d_model)
        context = torch.randn(batch_size, context_len, d_model)

        output = cross_attn(query, context, context)

        # Output should match query length
        assert output.shape == (batch_size, query_len, d_model)


class TestAttentionComparison:
    """Cross-attention mechanism comparison tests."""

    def test_all_attention_same_interface(self):
        """Test all attention mechanisms have same interface."""
        d_model, num_heads = 128, 8
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model)

        attention_classes = [
            MultiHeadAttention(d_model, num_heads),
            MultiQueryAttention(d_model, num_heads),
            GroupedQueryAttention(d_model, num_heads, num_kv_heads=2),
            FlashAttention(d_model, num_heads),
            LinearAttention(d_model, num_heads),
            SlidingWindowAttention(d_model, num_heads, window_size=8),
            SparseAttention(d_model, num_heads),
            CrossAttention(d_model, num_heads),
        ]

        for attn in attention_classes:
            output = attn(x, x, x)
            assert output.shape == (batch_size, seq_len, d_model), \
                f"Failed for {attn.__class__.__name__}"

    def test_attention_output_not_nan(self):
        """Test all attention mechanisms produce valid outputs."""
        d_model, num_heads = 128, 8
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model)

        attention_classes = [
            MultiHeadAttention(d_model, num_heads),
            MultiQueryAttention(d_model, num_heads),
            GroupedQueryAttention(d_model, num_heads, num_kv_heads=2),
            FlashAttention(d_model, num_heads),
            LinearAttention(d_model, num_heads),
        ]

        for attn in attention_classes:
            output = attn(x, x, x)
            assert not torch.isnan(output).any(), \
                f"NaN output from {attn.__class__.__name__}"
            assert not torch.isinf(output).any(), \
                f"Inf output from {attn.__class__.__name__}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
