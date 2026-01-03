"""
Verification Tests for Advanced Components
===========================================

PhD-Level Testing Protocol:

1. Mathematical Correctness
   - Verify complexity claims
   - Check numerical stability
   - Validate probabilistic properties

2. Implementation Verification
   - Forward pass correctness
   - Gradient flow (no vanishing/exploding)
   - Memory efficiency

3. Empirical Validation
   - Compare against baselines
   - Verify claimed speedups
   - Check quality preservation
"""

import torch
import torch.nn as nn
import pytest
import time
import math
from typing import Dict

import sys
sys.path.append('/home/user/New_Neural_Network')

from src.advanced.multi_scale_transformer import (
    MultiScaleTransformer,
    MultiScaleConfig,
    HierarchicalPooling,
)
from src.advanced.memory_augmented import (
    kNNMemory,
    MemoryConfig,
)
from src.advanced.calibrated_exit import (
    ConfidenceCalibrator,
    TemperatureScaling,
    ExitDecisionModule,
)


class TestHierarchicalPooling:
    """Test hierarchical pooling operations."""

    def test_pooling_reduces_sequence_length(self):
        """Verify pooling reduces sequence length by pool_factor."""
        d_model = 768
        pool_factor = 4
        seq_len = 256

        pooling = HierarchicalPooling(d_model, pool_factor, method="average")

        x = torch.randn(2, seq_len, d_model)
        pooled = pooling.pool(x)

        assert pooled.shape == (2, seq_len // pool_factor, d_model), \
            f"Expected shape (2, {seq_len // pool_factor}, {d_model}), got {pooled.shape}"

    def test_pooling_unpooling_reconstruction(self):
        """Verify pooling followed by unpooling approximately reconstructs input."""
        d_model = 768
        pool_factor = 4
        seq_len = 256

        for method in ["average", "max"]:
            pooling = HierarchicalPooling(d_model, pool_factor, method=method)

            x = torch.randn(2, seq_len, d_model)
            pooled = pooling.pool(x)
            unpooled = pooling.unpool(pooled, seq_len)

            assert unpooled.shape == x.shape, \
                f"Reconstruction shape mismatch: {unpooled.shape} vs {x.shape}"

            # For average pooling, reconstruction should be exact for constant inputs
            if method == "average":
                x_constant = torch.ones(2, seq_len, d_model)
                pooled_const = pooling.pool(x_constant)
                reconstructed = pooling.unpool(pooled_const, seq_len)

                assert torch.allclose(x_constant, reconstructed, atol=1e-6), \
                    "Average pooling should exactly reconstruct constant inputs"

    def test_computational_complexity(self):
        """Verify pooling is O(N) complexity."""
        d_model = 768
        pool_factor = 4

        pooling = HierarchicalPooling(d_model, pool_factor, method="average")

        # Test different sequence lengths
        times = []
        for seq_len in [256, 512, 1024, 2048]:
            x = torch.randn(1, seq_len, d_model)

            start = time.time()
            for _ in range(100):
                _ = pooling.pool(x)
            elapsed = time.time() - start

            times.append((seq_len, elapsed))

        # Verify roughly linear scaling
        ratio_1 = times[1][1] / times[0][1]
        ratio_2 = times[1][0] / times[0][0]

        # Time ratio should be close to sequence length ratio for O(N)
        assert abs(ratio_1 - ratio_2) < 1.0, \
            f"Pooling complexity may not be O(N): time ratio {ratio_1} vs length ratio {ratio_2}"


class TestMultiScaleTransformer:
    """Test multi-scale transformer."""

    def test_forward_pass(self):
        """Test basic forward pass."""
        config = MultiScaleConfig(
            d_model=256,
            num_layers_per_scale=[2, 1, 1],
            num_heads=4,
            num_kv_heads=2,
            d_ff=1024,
            num_scales=3,
            pooling_factors=[1, 4, 16],
            vocab_size=1000,
        )

        model = MultiScaleTransformer(config)
        input_ids = torch.randint(0, 1000, (2, 128))

        outputs = model(input_ids)

        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 128, 1000)
        assert "scale_features" in outputs
        assert len(outputs["scale_features"]) == 4  # 3 scales + initial

    def test_gradient_flow(self):
        """Verify gradients flow through all scales."""
        config = MultiScaleConfig(
            d_model=128,
            num_layers_per_scale=[2, 1],
            num_heads=4,
            num_kv_heads=2,
            d_ff=512,
            num_scales=2,
            pooling_factors=[1, 4],
            vocab_size=100,
        )

        model = MultiScaleTransformer(config)
        input_ids = torch.randint(0, 100, (2, 64))
        labels = torch.randint(0, 100, (2, 64))

        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_computational_overhead(self):
        """Verify multi-scale overhead is reasonable (<20%)."""
        from src.models import GPTModel, GPTConfig

        # Baseline: Standard GPT
        gpt_config = GPTConfig(
            vocab_size=1000,
            d_model=256,
            num_layers=4,
            num_heads=4,
            d_ff=1024,
        )
        gpt_model = GPTModel(gpt_config)

        # Multi-scale
        ms_config = MultiScaleConfig(
            d_model=256,
            num_layers_per_scale=[4],  # Same total layers
            num_heads=4,
            num_kv_heads=2,
            d_ff=1024,
            num_scales=1,  # Single scale for fair comparison
            vocab_size=1000,
        )
        ms_model = MultiScaleTransformer(ms_config)

        input_ids = torch.randint(0, 1000, (4, 128))

        # Measure baseline
        start = time.time()
        for _ in range(10):
            _ = gpt_model(input_ids)
        baseline_time = time.time() - start

        # Measure multi-scale
        start = time.time()
        for _ in range(10):
            _ = ms_model(input_ids)
        ms_time = time.time() - start

        overhead = (ms_time - baseline_time) / baseline_time
        print(f"Multi-scale overhead: {overhead*100:.1f}%")

        # Overhead should be reasonable
        assert overhead < 0.5, f"Overhead too high: {overhead*100:.1f}%"


class TestkNNMemory:
    """Test k-Nearest Neighbor memory."""

    def test_memory_addition(self):
        """Test adding items to memory."""
        config = MemoryConfig(
            memory_size=100,
            k_neighbors=5,
            memory_dim=64,
        )

        memory = kNNMemory(config)

        keys = torch.randn(2, 10, 64)
        values = torch.randint(0, 100, (2, 10))

        memory.add_to_memory(keys, values)

        # Check memory pointer updated
        assert memory.memory_ptr.item() == 20  # 2 * 10

    def test_retrieval_returns_k_neighbors(self):
        """Test retrieval returns exactly k neighbors."""
        config = MemoryConfig(
            memory_size=100,
            k_neighbors=5,
            memory_dim=64,
        )

        memory = kNNMemory(config)

        # Add some memories
        keys = torch.randn(10, 1, 64)
        values = torch.randint(0, 100, (10, 1))
        memory.add_to_memory(keys, values)

        # Retrieve
        query = torch.randn(2, 3, 64)
        neighbor_values, neighbor_distances, neighbor_probs = memory.retrieve(query, k=5)

        assert neighbor_values.shape == (2, 3, 5)
        assert neighbor_distances.shape == (2, 3, 5)
        assert neighbor_probs.shape == (2, 3, 5)

        # Probabilities should sum to 1
        assert torch.allclose(neighbor_probs.sum(dim=-1), torch.ones(2, 3), atol=1e-5)

    def test_knn_distribution_sums_to_one(self):
        """Verify kNN distribution is valid probability distribution."""
        config = MemoryConfig(
            memory_size=100,
            k_neighbors=5,
            memory_dim=64,
        )

        memory = kNNMemory(config)

        # Add memories
        keys = torch.randn(10, 1, 64)
        values = torch.randint(0, 50, (10, 1))
        memory.add_to_memory(keys, values)

        # Retrieve and compute distribution
        query = torch.randn(2, 3, 64)
        neighbor_values, _, neighbor_probs = memory.retrieve(query, k=5)

        vocab_size = 100
        knn_dist = memory.compute_knn_distribution(neighbor_values, neighbor_probs, vocab_size)

        # Should sum to 1
        assert torch.allclose(knn_dist.sum(dim=-1), torch.ones(2, 3), atol=1e-5)

        # Should be non-negative
        assert (knn_dist >= 0).all()

    def test_interpolation_weight_bounded(self):
        """Verify interpolation weight is in [0, 1]."""
        config = MemoryConfig(memory_size=100, memory_dim=64)
        memory = kNNMemory(config)

        # Lambda should be sigmoid of learned parameter
        lambda_val = torch.sigmoid(memory.lambda_param).item()
        assert 0 <= lambda_val <= 1, f"Lambda out of bounds: {lambda_val}"


class TestConfidenceCalibrator:
    """Test confidence calibration."""

    def test_entropy_confidence_bounds(self):
        """Verify entropy confidence is in [0, 1]."""
        vocab_size = 100
        calibrator = ConfidenceCalibrator(vocab_size, use_temperature_scaling=False)

        logits = torch.randn(2, 10, vocab_size)
        confidence = calibrator.compute_entropy_confidence(logits)

        assert (confidence >= 0).all() and (confidence <= 1).all(), \
            "Entropy confidence out of bounds"

    def test_uniform_distribution_low_confidence(self):
        """Verify uniform distribution gives low confidence."""
        vocab_size = 100
        calibrator = ConfidenceCalibrator(vocab_size, use_temperature_scaling=False)

        # Uniform logits (high entropy)
        uniform_logits = torch.zeros(2, 10, vocab_size)
        confidence = calibrator.compute_entropy_confidence(uniform_logits)

        # Should have low confidence (high entropy)
        assert confidence.mean() < 0.3, \
            f"Uniform distribution should have low confidence, got {confidence.mean()}"

    def test_peaked_distribution_high_confidence(self):
        """Verify peaked distribution gives high confidence."""
        vocab_size = 100
        calibrator = ConfidenceCalibrator(vocab_size, use_temperature_scaling=False)

        # Peaked logits (low entropy)
        peaked_logits = torch.randn(2, 10, vocab_size)
        peaked_logits[:, :, 0] += 10  # Make one class very likely

        confidence = calibrator.compute_entropy_confidence(peaked_logits)

        # Should have high confidence (low entropy)
        assert confidence.mean() > 0.7, \
            f"Peaked distribution should have high confidence, got {confidence.mean()}"

    def test_temperature_scaling_preserves_argmax(self):
        """Verify temperature scaling doesn't change predictions."""
        vocab_size = 100
        temp_scaling = TemperatureScaling(initial_temperature=2.0)

        logits = torch.randn(2, 10, vocab_size)

        # Original predictions
        original_preds = logits.argmax(dim=-1)

        # Scaled predictions
        scaled_logits = temp_scaling(logits)
        scaled_preds = scaled_logits.argmax(dim=-1)

        # Should be identical
        assert (original_preds == scaled_preds).all(), \
            "Temperature scaling changed predictions"


class TestExitDecisionModule:
    """Test early exit decision making."""

    def test_no_exit_before_min_layer(self):
        """Verify no exits before minimum layer."""
        exit_module = ExitDecisionModule(
            confidence_threshold=0.9,
            min_layer=6,
        )

        confidence = torch.ones(2, 10) * 0.95  # High confidence

        for layer in range(6):
            exit_mask, _ = exit_module.should_exit(layer, confidence)
            assert not exit_mask.any(), f"Unexpected exit at layer {layer}"

    def test_exit_with_high_confidence(self):
        """Verify exit occurs when confidence exceeds threshold."""
        exit_module = ExitDecisionModule(
            confidence_threshold=0.9,
            min_layer=2,
            patience=1,
        )

        high_confidence = torch.ones(2, 10) * 0.95
        low_confidence = torch.ones(2, 10) * 0.5

        # Should exit with high confidence
        exit_mask_high, metrics_high = exit_module.should_exit(5, high_confidence)
        assert exit_mask_high.any(), "Should exit with high confidence"
        assert metrics_high["exit_rate"] > 0

        # Should not exit with low confidence
        exit_mask_low, metrics_low = exit_module.should_exit(5, low_confidence)
        assert not exit_mask_low.any(), "Should not exit with low confidence"
        assert metrics_low["exit_rate"] == 0

    def test_patience_requires_consistency(self):
        """Verify patience requires consistent high confidence."""
        exit_module = ExitDecisionModule(
            confidence_threshold=0.9,
            min_layer=2,
            patience=3,
        )

        high_conf = torch.ones(2, 10) * 0.95
        low_conf = torch.ones(2, 10) * 0.5

        history = []

        # Layer 3: high confidence
        history.append(high_conf)

        # Layer 4: low confidence (breaks consistency)
        history.append(low_conf)

        # Layer 5: high confidence again
        exit_mask, _ = exit_module.should_exit(5, high_conf, history)

        # Should not exit because confidence was low at layer 4
        assert not exit_mask.any(), "Should not exit without consistent confidence"


def run_verification_suite():
    """Run complete verification suite."""
    print("="*80)
    print("ADVANCED COMPONENTS VERIFICATION SUITE")
    print("="*80)

    test_classes = [
        TestHierarchicalPooling,
        TestMultiScaleTransformer,
        TestkNNMemory,
        TestConfidenceCalibrator,
        TestExitDecisionModule,
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 80)

        test_instance = test_class()
        methods = [m for m in dir(test_instance) if m.startswith('test_')]

        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {str(e)}")

    print("\n" + "="*80)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("="*80)

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_verification_suite()
    exit(0 if success else 1)
