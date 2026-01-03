"""
Tests for Tokenization
=======================

Tests for BPE, WordPiece, and Character-level tokenizers.
"""

import torch
import pytest
from src.tokenization import BPETokenizer, WordPieceTokenizer, CharacterTokenizer


class TestBPETokenizer:
    """Tests for Byte Pair Encoding tokenizer."""

    def test_bpe_initialization(self):
        """Test BPE tokenizer can be initialized."""
        tokenizer = BPETokenizer(vocab_size=1000)
        assert tokenizer is not None
        assert tokenizer.vocab_size == 1000

    def test_bpe_train(self):
        """Test BPE tokenizer training."""
        texts = [
            "hello world",
            "hello there",
            "world of machines",
        ]

        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(texts)

        # Should have learned merges
        assert len(tokenizer.merges) > 0
        assert len(tokenizer.vocab) > 0

    def test_bpe_encode_decode(self):
        """Test BPE encode and decode."""
        texts = [
            "hello world hello world",
            "machine learning is amazing",
        ]

        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(texts)

        # Encode
        text = "hello world"
        encoded = tokenizer.encode(text)
        assert isinstance(encoded, list)
        assert len(encoded) > 0

        # Decode
        decoded = tokenizer.decode(encoded)
        # Should recover original text (with possible spacing differences)
        assert "hello" in decoded.lower()
        assert "world" in decoded.lower()

    def test_bpe_unknown_tokens(self):
        """Test BPE handling of unknown tokens."""
        texts = ["hello world"]

        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(texts)

        # Encode text with unknown words
        # BPE should fall back to character level
        encoded = tokenizer.encode("xyz")
        assert isinstance(encoded, list)


class TestWordPieceTokenizer:
    """Tests for WordPiece tokenizer."""

    def test_wordpiece_initialization(self):
        """Test WordPiece tokenizer can be initialized."""
        tokenizer = WordPieceTokenizer(vocab_size=1000)
        assert tokenizer is not None
        assert tokenizer.vocab_size == 1000

    def test_wordpiece_train(self):
        """Test WordPiece tokenizer training."""
        texts = [
            "running runner run",
            "playing player play",
        ]

        tokenizer = WordPieceTokenizer(vocab_size=100)
        tokenizer.train(texts)

        assert len(tokenizer.vocab) > 0

    def test_wordpiece_encode_decode(self):
        """Test WordPiece encode and decode."""
        texts = [
            "running runner run running",
            "playing player play playing",
        ]

        tokenizer = WordPieceTokenizer(vocab_size=100)
        tokenizer.train(texts)

        # Encode
        text = "running"
        encoded = tokenizer.encode(text)
        assert isinstance(encoded, list)
        assert len(encoded) > 0

        # Decode
        decoded = tokenizer.decode(encoded)
        assert "run" in decoded.lower()

    def test_wordpiece_subwords(self):
        """Test WordPiece creates subword tokens."""
        texts = [
            "unwanted unbelievable unforgettable",
            "wanted believable forgettable",
        ]

        tokenizer = WordPieceTokenizer(vocab_size=100)
        tokenizer.train(texts)

        # Should learn "un" as a prefix
        # Check that vocab contains subword units
        vocab_str = " ".join(tokenizer.vocab.keys())
        # At least some subword units should exist
        assert len(tokenizer.vocab) > 10


class TestCharacterTokenizer:
    """Tests for Character-level tokenizer."""

    def test_character_initialization(self):
        """Test Character tokenizer can be initialized."""
        tokenizer = CharacterTokenizer()
        assert tokenizer is not None

    def test_character_train(self):
        """Test Character tokenizer training."""
        texts = [
            "hello world",
            "goodbye world",
        ]

        tokenizer = CharacterTokenizer()
        tokenizer.train(texts)

        # Should have character vocabulary
        assert len(tokenizer.vocab) > 0
        # Should contain common characters
        assert 'h' in tokenizer.char_to_idx or ' h' in tokenizer.vocab
        assert 'e' in tokenizer.char_to_idx or ' e' in tokenizer.vocab

    def test_character_encode_decode(self):
        """Test Character encode and decode."""
        texts = ["hello world"]

        tokenizer = CharacterTokenizer()
        tokenizer.train(texts)

        # Encode
        text = "hello"
        encoded = tokenizer.encode(text)
        assert isinstance(encoded, list)
        assert len(encoded) == len(text)  # One token per character

        # Decode
        decoded = tokenizer.decode(encoded)
        assert decoded == text or decoded.replace(" ", "") == text.replace(" ", "")

    def test_character_all_chars_encoded(self):
        """Test Character tokenizer encodes all characters."""
        texts = ["abcdefghijklmnopqrstuvwxyz"]

        tokenizer = CharacterTokenizer()
        tokenizer.train(texts)

        # All characters should be encodable
        for char in "abcdefghijklmnopqrstuvwxyz":
            encoded = tokenizer.encode(char)
            assert len(encoded) == 1


class TestTokenizerComparison:
    """Cross-tokenizer comparison tests."""

    def test_all_tokenizers_encode_decode(self):
        """Test all tokenizers can encode and decode."""
        texts = [
            "machine learning is powerful",
            "natural language processing",
        ]

        tokenizers = [
            BPETokenizer(vocab_size=100),
            WordPieceTokenizer(vocab_size=100),
            CharacterTokenizer(),
        ]

        test_text = "machine learning"

        for tokenizer in tokenizers:
            tokenizer.train(texts)
            encoded = tokenizer.encode(test_text)
            decoded = tokenizer.decode(encoded)

            # Check encoding produced tokens
            assert len(encoded) > 0, f"Failed for {tokenizer.__class__.__name__}"

            # Check decoding produces text
            assert isinstance(decoded, str), f"Failed for {tokenizer.__class__.__name__}"
            assert len(decoded) > 0, f"Failed for {tokenizer.__class__.__name__}"

    def test_tokenizer_vocab_sizes(self):
        """Test tokenizers respect vocabulary size limits."""
        texts = ["a b c d e f g h i j k l m n o p q r s t u v w x y z"] * 10

        # BPE with small vocab
        bpe = BPETokenizer(vocab_size=50)
        bpe.train(texts)
        assert len(bpe.vocab) <= 50

        # WordPiece with small vocab
        wp = WordPieceTokenizer(vocab_size=50)
        wp.train(texts)
        assert len(wp.vocab) <= 50

        # Character tokenizer vocab is bounded by character set
        char = CharacterTokenizer()
        char.train(texts)
        # Should have roughly alphabet size + special tokens
        assert len(char.vocab) < 100


class TestTokenizerEdgeCases:
    """Edge case tests for tokenizers."""

    def test_empty_text_encoding(self):
        """Test encoding empty text."""
        texts = ["hello world"]

        tokenizers = [
            BPETokenizer(vocab_size=100),
            WordPieceTokenizer(vocab_size=100),
            CharacterTokenizer(),
        ]

        for tokenizer in tokenizers:
            tokenizer.train(texts)

            # Encode empty string
            encoded = tokenizer.encode("")
            # Should return empty list or just special tokens
            assert isinstance(encoded, list)

    def test_single_character_encoding(self):
        """Test encoding single character."""
        texts = ["abcdefg"]

        tokenizers = [
            BPETokenizer(vocab_size=100),
            WordPieceTokenizer(vocab_size=100),
            CharacterTokenizer(),
        ]

        for tokenizer in tokenizers:
            tokenizer.train(texts)

            # Encode single character
            encoded = tokenizer.encode("a")
            assert len(encoded) >= 1

    def test_repeated_text_encoding(self):
        """Test encoding repeated patterns."""
        texts = ["abc abc abc"]

        tokenizers = [
            BPETokenizer(vocab_size=100),
            WordPieceTokenizer(vocab_size=100),
            CharacterTokenizer(),
        ]

        for tokenizer in tokenizers:
            tokenizer.train(texts)

            # Encode repeated pattern
            encoded = tokenizer.encode("abc abc")
            assert len(encoded) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
