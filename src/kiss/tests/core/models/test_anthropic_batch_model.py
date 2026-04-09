"""Tests for AnthropicBatchModel — Anthropic Batch API integration.

Tests cover initialization, parameter handling, and an integration test
that submits a real batch request to the Anthropic API.
"""

import pytest

from kiss.core.models.anthropic_batch_model import AnthropicBatchModel
from kiss.core.models.model_info import MODEL_INFO, model
from kiss.tests.conftest import requires_anthropic_api_key


class TestAnthropicBatchModelInit:
    """Tests for AnthropicBatchModel initialization and parameter handling."""

    def test_str_repr(self) -> None:
        """__str__ and __repr__ show the batch model name."""
        m = AnthropicBatchModel("batch/claude-haiku-4-5", api_key="test-key")
        assert "batch/claude-haiku-4-5" in str(m)
        assert "AnthropicBatchModel" in repr(m)

    def test_batch_model_info_entries_exist(self) -> None:
        """All batch/ model entries exist in MODEL_INFO."""
        batch_models = [k for k in MODEL_INFO if k.startswith("batch/")]
        assert len(batch_models) >= 5
        for name in batch_models:
            info = MODEL_INFO[name]
            assert info.is_generation_supported
            assert info.is_function_calling_supported
            # Batch pricing should be ~50% of standard API pricing
            base_name = name.removeprefix("batch/")
            if base_name in MODEL_INFO:
                base_info = MODEL_INFO[base_name]
                assert info.input_price_per_1M == pytest.approx(
                    base_info.input_price_per_1M * 0.5, rel=0.01
                )
                assert info.output_price_per_1M == pytest.approx(
                    base_info.output_price_per_1M * 0.5, rel=0.01
                )

    def test_batch_model_cache_pricing(self) -> None:
        """Batch models get Anthropic-style cache pricing (10% read, 125% write)."""
        info = MODEL_INFO["batch/claude-opus-4-6"]
        assert info.cache_read_price_per_1M is not None
        assert info.cache_read_price_per_1M == pytest.approx(
            info.input_price_per_1M * 0.1, rel=0.01
        )
        assert info.cache_write_price_per_1M is not None
        assert info.cache_write_price_per_1M == pytest.approx(
            info.input_price_per_1M * 1.25, rel=0.01
        )


@requires_anthropic_api_key
class TestAnthropicBatchModelIntegration:
    """Integration tests that make real Anthropic Batch API calls."""

    @pytest.mark.timeout(600)
    def test_batch_generate(self) -> None:
        """Submit a real batch request and verify the response."""
        m = model(
            "batch/claude-haiku-4-5",
            model_config={"poll_interval": 2.0},
        )
        assert isinstance(m, AnthropicBatchModel)
        m.initialize("Reply with exactly the word 'hello' and nothing else.")
        content, response = m.generate()
        assert "hello" in content.lower()
        # Verify token counts are extracted
        inp, out, cr, cw = m.extract_input_output_token_counts_from_response(response)
        assert inp > 0
        assert out > 0
        # Verify conversation was updated
        assert len(m.conversation) == 2
        assert m.conversation[1]["role"] == "assistant"
