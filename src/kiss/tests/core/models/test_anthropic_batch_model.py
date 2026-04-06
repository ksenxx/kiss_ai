"""Tests for AnthropicBatchModel — Anthropic Batch API integration.

Tests cover initialization, parameter handling, and an integration test
that submits a real batch request to the Anthropic API.
"""

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_batch_model import AnthropicBatchModel
from kiss.core.models.model_info import MODEL_INFO, calculate_cost, model
from kiss.tests.conftest import requires_anthropic_api_key


class TestAnthropicBatchModelInit:
    """Tests for AnthropicBatchModel initialization and parameter handling."""

    def test_model_factory_creates_batch_model(self) -> None:
        """model('batch/...') returns an AnthropicBatchModel."""
        m = model("batch/claude-haiku-4-5")
        assert isinstance(m, AnthropicBatchModel)

    def test_model_name_keeps_batch_prefix(self) -> None:
        """model_name keeps the 'batch/' prefix for correct pricing lookup."""
        m = AnthropicBatchModel("batch/claude-opus-4-6", api_key="test-key")
        assert m.model_name == "batch/claude-opus-4-6"
        assert m._api_model_name == "claude-opus-4-6"

    def test_str_repr(self) -> None:
        """__str__ and __repr__ show the batch model name."""
        m = AnthropicBatchModel("batch/claude-haiku-4-5", api_key="test-key")
        assert "batch/claude-haiku-4-5" in str(m)
        assert "AnthropicBatchModel" in repr(m)

    def test_token_callback_ignored(self) -> None:
        """token_callback is set to None since batch API doesn't stream."""
        calls: list[str] = []
        m = AnthropicBatchModel(
            "batch/claude-haiku-4-5",
            api_key="test-key",
            token_callback=lambda t: calls.append(t),
        )
        assert m.token_callback is None

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

    def test_calculate_cost_for_batch_model(self) -> None:
        """calculate_cost works for batch/ models with 50% pricing."""
        # 200K input + 60K output for batch/claude-opus-4-6
        # input: 200K * 2.50 / 1M = 0.50, output: 60K * 12.50 / 1M = 0.75
        cost = calculate_cost("batch/claude-opus-4-6", 200_000, 60_000)
        expected = (200_000 * 2.50 + 60_000 * 12.50) / 1_000_000
        assert cost == pytest.approx(expected, rel=0.01)

    def test_build_create_kwargs_inherits_anthropic(self) -> None:
        """_build_create_kwargs works correctly (inherited from AnthropicModel).

        The inherited method uses self.model_name which is now the batch-prefixed
        name, but _create_message overrides the 'model' key with _api_model_name.
        """
        m = AnthropicBatchModel("batch/claude-haiku-4-5", api_key="test-key")
        m.initialize("Hello")
        kwargs = m._build_create_kwargs()
        assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert "max_tokens" in kwargs

    def test_get_embedding_raises(self) -> None:
        """get_embedding raises KISSError (Anthropic has no embedding API)."""
        m = AnthropicBatchModel("batch/claude-haiku-4-5", api_key="test-key")
        m.initialize("test")
        with pytest.raises(KISSError, match="(?i)embedding"):
            m.get_embedding("test text")


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
