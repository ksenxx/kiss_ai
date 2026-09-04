# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test suite for model implementation coverage.

These tests verify the actual model implementations (AnthropicModel, GeminiModel,
OpenAICompatibleModel) using real API calls. No mocks are used.
"""

import pytest

from kiss.core import config as config_module
from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model_info import (
    calculate_cost,
    model,
)
from kiss.core.models.openai_compatible_model import OpenAICompatibleBase
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
from kiss.tests.conftest import (
    requires_anthropic_api_key,
    requires_openai_api_key,
)


@requires_anthropic_api_key
class TestAnthropicModel:

    @pytest.mark.timeout(60)
    def test_get_embedding_raises_error(self):
        m = model("claude-haiku-4-5")
        assert isinstance(m, AnthropicModel)
        m.initialize("test")
        with pytest.raises(KISSError, match="(?i)embedding"):
            m.get_embedding("test text")


class TestModelInfo:

    def test_zai_and_moonshot_api_key_routing(self):
        from kiss.tests.conftest import get_required_api_key_for_model

        assert get_required_api_key_for_model("glm-4.6") == "ZAI_API_KEY"
        assert get_required_api_key_for_model("glm-4.5-flash") == "ZAI_API_KEY"
        assert get_required_api_key_for_model("kimi-k2.6") == "MOONSHOT_API_KEY"
        assert get_required_api_key_for_model("moonshot-v1-32k") == "MOONSHOT_API_KEY"

    def test_native_provider_and_keyless_routing(self):
        """The non-OpenAI-compatible branches route the same way ``model()`` does."""
        from kiss.tests.conftest import get_required_api_key_for_model

        assert get_required_api_key_for_model("claude-fable-5") == "ANTHROPIC_API_KEY"
        assert get_required_api_key_for_model("gemini-3-pro") == "GEMINI_API_KEY"
        # The subscription CLIs authenticate through a local executable,
        # so a test asking for one must not be skipped for a missing key.
        assert get_required_api_key_for_model("cc/opus-4-8") is None
        assert get_required_api_key_for_model("codex/gpt-5.6-sol") is None
        assert get_required_api_key_for_model("no-such-vendor/model") is None


class TestCachePricing:

    def test_calculate_cost_unknown_model_with_cache_tokens(self):
        with pytest.raises(KISSError, match="unknown model"):
            calculate_cost("unknown-model-xyz", 1000, 1000, 500, 500)
        assert calculate_cost("unknown-model-xyz", 0, 0, 0, 0) == 0.0


class TestModelConfigBaseUrlOverride:

    @pytest.mark.timeout(60)
    @requires_openai_api_key
    def test_base_url_and_api_key_override_calls_endpoint_and_returns_response(self):
        api_key = config_module.DEFAULT_CONFIG.OPENAI_API_KEY
        m = model(
            "gpt-4.1-mini",
            model_config={
                "base_url": "https://api.openai.com/v1",
                "api_key": api_key,
            },
        )
        # gpt-4.1-mini carries the live-verified catalog use_responses_api
        # flag and the override IS the registered OpenAI endpoint, so the
        # factory builds the v2 Responses adapter here.
        assert isinstance(m, OpenAICompatibleBase)
        m.initialize("Reply with exactly the word OK and nothing else.")
        text, _ = m.generate()
        assert isinstance(text, str)
        assert len(text) > 0
        assert "ok" in text.lower().strip()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
