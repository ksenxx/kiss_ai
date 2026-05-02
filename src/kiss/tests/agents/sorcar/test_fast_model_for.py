"""Unit tests for get_fast_model() API-key-based fast model selection."""

from __future__ import annotations

import pytest

from kiss.core.models.model_info import get_fast_model


class TestFastModelFor:
    """Verify get_fast_model() selects the correct fast model per available API key."""

    @pytest.fixture(autouse=True)
    def _clear_api_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure all API key env vars are unset before each test."""
        for key in (
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "TOGETHER_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "MINIMAX_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        from kiss.core import config as _cfg

        monkeypatch.setattr(_cfg, "DEFAULT_CONFIG", _cfg.Config())

    def _set_key(self, monkeypatch: pytest.MonkeyPatch, key: str) -> None:
        monkeypatch.setenv(key, "test-key")
        from kiss.core import config as _cfg

        monkeypatch.setattr(_cfg, "DEFAULT_CONFIG", _cfg.Config())

    def test_openrouter_key_returns_openrouter_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_key(monkeypatch, "OPENROUTER_API_KEY")
        assert get_fast_model() == "openrouter/anthropic/claude-haiku-4.5"

    def test_together_key_returns_together_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_key(monkeypatch, "TOGETHER_API_KEY")
        assert get_fast_model() == "deepseek-ai/DeepSeek-R1-0528"

    def test_openai_key_returns_gpt4o(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_key(monkeypatch, "OPENAI_API_KEY")
        assert get_fast_model() == "gpt-4o"

    def test_no_keys_returns_haiku_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no API keys are set, falls back to cc/haiku or claude-haiku-4-5."""
        import shutil

        if shutil.which("claude") is not None:
            assert get_fast_model() == "cc/haiku"
        else:
            assert get_fast_model() == "claude-haiku-4-5"

    def test_priority_openai_over_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OpenAI key takes priority over Gemini key."""
        self._set_key(monkeypatch, "GEMINI_API_KEY")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from kiss.core import config as _cfg

        monkeypatch.setattr(_cfg, "DEFAULT_CONFIG", _cfg.Config())
        assert get_fast_model() == "gpt-4o"
