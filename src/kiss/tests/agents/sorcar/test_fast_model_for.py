"""Unit tests for fast_model_for() API-key-based fast model selection."""

from __future__ import annotations

import pytest

from kiss.agents.vscode.helpers import fast_model_for


class TestFastModelFor:
    """Verify fast_model_for() selects the correct fast model per available API key."""

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
        # Force Config to re-read env vars
        from kiss.core import config as _cfg

        monkeypatch.setattr(_cfg, "DEFAULT_CONFIG", _cfg.Config())

    def _set_key(self, monkeypatch: pytest.MonkeyPatch, key: str) -> None:
        monkeypatch.setenv(key, "test-key")
        from kiss.core import config as _cfg

        monkeypatch.setattr(_cfg, "DEFAULT_CONFIG", _cfg.Config())

    def test_anthropic_key_returns_haiku(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_key(monkeypatch, "ANTHROPIC_API_KEY")
        assert fast_model_for() == "claude-haiku-4-5"

    def test_openrouter_key_returns_openrouter_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_key(monkeypatch, "OPENROUTER_API_KEY")
        assert fast_model_for() == "openrouter/anthropic/claude-haiku-4.5"

    def test_together_key_returns_together_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_key(monkeypatch, "TOGETHER_API_KEY")
        assert fast_model_for() == "deepseek-ai/DeepSeek-R1-0528"

    def test_gemini_key_returns_gemini_pro(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_key(monkeypatch, "GEMINI_API_KEY")
        assert fast_model_for() == "gemini-2.5-pro"

    def test_openai_key_returns_gpt4o(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_key(monkeypatch, "OPENAI_API_KEY")
        assert fast_model_for() == "gpt-4o"

    def test_no_keys_returns_haiku_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no API keys are set, falls back to claude-haiku-4-5."""
        assert fast_model_for() == "claude-haiku-4-5"

    def test_priority_anthropic_over_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Anthropic key takes priority over Gemini key."""
        self._set_key(monkeypatch, "ANTHROPIC_API_KEY")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from kiss.core import config as _cfg

        monkeypatch.setattr(_cfg, "DEFAULT_CONFIG", _cfg.Config())
        assert fast_model_for() == "claude-haiku-4-5"

    def test_priority_gemini_over_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Gemini key takes priority over OpenAI key."""
        self._set_key(monkeypatch, "GEMINI_API_KEY")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from kiss.core import config as _cfg

        monkeypatch.setattr(_cfg, "DEFAULT_CONFIG", _cfg.Config())
        assert fast_model_for() == "gemini-2.5-pro"

    def test_minimax_only_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MiniMax key alone is not enough — falls back to haiku."""
        self._set_key(monkeypatch, "MINIMAX_API_KEY")
        assert fast_model_for() == "claude-haiku-4-5"
