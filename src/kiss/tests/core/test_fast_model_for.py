# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Unit tests for get_fast_model() API-key-based fast model selection."""

from __future__ import annotations

import pytest

from kiss.core.models.model_info import MODEL_INFO, get_fast_model


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
            "ZAI_API_KEY",
            "MOONSHOT_API_KEY",
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
        assert get_fast_model() == "openrouter/anthropic/claude-sonnet-5.5"
        assert get_fast_model() in MODEL_INFO

    def test_together_key_returns_together_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_key(monkeypatch, "TOGETHER_API_KEY")
        assert get_fast_model() == "deepseek-ai/DeepSeek-V4.1-Flash"
        assert get_fast_model() in MODEL_INFO

    def test_openai_key_returns_luna(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_key(monkeypatch, "OPENAI_API_KEY")
        assert get_fast_model() == "gpt-6-luna"
        assert get_fast_model() in MODEL_INFO

    def test_gemini_key_returns_flash_lite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_key(monkeypatch, "GEMINI_API_KEY")
        assert get_fast_model() == "gemini-3.5-flash-lite"
        assert get_fast_model() in MODEL_INFO

    def test_anthropic_key_returns_sonnet(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_key(monkeypatch, "ANTHROPIC_API_KEY")
        assert get_fast_model() == "claude-sonnet-5-5"
        assert get_fast_model() in MODEL_INFO

    def test_no_keys_returns_cli_or_no_model_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No keys and no authenticated CLI result in No model."""
        assert get_fast_model() == "No model"

    def test_priority_openai_over_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OpenAI key takes priority over Gemini key."""
        self._set_key(monkeypatch, "GEMINI_API_KEY")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from kiss.core import config as _cfg

        monkeypatch.setattr(_cfg, "DEFAULT_CONFIG", _cfg.Config())
        assert get_fast_model() == "gpt-6-luna"
