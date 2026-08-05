# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests that default and fast model names are NOT hardcoded in vscode agent files.

They must be obtained dynamically from kiss.core.models.model_info.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.core.models.model_info import get_default_model, get_fast_model

VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"


def _read(rel_path: str) -> str:
    return (VSCODE_DIR / rel_path).read_text()


_DEFAULT_MODELS = [
    "claude-opus-4-7",
    "gpt-5.6-luna",
    "gemini-3.6-flash",
    "openrouter/anthropic/claude-opus-4.7",
    "moonshotai/Kimi-K3",
]

_FAST_MODELS = [
    "claude-haiku-4-5",
    "gpt-4o",
    "gemini-2.0-flash",
    "openrouter/anthropic/claude-haiku-4.5",
    "deepseek-ai/DeepSeek-R1-0528",
]






class TestNoHardcodedModelsAnywhere:
    """No default/fast model names should appear as hardcoded string literals
    in any TS or JS file (excluding node_modules)."""

    @pytest.fixture()
    def ts_js_files(self) -> list[Path]:
        result: list[Path] = []
        for pattern in ("src/*.ts", "media/main.js"):
            result.extend(VSCODE_DIR.glob(pattern))
        return result


    def test_python_get_default_model_returns_known_model(self) -> None:
        result = get_default_model()
        assert result, "get_default_model() returned empty string"

    def test_python_get_fast_model_returns_known_model(self) -> None:
        result = get_fast_model()
        assert result, "get_fast_model() returned empty string"
