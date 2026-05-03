"""Tests that default and fast model names are NOT hardcoded in vscode agent files.

They must be obtained dynamically from kiss.core.models.model_info.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kiss.core.models.model_info import get_default_model, get_fast_model

VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"


def _read(rel_path: str) -> str:
    return (VSCODE_DIR / rel_path).read_text()


# Known model names from the Python canonical source that must NOT appear
# as hardcoded literals in the TS/JS frontend.
_DEFAULT_MODELS = [
    "claude-opus-4-7",
    "gpt-5.5",
    "gemini-3.1-pro-preview",
    "openrouter/anthropic/claude-opus-4.7",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
]

_FAST_MODELS = [
    "claude-haiku-4-5",
    "gpt-4o",
    "gemini-2.0-flash",
    "openrouter/anthropic/claude-haiku-4.5",
    "deepseek-ai/DeepSeek-R1-0528",
]


class TestNoHardcodedModelsInDependencyInstaller:
    """DependencyInstaller.ts getDefaultModel() must not hardcode model names."""

    def test_no_hardcoded_default_models(self) -> None:
        src = _read("src/DependencyInstaller.ts")
        # Extract the getDefaultModel function body
        match = re.search(
            r"export function getDefaultModel\(\).*?\{(.*?)\n\}",
            src,
            re.DOTALL,
        )
        assert match, "getDefaultModel() not found in DependencyInstaller.ts"
        body = match.group(1)
        for model in _DEFAULT_MODELS:
            assert (
                f"'{model}'" not in body and f'"{model}"' not in body
            ), f"Hardcoded model '{model}' found in getDefaultModel() body"

    def test_calls_python_get_default_model(self) -> None:
        src = _read("src/DependencyInstaller.ts")
        match = re.search(
            r"export function getDefaultModel\(\).*?\{(.*?)\n\}",
            src,
            re.DOTALL,
        )
        assert match, "getDefaultModel() not found"
        body = match.group(1)
        assert "get_default_model" in body, (
            "getDefaultModel() must call Python's get_default_model()"
        )

    def test_uses_findUvPath_and_findKissProject(self) -> None:
    def test_uses_findUvPath_and_findKissProject(self) -> None:
    def test_uses_find_uv_path_and_find_kiss_project(self) -> None:
        src = _read("src/DependencyInstaller.ts")
        match = re.search(
            r"export function getDefaultModel\(\).*?\{(.*?)\n\}",
            src,
            re.DOTALL,
        )
        assert match
        body = match.group(1)
        assert "findUvPath" in body, "Must use findUvPath()"
        assert "findKissProject" in body, "Must use findKissProject()"


class TestNoHardcodedModelsInMainJs:
    """main.js must not hardcode default or fast model names."""

    def test_no_hardcoded_initial_selectedModel(self) -> None:
    def test_no_hardcoded_initial_selectedModel(self) -> None:
    def test_no_hardcoded_initial_selected_model(self) -> None:
        src = _read("media/main.js")
        # Check that the initial selectedModel declaration doesn't use a model name
        pattern = re.compile(r"let selectedModel\s*=\s*'[a-zA-Z]")
        assert not pattern.search(src), (
            "selectedModel must not be initialized with a hardcoded model name"
        )

    def test_no_hardcoded_restoreTab_fallback(self) -> None:
    def test_no_hardcoded_restore_tab_fallback(self) -> None:
    def test_no_hardcoded_restoreTab_fallback(self) -> None:
        src = _read("media/main.js")
        for model in _DEFAULT_MODELS:
            # Check for patterns like: tab.selectedModel || 'claude-opus-4-7'
            assert f"|| '{model}'" not in src, (
                f"Hardcoded fallback '{model}' in restoreTab"
            )

    def test_selectedModel_initialized_from_dom(self) -> None:
    def test_selected_model_initialized_from_dom(self) -> None:
        src = _read("media/main.js")
    def test_selectedModel_initialized_from_dom(self) -> None:
        # Should read the model name from the DOM element injected by the template
        assert "modelName" in src and "selectedModel" in src
        # Check that there's code reading from model-name element
        assert re.search(
            r"modelName.*textContent.*selectedModel|selectedModel.*modelName.*textContent",
            src,
        ), "selectedModel should be initialized from DOM model-name element"


class TestNoHardcodedModelsAnywhere:
    """No default/fast model names should appear as hardcoded string literals
    in any TS or JS file (excluding node_modules)."""

    @pytest.fixture()
    def ts_js_files(self) -> list[Path]:
        result = []
        result: list[Path] = []
        for pattern in ("src/*.ts", "media/main.js"):
            result.extend(VSCODE_DIR.glob(pattern))
        result = []

    def test_no_hardcoded_default_model_literals(
        self, ts_js_files: list[Path]
    ) -> None:
        for fp in ts_js_files:
            src = fp.read_text()
            for model in _DEFAULT_MODELS:
                # Allow model names in comments (docstrings) but not in code
                for line in src.splitlines():
                    stripped = line.lstrip()
                    if stripped.startswith("//") or stripped.startswith("*"):
                        continue
                    assert f"'{model}'" not in line and f'"{model}"' not in line, (
                        f"Hardcoded model '{model}' in {fp.name}: {line.strip()}"
                    )

    def test_python_get_default_model_returns_known_model(self) -> None:
        result = get_default_model()
        # Should return a non-empty string (either a real model or "No model")
        assert result, "get_default_model() returned empty string"

    def test_python_get_fast_model_returns_known_model(self) -> None:
        result = get_fast_model()
        assert result, "get_fast_model() returned empty string"
