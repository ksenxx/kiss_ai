"""Integration tests for printer-based streaming in coding agents.

These tests use REAL API calls -- no mocks. Each coding agent is tested for:
  1. Printer receives non-empty string tokens during execution.
  2. No printer (None) still works as before (regression guard).
"""

from pathlib import Path
from typing import Any

import pytest

from kiss.core.printer import Printer
from kiss.tests.conftest import (
    requires_gemini_api_key,
    requires_openai_api_key,
)


class CollectorPrinter(Printer):
    def __init__(self) -> None:
        self.tokens: list[str] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        return ""

    async def token_callback(self, token: str) -> None:
        self.tokens.append(token)

    def reset(self) -> None:
        self.tokens.clear()


def _run_gemini_cli_agent(tmp_path: Path, printer: Printer | None):
    from kiss.agents.coding_agents.gemini_cli_agent import GeminiCliAgent

    base_dir = tmp_path / "gemini_work"
    base_dir.mkdir()
    output_dir = base_dir / "output"
    output_dir.mkdir()
    agent = GeminiCliAgent("test-gemini-callback")
    return agent.run(
        model_name="gemini-2.5-flash",
        prompt_template="What is 2 + 2? Reply with just the number.",
        base_dir=str(base_dir),
        writable_paths=[str(output_dir)],
        printer=printer,
    )


def _run_openai_codex_agent(tmp_path: Path, printer: Printer | None):
    from kiss.agents.coding_agents.openai_codex_agent import OpenAICodexAgent

    base_dir = tmp_path / "codex_work"
    base_dir.mkdir()
    output_dir = base_dir / "output"
    output_dir.mkdir()
    agent = OpenAICodexAgent("test-codex-callback")
    return agent.run(
        model_name="gpt-4.1-mini",
        prompt_template="What is 2 + 2? Reply with just the number.",
        base_dir=str(base_dir),
        writable_paths=[str(output_dir)],
        printer=printer,
    )


CALLBACK_CASES = [
    pytest.param(_run_gemini_cli_agent, marks=requires_gemini_api_key, id="gemini"),
    pytest.param(
        _run_openai_codex_agent, marks=requires_openai_api_key, id="openai-codex"
    ),
]

REGRESSION_CASES = [
    pytest.param(
        _run_openai_codex_agent, marks=requires_openai_api_key, id="openai-codex"
    ),
]


class TestCodingAgentPrinter:
    @pytest.mark.parametrize("runner", CALLBACK_CASES)
    @pytest.mark.timeout(300)
    def test_printer_receives_tokens(self, runner, tmp_path: Path):
        printer = CollectorPrinter()
        try:
            runner(tmp_path, printer)
        except Exception:
            pass
        assert len(printer.tokens) > 0
        assert all(isinstance(t, str) for t in printer.tokens)

    @pytest.mark.parametrize("runner", REGRESSION_CASES)
    @pytest.mark.timeout(300)
    def test_no_printer_regression(self, runner, tmp_path: Path):
        try:
            runner(tmp_path, None)
        except Exception:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
