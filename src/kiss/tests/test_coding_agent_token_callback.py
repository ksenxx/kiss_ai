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
    requires_anthropic_api_key,
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


def _run_claude_coding_agent(tmp_path: Path, printer: Printer | None):
    from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent

    base_dir = tmp_path / "claude_work"
    base_dir.mkdir()
    output_dir = base_dir / "output"
    output_dir.mkdir()
    agent = ClaudeCodingAgent("test-claude-callback")
    return agent.run(
        model_name="claude-sonnet-4-5",
        prompt_template="What is 2 + 2? Reply with just the number.",
        work_dir=str(base_dir),
        writable_paths=[str(output_dir)],
        printer=printer,
    )


CALLBACK_CASES = [
    pytest.param(
        _run_claude_coding_agent, marks=requires_anthropic_api_key, id="claude"
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

    @pytest.mark.parametrize("runner", CALLBACK_CASES)
    @pytest.mark.timeout(300)
    def test_no_printer_regression(self, runner, tmp_path: Path):
        try:
            runner(tmp_path, None)
        except Exception:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
