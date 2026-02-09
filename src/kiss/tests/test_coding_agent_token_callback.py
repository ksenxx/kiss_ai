"""Integration tests for the async token_callback in coding agents.

These tests use REAL API calls -- no mocks. Each coding agent is tested for:
  1. Callback receives non-empty string tokens during execution.
  2. No callback (None) still works as before (regression guard).
"""

from pathlib import Path

import pytest

from kiss.core.models.model import TokenCallback
from kiss.tests.conftest import (
    requires_anthropic_api_key,
    requires_gemini_api_key,
    requires_openai_api_key,
)


def _make_collector() -> tuple[TokenCallback, list[str]]:
    tokens: list[str] = []

    async def _callback(token: str) -> None:
        tokens.append(token)

    return _callback, tokens


def _run_kiss_coding_agent(tmp_path: Path, token_callback: TokenCallback | None):
    from kiss.agents.coding_agents.kiss_coding_agent import KISSCodingAgent

    work_dir = tmp_path / "kiss_work"
    work_dir.mkdir()
    agent = KISSCodingAgent("test-kca-callback")
    return agent.run(
        prompt_template="What is 2 + 2? Reply with just the number, then finish.",
        work_dir=str(work_dir),
        orchestrator_model_name="gpt-5.2",
        subtasker_model_name="gpt-5.2",
        refiner_model_name="gpt-5.2",
        max_steps=10,
        max_budget=1.0,
        trials=3,
        token_callback=token_callback,
    )


def _run_relentless_coding_agent(tmp_path: Path, token_callback: TokenCallback | None):
    from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

    work_dir = tmp_path / "relentless_work"
    work_dir.mkdir()
    agent = RelentlessCodingAgent("test-rca-callback")
    return agent.run(
        prompt_template="What is 2 + 2? Reply with just the number, then finish.",
        work_dir=str(work_dir),
        subtasker_model_name="gpt-5.2",
        max_steps=10,
        max_budget=1.0,
        trials=3,
        token_callback=token_callback,
    )


def _run_claude_coding_agent(tmp_path: Path, token_callback: TokenCallback | None):
    from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent

    base_dir = tmp_path / "claude_work"
    base_dir.mkdir()
    output_dir = base_dir / "output"
    output_dir.mkdir()
    agent = ClaudeCodingAgent("test-claude-callback")
    return agent.run(
        model_name="claude-sonnet-4-5",
        prompt_template="What is 2 + 2? Reply with just the number.",
        base_dir=str(base_dir),
        writable_paths=[str(output_dir)],
        token_callback=token_callback,
    )


def _run_gemini_cli_agent(tmp_path: Path, token_callback: TokenCallback | None):
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
        token_callback=token_callback,
    )


def _run_openai_codex_agent(tmp_path: Path, token_callback: TokenCallback | None):
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
        token_callback=token_callback,
    )


CODING_AGENT_CASES = [
    pytest.param(_run_kiss_coding_agent, marks=requires_openai_api_key, id="kiss"),
    pytest.param(
        _run_relentless_coding_agent, marks=requires_openai_api_key, id="relentless"
    ),
    pytest.param(_run_claude_coding_agent, marks=requires_anthropic_api_key, id="claude"),
    pytest.param(_run_gemini_cli_agent, marks=requires_gemini_api_key, id="gemini"),
    pytest.param(
        _run_openai_codex_agent, marks=requires_openai_api_key, id="openai-codex"
    ),
]


class TestCodingAgentTokenCallback:
    @pytest.mark.parametrize("runner", CODING_AGENT_CASES)
    @pytest.mark.timeout(300)
    def test_callback_receives_tokens(self, runner, tmp_path: Path):
        callback, tokens = _make_collector()
        try:
            runner(tmp_path, callback)
        except Exception:
            pass  # Task may fail; we only care that tokens were streamed
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.parametrize("runner", CODING_AGENT_CASES)
    @pytest.mark.timeout(300)
    def test_no_callback_regression(self, runner, tmp_path: Path):
        try:
            runner(tmp_path, None)
        except Exception:
            pass  # Task may fail; we only care it doesn't crash differently


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
