"""Integration tests for the async token_callback in coding agents.

These tests use REAL API calls -- no mocks.  Each coding agent is tested for:
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_collector() -> tuple[TokenCallback, list[str]]:
    """Return an async callback and the list it appends tokens to."""
    tokens: list[str] = []

    async def _callback(token: str) -> None:
        tokens.append(token)

    return _callback, tokens


# ---------------------------------------------------------------------------
# KISSCodingAgent tests
# ---------------------------------------------------------------------------


@requires_openai_api_key
class TestKISSCodingAgentTokenCallback:
    """Token callback tests for KISSCodingAgent."""

    @pytest.fixture(autouse=True)
    def _setup_dirs(self, tmp_path: Path):
        self.work_dir = tmp_path / "work"
        self.work_dir.mkdir()

    @pytest.mark.timeout(300)
    def test_callback_receives_tokens(self):
        """KISSCodingAgent.run() should invoke the callback with streamed tokens."""
        from kiss.agents.coding_agents.kiss_coding_agent import KISSCodingAgent

        callback, tokens = _make_collector()
        agent = KISSCodingAgent("test-kca-callback")
        result = agent.run(
            prompt_template="Write a Python function that returns 42. Then finish.",
            work_dir=str(self.work_dir),
            orchestrator_model_name="gpt-4.1-mini",
            subtasker_model_name="gpt-4.1-mini",
            refiner_model_name="gpt-4.1-mini",
            max_steps=10,
            max_budget=0.50,
            trials=1,
            token_callback=callback,
        )
        assert result is not None
        assert len(tokens) > 0, "Expected at least one callback invocation"
        assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.timeout(300)
    def test_no_callback_regression(self):
        """KISSCodingAgent.run() without callback should still work."""
        from kiss.agents.coding_agents.kiss_coding_agent import KISSCodingAgent

        agent = KISSCodingAgent("test-kca-no-callback")
        result = agent.run(
            prompt_template="Write a Python function that returns 42. Then finish.",
            work_dir=str(self.work_dir),
            orchestrator_model_name="gpt-4.1-mini",
            subtasker_model_name="gpt-4.1-mini",
            refiner_model_name="gpt-4.1-mini",
            max_steps=10,
            max_budget=0.50,
            trials=1,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# RelentlessCodingAgent tests
# ---------------------------------------------------------------------------


@requires_openai_api_key
class TestRelentlessCodingAgentTokenCallback:
    """Token callback tests for RelentlessCodingAgent."""

    @pytest.fixture(autouse=True)
    def _setup_dirs(self, tmp_path: Path):
        self.work_dir = tmp_path / "work"
        self.work_dir.mkdir()

    @pytest.mark.timeout(300)
    def test_callback_receives_tokens(self):
        """RelentlessCodingAgent.run() should invoke the callback with streamed tokens."""
        from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

        callback, tokens = _make_collector()
        agent = RelentlessCodingAgent("test-rca-callback")
        result = agent.run(
            prompt_template="Write a Python function that returns 42. Then finish.",
            work_dir=str(self.work_dir),
            orchestrator_model_name="gpt-4.1-mini",
            subtasker_model_name="gpt-4.1-mini",
            max_steps=10,
            max_budget=0.50,
            trials=1,
            token_callback=callback,
        )
        assert result is not None
        assert len(tokens) > 0, "Expected at least one callback invocation"
        assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.timeout(300)
    def test_no_callback_regression(self):
        """RelentlessCodingAgent.run() without callback should still work."""
        from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

        agent = RelentlessCodingAgent("test-rca-no-callback")
        result = agent.run(
            prompt_template="Write a Python function that returns 42. Then finish.",
            work_dir=str(self.work_dir),
            orchestrator_model_name="gpt-4.1-mini",
            subtasker_model_name="gpt-4.1-mini",
            max_steps=10,
            max_budget=0.50,
            trials=1,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# ClaudeCodingAgent tests
# ---------------------------------------------------------------------------


@requires_anthropic_api_key
class TestClaudeCodingAgentTokenCallback:
    """Token callback tests for ClaudeCodingAgent."""

    @pytest.fixture(autouse=True)
    def _setup_dirs(self, tmp_path: Path):
        self.temp_dir = tmp_path / "claude_work"
        self.temp_dir.mkdir()
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()

    @pytest.mark.timeout(300)
    def test_callback_receives_tokens(self):
        """ClaudeCodingAgent.run() should invoke the callback with streamed tokens."""
        from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent

        callback, tokens = _make_collector()
        agent = ClaudeCodingAgent("test-claude-callback")
        result = agent.run(
            model_name="claude-sonnet-4-5",
            prompt_template="What is 2 + 2? Reply with just the number.",
            base_dir=str(self.temp_dir),
            writable_paths=[str(self.output_dir)],
            token_callback=callback,
        )
        assert result is not None
        assert len(tokens) > 0, "Expected at least one callback invocation"
        assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.timeout(300)
    def test_no_callback_regression(self):
        """ClaudeCodingAgent.run() without callback should still work."""
        from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent

        agent = ClaudeCodingAgent("test-claude-no-callback")
        result = agent.run(
            model_name="claude-sonnet-4-5",
            prompt_template="What is 2 + 2? Reply with just the number.",
            base_dir=str(self.temp_dir),
            writable_paths=[str(self.output_dir)],
        )
        assert result is not None


# ---------------------------------------------------------------------------
# GeminiCliAgent tests
# ---------------------------------------------------------------------------


@requires_gemini_api_key
class TestGeminiCliAgentTokenCallback:
    """Token callback tests for GeminiCliAgent."""

    @pytest.fixture(autouse=True)
    def _setup_dirs(self, tmp_path: Path):
        self.temp_dir = tmp_path / "gemini_work"
        self.temp_dir.mkdir()
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()

    @pytest.mark.timeout(300)
    def test_callback_receives_tokens(self):
        """GeminiCliAgent.run() should invoke the callback with streamed tokens."""
        from kiss.agents.coding_agents.gemini_cli_agent import GeminiCliAgent

        callback, tokens = _make_collector()
        agent = GeminiCliAgent("test_gemini_callback")
        result = agent.run(
            model_name="gemini-2.5-flash",
            prompt_template="What is 2 + 2? Reply with just the number.",
            base_dir=str(self.temp_dir),
            writable_paths=[str(self.output_dir)],
            token_callback=callback,
        )
        assert result is not None
        assert len(tokens) > 0, "Expected at least one callback invocation"
        assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.timeout(300)
    def test_no_callback_regression(self):
        """GeminiCliAgent.run() without callback should still work."""
        from kiss.agents.coding_agents.gemini_cli_agent import GeminiCliAgent

        agent = GeminiCliAgent("test_gemini_no_callback")
        result = agent.run(
            model_name="gemini-2.5-flash",
            prompt_template="What is 2 + 2? Reply with just the number.",
            base_dir=str(self.temp_dir),
            writable_paths=[str(self.output_dir)],
        )
        assert result is not None


# ---------------------------------------------------------------------------
# OpenAICodexAgent tests
# ---------------------------------------------------------------------------


@requires_openai_api_key
class TestOpenAICodexAgentTokenCallback:
    """Token callback tests for OpenAICodexAgent."""

    @pytest.fixture(autouse=True)
    def _setup_dirs(self, tmp_path: Path):
        self.temp_dir = tmp_path / "codex_work"
        self.temp_dir.mkdir()
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()

    @pytest.mark.timeout(300)
    def test_callback_receives_tokens(self):
        """OpenAICodexAgent.run() should invoke the callback with streamed tokens."""
        from kiss.agents.coding_agents.openai_codex_agent import OpenAICodexAgent

        callback, tokens = _make_collector()
        agent = OpenAICodexAgent("test-codex-callback")
        result = agent.run(
            model_name="gpt-4.1-mini",
            prompt_template="What is 2 + 2? Reply with just the number.",
            base_dir=str(self.temp_dir),
            writable_paths=[str(self.output_dir)],
            token_callback=callback,
        )
        assert result is not None
        assert len(tokens) > 0, "Expected at least one callback invocation"
        assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.timeout(300)
    def test_no_callback_regression(self):
        """OpenAICodexAgent.run() without callback should still work."""
        from kiss.agents.coding_agents.openai_codex_agent import OpenAICodexAgent

        agent = OpenAICodexAgent("test-codex-no-callback")
        result = agent.run(
            model_name="gpt-4.1-mini",
            prompt_template="What is 2 + 2? Reply with just the number.",
            base_dir=str(self.temp_dir),
            writable_paths=[str(self.output_dir)],
        )
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
