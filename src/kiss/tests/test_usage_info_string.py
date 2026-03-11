"""Tests for _get_usage_info_string with session_info and token capping."""

from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import get_max_context_length


class TestUsageInfoString:
    """Verify usage info includes session info and caps tokens."""

    def _make_agent(self, session_info: str = "") -> KISSAgent:
        agent = KISSAgent("test-agent")
        agent._reset(
            model_name="claude-sonnet-4-20250514",
            is_agentic=True,
            max_steps=100,
            max_budget=200.0,
            model_config=None,
            session_info=session_info,
        )
        return agent

    def test_session_info_included(self) -> None:
        agent = self._make_agent(session_info="Session: 2/5")
        result = agent._get_usage_info_string()
        assert result.startswith("Session: 2/5, ")
        assert "Steps:" in result
        assert "Tokens:" in result

    def test_no_session_info(self) -> None:
        agent = self._make_agent(session_info="")
        result = agent._get_usage_info_string()
        assert result.startswith("Steps:")
        assert "Session" not in result

    def test_tokens_capped_at_max_context(self) -> None:
        agent = self._make_agent()
        max_tokens = get_max_context_length("claude-sonnet-4-20250514")
        agent.total_tokens_used = max_tokens + 50000
        result = agent._get_usage_info_string()
        # Token display should be capped at max, not show the larger value
        assert f"Tokens: {max_tokens}/{max_tokens}" in result

    def test_tokens_below_max_shown_as_is(self) -> None:
        agent = self._make_agent()
        agent.total_tokens_used = 5000
        result = agent._get_usage_info_string()
        max_tokens = get_max_context_length("claude-sonnet-4-20250514")
        assert f"Tokens: 5000/{max_tokens}" in result

    def test_session_info_with_capped_tokens(self) -> None:
        agent = self._make_agent(session_info="Session: 1/3")
        max_tokens = get_max_context_length("claude-sonnet-4-20250514")
        agent.total_tokens_used = max_tokens + 100
        result = agent._get_usage_info_string()
        assert result.startswith("Session: 1/3, ")
        assert f"Tokens: {max_tokens}/{max_tokens}" in result
