# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the ``run_parallel`` tool offering in sorcar_agent.py.

No mocks, patches, fakes, or test doubles.  All tests exercise the real
SorcarAgent code path.
"""

from __future__ import annotations

from kiss.agents.sorcar.sorcar_agent import SorcarAgent


class TestRunParallelTool:
    """``run_parallel`` is offered exactly when ``_is_parallel`` is true."""

    def test_run_parallel_tool_in_agent_tools_when_parallel(self) -> None:
        """The run_parallel tool is included when is_parallel is True."""
        agent = SorcarAgent("test")
        agent._use_web_tools = False
        agent._is_parallel = True
        tools = agent._get_tools()
        tool_names = [getattr(t, "__name__", "") for t in tools]
        assert "run_parallel" in tool_names

    def test_run_parallel_tool_included_by_default(self) -> None:
        """The run_parallel tool is included by default (is_parallel=True)."""
        agent = SorcarAgent("test")
        agent._use_web_tools = False
        tools = agent._get_tools()
        tool_names = [getattr(t, "__name__", "") for t in tools]
        assert "run_parallel" in tool_names

    def test_run_parallel_tool_excluded_when_disabled(self) -> None:
        """The run_parallel tool is excluded when is_parallel is False."""
        agent = SorcarAgent("test")
        agent._use_web_tools = False
        agent._is_parallel = False
        tools = agent._get_tools()
        tool_names = [getattr(t, "__name__", "") for t in tools]
        assert "run_parallel" not in tool_names
