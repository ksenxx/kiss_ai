# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the ``run_parallel`` tool as offered by ``SorcarAgent``.

``run_parallel`` is N ``run_agent`` calls: each child is a daemon
sub-task, so the tool itself is checked here without a daemon — when
it is offered, its signature, and the refusals it shares with
``run_agent``.

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import inspect

from kiss.agents.sorcar.agent_dispatch import make_run_agent_tool, make_run_parallel_tool
from kiss.agents.sorcar.sorcar_agent import SorcarAgent


class TestRunParallelTool:
    """The ``run_parallel`` tool of a ``SorcarAgent``."""

    def test_run_parallel_tool_not_available_when_disabled(self) -> None:
        """run_parallel is NOT in tool list when is_parallel=False."""
        agent = SorcarAgent("test-no-parallel")
        agent._use_web_tools = False
        agent._is_parallel = False
        tools = agent._get_tools()
        names = [getattr(t, "__name__", "") for t in tools]
        assert "run_parallel" not in names

    def test_run_parallel_tool_available_when_enabled(self) -> None:
        """run_parallel IS in tool list when is_parallel=True."""
        agent = SorcarAgent("test-yes-parallel")
        agent._use_web_tools = False
        agent._is_parallel = True
        tools = agent._get_tools()
        names = [getattr(t, "__name__", "") for t in tools]
        assert "run_parallel" in names

    def test_run_parallel_tool_signature(self) -> None:
        """The run_parallel tool takes run_agent's arguments and no max_workers."""
        agent = SorcarAgent("test-sig")
        agent._use_web_tools = False
        agent._is_parallel = True
        tools = agent._get_tools()
        rp = [t for t in tools if getattr(t, "__name__", "") == "run_parallel"][0]
        params = list(inspect.signature(rp).parameters.keys())
        assert params == [
            "tasks", "agent", "model", "tool_profile", "max_budget", "timeout", "options",
        ]

    def test_empty_tasks_array_is_refused(self) -> None:
        """An empty JSON array starts nothing and returns the empty-array error."""
        run_parallel = make_run_parallel_tool("/tmp")
        assert run_parallel("[]") == "Error: tasks is an empty array; nothing to run."

    def test_unknown_agent_is_refused_like_run_agent(self) -> None:
        """An unknown agent gets the same error from run_parallel as from run_agent."""
        run_parallel = make_run_parallel_tool("/tmp")
        run_agent = make_run_agent_tool("/tmp")
        parallel_error = run_parallel('["x"]', agent="no_such_agent_zz")
        assert parallel_error.startswith("Error: unknown agent 'no_such_agent_zz'")
        assert parallel_error == run_agent("x", agent="no_such_agent_zz")
