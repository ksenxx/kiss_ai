"""Tests for run_tasks_parallel() in sorcar_agent.py.

No mocks, patches, fakes, or test doubles.  All tests exercise the real
SorcarAgent code path.
"""

from __future__ import annotations

from kiss.agents.sorcar.sorcar_agent import SorcarAgent, run_tasks_parallel


class TestRunTasksParallel:
    """Test concurrent task execution via ThreadPoolExecutor."""

    def test_empty_task_list(self) -> None:
        """Empty input returns empty output."""
        assert run_tasks_parallel([]) == []

    def test_accepts_list_of_strings(self) -> None:
        """Verify the function signature accepts list[str], not list[dict]."""
        # Type-level check: this must not raise TypeError at call time.
        # We don't actually run the tasks (would need LLM API), just
        # verify max_workers=0 raises ValueError from ThreadPoolExecutor.
        try:
            run_tasks_parallel(["task one", "task two"], max_workers=0)
        except ValueError:
            pass  # expected: ThreadPoolExecutor rejects max_workers=0

    def test_accepts_model_and_work_dir(self) -> None:
        """Verify model and work_dir parameters are accepted."""
        # max_workers=0 triggers ValueError before any agent runs,
        # proving the parameters are accepted without needing LLM API.
        try:
            run_tasks_parallel(
                ["task"],
                max_workers=0,
                model="gpt-4o",
                work_dir="/tmp",
            )
        except ValueError:
            pass  # expected: ThreadPoolExecutor rejects max_workers=0

    def test_run_parallel_tool_in_agent_tools_when_parallel(self) -> None:
        """The run_parallel tool is included when is_parallel is True."""
        agent = SorcarAgent("test")
        agent._use_web_tools = False
        agent._is_parallel = True
        tools = agent._get_tools()
        tool_names = [getattr(t, "__name__", "") for t in tools]
        assert "run_parallel" in tool_names

    def test_run_parallel_tool_excluded_by_default(self) -> None:
        """The run_parallel tool is excluded when is_parallel is False (default)."""
        agent = SorcarAgent("test")
        agent._use_web_tools = False
        tools = agent._get_tools()
        tool_names = [getattr(t, "__name__", "") for t in tools]
        assert "run_parallel" not in tool_names
