# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: ``run_parallel`` never iterates a bare-string ``tasks``.

An early fan-out iterated *tasks* with ``enumerate(...)``; when the
LLM passed a bare string (e.g. ``"hello"``) instead of a JSON array,
Python iterated it character by character and one sub-agent per
character was spawned.  The ``run_parallel`` tool now parses ``tasks``
strictly (:func:`kiss.agents.sorcar.fanout_guard.parse_tasks_json`)
and refuses anything but a JSON array of strings before a single
child is started.
"""

from __future__ import annotations

from kiss.agents.sorcar.sorcar_agent import SorcarAgent


class TestRunParallelClosureRejectsBareString:
    """The ``run_parallel`` tool exposed to the LLM never iterates a string.

    A bare-string ``tasks`` argument is refused outright with an
    ``Error:`` result (see :mod:`kiss.agents.sorcar.fanout_guard`), so
    no sub-agent — let alone one per character — is ever spawned.
    """

    def test_run_parallel_tool_rejects_bare_string(self) -> None:
        agent = SorcarAgent("test-string-bug")
        agent._use_web_tools = False
        agent._is_parallel = True
        tools = agent._get_tools()
        run_parallel = next(
            t for t in tools if getattr(t, "__name__", "") == "run_parallel"
        )

        result = run_parallel("hello world")
        assert result.startswith("Error: tasks must be a JSON array")
