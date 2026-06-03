# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for ``_coerce_tasks`` (str-vs-list[str] handling).

``run_tasks_parallel(tasks, ...)`` originally iterated *tasks* with
``enumerate(...)``.  When the LLM mistakenly passed a bare string
(e.g. ``"hello"``) instead of a list, Python iterated the string
character-by-character and the function spawned one sub-agent per
character.

The fix routes every entry point through :func:`_coerce_tasks`:

* JSON-encoded list strings such as ``'["a", "b"]'`` are parsed into
  ``["a", "b"]``.
* Bare strings (including strings that *happen* to start with ``[`` but
  are not valid JSON) are wrapped into a one-element list.
* Other non-``list[str]`` inputs raise :class:`TypeError`.

These tests verify the coercion behaviour directly — the wider
character-iteration bug is impossible whenever ``_coerce_tasks`` returns
a list of length 1 for a bare string and the right list for a JSON
payload, regardless of how the resulting list is later dispatched.
"""

from __future__ import annotations

import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _coerce_tasks


class TestCoerceTasks:
    """``_coerce_tasks`` normalises every supported tasks-argument shape."""

    def test_bare_string_is_wrapped_in_single_element_list(self) -> None:
        """A bare string must NOT be iterated character-by-character."""
        assert _coerce_tasks("hello") == ["hello"]

    def test_long_bare_string_is_wrapped(self) -> None:
        """Realistic single-task strings are wrapped, not iterated."""
        task = "Summarize file foo.py and reply with the result."
        assert _coerce_tasks(task) == [task]

    def test_list_of_strings_is_returned_as_is(self) -> None:
        """Already-correct ``list[str]`` inputs pass through unchanged."""
        assert _coerce_tasks(["task A", "task B"]) == ["task A", "task B"]

    def test_json_encoded_two_task_list_is_parsed(self) -> None:
        """``'["a", "b"]'`` parses to a proper two-element list."""
        assert _coerce_tasks('["task A", "task B"]') == ["task A", "task B"]

    def test_json_encoded_three_task_list_is_parsed(self) -> None:
        """``'["a", "b", "c"]'`` parses to a three-element list."""
        assert _coerce_tasks('["a", "b", "c"]') == ["a", "b", "c"]

    def test_bracket_string_that_is_not_json_is_wrapped(self) -> None:
        """A bare task that happens to start with ``[`` falls back to wrap."""
        bad_json = "[bug] fix X and reply [ok]"
        assert _coerce_tasks(bad_json) == [bad_json]

    def test_non_string_non_list_input_raises_typeerror(self) -> None:
        """Other non-list inputs (e.g. dict, int) must raise ``TypeError``."""
        with pytest.raises(TypeError):
            _coerce_tasks({"k": "v"})  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            _coerce_tasks(42)  # type: ignore[arg-type]

    def test_list_containing_non_string_raises_typeerror(self) -> None:
        """A list with non-string elements is rejected."""
        with pytest.raises(TypeError):
            _coerce_tasks([1, 2, 3])  # type: ignore[list-item]


class TestRunParallelClosureUsesCoercion:
    """The ``run_parallel`` tool exposed to the LLM also runs through coercion.

    With a bare-string ``tasks`` argument and ``max_workers="0"`` the closure
    must raise :class:`ValueError` from ``ThreadPoolExecutor(max_workers=0)``
    **after** coercion — proving the closure does not iterate the string
    character-by-character before reaching the executor.
    """

    def test_run_parallel_tool_does_not_iterate_string(self) -> None:
        """The closure surfaces ``ValueError`` from the executor, not a
        character-iteration loop."""
        agent = SorcarAgent("test-string-bug")
        agent._use_web_tools = False
        agent._is_parallel = True
        tools = agent._get_tools()
        run_parallel = next(
            t for t in tools if getattr(t, "__name__", "") == "run_parallel"
        )

        # ``max_workers="0"`` coerces to ``int(0)`` and surfaces
        # ``ValueError`` from ``ThreadPoolExecutor`` — but only AFTER
        # ``_coerce_tasks`` wraps ``"hello world"`` into a single-element
        # list.  Character iteration would not raise here.
        with pytest.raises(ValueError):
            run_parallel("hello world", max_workers="0")
