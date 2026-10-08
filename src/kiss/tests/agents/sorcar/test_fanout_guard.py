# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the sub-agent fan-out checks.

Rules from :mod:`kiss.agents.sorcar.fanout_guard`, exercised through the
real ``run_parallel`` tool closure and the real ``run_agent`` dispatch
helper:

* ``tasks`` must be a JSON array of non-empty strings (a literal
  ``"$(cat tasks.json)"`` is refused with a hint).
* There is no cap on review fan-outs and no rule against a reviewer
  spawning reviewers: those guardrails were removed, so the tests here
  pin that every such fan-out is dispatched.

No mocks: argument refusals happen before any sub-agent exists, and
the dispatch tests run without a daemon (the session's isolated
``KISS_HOME`` has no endpoint file), so every child comes back as the
per-child dispatch error rather than a refusal of the call.  The only
patching is ``DEFAULT_CONFIG.tool_profiles = False`` in the test that
gives a reviewer the ``run_parallel`` tool, because a reviewer running
with tool profiles on has no ``run_parallel`` tool at all.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from kiss.agents.sorcar.agent_dispatch import dispatch_result
from kiss.agents.sorcar.fanout_guard import is_review_task, parse_tasks_json
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.config import DEFAULT_CONFIG

NO_DAEMON = "Error: the sorcar agent task could not run"


def _dispatched_children(result: str, count: int) -> list[str]:
    """The per-child results of a ``run_parallel`` call that was
    dispatched (not refused): a YAML list of *count* strings, each the
    child's own dispatch error from the missing daemon."""
    assert not result.startswith("Error:"), result
    children = yaml.safe_load(result)
    assert isinstance(children, list) and len(children) == count, result
    for child in children:
        assert isinstance(child, str) and child.startswith(NO_DAEMON), child
        assert "reviewer sub-agent" not in child
    return children


def _run_parallel_tool(agent: SorcarAgent):
    agent._use_web_tools = False
    agent._is_parallel = True
    agent.max_budget = 10.0
    agent.budget_used = 0.0
    return next(
        t for t in agent._get_tools() if getattr(t, "__name__", "") == "run_parallel"
    )


def _mark_reviewer(agent: Any) -> None:
    agent._subagent_info = {
        "parent_task_id": "", "parent_tab_id": "", "reviewer": True,
    }


class TestIsReviewTask:
    def test_reviewer_stems_match_case_insensitively(self) -> None:
        assert is_review_task("Do a read-only REVIEW of the diff")
        assert is_review_task("act as a reviewer of src/x.py")
        assert is_review_task("Audit the concurrency of the fan-out")
        assert is_review_task("critique this design")
        assert is_review_task("Inspect the current diff for correctness defects")

    def test_bug_hunt_phrasings_match(self) -> None:
        assert is_review_task("Find bugs and security vulnerabilities")
        assert is_review_task("Check the patch for regressions")
        assert is_review_task("hunt for defects in the parser")
        assert is_review_task("scan for issues in the new module")
        assert is_review_task("adversarial testing of the fan-out engine")

    def test_non_review_tasks_do_not_match(self) -> None:
        assert not is_review_task("Run the bash command uv run pytest and report")
        assert not is_review_task("Summarize src/foo.py")
        assert not is_review_task("Implement the parser and write tests")
        # A substring inside another word is not a review request.
        assert not is_review_task("Preview the rendered page")


class TestParseTasksJson:
    def test_valid_array_is_returned(self) -> None:
        assert parse_tasks_json(' ["a", "b"] ') == ["a", "b"]

    def test_shell_substitution_is_refused_with_hint(self) -> None:
        with pytest.raises(ValueError) as info:
            parse_tasks_json("$(cat ./tmp/tasks.json)")
        message = str(info.value)
        assert "JSON array" in message
        assert "Shell substitutions are not expanded" in message

    def test_backtick_substitution_is_refused_with_hint(self) -> None:
        with pytest.raises(ValueError, match="Shell substitutions"):
            parse_tasks_json("`cat tasks.json`")

    def test_bare_prose_is_refused_without_shell_hint(self) -> None:
        with pytest.raises(ValueError) as info:
            parse_tasks_json("Summarize file A and file B")
        assert "Shell substitutions" not in str(info.value)

    def test_json_object_is_refused(self) -> None:
        with pytest.raises(ValueError, match="got a JSON dict"):
            parse_tasks_json('{"tasks": ["a"]}')

    def test_empty_array_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty array"):
            parse_tasks_json("[]")

    def test_non_string_or_blank_elements_are_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty strings"):
            parse_tasks_json("[1, 2]")
        with pytest.raises(ValueError, match="non-empty strings"):
            parse_tasks_json('["a", "  "]')


class TestRunParallelToolArguments:
    """Argument refusals come back as ``Error:`` strings before any
    child exists; everything else is dispatched."""

    def test_shell_substitution_string_is_rejected(self) -> None:
        run_parallel = _run_parallel_tool(SorcarAgent("guard-json"))
        result = run_parallel("$(cat ./tmp/tasks.json)")
        assert result.startswith("Error: tasks must be a JSON array")
        assert "Shell substitutions" in result

    def test_bare_string_is_rejected_not_dispatched(self) -> None:
        run_parallel = _run_parallel_tool(SorcarAgent("guard-bare"))
        result = run_parallel("hello world")
        assert result.startswith("Error: tasks must be a JSON array")

    def test_reviewer_subagent_has_no_run_parallel_tool(self) -> None:
        """With tool profiles on, a reviewer's toolset has no fan-out at all."""
        agent = SorcarAgent("guard-reviewer-profile")
        _mark_reviewer(agent)
        agent._use_web_tools = False
        agent._is_parallel = True
        names = {getattr(t, "__name__", "") for t in agent._get_tools()}
        assert "run_parallel" not in names and "Edit" not in names

    def test_reviewer_subagent_with_full_toolset_may_spawn_reviewers(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A reviewer running with the full toolset is not refused when it
        fans out another review task: the child is dispatched."""
        monkeypatch.setattr(DEFAULT_CONFIG, "tool_profiles", False)
        agent = SorcarAgent("guard-reviewer")
        _mark_reviewer(agent)
        run_parallel = _run_parallel_tool(agent)
        result = run_parallel('["Review src/x.py for regressions"]')
        _dispatched_children(result, 1)

    def test_review_fanouts_are_not_capped(self) -> None:
        """Five consecutive review fan-outs are all dispatched; there is
        no per-task-tree round limit any more."""
        run_parallel = _run_parallel_tool(SorcarAgent("guard-rounds"))
        for _ in range(5):
            result = run_parallel('["Review the diff read-only"]')
            _dispatched_children(result, 1)

    def test_mixed_review_and_plain_tasks_are_all_dispatched(self) -> None:
        """A review task and a plain task in one call each get their own
        child; the result keeps the tasks' order."""
        run_parallel = _run_parallel_tool(SorcarAgent("guard-mixed"))
        result = run_parallel('["Review the diff for regressions", "Run the tests"]')
        _dispatched_children(result, 2)


class TestRunAgentDispatch:
    """``run_agent`` dispatch no longer refuses reviews from a reviewer."""

    def test_reviewer_may_dispatch_a_review_task(self) -> None:
        parent = SorcarAgent("dispatch-reviewer")
        _mark_reviewer(parent)
        result = dispatch_result(
            name="helper", prompt="Review the diff for regressions",
            sea_path="/nonexistent/agent.py", work_dir="/tmp",
            model_name="", budget=None, timeout=1.0, parent_agent=parent,
        )
        # Without a daemon the dispatch itself fails; what matters is
        # that the failure is the dispatch error, not a spawn refusal.
        assert isinstance(result, str)
        assert "reviewer sub-agent" not in result
        assert result.startswith("Error: the helper agent task could not run")
