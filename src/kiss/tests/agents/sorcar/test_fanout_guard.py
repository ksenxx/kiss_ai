# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the sub-agent fan-out checks.

Rules from :mod:`kiss.agents.sorcar.fanout_guard`, exercised through the
real ``run_parallel`` tool closure, the real fan-out engine, and the
real ``run_agent`` dispatch helper:

* ``tasks`` must be a JSON array of non-empty strings (a literal
  ``"$(cat tasks.json)"`` is refused with a hint).
* A review task puts its child on the read-only ``review`` tool
  profile, and the reviewer marker is inherited down the sub-tree.
* There is no cap on review fan-outs and no rule against a reviewer
  spawning reviewers: those guardrails were removed, so the tests here
  pin that every such fan-out is dispatched.

No mocks: argument refusals happen before any sub-agent exists, and the
dispatch tests drive the real engine with an unknown model name so each
child fails fast without a network call.  The only patching is
``DEFAULT_CONFIG.tool_profiles = False`` in the tests that give a
reviewer the ``run_parallel`` tool, because a reviewer running with tool
profiles on has no ``run_parallel`` tool at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.agents.sorcar.agent_dispatch import _dispatch
from kiss.agents.sorcar.fanout_guard import is_review_task, parse_tasks_json
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _LiveUsageMonitor,
    run_tasks_parallel,
)
from kiss.core.config import DEFAULT_CONFIG

UNKNOWN_MODEL = "no-such-model-fanout-guard"


def _run_parallel_tool(agent: SorcarAgent):
    agent._use_web_tools = False
    agent._is_parallel = True
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

    def test_bad_max_workers_is_rejected_before_dispatch(self) -> None:
        run_parallel = _run_parallel_tool(SorcarAgent("guard-workers"))
        assert run_parallel('["Summarize a"]', max_workers="0").startswith(
            "Error: max_workers must be at least 1"
        )
        assert run_parallel('["Summarize a"]', max_workers="two").startswith(
            "Error: max_workers must be an integer string"
        )

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
        fans out another review task: the children are dispatched (and
        fail fast on the unknown model)."""
        monkeypatch.setattr(DEFAULT_CONFIG, "tool_profiles", False)
        agent = SorcarAgent("guard-reviewer")
        _mark_reviewer(agent)
        agent.model_name = UNKNOWN_MODEL
        run_parallel = _run_parallel_tool(agent)
        result = run_parallel('["Review src/x.py for regressions"]')
        assert "Unknown model name" in result
        assert not result.startswith("Error:")

    def test_review_fanouts_are_not_capped(self) -> None:
        """Five consecutive review fan-outs all reach the engine; there is
        no per-task-tree round limit any more."""
        agent = SorcarAgent("guard-rounds")
        agent.model_name = UNKNOWN_MODEL
        run_parallel = _run_parallel_tool(agent)
        for _ in range(5):
            result = run_parallel('["Review the diff read-only"]')
            assert "Unknown model name" in result
            assert not result.startswith("Error:")

    def test_review_share_in_prompt_does_not_clip_children(
        self, tmp_path,
    ) -> None:
        """A prompt naming a review share used to cap reviewer children's
        budgets; now every child gets the plain remaining-budget share."""
        parent = SorcarAgent("guard-share")
        parent.max_budget = 10.0
        parent.budget_used = 0.0
        monitor = _LiveUsageMonitor(parent, None)
        results = run_tasks_parallel(
            ["Review the diff for regressions", "Run the tests"],
            max_workers=1, model_name=UNKNOWN_MODEL, usage_monitor=monitor,
            parent_agent=parent, max_budget=parent._subagent_budget_share(2),
            work_dir=str(tmp_path),
        )
        assert all("Unknown model name" in r for r in results)
        assert [a.max_budget for a in monitor._agents] == [10.0 / 3, 10.0 / 3]


class TestReviewerMarkerAcrossTree:
    """``run_tasks_parallel`` stamps the reviewer marker on each child
    so a reviewer's helpers get the read-only tool profile.  The
    unknown model makes each child fail fast (no network) after the
    marker has been applied."""

    @staticmethod
    def _spawn(parent: SorcarAgent, tasks: list[str]) -> list[Any]:
        monitor = _LiveUsageMonitor(parent, None)
        results = run_tasks_parallel(
            tasks, max_workers=1, model_name=UNKNOWN_MODEL,
            usage_monitor=monitor, parent_agent=parent,
        )
        assert all("Unknown model name" in r for r in results)
        return list(monitor._agents)

    def test_review_task_child_is_marked_reviewer(self) -> None:
        children = self._spawn(
            SorcarAgent("stamp-parent"), ["Review src/a.py", "Run tests"],
        )
        infos = [c._subagent_info for c in children]
        assert [i["reviewer"] for i in infos] == [True, False]

    def test_reviewer_parent_marks_every_child_reviewer(self) -> None:
        parent = SorcarAgent("stamp-reviewer")
        _mark_reviewer(parent)
        children = self._spawn(parent, ["Run tests"])
        assert children[0]._subagent_info["reviewer"] is True

    def test_top_level_parent_children_default_to_not_reviewer(self) -> None:
        children = self._spawn(SorcarAgent("stamp-plain"), ["Run tests"])
        assert children[0]._subagent_info["reviewer"] is False


class TestRunAgentDispatch:
    """``run_agent`` dispatch no longer refuses reviews from a reviewer."""

    def test_reviewer_may_dispatch_a_review_task(self) -> None:
        parent = SorcarAgent("dispatch-reviewer")
        _mark_reviewer(parent)
        result = _dispatch(
            name="helper", prompt="Review the diff for regressions",
            agent_path="/nonexistent/agent.py", work_dir="/tmp",
            model_name="", budget=None, timeout=1.0, parent_agent=parent,
        )
        # Without a daemon the dispatch itself fails; what matters is
        # that the failure is the dispatch error, not a spawn refusal.
        assert "reviewer sub-agent" not in result
        assert result.startswith("Error: the helper agent task could not run")
