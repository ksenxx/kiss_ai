# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the sub-agent spawning guardrails.

Rules from :mod:`kiss.agents.sorcar.fanout_guard`, exercised through the
real ``run_parallel`` tool closure, the real fan-out engine, and the
real ``run_agent`` dispatch helper:

* ``tasks`` must be a JSON array of non-empty strings (a literal
  ``"$(cat tasks.json)"`` is refused with a hint).
* One task tree launches at most ``MAX_REVIEW_ROUNDS`` review fan-outs;
  the quota is one shared, atomic :class:`ReviewQuota` inherited by
  every child, and a fan-out refused before any child exists (bad
  ``max_workers``, budget preflight) does not burn a round.
* A reviewer sub-agent — and anything under it — may not spawn further
  reviewers, through ``run_parallel`` or ``run_agent``.

No mocks or patches: refusals happen before any sub-agent exists, the
dispatch/round-consumption tests drive the real engine with an unknown
model name so each child fails fast without a network call, and the
LLM-driven test runs on a real cheap model (skipped without a key).
"""

from __future__ import annotations

import os
import threading
from typing import Any

import pytest

from kiss.agents.sorcar.agent_dispatch import _dispatch
from kiss.agents.sorcar.fanout_guard import (
    MAX_REVIEW_ROUNDS,
    ReviewQuota,
    is_review_task,
    parse_tasks_json,
)
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _LiveUsageMonitor,
    run_tasks_parallel,
)
from kiss.core.config import DEFAULT_CONFIG

FAST_MODEL = "claude-haiku-4-5"
UNKNOWN_MODEL = "no-such-model-fanout-guard"
skip_no_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)


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


class TestReviewQuota:
    def test_reserve_up_to_limit_then_refuse(self) -> None:
        quota = ReviewQuota(limit=2)
        assert quota.try_reserve() and quota.try_reserve()
        assert not quota.try_reserve()
        assert quota.used == 2

    def test_concurrent_reservations_never_exceed_limit(self) -> None:
        quota = ReviewQuota()
        outcomes: list[bool] = []
        lock = threading.Lock()

        def reserve() -> None:
            got = quota.try_reserve()
            with lock:
                outcomes.append(got)

        threads = [threading.Thread(target=reserve) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(outcomes) == MAX_REVIEW_ROUNDS
        assert quota.used == MAX_REVIEW_ROUNDS


class TestRunParallelToolRefusals:
    """Refusals come back as ``Error:`` strings before any child exists."""

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

    def test_reviewer_subagent_cannot_spawn_reviewers(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The refusal still guards a reviewer that runs with the full toolset."""
        monkeypatch.setattr(DEFAULT_CONFIG, "tool_profiles", False)
        agent = SorcarAgent("guard-reviewer")
        _mark_reviewer(agent)
        run_parallel = _run_parallel_tool(agent)
        result = run_parallel('["Review src/x.py for regressions"]')
        assert result.startswith("Error: You are a reviewer sub-agent")
        assert agent._review_quota is None  # nothing reserved

    def test_reviewer_subagent_may_still_run_non_review_fanouts(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only reviewer-spawning is forbidden; helper fan-outs still
        dispatch (children fail fast on the unknown model)."""
        monkeypatch.setattr(DEFAULT_CONFIG, "tool_profiles", False)
        agent = SorcarAgent("guard-reviewer-helper")
        _mark_reviewer(agent)
        agent.model_name = UNKNOWN_MODEL
        run_parallel = _run_parallel_tool(agent)
        result = run_parallel('["Run the test split 3 and report PASS/FAIL"]')
        assert "Unknown model name" in result

    def test_fourth_review_round_is_refused(self) -> None:
        agent = SorcarAgent("guard-rounds")
        agent.model_name = UNKNOWN_MODEL
        run_parallel = _run_parallel_tool(agent)
        for expected in range(1, MAX_REVIEW_ROUNDS + 1):
            result = run_parallel('["Review the diff read-only"]')
            assert "Unknown model name" in result  # dispatch happened
            assert agent._review_quota is not None
            assert agent._review_quota.used == expected
        result = run_parallel('["Review the diff read-only"]')
        assert result.startswith("Error: Review-round cap reached")
        assert agent._review_quota is not None
        assert agent._review_quota.used == MAX_REVIEW_ROUNDS

    def test_non_review_fanouts_do_not_consume_rounds(self) -> None:
        agent = SorcarAgent("guard-no-round")
        agent.model_name = UNKNOWN_MODEL
        run_parallel = _run_parallel_tool(agent)
        result = run_parallel('["Summarize README.md"]')
        assert "Unknown model name" in result
        assert agent._review_quota is None

    def test_budget_refused_review_fanout_does_not_burn_quota(self) -> None:
        """The zero-child budget preflight raises BEFORE a round is
        reserved, so a later viable review fan-out still runs."""
        from kiss.core.kiss_error import KISSError

        agent = SorcarAgent("guard-preflight")
        agent.max_budget = 1.0
        agent.budget_used = 0.9  # remaining $0.10 -> share below minimum
        run_parallel = _run_parallel_tool(agent)
        with pytest.raises(KISSError, match="Refusing to spawn"):
            run_parallel('["Review the diff"]')
        assert agent._review_quota is None


class TestQuotaSharedAcrossTree:
    """``run_tasks_parallel`` hands the parent's quota to every child,
    so helper children cannot each mint a fresh 3-round budget.  The
    unknown model makes each child fail fast (no network) after the
    inheritance has been applied."""

    @staticmethod
    def _spawn(parent: SorcarAgent, tasks: list[str]) -> list[Any]:
        monitor = _LiveUsageMonitor(parent, None)
        results = run_tasks_parallel(
            tasks, max_workers=1, model_name=UNKNOWN_MODEL,
            usage_monitor=monitor, parent_agent=parent,
        )
        assert all("Unknown model name" in r for r in results)
        return list(monitor._agents)

    def test_children_inherit_the_parent_quota_instance(self) -> None:
        parent = SorcarAgent("quota-parent")
        parent._review_quota = ReviewQuota()
        children = self._spawn(parent, ["Run tests", "Summarize a file"])
        assert all(c._review_quota is parent._review_quota for c in children)

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


class TestRunAgentDispatchGuard:
    """``run_agent`` dispatch honours the reviewer sub-tree rule."""

    def test_reviewer_may_not_dispatch_a_review_task(self) -> None:
        parent = SorcarAgent("dispatch-reviewer")
        _mark_reviewer(parent)
        result = _dispatch(
            name="helper", prompt="Review the diff for regressions",
            agent_path="/nonexistent/agent.py", work_dir="/tmp",
            model_name="", budget=None, timeout=1.0, parent_agent=parent,
        )
        assert result.startswith("Error: You are a reviewer sub-agent")

    def test_review_dispatch_draws_from_the_shared_quota(self) -> None:
        """A ``run_agent`` review dispatch is not a free side door: it
        reserves one round from the same task-tree quota, and an
        exhausted quota refuses the dispatch outright."""
        parent = SorcarAgent("dispatch-quota")
        parent._review_quota = ReviewQuota()
        result = _dispatch(
            name="helper", prompt="Review the diff for regressions",
            agent_path="/nonexistent/agent.py", work_dir="/tmp",
            model_name="", budget=None, timeout=1.0, parent_agent=parent,
        )
        # The reservation happened before the (failing, daemonless)
        # dispatch attempt; the result is a dispatch error, not a
        # guardrail refusal.
        assert parent._review_quota.used == 1
        assert not result.startswith("Error: Review-round cap reached")

        while parent._review_quota.try_reserve():
            pass
        refused = _dispatch(
            name="helper", prompt="Review the diff for regressions",
            agent_path="/nonexistent/agent.py", work_dir="/tmp",
            model_name="", budget=None, timeout=1.0, parent_agent=parent,
        )
        assert refused.startswith("Error: Review-round cap reached")

    def test_non_review_dispatch_ignores_the_quota(self) -> None:
        parent = SorcarAgent("dispatch-plain")
        parent._review_quota = ReviewQuota()
        _dispatch(
            name="helper", prompt="Run the tests and report PASS/FAIL",
            agent_path="/nonexistent/agent.py", work_dir="/tmp",
            model_name="", budget=None, timeout=1.0, parent_agent=parent,
        )
        assert parent._review_quota.used == 0


@skip_no_key
class TestReviewerSubtreeWithRealModel:
    @pytest.mark.slow
    def test_reviewer_child_is_refused_when_it_spawns_a_reviewer(
        self, tmp_path,
    ) -> None:
        """A real child of a reviewer asks ``run_parallel`` for another
        reviewer and must receive the refusal as its tool result."""
        parent = SorcarAgent("real-reviewer-parent")
        _mark_reviewer(parent)
        task = (
            "Call the run_parallel tool exactly once with "
            'tasks=\'["Review README.md for typos"]\'. Then finish; your '
            "summary must quote the tool result verbatim."
        )
        results = run_tasks_parallel(
            [task], max_workers=1, model_name=FAST_MODEL,
            work_dir=str(tmp_path), parent_agent=parent, max_budget=0.5,
            web_tools=False,
        )
        assert "may not spawn further reviewers" in results[0]
