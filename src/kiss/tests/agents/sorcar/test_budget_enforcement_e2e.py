# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: sub-agent budget attribution and fair budget distribution.

Covers the production bug where parallel sub-agents spawned via
``run_parallel`` received NO budget cap (defaulting to the full configured
budget), so a single sub-agent could spend the entire budget of the main
task.  Sub-agents must now receive a meaningful share: the parent's
remaining budget divided across the tasks.  Also verifies that spend
attributed to the parent task by parallel sub-agents
(``_attribute_sub_usage``) is enforced mid-session by ``RelentlessAgent``.

The ``KISSAgent``-only half of the mid-step enforcement fix lives in
``kiss.tests.core.test_budget_enforcement_e2e``, whose fake
OpenAI-compatible HTTP harness this file reuses.

All tests drive real agents over real HTTP against a local
OpenAI-chat-completions-compatible server.  No mocks, patches, fakes, or
test doubles.
"""

from __future__ import annotations

import json
import tempfile
from http.server import BaseHTTPRequestHandler

import pytest
import yaml

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.agents.sorcar.sorcar_agent import (
    MIN_SUBAGENT_BUDGET,
    SorcarAgent,
    _attribute_sub_usage,
    run_tasks_parallel,
)
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import BudgetExceededError, KISSError
from kiss.tests.core.test_budget_enforcement_e2e import (
    _CHEAP,
    _EXPENSIVE,
    _ExpensiveNoopHandler,
    _read_body,
    _send_json,
    _start_server,
    _tool_call_response,
)


class _CheapSubSpendHandler(BaseHTTPRequestHandler):
    """Always returns a cheap ``sub_spend`` tool call and counts requests."""

    requests = 0

    def do_POST(self) -> None:  # noqa: N802
        _read_body(self)
        type(self).requests += 1
        _send_json(self, _tool_call_response("sub_spend", "{}", *_CHEAP))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _ParallelParentHandler(BaseHTTPRequestHandler):
    """Routes parent vs sub-agent requests for the distribution tests.

    * Requests whose conversation already has a tool-role message (the
      parent after ``run_parallel`` returned) -> cheap ``finish``.
    * Requests mentioning BUDGETPROBE (the sub-agents' task prompts)
      -> EXPENSIVE non-finish tool call, so each sub-agent immediately
      blows through any small budget share it was given.
    * Everything else (the parent's first call, or the summarizer)
      -> cheap ``run_parallel`` call spawning two BUDGETPROBE tasks.
    """

    def do_POST(self) -> None:  # noqa: N802
        body = _read_body(self)
        try:
            messages = json.loads(body).get("messages", [])
        except Exception:
            messages = []
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        text = json.dumps(messages)
        if has_tool_result:
            resp = _tool_call_response(
                "finish", '{"result": "parent-done"}', *_CHEAP
            )
        elif "BUDGETPROBE" in text:
            resp = _tool_call_response("noop", "{}", *_EXPENSIVE)
        else:
            args = json.dumps(
                {"tasks": '["BUDGETPROBE alpha", "BUDGETPROBE beta"]'}
            )
            resp = _tool_call_response("run_parallel", args, *_CHEAP)
        _send_json(self, resp)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass



class TestParentAttributedSpendEnforcedMidSession:
    """Sub-agent spend lands on the relentless parent via
    ``_attribute_sub_usage``; the live executor must observe it and stop
    within one step instead of running to the end of the session."""

    def test_relentless_stops_promptly_after_attributed_spend(self) -> None:
        """A tool that attributes $5 of sub-agent spend to a parent with a
        $1 budget must stop the run within roughly one step."""
        _CheapSubSpendHandler.requests = 0
        srv, url = _start_server(_CheapSubSpendHandler)
        agent = RelentlessAgent("attributed-spend")

        def sub_spend() -> str:
            """Attribute $5.00 of sub-agent spend to the parent task,
            exactly as ``run_parallel`` does in production."""
            _attribute_sub_usage(agent, 5.0, 1_000, 3)
            return "sub-agents finished"

        try:
            with tempfile.TemporaryDirectory() as td:
                with pytest.raises(KISSError, match="budget"):
                    agent.run(
                        model_name="gpt-4o-mini",
                        prompt_template="Spawn sub-agents.",
                        tools=[sub_spend],
                        max_steps=5,
                        max_budget=1.0,
                        max_sub_sessions=3,
                        work_dir=td,
                        verbose=False,
                        model_config={"base_url": url, "api_key": "test-key"},
                    )
            assert agent.budget_used >= 5.0
            assert agent.budget_used < 5.5, (
                f"Total spend ${agent.budget_used:.4f}: the executor kept "
                f"attributing sub-agent spend after the $1.00 budget was "
                f"exceeded — mid-session enforcement is missing."
            )
            assert _CheapSubSpendHandler.requests == 1, (
                f"{_CheapSubSpendHandler.requests} model requests ran — a "
                f"budget failure launched more model work (likely the "
                f"RelentlessAgent summarizer)."
            )
            assert agent.total_steps == 4
        finally:
            srv.shutdown()

    def test_check_total_budget_direct(self) -> None:
        """``_check_total_budget`` must work with and without a live
        executor and include the executor's own live spend."""
        agent = RelentlessAgent("hook-direct")
        agent.max_budget = 1.0
        agent.budget_used = 0.4
        agent._current_executor = None
        agent._check_total_budget()

        agent.budget_used = 1.2
        with pytest.raises(KISSError, match="budget exceeded"):
            agent._check_total_budget()

        executor = KISSAgent("hook-executor")
        executor.budget_used = 0.7
        agent.budget_used = 0.4
        agent._current_executor = executor
        with pytest.raises(KISSError, match="budget exceeded"):
            agent._check_total_budget()

        executor.budget_used = 0.5
        agent._check_total_budget()



class TestSubagentBudgetShare:
    """The parent's remaining budget must be split across sub-tasks."""

    def test_share_divides_remaining_budget(self) -> None:
        agent = SorcarAgent("share")
        agent.max_budget = 6.0
        agent.budget_used = 1.0
        agent._current_executor = None
        assert agent._subagent_budget_share(4) == pytest.approx(1.0)

        executor = KISSAgent("share-executor")
        executor.budget_used = 1.0
        agent._current_executor = executor
        assert agent._subagent_budget_share(2) == pytest.approx(4.0 / 3)

    def test_single_subagent_cannot_consume_parent_remainder(self) -> None:
        """Even a one-item fan-out must reserve budget for the main agent
        to process the result and finish; otherwise that one sub-agent can
        consume the entire remaining main-task budget."""
        agent = SorcarAgent("share-single")
        agent.max_budget = 2.2
        agent.budget_used = 0.2
        agent._current_executor = None
        assert agent._subagent_budget_share(1) == pytest.approx(1.0)

    def test_share_guards_zero_tasks(self) -> None:
        agent = SorcarAgent("share-zero")
        agent.max_budget = 1.0
        agent.budget_used = 0.0
        agent._current_executor = None
        assert agent._subagent_budget_share(0) == pytest.approx(1.0)

    def test_share_raises_when_no_budget_left(self) -> None:
        agent = SorcarAgent("share-exhausted")
        agent.max_budget = 1.0
        agent.budget_used = 1.0
        agent._current_executor = None
        with pytest.raises(BudgetExceededError, match="budget"):
            agent._subagent_budget_share(2)


class TestSubagentBudgetFloor:
    """Fan-outs whose per-child share is below ``MIN_SUBAGENT_BUDGET``
    must be refused instead of spawning doomed sub-agents.

    Reproduces the production stall where recursive ``run_parallel``
    fan-outs split a parent's remainder down to ~$0.03 per child; each
    child's FIRST LLM step (it resumes the parent's chat context) cost
    more than its whole budget, so dozens of sub-agents were killed on
    step 1 having done nothing.
    """

    def test_share_below_floor_refused_with_plain_kiss_error(self) -> None:
        """A sliver share must raise a plain ``KISSError`` — NOT
        ``BudgetExceededError``, which ``_execute_tool`` re-raises and
        thereby kills the whole parent agent.  A plain ``KISSError``
        becomes an actionable tool-error string so the parent model can
        do the work inline instead."""
        agent = SorcarAgent("share-starved")
        agent.max_budget = 0.25
        agent.budget_used = 0.0
        agent._current_executor = None
        with pytest.raises(KISSError, match="Refusing to spawn") as exc_info:
            agent._subagent_budget_share(7)
        assert not isinstance(exc_info.value, BudgetExceededError)
        assert "inline" in str(exc_info.value)

    def test_share_at_floor_allowed(self) -> None:
        """A share exactly at the floor must still be handed out."""
        agent = SorcarAgent("share-at-floor")
        agent.max_budget = MIN_SUBAGENT_BUDGET * 3
        agent.budget_used = 0.0
        agent._current_executor = None
        assert agent._subagent_budget_share(2) == pytest.approx(
            MIN_SUBAGENT_BUDGET
        )

    def test_share_at_floor_allowed_despite_float_rounding(self) -> None:
        """A mathematically exact floor share must not be refused when
        float subtraction rounds it a few ULPs below the floor:
        ``1.13 - 0.13`` is ``0.9999999999999999``, so a one-child share
        computes as ``0.49999999999999994`` — still $0.50 in currency
        terms."""
        agent = SorcarAgent("share-floor-rounding")
        agent.max_budget = 1.13
        agent.budget_used = 0.13
        agent._current_executor = None
        assert agent._subagent_budget_share(1) == pytest.approx(
            MIN_SUBAGENT_BUDGET
        )

    def test_live_executor_spend_counts_towards_floor(self) -> None:
        """The live executor session's own spend shrinks the remainder,
        so it can push a previously viable share below the floor."""
        agent = SorcarAgent("share-live-floor")
        agent.max_budget = 2.0
        agent.budget_used = 0.0
        executor = KISSAgent("share-live-floor-executor")
        executor.budget_used = 1.5
        agent._current_executor = executor
        with pytest.raises(KISSError, match="Refusing to spawn"):
            agent._subagent_budget_share(2)



class TestRunTasksParallelBudgetCap:
    """Each spawned sub-agent must run under the per-task ``max_budget``."""

    def test_each_subagent_capped(self) -> None:
        srv, url = _start_server(_ExpensiveNoopHandler)
        try:
            with tempfile.TemporaryDirectory() as td:
                totals: dict[str, float] = {}
                results = run_tasks_parallel(
                    ["BUDGETPROBE alpha", "BUDGETPROBE beta"],
                    model_name="gpt-4o-mini",
                    work_dir=td,
                    max_budget=0.01,
                    model_config={"base_url": url, "api_key": "test-key"},
                    totals_out=totals,
                )
            assert len(results) == 2
            for res in results:
                payload = yaml.safe_load(res)
                assert payload["success"] is False
                assert "budget exceeded" in str(payload["summary"]).lower()
            assert 0.7 < totals["budget_used"] < 1.0, (
                f"Sub-agents spent ${totals['budget_used']:.4f} — the "
                f"$0.01 per-task cap was not enforced."
            )
        finally:
            srv.shutdown()



def _assert_distributed(parent: SorcarAgent, url: str, td: str) -> None:
    """Run *parent* with a $2.00 budget; its run_parallel spawns two
    expensive sub-agents, each capped to ~a third of the remaining
    budget (~$0.66 — comfortably above ``MIN_SUBAGENT_BUDGET``, so the
    fan-out is allowed).  Each sub-agent stops after ONE $0.375 call
    (the fake 500k-token response also exhausts the model's context
    window), and both subs' spend must be attributed back to the
    parent (~$0.75 total)."""
    try:
        parent.run(
            prompt_template="Run two probes in parallel.",
            model_name="gpt-4o-mini",
            model_config={"base_url": url, "api_key": "test-key"},
            work_dir=td,
            is_parallel=True,
            max_steps=5,
            max_sub_sessions=2,
            max_budget=2.0,
        )
    except KISSError:
        pass
    assert parent.budget_used > 0.7, (
        f"Parent budget_used ${parent.budget_used:.4f}: sub-agent spend was "
        f"not attributed back to the parent task."
    )
    assert parent.budget_used < 1.6, (
        f"Parent budget_used ${parent.budget_used:.4f}: sub-agents were not "
        f"capped to a share of the parent's $2.00 budget — a sub-agent "
        f"could spend the whole configured budget."
    )


class TestParallelBudgetDistributionE2E:
    """Full agent -> run_parallel -> sub-agents budget distribution."""

    def test_sorcar_agent_distributes_budget(self) -> None:
        srv, url = _start_server(_ParallelParentHandler)
        try:
            with tempfile.TemporaryDirectory() as td:
                _assert_distributed(SorcarAgent("dist-parent"), url, td)
        finally:
            srv.shutdown()

    def test_chat_sorcar_agent_distributes_budget(self) -> None:
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

        srv, url = _start_server(_ParallelParentHandler)
        try:
            with tempfile.TemporaryDirectory() as td:
                _assert_distributed(ChatSorcarAgent("dist-chat-parent"), url, td)
        finally:
            srv.shutdown()


class _CountingParallelParentHandler(_ParallelParentHandler):
    """``_ParallelParentHandler`` that counts served sub-agent requests."""

    probe_requests = 0

    def do_POST(self) -> None:  # noqa: N802
        # Peek at the body via the parent's routing by re-implementing
        # only the counter; the response logic stays in the parent.
        # BaseHTTPRequestHandler bodies can only be read once, so count
        # here and delegate the already-parsed decision to a copy of the
        # parent's logic.
        body = _read_body(self)
        try:
            messages = json.loads(body).get("messages", [])
        except Exception:
            messages = []
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        text = json.dumps(messages)
        if has_tool_result:
            resp = _tool_call_response(
                "finish", '{"result": "parent-done"}', *_CHEAP
            )
        elif "BUDGETPROBE" in text:
            type(self).probe_requests += 1
            resp = _tool_call_response("noop", "{}", *_EXPENSIVE)
        else:
            args = json.dumps(
                {"tasks": '["BUDGETPROBE alpha", "BUDGETPROBE beta"]'}
            )
            resp = _tool_call_response("run_parallel", args, *_CHEAP)
        _send_json(self, resp)


class TestStarvedFanOutRefusedE2E:
    """End-to-end reproduction of the recursive-fan-out stall: a parent
    whose remaining budget cannot fund viable sub-agents must NOT spawn
    them.  ``run_parallel`` returns an actionable tool error instead,
    and the parent survives to finish inline."""

    def test_run_parallel_refused_no_subagents_spawned(self) -> None:
        _CountingParallelParentHandler.probe_requests = 0
        srv, url = _start_server(_CountingParallelParentHandler)
        parent = SorcarAgent("starved-parent")
        try:
            with tempfile.TemporaryDirectory() as td:
                result = parent.run(
                    prompt_template="Run two probes in parallel.",
                    model_name="gpt-4o-mini",
                    model_config={"base_url": url, "api_key": "test-key"},
                    work_dir=td,
                    is_parallel=True,
                    max_steps=5,
                    max_sub_sessions=2,
                    max_budget=0.30,
                )
            assert _CountingParallelParentHandler.probe_requests == 0, (
                f"{_CountingParallelParentHandler.probe_requests} sub-agent "
                f"model requests were served — doomed sub-agents were "
                f"spawned despite a ${0.30/3:.2f} per-child share below the "
                f"${MIN_SUBAGENT_BUDGET:.2f} floor."
            )
            assert "parent-done" in result, (
                f"Parent did not finish inline after the refusal: {result!r}"
            )
            assert parent.budget_used < 0.30, (
                f"Parent spent ${parent.budget_used:.4f} — sub-agents burned "
                f"budget despite the fan-out refusal."
            )
        finally:
            srv.shutdown()
