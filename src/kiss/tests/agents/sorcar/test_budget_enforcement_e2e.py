# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: sub-agent budget attribution and fair budget distribution.

Covers the production bug where parallel sub-agents spawned via
``run_parallel`` received NO budget cap (defaulting to the full configured
budget), so a single sub-agent could spend the entire budget of the main
task.  Sub-agents must receive a meaningful share: the parent's
remaining budget divided across the tasks plus the parent
(``SorcarAgent._subagent_budget_share``).  Also verifies that spend
attributed to the parent task by sub-agents (``_attribute_sub_usage``)
is enforced mid-session by ``RelentlessAgent``.

The ``KISSAgent``-only half of the mid-step enforcement fix lives in
``kiss.tests.core.test_budget_enforcement_e2e``, whose fake
OpenAI-compatible HTTP harness this file reuses.

All tests drive real agents over real HTTP against a local
OpenAI-chat-completions-compatible server.  No mocks, patches, fakes, or
test doubles.
"""

from __future__ import annotations

import tempfile
from http.server import BaseHTTPRequestHandler

import pytest

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, _attribute_sub_usage
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import BudgetExceededError, KISSError
from kiss.tests.core.test_budget_enforcement_e2e import (
    _CHEAP,
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


class TestSubagentBudgetShareHasNoFloor:
    """Small per-child shares are handed out, not refused.

    An earlier ``MIN_SUBAGENT_BUDGET`` floor turned a sliver share into a
    ``KISSError``; that guardrail is gone, so the share is whatever the
    remaining budget divides to and the children's own budget checks
    decide their fate.
    """

    def test_share_below_half_a_dollar_is_returned(self) -> None:
        agent = SorcarAgent("share-small")
        agent.max_budget = 0.25
        agent.budget_used = 0.0
        agent._current_executor = None
        assert agent._subagent_budget_share(7) == pytest.approx(0.25 / 8)

    def test_live_executor_spend_only_shrinks_the_share(self) -> None:
        agent = SorcarAgent("share-live-small")
        agent.max_budget = 2.0
        agent.budget_used = 0.0
        executor = KISSAgent("share-live-small-executor")
        executor.budget_used = 1.5
        agent._current_executor = executor
        assert agent._subagent_budget_share(2) == pytest.approx(0.5 / 3)
