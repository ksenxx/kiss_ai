# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: prompt budget enforcement and fair sub-agent budget distribution.

Covers the two production bugs:

1. Agents kept working after exceeding ``max_budget`` (the settings-panel
   budget): the budget was only checked at the top of each step, so an
   over-budget response still executed a full round of tools, and spend
   attributed to the parent task by parallel sub-agents
   (``_attribute_sub_usage``) was never enforced mid-session at all.

2. Parallel sub-agents spawned via ``run_parallel`` received NO budget cap
   (defaulting to the full configured budget), so a single sub-agent could
   spend the entire budget of the main task.  Sub-agents must now receive a
   meaningful share: the parent's remaining budget divided across the tasks.

All tests drive real agents over real HTTP against a local
OpenAI-chat-completions-compatible server.  No mocks, patches, fakes, or
test doubles.
"""

from __future__ import annotations

import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _attribute_sub_usage,
    run_tasks_parallel,
)
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.relentless_agent import RelentlessAgent

# ---------------------------------------------------------------------------
# OpenAI chat-completions response builders
# ---------------------------------------------------------------------------


def _tool_calls_response(
    calls: list[tuple[str, str]],
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    """Build a chat-completions response containing one or more tool calls."""
    return {
        "id": "chatcmpl-budget-e2e",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                        for index, (name, arguments) in enumerate(calls, start=1)
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _tool_call_response(
    name: str,
    arguments: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    """Build a chat-completions response containing one tool call."""
    return _tool_calls_response(
        [(name, arguments)], prompt_tokens, completion_tokens
    )


def _send_json(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
    """Write *payload* as a JSON HTTP 200 response."""
    body = json.dumps(payload).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> str:
    """Read and return the request body as text."""
    cl = int(handler.headers.get("Content-Length", 0))
    return handler.rfile.read(cl).decode() if cl else ""


def _start_server(
    handler_cls: type[BaseHTTPRequestHandler],
) -> tuple[ThreadingHTTPServer, str]:
    """Start a local HTTP server; return (server, base_url)."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


# At gpt-4o-mini rates 500k prompt + 500k completion tokens cost $0.375.
_EXPENSIVE = (500_000, 500_000)
# 10 prompt + 5 completion tokens cost ~$0.0000045 (effectively free).
_CHEAP = (10, 5)


class _ExpensiveNoopHandler(BaseHTTPRequestHandler):
    """Always returns a non-finish ``noop`` tool call costing $0.375."""

    def do_POST(self) -> None:  # noqa: N802
        _read_body(self)
        _send_json(self, _tool_call_response("noop", "{}", *_EXPENSIVE))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _ExpensiveFinishHandler(BaseHTTPRequestHandler):
    """Always returns a ``finish`` tool call costing $0.375."""

    def do_POST(self) -> None:  # noqa: N802
        _read_body(self)
        _send_json(
            self,
            _tool_call_response("finish", '{"result": "done"}', *_EXPENSIVE),
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _CheapSubSpendHandler(BaseHTTPRequestHandler):
    """Always returns a cheap ``sub_spend`` tool call and counts requests."""

    requests = 0

    def do_POST(self) -> None:  # noqa: N802
        _read_body(self)
        type(self).requests += 1
        _send_json(self, _tool_call_response("sub_spend", "{}", *_CHEAP))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _ExpensiveMixedFinishHandler(BaseHTTPRequestHandler):
    """Returns an expensive non-finish tool followed by ``finish``."""

    def do_POST(self) -> None:  # noqa: N802
        _read_body(self)
        _send_json(
            self,
            _tool_calls_response(
                [("noop", "{}"), ("finish", '{"result": "done"}')],
                *_EXPENSIVE,
            ),
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _CheapTwoToolsHandler(BaseHTTPRequestHandler):
    """Returns ``sub_spend`` followed by a tool that must be blocked."""

    def do_POST(self) -> None:  # noqa: N802
        _read_body(self)
        _send_json(
            self,
            _tool_calls_response(
                [("sub_spend", "{}"), ("must_not_run", "{}")], *_CHEAP
            ),
        )

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


# ---------------------------------------------------------------------------
# 1. KISSAgent must stop the moment a response takes it over budget
# ---------------------------------------------------------------------------


class TestMidStepBudgetEnforcement:
    """The agent must not execute further tools once over budget."""

    def test_tools_not_executed_once_over_budget(self) -> None:
        """The very response that exceeds the budget must abort the step
        BEFORE its (non-finish) tool calls are executed — previously the
        agent executed one more full round of tools and only stopped at
        the top of the next step."""
        srv, url = _start_server(_ExpensiveNoopHandler)
        calls: list[str] = []

        def noop() -> str:
            """A no-op tool that records that it was called."""
            calls.append("noop")
            return "ok"

        try:
            agent = KISSAgent("mid-step-budget")
            with pytest.raises(KISSError, match="budget exceeded"):
                agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Call noop.",
                    tools=[noop],
                    is_agentic=True,
                    max_steps=10,
                    max_budget=0.01,
                    verbose=False,
                    model_config={"base_url": url, "api_key": "test-key"},
                )
            assert agent.budget_used > 0.01
            assert calls == [], (
                f"Tools were executed {len(calls)} time(s) AFTER the budget "
                f"was already exceeded — the agent must stop immediately."
            )
        finally:
            srv.shutdown()

    def test_finish_result_returned_even_when_over_budget(self) -> None:
        """When the over-budget response contains ONLY a ``finish`` call the
        agent IS stopping — the result must be returned, not discarded."""
        srv, url = _start_server(_ExpensiveFinishHandler)
        try:
            agent = KISSAgent("finish-over-budget")
            result = agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Finish immediately.",
                is_agentic=True,
                max_steps=10,
                max_budget=0.01,
                verbose=False,
                model_config={"base_url": url, "api_key": "test-key"},
            )
            assert result == "done"
            assert agent.budget_used > 0.01
        finally:
            srv.shutdown()

    def test_finish_does_not_bypass_nonfinish_tool_budget_check(self) -> None:
        """A response containing both a normal tool and ``finish`` must
        not use the finish call as a loophole to execute the normal tool
        after the model response has already exceeded the budget."""
        srv, url = _start_server(_ExpensiveMixedFinishHandler)
        calls: list[str] = []

        def noop() -> str:
            """Record an invocation that must never happen."""
            calls.append("noop")
            return "ok"

        try:
            agent = KISSAgent("mixed-finish-over-budget")
            with pytest.raises(KISSError, match="budget exceeded"):
                agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Call noop and then finish.",
                    tools=[noop],
                    max_steps=10,
                    max_budget=0.01,
                    verbose=False,
                    model_config={"base_url": url, "api_key": "test-key"},
                )
            assert calls == []
        finally:
            srv.shutdown()

    def test_attributed_spend_blocks_later_tool_in_same_response(self) -> None:
        """After one tool attributes sub-agent spend over the parent
        limit, no later tool from that same model response may execute."""
        srv, url = _start_server(_CheapTwoToolsHandler)
        parent_spend = 0.0
        forbidden_calls: list[str] = []

        def sub_spend() -> str:
            """Simulate spend attributed by ``run_parallel``."""
            nonlocal parent_spend
            parent_spend = 5.0
            return "spent"

        def must_not_run() -> str:
            """Record a post-budget tool invocation that is forbidden."""
            forbidden_calls.append("ran")
            return "bad"

        agent = KISSAgent("same-response-parent-budget")

        def check_parent_budget() -> None:
            if parent_spend > 1.0:
                raise KISSError("Parent budget exceeded.")

        agent.budget_check_hook = check_parent_budget
        try:
            with pytest.raises(KISSError, match="budget exceeded"):
                agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Spend, then run the second tool.",
                    tools=[sub_spend, must_not_run],
                    max_steps=10,
                    max_budget=10.0,
                    verbose=False,
                    model_config={"base_url": url, "api_key": "test-key"},
                )
            assert forbidden_calls == []
        finally:
            srv.shutdown()


# ---------------------------------------------------------------------------
# 2. Mid-session enforcement of spend attributed to the relentless parent
# ---------------------------------------------------------------------------


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
            # Exactly one executor model request/step is allowed.  The
            # budget check immediately after ``sub_spend`` must abort the
            # session, and a budget failure must NEVER launch the LLM
            # summarizer (which would spend more after the limit).
            assert _CheapSubSpendHandler.requests == 1, (
                f"{_CheapSubSpendHandler.requests} model requests ran — a "
                f"budget failure launched more model work (likely the "
                f"RelentlessAgent summarizer)."
            )
            # One executor step plus the three steps attributed by the
            # simulated sub-agent spend.
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
        agent._check_total_budget()  # 0.4 < 1.0 — no raise

        agent.budget_used = 1.2
        with pytest.raises(KISSError, match="budget exceeded"):
            agent._check_total_budget()

        executor = KISSAgent("hook-executor")
        executor.budget_used = 0.7
        agent.budget_used = 0.4
        agent._current_executor = executor
        with pytest.raises(KISSError, match="budget exceeded"):
            agent._check_total_budget()  # 0.4 + 0.7 > 1.0

        executor.budget_used = 0.5
        agent._check_total_budget()  # 0.4 + 0.5 < 1.0 — no raise


# ---------------------------------------------------------------------------
# 3. Fair budget share computation for parallel sub-agents
# ---------------------------------------------------------------------------


class TestSubagentBudgetShare:
    """The parent's remaining budget must be split across sub-tasks."""

    def test_share_divides_remaining_budget(self) -> None:
        agent = SorcarAgent("share")
        agent.max_budget = 1.2
        agent.budget_used = 0.2
        agent._current_executor = None
        assert agent._subagent_budget_share(4) == pytest.approx(0.25)

        executor = KISSAgent("share-executor")
        executor.budget_used = 0.2
        agent._current_executor = executor
        assert agent._subagent_budget_share(2) == pytest.approx(0.4)

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
        with pytest.raises(KISSError, match="budget"):
            agent._subagent_budget_share(2)


# ---------------------------------------------------------------------------
# 4. run_tasks_parallel must cap every sub-agent at the given budget
# ---------------------------------------------------------------------------


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
            # Each sub-agent stopped after ONE $0.375 model call.
            assert 0.7 < totals["budget_used"] < 1.0, (
                f"Sub-agents spent ${totals['budget_used']:.4f} — the "
                f"$0.01 per-task cap was not enforced."
            )
        finally:
            srv.shutdown()


# ---------------------------------------------------------------------------
# 5. End-to-end: a parent's run_parallel distributes its remaining budget
# ---------------------------------------------------------------------------


def _assert_distributed(parent: SorcarAgent, url: str, td: str) -> None:
    """Run *parent* with a $0.10 budget; its run_parallel spawns two
    expensive sub-agents.  Each sub-agent must be capped to ~half the
    remaining budget (stopping after ONE $0.375 call), and the parent must
    stop once the attributed spend exceeds its budget."""
    try:
        parent.run(
            prompt_template="Run two probes in parallel.",
            model_name="gpt-4o-mini",
            model_config={"base_url": url, "api_key": "test-key"},
            work_dir=td,
            is_parallel=True,
            max_steps=5,
            max_sub_sessions=2,
            max_budget=0.10,
        )
    except KISSError:
        pass  # over-budget termination is the expected outcome
    assert parent.budget_used > 0.7, (
        f"Parent budget_used ${parent.budget_used:.4f}: sub-agent spend was "
        f"not attributed back to the parent task."
    )
    assert parent.budget_used < 1.6, (
        f"Parent budget_used ${parent.budget_used:.4f}: sub-agents were not "
        f"capped to a share of the parent's $0.10 budget — a sub-agent "
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
