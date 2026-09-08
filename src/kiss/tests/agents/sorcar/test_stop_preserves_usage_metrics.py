# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: a stop must not zero the task's usage metrics.

The VS Code daemon stops a task by injecting ``KeyboardInterrupt``
into the agent thread. ``RelentlessAgent.perform_task`` folded the live
executor session's spend into ``agent.budget_used`` /
``agent.total_tokens_used`` / ``agent.total_steps`` on
``BudgetExceededError`` and on ``except Exception`` — but a
``KeyboardInterrupt`` is a ``BaseException``, so it propagated WITHOUT
:meth:`RelentlessAgent._accumulate_usage`. The task runner's
``_subtask_metrics`` then read zeros, and the "Task stopped by user"
``result`` event (and the persisted ``task_history`` row) reported
``$0.0000`` / 0 tokens for a task that had burned real money — the
chat webview showed "Cost $0.00".

This test drives a real :class:`RelentlessAgent` against a real
``ThreadingHTTPServer`` speaking the OpenAI chat-completions protocol.
A registered tool raises ``KeyboardInterrupt`` on its second call —
exactly how a stop surfaces mid-step (``KISSAgent``'s tool wrapper
catches only ``(Exception, SystemExit)``, so the interrupt propagates
like the watchdog's injection). No mocks, patches, fakes, or doubles.
"""

from __future__ import annotations

import ctypes
import json
import tempfile
import threading
import time
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.core.models.model_info import calculate_cost
from kiss.server.task_runner import _subtask_metrics

_PROMPT_TOKENS = 1000
_COMPLETION_TOKENS = 100
_MODEL = "gpt-4o-mini"

_http_calls: dict[str, int] = {"count": 0}
_poke_calls: dict[str, int] = {"count": 0}
_lock = threading.Lock()


def poke() -> str:
    """No-op tool; raises ``KeyboardInterrupt`` on its second call.

    Returns:
        A fixed acknowledgement string on the first call.

    Raises:
        KeyboardInterrupt: On the second and later calls, standing in
            for the daemon's stop injection landing inside a tool.
    """
    with _lock:
        _poke_calls["count"] += 1
        n = _poke_calls["count"]
    if n >= 2:
        raise KeyboardInterrupt("user stop")
    return "ok"


def _poke_response() -> dict:
    """A chat completion that calls the ``poke`` tool."""
    return {
        "id": "chatcmpl-poke",
        "object": "chat.completion",
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Poking.",
                    "tool_calls": [
                        {
                            "id": "call_poke",
                            "type": "function",
                            "function": {"name": "poke", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": _PROMPT_TOKENS,
            "completion_tokens": _COMPLETION_TOKENS,
            "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
        },
    }


class _StopUsageHandler(BaseHTTPRequestHandler):
    """Always answers with a ``poke`` tool call and counts the calls."""

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length:
            self.rfile.read(content_length)
        with _lock:
            _http_calls["count"] += 1
        body = json.dumps(_poke_response()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def stop_usage_server() -> Generator[str]:
    """Start a real HTTP server and reset the per-test counters."""
    with _lock:
        _http_calls["count"] = 0
        _poke_calls["count"] = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StopUsageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()


class TestStopPreservesUsageMetrics:
    """A ``KeyboardInterrupt`` exit must keep the run's real usage."""

    def test_interrupt_keeps_cost_tokens_and_steps(
        self, stop_usage_server: str
    ) -> None:
        """Interrupt the executor mid-run and verify the counters.

        The model answers every call with a ``poke`` tool call; the
        tool raises ``KeyboardInterrupt`` on its second invocation, so
        the executor makes exactly two paid HTTP calls before the
        interrupt unwinds ``RelentlessAgent.run``. The agent's grand
        totals — the very counters ``_subtask_metrics`` feeds into the
        stopped task's ``result`` event and ``task_history`` row —
        must reflect both calls.
        """
        agent = RelentlessAgent("stop-preserves-usage")
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(KeyboardInterrupt):
                agent.run(
                    model_name=_MODEL,
                    prompt_template="Keep poking.",
                    max_steps=50,
                    max_budget=1.00,
                    max_sub_sessions=1,
                    work_dir=td,
                    verbose=False,
                    tools=[poke],
                    model_config={
                        "base_url": stop_usage_server,
                        "api_key": "test-key",
                    },
                )

        with _lock:
            http_calls = _http_calls["count"]
            poke_calls = _poke_calls["count"]
        assert poke_calls == 2, "the tool must have been interrupted mid-run"
        assert http_calls >= 2, "the executor must have made paid calls"

        cost_per_call = calculate_cost(
            _MODEL, _PROMPT_TOKENS, _COMPLETION_TOKENS
        )
        expected_cost = http_calls * cost_per_call
        expected_tokens = http_calls * (_PROMPT_TOKENS + _COMPLETION_TOKENS)

        assert agent.budget_used == pytest.approx(expected_cost, rel=1e-6), (
            f"stop lost the live session's spend: budget_used="
            f"{agent.budget_used}, expected {expected_cost}"
        )
        assert agent.total_tokens_used == expected_tokens

        # The exact values the task runner broadcasts in the stopped
        # task's `result` event and persists on the task_history row.
        tokens, cost, steps = _subtask_metrics(agent)
        assert cost == pytest.approx(expected_cost, rel=1e-6)
        assert tokens == expected_tokens
        assert steps >= 2, "both completed steps must be counted"


_summarizer_started = threading.Event()
_executor_http_calls: dict[str, int] = {"count": 0}


def poke_forever() -> str:
    """No-op tool that always succeeds (keeps the executor looping).

    Returns:
        A fixed acknowledgement string.
    """
    return "ok"


def _poke_forever_response() -> dict:
    """A chat completion that calls the ``poke_forever`` tool."""
    return {
        "id": "chatcmpl-poke-forever",
        "object": "chat.completion",
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Poking.",
                    "tool_calls": [
                        {
                            "id": "call_poke_forever",
                            "type": "function",
                            "function": {
                                "name": "poke_forever",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": _PROMPT_TOKENS,
            "completion_tokens": _COMPLETION_TOKENS,
            "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
        },
    }


def _summarizer_finish_response() -> dict:
    """``finish(success=True, summary=...)`` so the summarizer returns."""
    args = json.dumps({"success": True, "summary": "summary-from-test"})
    return {
        "id": "chatcmpl-sum",
        "object": "chat.completion",
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_fin",
                            "type": "function",
                            "function": {"name": "finish", "arguments": args},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": _PROMPT_TOKENS,
            "completion_tokens": _COMPLETION_TOKENS,
            "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
        },
    }


class _StopDuringSummarizerHandler(BaseHTTPRequestHandler):
    """Executor calls loop on ``poke``; the summarizer call stalls.

    The stall gives the test time to inject ``KeyboardInterrupt`` into
    the agent thread while the summarizer is mid-request — the interrupt
    materializes as soon as the thread resumes running bytecode, i.e.
    inside the summarizer's session.
    """

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length else b""
        is_summarizer = "Summarizer" in body_bytes.decode(errors="ignore")
        if is_summarizer:
            _summarizer_started.set()
            time.sleep(1.2)
            payload = _summarizer_finish_response()
        else:
            with _lock:
                _executor_http_calls["count"] += 1
            payload = _poke_forever_response()
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def stop_during_summarizer_server() -> Generator[str]:
    """Start a real HTTP server for the stop-during-summarizer test."""
    _summarizer_started.clear()
    with _lock:
        _executor_http_calls["count"] = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StopDuringSummarizerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()


class TestStopDuringSummarizerPreservesUsage:
    """A stop landing in the failed-session summarizer keeps the spend."""

    def test_interrupt_during_summarizer_keeps_executor_cost(
        self, stop_during_summarizer_server: str
    ) -> None:
        """Inject the stop while the summarizer's model call is stalled.

        With ``max_steps=3`` the executor exceeds its step limit, so
        ``perform_task`` enters the summarizer branch. The daemon's
        Stop watchdog injects ``KeyboardInterrupt`` into the agent
        thread with ``PyThreadState_SetAsyncExc`` — this test does
        exactly the same (no mocks) while the summarizer's HTTP call is
        stalled server-side, so the interrupt lands inside the
        summarizer session. Before the fix, the FAILED executor
        session's spend was banked only after the summarizer returned,
        and the interrupt lost it: the stopped task reported only the
        summarizer's own usage (or nothing).
        """
        agent = RelentlessAgent("stop-during-summarizer")
        outcome: dict[str, BaseException | None] = {"exc": None}
        thread_ids: dict[str, int] = {}
        ready = threading.Event()

        def run_agent(td: str) -> None:
            """Run the agent in a worker thread, capturing its exit."""
            thread_ids["agent"] = threading.get_ident()
            ready.set()
            try:
                agent.run(
                    model_name=_MODEL,
                    prompt_template="Keep poking.",
                    max_steps=3,
                    max_budget=1.00,
                    max_sub_sessions=1,
                    work_dir=td,
                    verbose=False,
                    tools=[poke_forever],
                    model_config={
                        "base_url": stop_during_summarizer_server,
                        "api_key": "test-key",
                    },
                )
            except BaseException as exc:  # noqa: BLE001 — capture for asserts
                outcome["exc"] = exc

        with tempfile.TemporaryDirectory() as td:
            worker = threading.Thread(target=run_agent, args=(td,))
            worker.start()
            assert ready.wait(timeout=10), "agent thread never started"
            assert _summarizer_started.wait(timeout=60), (
                "the summarizer never called the model — the test cannot "
                "exercise the stop-during-summarizer path"
            )
            # The summarizer's HTTP call is stalled for another second;
            # inject now, exactly like the daemon's stop watchdog.
            time.sleep(0.1)
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread_ids["agent"]),
                ctypes.py_object(KeyboardInterrupt),
            )
            assert res == 1, "async exception injection must hit one thread"
            worker.join(timeout=60)
            assert not worker.is_alive(), "agent thread must have exited"

        assert isinstance(outcome["exc"], KeyboardInterrupt), (
            f"the stop must unwind the agent, got {outcome['exc']!r}"
        )

        with _lock:
            executor_calls = _executor_http_calls["count"]
        assert executor_calls >= 3, "the executor must have made paid calls"

        cost_per_call = calculate_cost(
            _MODEL, _PROMPT_TOKENS, _COMPLETION_TOKENS
        )
        executor_cost = executor_calls * cost_per_call
        # The failed session is banked BEFORE the summarizer runs, so
        # the interrupt cannot lose it. The summarizer's own partial
        # spend may or may not have been counted by injection time,
        # hence >= rather than ==.
        assert agent.budget_used >= executor_cost * (1 - 1e-6), (
            f"stop during the summarizer lost the failed session's spend: "
            f"budget_used={agent.budget_used}, executor portion "
            f"{executor_cost}"
        )
        assert agent.total_tokens_used >= executor_calls * (
            _PROMPT_TOKENS + _COMPLETION_TOKENS
        )
