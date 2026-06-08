# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: RelentlessAgent must include the summarizer's cost.

When an executor sub-session exhausts its step limit, ``RelentlessAgent``
invokes a separate :class:`KISSAgent` summarizer to compress the
trajectory into a continuation summary. The summarizer's own model
calls cost real money. Before the fix, that cost was silently dropped:
:meth:`RelentlessAgent.perform_task` only added
``executor.budget_used`` (and ``executor.total_tokens_used`` /
``step_count``) to the parent agent's running totals and never folded
in ``summarizer_agent.budget_used``.

This test starts a real ThreadingHTTPServer that speaks the OpenAI
chat-completions protocol, distinguishes summarizer requests from
executor requests by inspecting the prompt body, and verifies that
every HTTP call the agent makes is reflected in ``agent.budget_used``
and ``agent.total_tokens_used``. No mocks, patches, fakes, or doubles.
"""

from __future__ import annotations

import json
import tempfile
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import calculate_cost
from kiss.core.relentless_agent import RelentlessAgent

# Per-call usage that the fake server reports back. Picked small enough
# that several summarizer + executor calls comfortably fit under
# ``max_budget`` (so the summarizer actually runs and is not short-
# circuited by ``summarizer_budget <= 0``).
_PROMPT_TOKENS = 1000
_COMPLETION_TOKENS = 100
_MODEL = "gpt-4o-mini"


def _executor_response() -> dict:
    """Non-finish tool call — keeps the executor looping until ``max_steps``."""
    return {
        "id": "chatcmpl-exec",
        "object": "chat.completion",
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Calling noop.",
                    "tool_calls": [
                        {
                            "id": "call_noop",
                            "type": "function",
                            "function": {"name": "noop", "arguments": "{}"},
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
    """``finish(success=True, summary="...")`` so the summarizer returns immediately."""
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


# Per-server counters mutated by the request handler thread(s). Module-
# level so the fixture can hand them to the test, which reads the totals
# after the agent finishes.
_call_counts: dict[str, int] = {"executor": 0, "summarizer": 0}
_call_counts_lock = threading.Lock()


class _RelentlessBudgetHandler(BaseHTTPRequestHandler):
    """Routes by prompt content: summarizer prompts get a finish call,
    everything else gets a non-finish tool call.
    """

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length else b""

        # Distinguish summarizer from executor by inspecting the prompt
        # body: the executor sends ``TASK_PROMPT`` (no "Summarizer"
        # token), while the summarizer sends ``SUMMARIZER_PROMPT`` whose
        # first heading is "# Summarizer".
        is_summarizer = "Summarizer" in body_bytes.decode(errors="ignore")

        with _call_counts_lock:
            if is_summarizer:
                _call_counts["summarizer"] += 1
            else:
                _call_counts["executor"] += 1

        payload = (
            _summarizer_finish_response()
            if is_summarizer
            else _executor_response()
        )
        body_out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body_out)))
        self.end_headers()
        self.wfile.write(body_out)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def relentless_budget_server() -> Generator[str]:
    """Start a real HTTP server and reset per-test call counters."""
    with _call_counts_lock:
        _call_counts["executor"] = 0
        _call_counts["summarizer"] = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RelentlessBudgetHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()


class TestRelentlessAgentSummarizerBudgetIncluded:
    """The summarizer's model cost must be added to ``agent.budget_used``."""

    def test_summarizer_cost_added_to_total_budget(
        self, relentless_budget_server: str
    ) -> None:
        """Force at least one executor → summarizer cycle and verify
        ``budget_used`` equals the cost of every HTTP call the agent made.

        With ``max_steps=3`` the executor's ``_check_limits`` raises
        ``KISSError`` once ``step_count > 3``. Because no cause is
        chained and ``step_count > 1``, ``perform_task`` enters the
        summarizer branch; the summarizer immediately calls ``finish``
        with ``success=True`` and the session returns
        ``is_continue=True``. The for-loop then exits after
        ``max_sub_sessions=1`` and raises the exhaustion ``KISSError``.

        Total expected cost = (executor_calls + summarizer_calls) * cost_per_call.
        Before the fix, ``summarizer_agent.budget_used`` was dropped, so
        ``agent.budget_used`` equalled only the executor portion.
        """
        agent = RelentlessAgent("summarizer-budget-undercount")
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(KISSError):
                agent.run(
                    model_name=_MODEL,
                    prompt_template="Do nothing forever.",
                    max_steps=3,
                    max_budget=1.00,
                    max_sub_sessions=1,
                    work_dir=td,
                    verbose=False,
                    model_config={
                        "base_url": relentless_budget_server,
                        "api_key": "test-key",
                    },
                )

        with _call_counts_lock:
            executor_calls = _call_counts["executor"]
            summarizer_calls = _call_counts["summarizer"]

        # Sanity: the summarizer branch must have been exercised at all
        # for this test to be meaningful.
        assert executor_calls > 0, "executor never called the model"
        assert summarizer_calls > 0, (
            "summarizer never called the model — the test cannot prove the "
            "undercount because there was no summarizer cost to drop"
        )

        cost_per_call = calculate_cost(_MODEL, _PROMPT_TOKENS, _COMPLETION_TOKENS)
        expected_budget = (executor_calls + summarizer_calls) * cost_per_call
        executor_only_budget = executor_calls * cost_per_call

        # If the bug is present, ``agent.budget_used`` ≈ ``executor_only_budget``
        # and the summarizer's cost is silently lost. After the fix,
        # ``agent.budget_used`` ≈ ``expected_budget``.
        assert agent.budget_used == pytest.approx(expected_budget, rel=1e-6), (
            f"budget undercount detected: agent.budget_used="
            f"${agent.budget_used:.6f} but expected ${expected_budget:.6f} "
            f"(executor-only would be ${executor_only_budget:.6f}). "
            f"Executor calls: {executor_calls}, summarizer calls: "
            f"{summarizer_calls}, cost/call: ${cost_per_call:.6f}."
        )

        expected_tokens = (
            (executor_calls + summarizer_calls) * (_PROMPT_TOKENS + _COMPLETION_TOKENS)
        )
        assert agent.total_tokens_used == expected_tokens, (
            f"token undercount: agent.total_tokens_used={agent.total_tokens_used} "
            f"but expected {expected_tokens}"
        )
