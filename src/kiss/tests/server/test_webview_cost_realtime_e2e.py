# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: the cost streamed to the chat webview is correct in real time.

Starts a real ThreadingHTTPServer speaking the OpenAI chat-completions
protocol whose responses carry known token usage (including cached
prompt tokens).  A KISSAgent runs the full agentic loop against it with
a JsonPrinter subclass that captures every broadcast event — exactly
the event stream WebPrinter fans out to the chat webview, where
``main.js`` renders ``'Cost: ' + ev.cost`` from each ``usage_info`` /
``result`` event.

The test recomputes the expected cumulative cost independently from the
known usage numbers and hard-coded published prices, and asserts that:

1. one ``usage_info`` event is broadcast per model call (real-time:
   after each step's model response, before the next step), and
2. each event's ``cost`` equals the independently computed cumulative
   spend at that point, and
3. the final ``result`` event repeats the exact total, and
4. per-task ``budget_offset`` (continued sessions / parallel sub-agent
   attribution) is added to every broadcast cost.

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.server.json_printer import JsonPrinter

MODEL = "gpt-4o-mini"

# Published OpenAI prices for gpt-4o-mini (USD per 1M tokens), hard-coded
# as an oracle INDEPENDENT of calculate_cost() so a pricing-table or
# formula regression cannot silently cancel out in this test.
_INPUT_PRICE = 0.15
_CACHED_INPUT_PRICE = 0.075
_OUTPUT_PRICE = 0.60

# (prompt_tokens, completion_tokens, cached_tokens) per model call.
# The third call returns the finish tool call.
_USAGE_PER_CALL: list[tuple[int, int, int]] = [
    (1_000, 200, 0),
    (2_000, 300, 800),
    (3_000, 100, 1_500),
]


def _response_for_call(call_index: int) -> dict[str, Any]:
    """Build the chat-completions response for the *call_index*-th request."""
    prompt, completion, cached = _USAGE_PER_CALL[call_index]
    is_last = call_index == len(_USAGE_PER_CALL) - 1
    if is_last:
        function = {"name": "finish", "arguments": '{"result": "done"}'}
    else:
        function = {"name": "noop", "arguments": "{}"}
    return {
        "id": f"chatcmpl-cost-{call_index}",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{call_index}",
                            "type": "function",
                            "function": function,
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


def _sse_chunks_for_call(index: int) -> list[dict[str, Any]]:
    """Convert the scripted full response into streaming SSE chunk payloads."""
    full = _response_for_call(index)
    message = full["choices"][0]["message"]
    tool_call = message["tool_calls"][0]
    base = {"id": full["id"], "object": "chat.completion.chunk", "model": MODEL}
    return [
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": tool_call["id"],
                                "type": "function",
                                "function": {
                                    "name": tool_call["function"]["name"],
                                    "arguments": tool_call["function"]["arguments"],
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
        {**base, "choices": [], "usage": full["usage"]},
    ]


class _CostServerHandler(BaseHTTPRequestHandler):
    """Serves the scripted responses in request order (streaming or not)."""

    call_count = 0
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        cl = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(cl) if cl else b"{}"
        try:
            wants_stream = bool(json.loads(raw.decode() or "{}").get("stream"))
        except Exception:
            wants_stream = False
        with _CostServerHandler.lock:
            index = min(_CostServerHandler.call_count, len(_USAGE_PER_CALL) - 1)
            _CostServerHandler.call_count += 1
        if wants_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in _sse_chunks_for_call(index):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        body = json.dumps(_response_for_call(index)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _CapturingWebPrinter(JsonPrinter):
    """JsonPrinter that captures the events the webview would receive.

    ``WebPrinter`` (the production subclass) forwards every broadcast
    event over its transport to the chat webview.  This subclass keeps
    the same recording/persistence path and appends each event to a
    list, giving the test the exact stream ``main.js`` consumes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record the event, then run the base recording/persistence path."""
        with self._events_lock:
            self.events.append(dict(event))
        super().broadcast(event)


@pytest.fixture()
def cost_server() -> Generator[str]:
    """Start the scripted OpenAI-compatible server on an ephemeral port."""
    _CostServerHandler.call_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CostServerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def _expected_cumulative_costs() -> list[float]:
    """Recompute the cumulative cost after each model call independently.

    Uses the hard-coded published gpt-4o-mini rates (NOT calculate_cost)
    so this test is an independent oracle for the whole pipeline:
    usage extraction -> pricing table -> cost accumulation -> event stream.
    """
    totals: list[float] = []
    cumulative = 0.0
    for prompt, completion, cached in _USAGE_PER_CALL:
        cumulative += (
            (prompt - cached) * _INPUT_PRICE
            + cached * _CACHED_INPUT_PRICE
            + completion * _OUTPUT_PRICE
        ) / 1_000_000
        totals.append(cumulative)
    return totals


def _run_agent(printer: _CapturingWebPrinter, base_url: str) -> KISSAgent:
    """Run the full agentic loop against the scripted server."""
    agent = KISSAgent("webview-cost-e2e")

    def noop() -> str:
        """A no-op tool that does nothing."""
        return "ok"

    result = agent.run(
        model_name=MODEL,
        prompt_template="Call noop twice, then finish.",
        tools=[noop],
        is_agentic=True,
        max_steps=10,
        max_budget=10.0,
        verbose=False,
        printer=printer,
        model_config={"base_url": base_url, "api_key": "test-key"},
    )
    assert result == "done"
    return agent


class TestWebviewCostRealtime:
    """The cost the webview renders must match the real accumulated spend."""

    def test_usage_info_costs_match_cumulative_spend(self, cost_server: str) -> None:
        """Each per-step usage_info event carries the exact cumulative cost."""
        printer = _CapturingWebPrinter()
        agent = _run_agent(printer, cost_server)

        usage_events = [e for e in printer.events if e.get("type") == "usage_info"]
        assert len(usage_events) == len(_USAGE_PER_CALL)

        expected = _expected_cumulative_costs()
        expected_tokens = 0
        for event, cumulative, (prompt, completion, _) in zip(
            usage_events, expected, _USAGE_PER_CALL, strict=True
        ):
            expected_tokens += prompt + completion
            assert event["cost"] == f"${cumulative:.4f}"
            assert event["total_tokens"] == expected_tokens

        # The agent's own accounting agrees with the last streamed cost.
        assert f"${agent.budget_used:.4f}" == usage_events[-1]["cost"]

    def test_usage_info_streams_in_real_time(self, cost_server: str) -> None:
        """usage_info for step N is broadcast before step N's tool result.

        This is what makes the webview header live: the cost updates as
        soon as each model response arrives, not only when the task ends.
        """
        printer = _CapturingWebPrinter()
        _run_agent(printer, cost_server)

        sequence = [
            e["type"]
            for e in printer.events
            if e.get("type") in ("usage_info", "tool_result", "result")
        ]
        assert sequence == [
            "usage_info",
            "tool_result",
            "usage_info",
            "tool_result",
            "usage_info",
            "result",
        ]

    def test_result_event_cost_matches_total(self, cost_server: str) -> None:
        """The final result event repeats the exact total spend."""
        printer = _CapturingWebPrinter()
        _run_agent(printer, cost_server)

        result_events = [e for e in printer.events if e.get("type") == "result"]
        assert len(result_events) == 1
        assert result_events[0]["cost"] == f"${_expected_cumulative_costs()[-1]:.4f}"

    def test_budget_offset_included_in_streamed_costs(self, cost_server: str) -> None:
        """Prior-session / sub-agent spend (budget_offset) is added live.

        RelentlessAgent snapshots ``printer.budget_offset = budget_used``
        at every session start and ``_attribute_sub_usage`` bumps it when
        parallel sub-agents finish; every subsequently broadcast cost
        must include it.
        """
        offset = 1.25
        printer = _CapturingWebPrinter()
        printer.budget_offset = offset
        _run_agent(printer, cost_server)

        usage_events = [e for e in printer.events if e.get("type") == "usage_info"]
        assert len(usage_events) == len(_USAGE_PER_CALL)
        for event, cumulative in zip(usage_events, _expected_cumulative_costs(), strict=True):
            assert event["cost"] == f"${cumulative + offset:.4f}"

        result_events = [e for e in printer.events if e.get("type") == "result"]
        assert result_events[0]["cost"] == (f"${_expected_cumulative_costs()[-1] + offset:.4f}")
