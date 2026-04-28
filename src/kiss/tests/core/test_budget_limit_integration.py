"""Integration test: KISSAgent hits budget limit via real HTTP calls.

Starts a real ThreadingHTTPServer that speaks the OpenAI chat-completions
protocol, then calls KISSAgent.run() with a tiny max_budget.  The agent's
agentic loop makes real HTTP requests, accumulates cost from usage data in
the responses, and _check_limits() raises KISSError when the budget is
exceeded — exactly as in production.

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError


def _chat_response_with_tool_call() -> dict:
    """Non-finish tool call so the agent keeps looping."""
    return {
        "id": "chatcmpl-budget",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Calling tool.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "noop",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 500_000,
            "completion_tokens": 500_000,
            "total_tokens": 1_000_000,
        },
    }


class _BudgetTestHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible handler that always returns a non-finish tool call
    with large token usage so the budget is exceeded quickly."""

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length:
            self.rfile.read(content_length)

        body = json.dumps(_chat_response_with_tool_call()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture(scope="module")
def budget_server() -> Generator[str]:
    """Start a real HTTP server for the budget integration test."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BudgetTestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


class TestBudgetLimitViaRealHTTP:
    """KISSAgent.run() exceeds max_budget through real HTTP calls."""

    def test_agent_stops_with_budget_exceeded(self, budget_server: str) -> None:
        """Run the full agentic loop with a tiny budget.

        After the first step the accumulated cost from the server's usage
        data exceeds max_budget, so _check_limits() on the second step
        raises KISSError with 'budget exceeded'.
        """
        agent = KISSAgent("budget-integration")

        def noop() -> str:
            """A no-op tool that does nothing."""
            return "ok"

        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        # Server returns 500k input + 500k output per call:
        #   cost = (500000*0.15 + 500000*0.60) / 1_000_000 = $0.375
        # max_budget=$0.01 → first step costs $0.375 → exceeds budget on step 2
        with pytest.raises(KISSError, match="budget exceeded"):
            agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Do nothing, just call noop.",
                tools=[noop],
                is_agentic=True,
                max_steps=50,
                max_budget=0.01,
                verbose=False,
                model_config={
                    "base_url": budget_server,
                    "api_key": "test-key",
                },
            )

        # The agent ran at least 1 step and accumulated real cost
        assert agent.step_count >= 1
        assert agent.budget_used > 0.01
        assert agent.total_tokens_used > 0

    def test_budget_exceeded_includes_agent_name(self, budget_server: str) -> None:
        """The KISSError message must include the agent's name."""
        agent = KISSAgent("my-named-agent")

        def noop() -> str:
            """A no-op tool."""
            return "ok"

        with pytest.raises(KISSError, match="my-named-agent") as exc_info:
            agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Call noop.",
                tools=[noop],
                is_agentic=True,
                max_steps=50,
                max_budget=0.01,
                verbose=False,
                model_config={
                    "base_url": budget_server,
                    "api_key": "test-key",
                },
            )
        assert "budget exceeded" in str(exc_info.value).lower()

    def test_sufficient_budget_allows_finish(self, budget_server: str) -> None:
        """When the budget is large enough, the agent should NOT raise
        KISSError for budget.  Here we use a server that returns a
        finish tool call so the agent completes normally."""
        # Start a second server that returns finish
        finish_response = {
            "id": "chatcmpl-fin",
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
                                "id": "call_fin",
                                "type": "function",
                                "function": {
                                    "name": "finish",
                                    "arguments": '{"result": "done"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        class _FinishHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                cl = int(self.headers.get("Content-Length", 0))
                if cl:
                    self.rfile.read(cl)
                body = json.dumps(finish_response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _FinishHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        url = f"http://127.0.0.1:{srv.server_port}/v1"

        try:
            agent = KISSAgent("budget-ok")
            result = agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Finish immediately.",
                is_agentic=True,
                max_steps=10,
                max_budget=10.0,
                verbose=False,
                model_config={"base_url": url, "api_key": "test-key"},
            )
            assert result == "done"
            assert agent.budget_used < 10.0
        finally:
            srv.shutdown()
