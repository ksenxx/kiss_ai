# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: RelentlessAgent budget enforcement across sub-sessions.

Starts a real ThreadingHTTPServer that speaks the OpenAI chat-completions
protocol, then calls RelentlessAgent.run() with a tiny max_budget.  The
agent makes real HTTP requests, accumulates cost from usage data in the
responses, and _check_limits() raises KISSError when the budget is
exceeded — exactly as in production.

The plain-KISSAgent half of this contract lives in
``kiss.tests.core.test_budget_limit_integration``.

No mocks, patches, fakes, or test doubles.
"""

from __future__ import annotations

import itertools
import json
import tempfile
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.core.kiss_error import KISSError


def note(text: str) -> str:
    """Record *text*: the tool each scripted session calls before continuing.

    A continuation that only calls ``finish`` is a zero-progress session
    (``relentless_agent.MAX_ZERO_PROGRESS_SESSIONS``); these tests need the
    loop to keep going until the *budget* ends it, so every session first
    makes one real tool call.
    """
    return f"noted: {text}"


def _tool_call_response(name: str, arguments: dict, tokens: int = 40_000) -> dict:
    """Response that calls tool *name* with *arguments* and reports large token usage.

    80k tokens per call (about $0.03 at gpt-4o-mini prices) burns the budget
    within a few sessions yet stays under KISSAgent's context-window limit
    (70% of 128k), which would otherwise end the session before ``finish``.
    """
    return {
        "id": "chatcmpl-cont",
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
                            "id": f"call_{name}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": tokens,
            "completion_tokens": tokens,
            "total_tokens": tokens * 2,
        },
    }


class _ContinueHandler(BaseHTTPRequestHandler):
    """Drives every sub-session as ``note`` then ``finish(is_continue=True)``.

    The reply depends only on the request: a session whose last message is
    not yet a tool result gets the ``note`` call; the request carrying
    ``note``'s result gets a continuation whose summary is numbered, so no
    two sessions look identical to the zero-progress guard.  Each reply
    carries large token usage so the budget, not the session count or the
    guard, ends the run.
    """

    continuations = itertools.count(1)

    def do_POST(self) -> None:  # noqa: N802
        cl = int(self.headers.get("Content-Length", 0))
        messages = json.loads(self.rfile.read(cl)).get("messages", []) if cl else []
        if messages and messages[-1].get("role") == "tool":
            summary = f"did some work (continuation {next(self.continuations)})"
            response = _tool_call_response(
                "finish", {"success": False, "is_continue": True, "summary_in_html": summary}
            )
        else:
            response = _tool_call_response("note", {"text": "working"})
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture(scope="module")
def continue_server() -> Generator[str]:
    """HTTP server that drives every sub-session as ``note`` then finish(is_continue=True)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ContinueHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


class TestRelentlessAgentBudgetAcrossSubSessions:
    """RelentlessAgent must enforce total budget across all sub-sessions.

    Before the fix, each sub-session got the full max_budget, allowing
    the total cost to be max_budget * num_sub_sessions. The fix passes
    remaining_budget to each sub-session's KISSAgent.
    """

    def test_total_budget_not_exceeded(self, continue_server: str) -> None:
        """With max_budget=$0.10, each sub-session costs ~$0.06 (two calls
        of 40k input + 40k output at gpt-4o-mini rates). The second
        sub-session's spend pushes the total past $0.10, so RelentlessAgent
        must raise KISSError during it — after ~$0.12, not many sessions later.

        Before the fix, every sub-session would get max_budget=$0.10 of its
        own and happily run, accumulating total cost far above $0.10.
        """
        agent = RelentlessAgent("budget-cross-session")
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(KISSError, match="budget"):
                agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Keep working.",
                    max_steps=5,
                    max_budget=0.10,
                    max_sub_sessions=10,
                    work_dir=td,
                    verbose=False,
                    tools=[note],
                    model_config={
                        "base_url": continue_server,
                        "api_key": "test-key",
                    },
                )

        assert agent.budget_used > 0.0
        assert agent.budget_used < 0.50, (
            f"Total cost ${agent.budget_used:.4f} far exceeds max_budget $0.10 — "
            f"budget was not enforced across sub-sessions"
        )

    def test_remaining_budget_passed_to_subsession(self, continue_server: str) -> None:
        """Each sub-session's KISSAgent should receive the remaining
        budget, not the full max_budget. Verify by checking that with
        a generous budget and cheap calls, multiple sub-sessions can
        run but eventually the budget is exhausted."""
        agent = RelentlessAgent("budget-remaining")
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(KISSError, match="budget"):
                agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Keep going.",
                    max_steps=5,
                    max_budget=0.50,
                    max_sub_sessions=20,
                    work_dir=td,
                    verbose=False,
                    tools=[note],
                    model_config={
                        "base_url": continue_server,
                        "api_key": "test-key",
                    },
                )

        assert agent.total_steps >= 2
        assert agent.budget_used < 1.00, (
            f"Total cost ${agent.budget_used:.4f} greatly exceeded $0.50"
        )
