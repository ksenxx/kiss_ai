# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: agent must not silently die with "(no result)" when the
model returns empty assistant turns.

Bug reproduction (claude-fable-5 production failures, tasks 3706/3707/3708/3710
in ~/.kiss/sorcar.db):

The model returns an assistant message with empty ``content`` and no tool
calls (e.g. after a streaming/reasoning-block parsing hiccup in the provider
adapter).  In ``KISSAgent._execute_step``, ``function_calls`` is empty and
``response_text`` is ``""``.  After ``MAX_CONSECUTIVE_NO_TOOL_CALLS`` (=2)
such turns, the loop returns ``str(response_text)`` → ``""``.

``JsonPrinter._broadcast_result`` then substitutes the literal string
``"(no result)"`` for the empty body, and downstream ``RelentlessAgent``
parses an empty YAML payload (``yaml.safe_load("") -> None -> {}``) and
returns ``""``, which ``task_runner`` finally persists as the string
``"No summary available"``.  The user sees nothing actionable.

Fix: when the agent has accumulated ``MAX_CONSECUTIVE_NO_TOOL_CALLS``
consecutive empty (whitespace-only, no tool-calls) responses, raise a
``KISSError`` with a clear diagnostic instead of silently returning ``""``.
``RelentlessAgent`` already routes ``KISSError`` into a visible
``success=False`` result, so the user sees the actual cause.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError


def _empty_assistant_response() -> dict[str, Any]:
    """OpenAI-compatible response with empty content and no tool_calls."""
    return {
        "id": "chatcmpl-test-empty",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "total_tokens": 10,
        },
    }


def _tool_call_response(call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test-tool",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20,
        },
    }


class _AlwaysEmptyHandler(BaseHTTPRequestHandler):
    """Always responds with an empty assistant turn (no text, no tool_calls)."""

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length:
            self.rfile.read(content_length)
        body = json.dumps(_empty_assistant_response()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class TestEmptyResponseSilentDeath:
    """Regression for claude-fable-5 ``"(no result)"`` silent task death."""

    def test_always_empty_response_raises_kiss_error(self) -> None:
        """When the model only ever returns empty assistant turns, the agent
        must raise a visible ``KISSError`` instead of silently returning the
        empty string (which downstream renders as ``"(no result)"``)."""
        server = HTTPServer(("127.0.0.1", 0), _AlwaysEmptyHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            agent = KISSAgent("test-always-empty")
            with pytest.raises(KISSError) as excinfo:
                agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="hi",
                    max_steps=20,
                    max_budget=1.0,
                    verbose=False,
                    model_config={
                        "base_url": f"http://127.0.0.1:{port}/v1",
                        "api_key": "sk-test",
                    },
                )
            msg = str(excinfo.value).lower()
            # Diagnostic must mention emptiness so the user can act
            assert "empty" in msg
            # Should have stopped within MAX_CONSECUTIVE_NO_TOOL_CALLS
            assert agent.step_count <= 2, (
                f"Agent took {agent.step_count} steps; expected ≤2 before "
                f"raising on consecutive empty responses"
            )
        finally:
            server.shutdown()

    def test_empty_after_tool_call_raises_kiss_error(self) -> None:
        """Reproduces the exact claude-fable-5 production failure: model
        successfully calls a non-finish tool once, then emits empty assistant
        turns.  The agent must raise ``KISSError``, not silently return ``""``.

        In the production failure (task 3710), the model called Read on a
        file, received the tool_result, and then returned an empty text turn
        — the loop returned ``""`` which rendered as ``"(no result)"`` and
        was persisted as ``"No summary available"``.
        """
        call_count = [0]

        class _ToolThenEmptyHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length:
                    self.rfile.read(content_length)
                idx = call_count[0]
                call_count[0] += 1
                if idx == 0:
                    # First call: model issues a non-finish tool call
                    # (use a built-in tool the agent will execute).
                    resp = _tool_call_response(
                        "call_ls",
                        "Bash",
                        {"command": "echo hi", "description": "test"},
                    )
                else:
                    # All subsequent turns: empty assistant turn
                    resp = _empty_assistant_response()
                body = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        server = HTTPServer(("127.0.0.1", 0), _ToolThenEmptyHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            agent = KISSAgent("test-empty-after-tool")

            def _bash(command: str, description: str) -> str:
                return f"ran: {command}"

            with pytest.raises(KISSError) as excinfo:
                agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="run a command",
                    tools=[_bash],
                    max_steps=20,
                    max_budget=1.0,
                    verbose=False,
                    model_config={
                        "base_url": f"http://127.0.0.1:{port}/v1",
                        "api_key": "sk-test",
                    },
                )
            msg = str(excinfo.value).lower()
            assert "empty" in msg
            # Expected: step 1 = tool call (resets counter), step 2 = empty
            # (counter=1), step 3 = empty (counter=2, raise).
            assert agent.step_count <= 3, (
                f"Agent took {agent.step_count} steps; expected ≤3"
            )
        finally:
            server.shutdown()
