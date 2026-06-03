# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: KISSAgent must not loop forever when the model never calls tools.

Bug: when a model (especially text-based tool-calling models like CodexModel)
responds with plain text but zero tool calls, ``_execute_step`` sends a retry
message and returns ``None``.  The agentic loop continues, the model responds
again without tool calls, and this repeats for up to ``max_steps`` iterations.
For example, sending "hi" to a codex model causes ~100 sequential CLI
invocations before the step limit is hit.

Fix: track consecutive no-tool-call responses.  After
``MAX_CONSECUTIVE_NO_TOOL_CALLS`` consecutive text-only responses (currently
2), treat the last response as an implicit finish and return it.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from kiss.core.kiss_agent import MAX_CONSECUTIVE_NO_TOOL_CALLS, KISSAgent


def _make_text_only_response(text: str = "Hi! How can I help you?") -> dict[str, Any]:
    """OpenAI-compatible response with text only, no tool calls."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


class _TextOnlyHandler(BaseHTTPRequestHandler):
    """Always responds with plain text, no tool calls."""

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length:
            self.rfile.read(content_length)
        body = json.dumps(_make_text_only_response()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class TestNoToolCallLoop:
    """KISSAgent must return quickly when the model never calls tools."""

    def test_agent_returns_after_consecutive_no_tool_call_responses(self) -> None:
        """When the model consistently returns text without tool calls, the
        agent must return the text as the result within a small number of
        steps, not loop for all max_steps."""
        server = HTTPServer(("127.0.0.1", 0), _TextOnlyHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            agent = KISSAgent("test-no-tool-loop")
            result = agent.run(
                model_name="test-model",
                prompt_template="hi",
                max_steps=20,
                max_budget=1.0,
                verbose=False,
                model_config={
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "api_key": "sk-test",
                },
            )
            # Should return within MAX_CONSECUTIVE_NO_TOOL_CALLS steps
            assert agent.step_count <= MAX_CONSECUTIVE_NO_TOOL_CALLS, (
                f"Agent looped {agent.step_count} times; expected "
                f"≤{MAX_CONSECUTIVE_NO_TOOL_CALLS} for a no-tool-call model"
            )
            # The result should contain the model's response text
            assert "Hi" in result
        finally:
            server.shutdown()

    def test_tool_call_resets_no_tool_call_counter(self) -> None:
        """A successful tool call should reset the consecutive no-tool-call
        counter, so a single text-only response after tool calls does not
        trigger early exit."""
        call_count = [0]

        class _MixedHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length:
                    self.rfile.read(content_length)
                idx = call_count[0]
                call_count[0] += 1
                if idx == 0:
                    # First call: model calls finish tool
                    resp: dict[str, Any] = {
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "finish",
                                                "arguments": json.dumps(
                                                    {"result": "Done!"}
                                                ),
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
                else:
                    resp = _make_text_only_response()
                body = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        server = HTTPServer(("127.0.0.1", 0), _MixedHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            agent = KISSAgent("test-mixed-calls")
            result = agent.run(
                model_name="test-model",
                prompt_template="do something",
                max_steps=20,
                max_budget=1.0,
                verbose=False,
                model_config={
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "api_key": "sk-test",
                },
            )
            # finish was called on the first step
            assert agent.step_count == 1
            assert "Done!" in result
        finally:
            server.shutdown()
