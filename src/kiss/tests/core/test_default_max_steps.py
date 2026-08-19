# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: the default step budget of ``KISSAgent`` is 10000.

When a caller does not pass ``max_steps``, ``KISSAgent`` must allow 10000
steps per session, and the per-step usage banner handed to the model must
advertise that same number.  The RelentlessAgent / SorcarAgent halves of
this contract live in ``kiss.tests.agents.sorcar.test_default_max_steps``,
which reuses this file's ``_ScriptedServer`` harness.
"""

from __future__ import annotations

import http.server
import json
import threading
import unittest

from kiss.core.kiss_agent import KISSAgent

DEFAULT_MAX_STEPS = 10000



def _tool_call_response(name: str, arguments: dict) -> dict:
    """Build a fake OpenAI chat completion that calls the given tool."""
    return {
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
                                "name": name,
                                "arguments": json.dumps(arguments),
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



def ping() -> str:
    """Return a fixed string; a harmless tool used to advance one agent step.

    Returns:
        The literal string ``"pong"``.
    """
    return "pong"



class _ScriptedServer:
    """Fake OpenAI-compatible server replaying a fixed list of tool calls."""

    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        bodies = [json.dumps(_tool_call_response(n, a)).encode() for n, a in calls]
        self.request_bodies: list[dict] = []
        captured = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                try:
                    captured.request_bodies.append(json.loads(raw))
                except json.JSONDecodeError:
                    pass
                body = bodies[min(len(captured.request_bodies) - 1, len(bodies) - 1)]
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        self._server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def model_config(self) -> dict[str, str]:
        """Return the model_config pointing an agent at this fake server."""
        return {"base_url": f"http://127.0.0.1:{self.port}/v1", "api_key": "sk-test"}

    def stop(self) -> None:
        """Shut down the HTTP server."""
        self._server.shutdown()



def _all_text(payload: object) -> str:
    """Flatten every string inside a JSON-like payload into one blob."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        return " ".join(_all_text(v) for v in payload.values())
    if isinstance(payload, list):
        return " ".join(_all_text(v) for v in payload)
    return ""


class TestKISSAgentDefaultMaxSteps(unittest.TestCase):
    """The resolved default of ``max_steps`` is 10000 for ``KISSAgent``."""

    def test_kiss_agent_default_and_banner(self) -> None:
        """``KISSAgent.run`` without ``max_steps`` allows 10000 steps."""
        server = _ScriptedServer(
            [("ping", {}), ("finish", {"result": "done"})]
        )
        try:
            agent = KISSAgent("DefaultStepsKiss")
            agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Call ping, then finish.",
                tools=[ping],
                model_config=server.model_config(),
                verbose=False,
            )
            self.assertEqual(agent.max_steps, DEFAULT_MAX_STEPS)
            self.assertIn(
                f"Steps: 1/{DEFAULT_MAX_STEPS}", _all_text(server.request_bodies)
            )
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
