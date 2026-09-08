# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 fix round (core-base): implicit-finish outcome and veto order.

Review findings #1 and #9 against the audit's implicit-finish change in
``KISSAgent._execute_step``:

* **#1** — the text-only net (``MAX_CONSECUTIVE_NO_TOOL_CALLS`` turns
  without a tool call) returned the structured ``finish`` contract with
  ``is_continue=True``.  ``RelentlessAgent`` resumes every
  ``is_continue=True`` session, so a model that only ever answers in
  text kept the task looping session after session until the budget or
  the sub-session cap.  The documented contract is "treat the last
  response as an implicit finish and return it": the text-only net is
  TERMINAL (``success=True, is_continue=False`` with the text as the
  summary).
* **#9** — ``_implicit_finish_allowed`` consulted ``tool_call_guard``
  before ``tool_call_hook``; a real ``finish`` call runs the hook first
  and skips the guard when the hook rejects (``test_run_hooks.py``).

Every test drives real agents against a local OpenAI-compatible HTTP
server; nothing is mocked.
"""

from __future__ import annotations

import json
import tempfile
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import yaml

from kiss.agents.sorcar.relentless_agent import RelentlessAgent
from kiss.core.kiss_agent import (
    MAX_CONSECUTIVE_NO_TOOL_CALLS,
    KISSAgent,
)
from kiss.core.utils import finish as structured_finish

_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def _text_response(text: str) -> dict[str, Any]:
    """OpenAI-compatible response with text only, no tool calls."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": _USAGE,
    }


def _serve(respond: Callable[[int, dict[str, Any]], dict[str, Any]]) -> HTTPServer:
    """Start a local chat-completions server; ``respond(turn, request)`` builds each reply."""
    turn_counter = [0]

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(content_length)) if content_length else {}
            body = json.dumps(respond(turn_counter[0], request)).encode()
            turn_counter[0] += 1
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _model_config(server: HTTPServer) -> dict[str, Any]:
    """``model_config`` pointing the OpenAI adapter at *server*."""
    return {
        "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
        "api_key": "sk-test",
    }


def _run(
    server: HTTPServer,
    agent: KISSAgent,
    tools: list[Callable[..., Any]] | None = None,
    tool_call_hook: Callable[[str, dict[str, Any]], str] | None = None,
) -> str:
    """Run *agent* against the local server and return its result."""
    return agent.run(
        model_name="gpt-4o-mini",
        prompt_template="Report the status.",
        tools=tools,
        max_steps=12,
        max_budget=5.0,
        verbose=False,
        model_config=_model_config(server),
        tool_call_hook=tool_call_hook,
    )


class TestTextOnlyImplicitFinishIsTerminal:
    """Review #1: the text-only net ends the task, it does not resume it."""

    def test_relentless_agent_terminates_after_one_session_with_text_only_model(self) -> None:
        """A model that ALWAYS answers in plain text must end a
        ``RelentlessAgent`` task after its first session, with the text as
        the result — not loop until ``max_sub_sessions`` is exhausted."""
        requests_seen = [0]

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            del request
            requests_seen[0] = turn + 1
            return _text_response("The deployment is healthy; nothing else to do.")

        server = _serve(respond)
        agent = RelentlessAgent("audit-fix-relentless-text-only")
        try:
            with tempfile.TemporaryDirectory() as td:
                result = agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Check the deployment.",
                    max_steps=10,
                    max_budget=5.0,
                    max_sub_sessions=4,
                    work_dir=td,
                    verbose=False,
                    model_config=_model_config(server),
                )
            payload = yaml.safe_load(result)
            assert isinstance(payload, dict), result
            assert payload["is_continue"] is False, (
                f"text-only implicit finish was resumable, RelentlessAgent kept looping: {result!r}"
            )
            assert payload["success"] is True
            assert "deployment is healthy" in payload["summary"]
            assert "Previous Session" not in payload["summary"], (
                "more than one sub-session ran for a text-only model"
            )
            assert agent.total_steps == MAX_CONSECUTIVE_NO_TOOL_CALLS
            assert requests_seen[0] == MAX_CONSECUTIVE_NO_TOOL_CALLS
        finally:
            server.shutdown()

    def test_kiss_agent_text_only_structured_contract_is_terminal_success(self) -> None:
        """With the structured ``utils.finish`` registered, the text-only net
        returns ``success=True, is_continue=False`` and the model's text."""
        server = _serve(lambda turn, request: _text_response("All checks passed."))
        try:
            agent = KISSAgent("audit-fix-text-only-contract")
            payload = yaml.safe_load(_run(server, agent, tools=[structured_finish]))
            assert payload == {
                "success": True,
                "is_continue": False,
                "summary": payload["summary"],
            }
            assert "All checks passed." in payload["summary"]
            assert agent.step_count == MAX_CONSECUTIVE_NO_TOOL_CALLS
        finally:
            server.shutdown()

class _VetoRecorder:
    """Real ``tool_call_hook`` + ``tool_call_guard`` pair that records call order."""

    def __init__(self, hook_verdicts: list[str]) -> None:
        self.calls: list[str] = []
        self._verdicts = iter(hook_verdicts)

    def hook(self, name: str, args: dict[str, Any]) -> str:
        """Record ``hook:<name>`` and answer with the next scripted verdict."""
        del args
        self.calls.append(f"hook:{name}")
        return next(self._verdicts)

    def guard(self, name: str, args: dict[str, Any]) -> str | None:
        """Record ``guard:<name>`` and never object."""
        del args
        self.calls.append(f"guard:{name}")
        return None


class TestImplicitFinishHookBeforeGuard:
    """Review #9: hook first; the guard runs only after an ``"OK"`` verdict."""

    def test_text_only_net_skips_guard_when_hook_rejects(self) -> None:
        """Turn 2 trips the text-only net: the hook says "not yet" and the
        guard must NOT be consulted.  Turn 3: the hook says OK, then the
        guard runs, and the run ends with the text."""
        recorder = _VetoRecorder(["not yet", "OK"])
        server = _serve(lambda turn, request: _text_response(f"Text {turn}."))
        try:
            agent = KISSAgent("audit-fix-veto-order-text")
            agent.tool_call_guard = recorder.guard
            result = _run(server, agent, tool_call_hook=recorder.hook)
            assert recorder.calls == ["hook:finish", "hook:finish", "guard:finish"], (
                f"guard consulted before / despite the hook's rejection: {recorder.calls}"
            )
            assert result == "Text 2."
        finally:
            server.shutdown()
