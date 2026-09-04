# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 (core-base): the text-only implicit finish must obey the
same rules as the stagnation implicit finish.

``KISSAgent._execute_step`` has two "the model is done but never called
``finish``" nets.  The stagnation net (identical tool calls with identical
results) consults ``tool_call_guard`` / ``tool_call_hook`` and returns the
registered ``finish`` tool's contract via ``_implicit_finish_result``.  The
text-only net (``MAX_CONSECUTIVE_NO_TOOL_CALLS`` turns without any tool
call) returned the raw response text unconditionally, which

* bypassed a guard that blocks ``finish`` (Sorcar blocks it while a user
  follow-up is queued, so the follow-up was silently dropped), and
* handed raw text to callers that registered the structured
  :func:`kiss.core.utils.finish` and parse the result as YAML
  (``RelentlessAgent``, ``agents/kiss.py``).

Every test runs a real ``KISSAgent`` against a local OpenAI-compatible
HTTP server; nothing is mocked.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import yaml

from kiss.core.kiss_agent import MAX_CONSECUTIVE_NO_TOOL_CALLS, KISSAgent
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


def _finish_call_response(result: str) -> dict[str, Any]:
    """OpenAI-compatible response calling the built-in ``finish`` tool."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "arguments": json.dumps({"result": result}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
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
        model_config={
            "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
            "api_key": "sk-test",
        },
        tool_call_hook=tool_call_hook,
    )


class _PendingMessageSimulator:
    """Reproduces Sorcar's pre-step drain + finish guard with a real queue.

    A user follow-up is queued while the first model call is in flight.
    The guard rejects ``finish`` while the queue is non-empty; the
    pre-step hook drains the queue into the conversation, exactly like
    ``SorcarAgent._drain_pending_user_messages`` /
    ``_block_finish_when_user_message_pending``.
    """

    def __init__(self) -> None:
        self.queue: list[str] = []
        self.drained: list[str] = []
        self.guard_calls: list[str] = []

    def pre_step(self, model: Any) -> None:
        """Drain queued follow-ups into *model*'s conversation."""
        while self.queue:
            msg = self.queue.pop(0)
            self.drained.append(msg)
            model.add_message_to_conversation("user", f"User says: {msg}.")

    def guard(self, name: str, args: dict[str, Any]) -> str | None:
        """Reject ``finish`` while a follow-up is still queued."""
        del args
        self.guard_calls.append(name)
        if name == "finish" and self.queue:
            return "Error: finish rejected — a user message is pending."
        return None


class TestTextOnlyImplicitFinishHonoursGuard:
    """The text-only net must never finish past a guard that blocks ``finish``."""

    def test_pending_user_message_is_drained_before_text_only_finish(self) -> None:
        """Model answers with text only on every turn.  A follow-up is queued
        while the model call that trips the text-only net is in flight (after
        that step's pre-step drain); the guard blocks ``finish`` until the
        pre-step hook has drained it.  The agent must take one more step
        (draining the message and letting the model see it) instead of
        returning the text while the follow-up is still queued."""
        sim = _PendingMessageSimulator()
        seen_followup_turn = [-1]

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if turn == MAX_CONSECUTIVE_NO_TOOL_CALLS - 1:
                sim.queue.append("also check the lint")
            if seen_followup_turn[0] < 0 and any(
                "also check the lint" in str(m.get("content", ""))
                for m in request.get("messages", [])
            ):
                seen_followup_turn[0] = turn
            return _text_response(f"Status report turn {turn}.")

        server = _serve(respond)
        try:
            agent = KISSAgent("audit-text-only-guard")
            agent.pre_step_hook = sim.pre_step
            agent.tool_call_guard = sim.guard
            result = _run(server, agent)
            assert sim.drained == ["also check the lint"], (
                "the queued follow-up was never drained: the text-only implicit "
                "finish returned before the pre-step hook could run"
            )
            assert seen_followup_turn[0] >= 0, "the model never saw the follow-up"
            assert "finish" in sim.guard_calls, "the guard was never consulted"
            assert result.startswith("Status report turn")
            assert agent.step_count > MAX_CONSECUTIVE_NO_TOOL_CALLS
        finally:
            server.shutdown()

    def test_guard_allows_text_only_finish_when_nothing_pending(self) -> None:
        """With an installed guard but nothing queued, the text-only net
        still finishes after ``MAX_CONSECUTIVE_NO_TOOL_CALLS`` turns."""
        sim = _PendingMessageSimulator()
        server = _serve(lambda turn, request: _text_response("All done."))
        try:
            agent = KISSAgent("audit-text-only-guard-clear")
            agent.pre_step_hook = sim.pre_step
            agent.tool_call_guard = sim.guard
            result = _run(server, agent)
            assert result == "All done."
            assert agent.step_count == MAX_CONSECUTIVE_NO_TOOL_CALLS
            assert sim.guard_calls == ["finish"]
        finally:
            server.shutdown()

    def test_tool_call_hook_can_veto_text_only_finish(self) -> None:
        """A ``tool_call_hook`` returning anything but ``"OK"`` for ``finish``
        suppresses the text-only implicit finish, mirroring its documented
        veto over the stagnation implicit finish.  Once it returns ``"OK"``
        the run ends with the text."""
        verdicts = iter(["not yet, keep going", "OK"])
        hook_calls: list[tuple[str, dict[str, Any]]] = []

        def hook(name: str, args: dict[str, Any]) -> str:
            hook_calls.append((name, args))
            return next(verdicts)

        server = _serve(lambda turn, request: _text_response(f"Text {turn}."))
        try:
            agent = KISSAgent("audit-text-only-hook")
            result = _run(server, agent, tool_call_hook=hook)
            assert hook_calls == [("finish", {}), ("finish", {})]
            assert result == "Text 2."
            assert agent.step_count == MAX_CONSECUTIVE_NO_TOOL_CALLS + 1
        finally:
            server.shutdown()

    def test_vetoed_text_only_finish_still_ends_on_real_finish(self) -> None:
        """A blocked text-only net must not swallow a later real ``finish``
        call: the model finishes normally on the next turn."""

        def guard(name: str, args: dict[str, Any]) -> str | None:
            del args
            return "blocked" if name == "finish" and turn_seen[0] < 2 else None

        turn_seen = [0]

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            turn_seen[0] = turn
            if turn < 2:
                return _text_response("Working...")
            return _finish_call_response("REAL_FINISH")

        server = _serve(respond)
        try:
            agent = KISSAgent("audit-text-only-then-finish")
            agent.tool_call_guard = guard
            result = _run(server, agent)
            assert result == "REAL_FINISH"
            assert agent.step_count == 3
        finally:
            server.shutdown()


class TestTextOnlyImplicitFinishContract:
    """The text-only net must return the registered ``finish`` tool's contract."""

    def test_structured_finish_contract_preserved(self) -> None:
        """With the production structured ``utils.finish`` registered (as
        RelentlessAgent does), a text-only implicit finish must produce
        that tool's YAML — never raw text that ``yaml.safe_load`` turns
        into a string and downstream code indexes as a dict.  The outcome
        is TERMINAL (``success=True, is_continue=False``): the text is the
        answer, and RelentlessAgent must not resume the session (review #1
        fix round; see ``test_audit0902_fix_core_implicit_finish.py``)."""
        server = _serve(
            lambda turn, request: _text_response("The build is green; nothing else to do.")
        )
        try:
            agent = KISSAgent("audit-text-only-contract")
            result = _run(server, agent, tools=[structured_finish])
            payload = yaml.safe_load(result)
            assert isinstance(payload, dict), (
                f"Text-only implicit finish broke the structured finish contract: {result!r}"
            )
            assert payload["success"] is True
            assert payload["is_continue"] is False
            assert "without calling finish" in payload["summary"]
            assert "build is green" in payload["summary"]
        finally:
            server.shutdown()

    def test_plain_finish_contract_returns_text(self) -> None:
        """With the built-in plain ``finish(result)`` the text itself is the
        result (unchanged behaviour)."""
        server = _serve(lambda turn, request: _text_response("Plain answer."))
        try:
            agent = KISSAgent("audit-text-only-plain")
            assert _run(server, agent) == "Plain answer."
        finally:
            server.shutdown()
