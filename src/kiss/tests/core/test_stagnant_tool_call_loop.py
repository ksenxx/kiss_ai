# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: KISSAgent must escape a done-but-not-finishing model.

Bug: the step loop's only implicit-finish net (two consecutive text-only
turns) is defeated by the loop's own "you MUST make a function call" retry
nudge.  A model that has finished its work but never calls ``finish`` learns
to attach one harmless read-only verification call to every status turn, and
each tool call resets ``_consecutive_no_tool_calls`` to zero, so the net can
never trip and the agent burns through ``max_steps``/budget.

Fix (two parts):

1. The retry nudge now tells the model to call ``finish`` if the task is
   complete, so the trained-in behavior is the correct exit rather than a
   token verification call.
2. A stagnation net that harmless calls cannot defeat: when the model repeats
   the *identical* tool call(s) and gets *identical* results for
   ``STAGNANT_TURNS_REMINDER`` consecutive turns it is reminded to call
   ``finish`` (or to vary its action if it is waiting on a slow process); at
   ``STAGNANT_TURNS_FINISH`` turns the agent implicitly finishes *through the
   registered finish tool's contract*: the structured
   :func:`kiss.core.utils.finish` contract ends the session as
   ``success=False, is_continue=True`` — resumable, never a fake success —
   while the built-in ``finish(result)`` contract returns the model's last
   status text.  Polling whose results change never counts as stagnant.

Every test here runs the real ``KISSAgent.run`` against a local HTTP server
speaking the OpenAI chat-completions protocol — no mocks or patches.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
import yaml

from kiss.core.kiss_agent import (
    STAGNANT_TURNS_FINISH,
    STAGNANT_TURNS_REMINDER,
    KISSAgent,
)
from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import MODEL_INFO, ModelInfo
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


def _tool_call_response(
    text: str, name: str, arguments: dict[str, Any] | None = None
) -> dict[str, Any]:
    """OpenAI-compatible response with text plus one tool call."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments or {}),
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


def _run_agent(server: HTTPServer, tools: list[Callable[..., Any]], max_steps: int = 30) -> tuple[
    str, KISSAgent
]:
    """Run a real KISSAgent against the local server and return (result, agent)."""
    agent = KISSAgent("test-stagnant-loop")
    result = agent.run(
        model_name="gpt-4o-mini",
        prompt_template="Verify the build is green and report the status.",
        tools=tools,
        max_steps=max_steps,
        max_budget=5.0,
        verbose=False,
        model_config={
            "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
            "api_key": "sk-test",
        },
    )
    return result, agent


def check_build() -> str:
    """Deterministic read-only verification tool: always reports success."""
    return "build: all 42 tests passed"


class TestStagnantToolCallLoop:
    """A done model padding every turn with a harmless call must still terminate."""

    def test_identical_call_every_turn_implicitly_finishes(self) -> None:
        """The exact failure mode from the bug report: every turn is status
        text plus the same harmless verification call, ``finish`` is never
        called.  The agent must implicitly finish with the status text
        instead of looping to max_steps."""

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            return _tool_call_response(
                "The task is complete; the build is green.", "check_build"
            )

        server = _serve(respond)
        try:
            result, agent = _run_agent(server, [check_build])
            assert agent.step_count <= STAGNANT_TURNS_FINISH + 1, (
                f"Agent looped {agent.step_count} steps; the stagnation net "
                f"should have tripped by step {STAGNANT_TURNS_FINISH}"
            )
            assert "complete" in result
        finally:
            server.shutdown()

    def test_oscillating_text_and_identical_call_terminates(self) -> None:
        """The oscillation variant: text-only turn (nudged) alternating with
        a status turn carrying one identical verification call.  The
        no-tool-call counter oscillates 0<->1 forever; the stagnation net
        must still trip because the calls and results never change."""

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if turn % 2 == 0:
                return _text_response("Everything looks good; work is done.")
            return _tool_call_response("Re-verifying just in case.", "check_build")

        server = _serve(respond)
        try:
            result, agent = _run_agent(server, [check_build])
            assert agent.step_count <= 2 * STAGNANT_TURNS_FINISH + 2, (
                f"Agent looped {agent.step_count} steps without terminating"
            )
            assert result.strip()
        finally:
            server.shutdown()

    def test_changing_results_are_not_stagnation(self) -> None:
        """Legitimate polling — the same call whose *results change* — must
        never trip the stagnation net; the agent runs until the model calls
        ``finish`` well past STAGNANT_TURNS_FINISH turns."""
        poll_turns = STAGNANT_TURNS_FINISH + 3
        tick = [0]

        def poll_build() -> str:
            """Polling tool whose result changes every call."""
            tick[0] += 1
            return f"build progress: {tick[0]}%"

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if turn < poll_turns:
                return _tool_call_response("Still building; polling again.", "poll_build")
            return _tool_call_response(
                "Build finished.", "finish", {"result": "BUILD_DONE"}
            )

        server = _serve(respond)
        try:
            result, agent = _run_agent(server, [poll_build])
            assert result == "BUILD_DONE"
            assert agent.step_count == poll_turns + 1, (
                f"Agent finished at step {agent.step_count}; expected "
                f"{poll_turns + 1} — a premature exit means polling was "
                f"wrongly treated as stagnation"
            )
        finally:
            server.shutdown()

    def test_stagnation_reminder_directs_model_to_finish(self) -> None:
        """After STAGNANT_TURNS_REMINDER identical turns the agent must
        inject a reminder naming the ``finish`` tool; a model that obeys it
        exits cleanly before the implicit-finish threshold."""
        saw_reminder_at = [0]

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            texts = [
                str(m.get("content")) for m in request.get("messages", []) if m.get("content")
            ]
            reminders = [
                t for t in texts if "identical results" in t and "consecutive turns" in t
            ]
            if reminders:
                assert all("finish" in t for t in reminders), (
                    "The stagnation reminder must direct the model to the finish tool"
                )
                saw_reminder_at[0] = turn
                return _tool_call_response("Done.", "finish", {"result": "REMINDED_DONE"})
            return _tool_call_response("Verifying again.", "check_build")

        server = _serve(respond)
        try:
            result, agent = _run_agent(server, [check_build])
            assert result == "REMINDED_DONE"
            assert saw_reminder_at[0] == STAGNANT_TURNS_REMINDER, (
                f"Reminder reached the model at turn {saw_reminder_at[0]}; "
                f"expected turn {STAGNANT_TURNS_REMINDER}"
            )
        finally:
            server.shutdown()

    def test_no_tool_call_nudge_mentions_finish(self) -> None:
        """The retry nudge for a text-only turn must direct a completed
        model to the ``finish`` tool (the correct exit), not merely demand
        an arbitrary function call."""

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            texts = [
                str(m.get("content")) for m in request.get("messages", []) if m.get("content")
            ]
            nudges = [t for t in texts if "MUST have at least one function call" in t]
            if nudges:
                assert all("finish" in t for t in nudges), (
                    "The retry nudge must tell the model to call finish when done"
                )
                return _tool_call_response("Done.", "finish", {"result": "NUDGED_DONE"})
            return _text_response("I believe the task is complete.")

        server = _serve(respond)
        try:
            result, agent = _run_agent(server, [check_build])
            assert result == "NUDGED_DONE"
            assert agent.step_count == 2
        finally:
            server.shutdown()

    def test_different_calls_reset_stagnation(self) -> None:
        """Turns whose tool calls differ from the previous turn must reset
        the stagnation counter: alternating two distinct verification calls
        never trips the net, and the run ends only via the model's finish."""
        distinct_turns = STAGNANT_TURNS_FINISH + 3

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if turn < distinct_turns:
                name = "check_build" if turn % 2 == 0 else "check_lint"
                return _tool_call_response("Checking.", name)
            return _tool_call_response("Done.", "finish", {"result": "ALL_CHECKED"})

        def check_lint() -> str:
            """Deterministic read-only lint check."""
            return "lint: clean"

        server = _serve(respond)
        try:
            result, agent = _run_agent(server, [check_build, check_lint])
            assert result == "ALL_CHECKED"
            assert agent.step_count == distinct_turns + 1
        finally:
            server.shutdown()

    def test_blocked_finish_never_escalates_to_implicit_finish(self) -> None:
        """A ``tool_call_guard`` that blocks ``finish`` (e.g. while a user
        message is pending) produces identical blocked calls and results,
        but the stagnation net must never implicitly finish past the guard;
        the run ends only when the guard finally allows finish."""
        blocked_turns = STAGNANT_TURNS_FINISH + 2
        finish_attempts = [0]

        def guard(name: str, args: dict[str, Any]) -> str | None:
            if name == "finish":
                finish_attempts[0] += 1
                if finish_attempts[0] <= blocked_turns:
                    return "finish blocked: a pending user message must be handled first"
            return None

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            return _tool_call_response("Done.", "finish", {"result": "GUARDED_DONE"})

        server = _serve(respond)
        try:
            agent = KISSAgent("test-guarded-finish")
            agent.tool_call_guard = guard
            result = agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Finish when allowed.",
                tools=[check_build],
                max_steps=30,
                max_budget=5.0,
                verbose=False,
                model_config={
                    "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                    "api_key": "sk-test",
                },
            )
            assert result == "GUARDED_DONE", (
                f"Got {result!r} — the stagnation net must not bypass the "
                f"finish-blocking guard"
            )
            assert agent.step_count == blocked_turns + 1
        finally:
            server.shutdown()

    def test_structured_finish_contract_preserved(self) -> None:
        """With the production structured ``utils.finish`` tool registered
        (as RelentlessAgent does), the implicit finish must return that
        tool's YAML contract — ``success: false, is_continue: true`` with
        the stall explained — never raw text that downstream YAML parsers
        would drop metadata from or mistake for a successful result."""

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            return _tool_call_response(
                "The task is complete; the build is green.", "check_build"
            )

        server = _serve(respond)
        try:
            result, agent = _run_agent(server, [structured_finish, check_build])
            payload = yaml.safe_load(result)
            assert isinstance(payload, dict), (
                f"Implicit finish broke the structured finish contract: {result!r}"
            )
            assert payload["success"] is False
            assert payload["is_continue"] is True
            assert "stalled" in payload["summary"]
            assert "complete" in payload["summary"]
        finally:
            server.shutdown()

    def test_fallback_swap_does_not_leak_primary_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After a mid-run fallback model swap, an implicit finish must
        never report stale status text from the failed primary model."""
        primary = "gpt-stagnant-primary-under-test"
        fallback = "gpt-stagnant-fallback-under-test"
        for name, fb in ((primary, fallback), (fallback, None)):
            monkeypatch.setitem(
                MODEL_INFO,
                name,
                ModelInfo(
                    context_length=128_000,
                    input_price_per_million=0.0,
                    output_price_per_million=0.0,
                    is_function_calling_supported=True,
                    is_embedding_supported=False,
                    is_generation_supported=True,
                    fallback=fb,
                ),
            )
        primary_turns = [0]

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if request.get("model") == primary:
                idx = primary_turns[0]
                primary_turns[0] += 1
                if idx == 0:
                    return _tool_call_response(
                        "PRIMARY-ONLY-STATUS: just starting the investigation.",
                        "check_build",
                    )
                return _text_response("")  # two empty turns → fallback swap
            # Fallback model: no text, identical stagnant verification calls.
            return _tool_call_response("", "check_build")

        server = _serve(respond)
        try:
            agent = KISSAgent("test-fallback-stale-text")
            result = agent.run(
                model_name=primary,
                prompt_template="Verify the build.",
                tools=[check_build],
                max_steps=30,
                max_budget=5.0,
                verbose=False,
                model_config={
                    "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                    "api_key": "sk-test",
                },
            )
            assert agent.model_name == fallback
            assert "PRIMARY-ONLY-STATUS" not in result, (
                "Implicit finish leaked stale text from the failed primary model"
            )
            assert "stalled" in result
        finally:
            server.shutdown()

    def test_max_steps_still_enforced_for_progressing_agent(self) -> None:
        """Sanity: an agent making real (non-stagnant) progress that never
        finishes still hits the max_steps ceiling as before."""
        tick = [0]

        def poll_build() -> str:
            """Polling tool whose result changes every call."""
            tick[0] += 1
            return f"build progress: {tick[0]}%"

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            return _tool_call_response("Working.", "poll_build")

        server = _serve(respond)
        try:
            with pytest.raises(KISSError, match="exceeded"):
                _run_agent(server, [poll_build], max_steps=8)
        finally:
            server.shutdown()
