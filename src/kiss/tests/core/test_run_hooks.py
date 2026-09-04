# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``KISSAgent.run``'s ``llm_call_hook`` / ``tool_call_hook``.

The two hooks are ``run()`` parameters:

* ``llm_call_hook`` is called before every ``generate_and_process_with_tools``
  LLM call with the list of NEW messages (added to the conversation since the
  previous LLM call) and returns a possibly modified list that is sent to the
  LLM instead.
* ``tool_call_hook`` is called before every tool call with the tool name and
  its arguments. A return of ``"OK"`` lets the tool execute as usual; any
  other string suppresses execution and becomes the tool's result. A
  stagnation-triggered implicit finish is likewise suppressed unless the hook
  returns ``"OK"`` for ``("finish", {})``.

Every test runs the real ``KISSAgent.run`` against a local HTTP server
speaking the OpenAI chat-completions protocol — no mocks or patches.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from kiss.core.kiss_agent import STAGNANT_TURNS_FINISH, KISSAgent
from kiss.core.kiss_error import KISSError

_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

_PROMPT = "Verify the build is green and report the status."


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


def _multi_tool_call_response(
    text: str, calls: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    """OpenAI-compatible response with text plus the given (name, args) tool calls."""
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
                            "id": f"call_{i + 1}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                        for i, (name, arguments) in enumerate(calls)
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": _USAGE,
    }


def _tool_call_response(
    text: str, name: str, arguments: dict[str, Any] | None = None
) -> dict[str, Any]:
    """OpenAI-compatible response with text plus one tool call."""
    return _multi_tool_call_response(text, [(name, arguments or {})])


def _serve(
    respond: Callable[[int, dict[str, Any]], dict[str, Any]],
    requests_log: list[dict[str, Any]] | None = None,
) -> HTTPServer:
    """Start a local chat-completions server; ``respond(turn, request)`` builds each reply.

    Args:
        respond: Builds the JSON reply for the given zero-based turn and request.
        requests_log: When given, every parsed request body is appended to it.

    Returns:
        The started ``HTTPServer`` (call ``shutdown()`` when done).
    """
    turn_counter = [0]

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(content_length)) if content_length else {}
            if requests_log is not None:
                requests_log.append(request)
            reply = respond(turn_counter[0], request)
            turn_counter[0] += 1
            if reply.get("__status__"):
                status = int(reply["__status__"])
                body = json.dumps({"error": {"message": "synthetic failure"}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = json.dumps(reply).encode()
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


def _run_agent(
    server: HTTPServer,
    tools: list[Callable[..., Any]],
    max_steps: int = 30,
    llm_call_hook: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    tool_call_hook: Callable[[str, dict[str, Any]], str] | None = None,
) -> tuple[str, KISSAgent]:
    """Run a real KISSAgent against the local server and return (result, agent)."""
    agent = KISSAgent("test-run-hooks")
    result = agent.run(
        model_name="gpt-4o-mini",
        prompt_template=_PROMPT,
        tools=tools,
        max_steps=max_steps,
        max_budget=5.0,
        verbose=False,
        model_config={
            "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
            "api_key": "sk-test",
        },
        llm_call_hook=llm_call_hook,
        tool_call_hook=tool_call_hook,
    )
    return result, agent


class _CountingTool:
    """A tool whose executions are counted, to prove suppression end to end."""

    def __init__(self) -> None:
        self.executions = 0

    def make(self) -> Callable[[], str]:
        """Return the ``check_build`` tool bound to this counter."""

        def check_build() -> str:
            """Deterministic read-only verification tool: always reports success."""
            self.executions += 1
            return "build: all 42 tests passed"

        return check_build


class TestLLMCallHook:
    """``llm_call_hook`` sees exactly the new messages and its output is sent."""

    def test_hook_receives_only_new_messages_each_call(self) -> None:
        """First call gets the initial prompt; later calls get only messages
        added since the previous LLM call (tool results), never the already
        sent prompt or assistant turns."""
        seen: list[list[dict[str, Any]]] = []

        def llm_call_hook(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            seen.append([dict(m) for m in messages])
            return messages

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if turn == 0:
                return _tool_call_response("Checking.", "check_build", {})
            return _tool_call_response("Done.", "finish", {"result": "GREEN"})

        counter = _CountingTool()
        server = _serve(respond)
        try:
            result, _ = _run_agent(
                server, [counter.make()], llm_call_hook=llm_call_hook
            )
        finally:
            server.shutdown()

        assert result == "GREEN"
        assert len(seen) == 2, f"Hook must run before each LLM call, ran {len(seen)}x"
        first_contents = " ".join(str(m.get("content", "")) for m in seen[0])
        assert _PROMPT in first_contents
        second = seen[1]
        assert second, "Second call must see the new tool-result messages"
        assert all(m.get("role") in ("tool", "user") for m in second), (
            f"Second call must only see messages added after the first LLM "
            f"call, got roles {[m.get('role') for m in second]}"
        )
        second_contents = " ".join(str(m.get("content", "")) for m in second)
        assert _PROMPT not in second_contents
        assert "42 tests passed" in second_contents

    def test_hook_modifications_are_sent_to_llm(self) -> None:
        """The list the hook returns replaces the new messages in the request
        the LLM actually receives — verified on the wire, not in memory."""
        marker = "INJECTED-BY-LLM-CALL-HOOK-7391"

        def llm_call_hook(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            modified = []
            for m in messages:
                m = dict(m)
                if m.get("role") == "user":
                    m["content"] = marker
                modified.append(m)
            return modified

        requests_log: list[dict[str, Any]] = []

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            return _tool_call_response("Done.", "finish", {"result": "OK_DONE"})

        server = _serve(respond, requests_log)
        try:
            result, _ = _run_agent(server, [], llm_call_hook=llm_call_hook)
        finally:
            server.shutdown()

        assert result == "OK_DONE"
        sent = requests_log[0]["messages"]
        sent_contents = [str(m.get("content", "")) for m in sent]
        assert any(marker in c for c in sent_contents), (
            f"The LLM request must carry the hook's replacement, got {sent_contents}"
        )
        assert not any(_PROMPT in c for c in sent_contents), (
            "The original prompt must have been replaced by the hook's output"
        )

    def test_hook_can_append_messages(self) -> None:
        """A hook that appends an extra message grows the request the LLM sees."""
        extra = "EXTRA-CONTEXT-FROM-HOOK-2214"

        def llm_call_hook(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [*messages, {"role": "user", "content": extra}]

        requests_log: list[dict[str, Any]] = []

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            return _tool_call_response("Done.", "finish", {"result": "APPENDED"})

        server = _serve(respond, requests_log)
        try:
            result, _ = _run_agent(server, [], llm_call_hook=llm_call_hook)
        finally:
            server.shutdown()

        assert result == "APPENDED"
        sent_contents = [str(m.get("content", "")) for m in requests_log[0]["messages"]]
        assert any(_PROMPT in c for c in sent_contents)
        assert any(extra in c for c in sent_contents)

    def test_hook_not_reapplied_after_retryable_provider_error(self) -> None:
        """When the LLM call itself raises (provider 500) and the agent
        retries, the hook must NOT be re-run over already-hooked messages:
        the next invocation sees only the retry message the loop appended."""
        seen: list[list[dict[str, Any]]] = []

        def llm_call_hook(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            seen.append([dict(m) for m in messages])
            return messages

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            contents = " ".join(
                str(m.get("content", "")) for m in request.get("messages", [])
            )
            if "Please try again" not in contents:
                return {"__status__": 500}
            return _tool_call_response("Done.", "finish", {"result": "RECOVERED"})

        server = _serve(respond)
        try:
            result, _ = _run_agent(server, [], llm_call_hook=llm_call_hook)
        finally:
            server.shutdown()

        assert result == "RECOVERED"
        assert len(seen) == 2, f"Hook must run once per agent LLM step, ran {len(seen)}x"
        first_contents = " ".join(str(m.get("content", "")) for m in seen[0])
        assert _PROMPT in first_contents
        second_contents = " ".join(str(m.get("content", "")) for m in seen[1])
        assert _PROMPT not in second_contents, (
            "Messages already passed to the hook must not be presented again "
            "after a failed LLM call"
        )
        assert "Please try again" in second_contents


class TestToolCallHook:
    """``tool_call_hook`` gates every tool execution on returning ``"OK"``."""

    def test_ok_verdict_executes_tool_as_before(self) -> None:
        """A hook returning "OK" observes every call but changes nothing."""
        hook_calls: list[tuple[str, dict[str, Any]]] = []

        def tool_call_hook(name: str, args: dict[str, Any]) -> str:
            hook_calls.append((name, dict(args)))
            return "OK"

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if turn == 0:
                return _tool_call_response("Checking.", "check_build", {})
            return _tool_call_response("Done.", "finish", {"result": "ALL_GREEN"})

        counter = _CountingTool()
        server = _serve(respond)
        try:
            result, _ = _run_agent(
                server, [counter.make()], tool_call_hook=tool_call_hook
            )
        finally:
            server.shutdown()

        assert result == "ALL_GREEN"
        assert counter.executions == 1, "An OK verdict must let the tool run"
        assert hook_calls == [
            ("check_build", {}),
            ("finish", {"result": "ALL_GREEN"}),
        ]

    def test_non_ok_verdict_suppresses_tool_and_becomes_result(self) -> None:
        """Any non-"OK" string blocks execution and is fed back to the model
        as the tool's result — verified on the wire in the next request."""
        rejection = "check_build denied: read-only mode"

        def tool_call_hook(name: str, args: dict[str, Any]) -> str:
            if name == "check_build":
                return rejection
            return "OK"

        requests_log: list[dict[str, Any]] = []

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if turn == 0:
                return _tool_call_response("Checking.", "check_build", {})
            return _tool_call_response("Done.", "finish", {"result": "GAVE_UP"})

        counter = _CountingTool()
        server = _serve(respond, requests_log)
        try:
            result, _ = _run_agent(
                server, [counter.make()], tool_call_hook=tool_call_hook
            )
        finally:
            server.shutdown()

        assert result == "GAVE_UP"
        assert counter.executions == 0, "A blocked tool must never execute"
        tool_messages = [
            str(m.get("content", ""))
            for m in requests_log[1]["messages"]
            if m.get("role") == "tool"
        ]
        assert any(rejection in c for c in tool_messages), (
            f"The rejection string must reach the model as the tool result, "
            f"got tool messages {tool_messages}"
        )

    def test_blocked_finish_does_not_end_run(self) -> None:
        """Blocking ``finish`` keeps the loop alive until the hook allows it."""
        finish_attempts = [0]

        def tool_call_hook(name: str, args: dict[str, Any]) -> str:
            if name == "finish":
                finish_attempts[0] += 1
                if finish_attempts[0] <= 2:
                    return "finish denied: task not yet verified"
            return "OK"

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            return _tool_call_response("Done.", "finish", {"result": "HOOKED_DONE"})

        server = _serve(respond)
        try:
            result, agent = _run_agent(server, [], tool_call_hook=tool_call_hook)
        finally:
            server.shutdown()

        assert result == "HOOKED_DONE"
        assert finish_attempts[0] == 3
        assert agent.step_count == 3

    def test_hook_called_before_every_call_even_when_guard_blocks(self) -> None:
        """The hook runs for every model-emitted tool call — including one the
        pre-existing ``tool_call_guard`` blocks. With an "OK" verdict the
        guard's block still stands (the tool never executes)."""
        hook_calls: list[str] = []

        def tool_call_hook(name: str, args: dict[str, Any]) -> str:
            hook_calls.append(name)
            return "OK"

        def guard(name: str, args: dict[str, Any]) -> str | None:
            if name == "check_build":
                return "guard: check_build blocked"
            return None

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if turn == 0:
                return _tool_call_response("Checking.", "check_build", {})
            return _tool_call_response("Done.", "finish", {"result": "GUARDED"})

        counter = _CountingTool()
        requests_log: list[dict[str, Any]] = []
        server = _serve(respond, requests_log)
        try:
            agent = KISSAgent("test-guard-and-hook")
            agent.tool_call_guard = guard
            result = agent.run(
                model_name="gpt-4o-mini",
                prompt_template=_PROMPT,
                tools=[counter.make()],
                max_steps=30,
                max_budget=5.0,
                verbose=False,
                model_config={
                    "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                    "api_key": "sk-test",
                },
                tool_call_hook=tool_call_hook,
            )
        finally:
            server.shutdown()

        assert result == "GUARDED"
        assert counter.executions == 0, "The guard's block must still suppress the tool"
        assert hook_calls == ["check_build", "finish"], (
            f"The hook must be called before every tool call, got {hook_calls}"
        )
        tool_messages = [
            str(m.get("content", ""))
            for m in requests_log[1]["messages"]
            if m.get("role") == "tool"
        ]
        assert any("guard: check_build blocked" in c for c in tool_messages)

    def test_hook_rejection_takes_precedence_over_guard(self) -> None:
        """A non-"OK" hook verdict is the tool result the model sees, even
        when the guard would also have blocked the call with its own string."""
        rejection = "hook: check_build denied"
        guard_calls: list[str] = []

        def tool_call_hook(name: str, args: dict[str, Any]) -> str:
            if name == "check_build":
                return rejection
            return "OK"

        def guard(name: str, args: dict[str, Any]) -> str | None:
            guard_calls.append(name)
            if name == "check_build":
                return "guard: check_build blocked"
            return None

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if turn == 0:
                return _tool_call_response("Checking.", "check_build", {})
            return _tool_call_response("Done.", "finish", {"result": "HOOK_WINS"})

        counter = _CountingTool()
        requests_log: list[dict[str, Any]] = []
        server = _serve(respond, requests_log)
        try:
            agent = KISSAgent("test-hook-precedence")
            agent.tool_call_guard = guard
            result = agent.run(
                model_name="gpt-4o-mini",
                prompt_template=_PROMPT,
                tools=[counter.make()],
                max_steps=30,
                max_budget=5.0,
                verbose=False,
                model_config={
                    "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                    "api_key": "sk-test",
                },
                tool_call_hook=tool_call_hook,
            )
        finally:
            server.shutdown()

        assert result == "HOOK_WINS"
        assert counter.executions == 0
        assert "check_build" not in guard_calls, (
            "The guard must not be consulted for a call the hook already rejected"
        )
        tool_messages = [
            str(m.get("content", ""))
            for m in requests_log[1]["messages"]
            if m.get("role") == "tool"
        ]
        assert any(rejection in c for c in tool_messages), (
            f"The hook's rejection must be the tool result, got {tool_messages}"
        )

    def test_multiple_tool_calls_in_one_turn_gated_independently(self) -> None:
        """With two tool calls in one model turn, the hook gates each call on
        its own: the allowed one executes, the rejected one is suppressed."""
        rejection = "second call denied"
        hook_calls: list[tuple[str, dict[str, Any]]] = []

        def tool_call_hook(name: str, args: dict[str, Any]) -> str:
            hook_calls.append((name, dict(args)))
            if args.get("target") == "flaky":
                return rejection
            return "OK"

        def check_build(target: str) -> str:
            """Verification tool taking a target name, to distinguish the two calls."""
            executed.append(target)
            return f"build of {target}: all 42 tests passed"

        executed: list[str] = []

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            if turn == 0:
                return _multi_tool_call_response(
                    "Checking both.",
                    [
                        ("check_build", {"target": "stable"}),
                        ("check_build", {"target": "flaky"}),
                    ],
                )
            return _tool_call_response("Done.", "finish", {"result": "MIXED"})

        requests_log: list[dict[str, Any]] = []
        server = _serve(respond, requests_log)
        try:
            result, _ = _run_agent(
                server, [check_build], tool_call_hook=tool_call_hook
            )
        finally:
            server.shutdown()

        assert result == "MIXED"
        assert executed == ["stable"], f"Only the allowed call may run, ran {executed}"
        assert ("check_build", {"target": "stable"}) in hook_calls
        assert ("check_build", {"target": "flaky"}) in hook_calls
        tool_messages = [
            str(m.get("content", ""))
            for m in requests_log[1]["messages"]
            if m.get("role") == "tool"
        ]
        assert any("build of stable" in c for c in tool_messages)
        assert any(rejection in c for c in tool_messages)

    def test_implicit_finish_requires_hook_ok(self) -> None:
        """The stagnation net must not bypass a hook that blocks ``finish``."""

        def tool_call_hook(name: str, args: dict[str, Any]) -> str:
            if name == "finish":
                return "finish denied: never allowed"
            return "OK"

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            return _tool_call_response("Still verifying.", "check_build", {})

        max_steps = STAGNANT_TURNS_FINISH + 4
        counter = _CountingTool()
        server = _serve(respond)
        try:
            with pytest.raises(KISSError, match="exceeded"):
                _run_agent(
                    server,
                    [counter.make()],
                    max_steps=max_steps,
                    tool_call_hook=tool_call_hook,
                )
        finally:
            server.shutdown()
        assert counter.executions == max_steps

    def test_implicit_finish_proceeds_on_hook_ok(self) -> None:
        """When the hook answers "OK" for finish, stagnation still finishes."""

        def tool_call_hook(name: str, args: dict[str, Any]) -> str:
            return "OK"

        def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
            return _tool_call_response("Build is green.", "check_build", {})

        counter = _CountingTool()
        server = _serve(respond)
        try:
            result, agent = _run_agent(
                server, [counter.make()], tool_call_hook=tool_call_hook
            )
        finally:
            server.shutdown()

        assert result == "Build is green."
        assert agent.step_count == STAGNANT_TURNS_FINISH
