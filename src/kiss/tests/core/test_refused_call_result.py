# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a refused tool call can never read like a successful one.

Whether a ``tool_call_hook`` or the ``tool_call_guard`` refuses a call,
the model reads a result that opens with "REFUSED: the X call did NOT
run and had no effect." (:func:`kiss.core.tool_verdict.refusal_result`),
and the next allowed call of another tool carries a reminder to
re-issue the refused one.  Every test runs the real ``KISSAgent.run``
against the local chat-completions server of ``test_run_hooks``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kiss.core.kiss_agent import KISSAgent
from kiss.core.tool_verdict import ALLOW, REISSUE_REMINDER, Verdict, refusal_result, refuse
from kiss.tests.core.test_run_hooks import (
    _multi_tool_call_response,
    _run_agent,
    _serve,
    _tool_call_response,
)


def _tools() -> tuple[list[str], list[Callable[..., str]]]:
    """Return ``(executions, [check_build, lint])``; a call's ``deny`` argument steers a hook."""
    executions: list[str] = []

    def check_build(deny: bool = False) -> str:
        """Verify the build."""
        executions.append("check_build")
        return "build: ok"

    def lint(deny: bool = False) -> str:
        """Lint the tree."""
        executions.append("lint")
        return "lint: clean"

    return executions, [check_build, lint]


def _results(request: dict[str, Any]) -> list[str]:
    """Return the tool-result texts of one request, without the ``Steps:`` footers."""
    return [
        str(m.get("content", "")).split("\n\nSteps:")[0]
        for m in request["messages"]
        if m.get("role") == "tool"
    ]


def _deny_hook(name: str, args: dict[str, Any]) -> Verdict:
    """Refuse a call whose ``deny`` argument is true."""
    return refuse(f"{name} denied.") if args.get("deny") else ALLOW


def test_hook_refusal_is_prefixed_and_the_next_other_tool_carries_the_reminder() -> None:
    """A refused ``check_build``, then ``lint``: the refusal says the call did not run,
    ``lint``'s result ends with the reminder, and the re-issued ``check_build``
    (which ran) carries nothing extra."""

    def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
        if turn == 0:
            return _tool_call_response("Checking.", "check_build", {"deny": True})
        if turn == 1:
            return _tool_call_response("Linting.", "lint", {})
        if turn == 2:
            return _tool_call_response("Checking again.", "check_build", {})
        return _tool_call_response("Done.", "finish", {"result": "OK"})

    executions, tools = _tools()
    requests_log: list[dict[str, Any]] = []
    server = _serve(respond, requests_log)
    try:
        result, _agent = _run_agent(server, tools, tool_call_hook=_deny_hook)
    finally:
        server.shutdown()

    assert result == "OK"
    assert executions == ["lint", "check_build"]
    refused, linted, checked = _results(requests_log[3])
    assert refused == refusal_result("check_build", "check_build denied.")
    assert refused.startswith("REFUSED: the check_build call did NOT run and had no effect. ")
    assert linted == "lint: clean" + REISSUE_REMINDER.format(name="check_build")
    assert checked == "build: ok"


def test_reissuing_the_refused_tool_settles_the_debt_without_a_reminder() -> None:
    """``check_build`` refused once by the guard, then re-issued at once: its own
    result is plain, and a later ``lint`` carries no reminder either."""

    def guard(name: str, args: dict[str, Any]) -> str | None:
        return "guard: not yet" if args.get("deny") else None

    def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
        if turn == 0:
            return _tool_call_response("Checking.", "check_build", {"deny": True})
        if turn == 1:
            return _tool_call_response("Checking again.", "check_build", {})
        if turn == 2:
            return _tool_call_response("Linting.", "lint", {})
        return _tool_call_response("Done.", "finish", {"result": "OK"})

    executions, tools = _tools()
    requests_log: list[dict[str, Any]] = []
    server = _serve(respond, requests_log)
    try:
        agent = KISSAgent("test-refusal-settled")
        agent.tool_call_guard = guard
        result = agent.run(
            model_name="gpt-4o-mini",
            prompt_template="Check the build.",
            tools=tools,
            max_steps=10,
            max_budget=5.0,
            verbose=False,
            model_config={
                "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                "api_key": "sk-test",
            },
        )
    finally:
        server.shutdown()

    assert result == "OK"
    assert executions == ["check_build", "lint"]
    assert _results(requests_log[3]) == [
        refusal_result("check_build", "guard: not yet"),
        "build: ok",
        "lint: clean",
    ]


def test_the_latest_refusal_is_the_one_reminded_and_only_once() -> None:
    """Two refusals in a row (``check_build`` then ``lint``) owe one reminder, for the
    latest; a step that refuses one call and allows another in parallel
    reminds in that same step; the reminder is not repeated afterwards."""

    def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
        if turn == 0:
            return _tool_call_response("Checking.", "check_build", {"deny": True})
        if turn == 1:
            return _multi_tool_call_response(
                "Both.", [("lint", {"deny": True}), ("check_build", {})]
            )
        if turn == 2:
            return _tool_call_response("Linting.", "lint", {})
        return _tool_call_response("Done.", "finish", {"result": "OK"})

    executions, tools = _tools()
    requests_log: list[dict[str, Any]] = []
    server = _serve(respond, requests_log)
    try:
        result, _agent = _run_agent(server, tools, tool_call_hook=_deny_hook)
    finally:
        server.shutdown()

    assert result == "OK"
    assert executions == ["check_build", "lint"]
    assert _results(requests_log[3]) == [
        refusal_result("check_build", "check_build denied."),
        refusal_result("lint", "lint denied."),
        "build: ok" + REISSUE_REMINDER.format(name="lint"),
        "lint: clean",
    ]
