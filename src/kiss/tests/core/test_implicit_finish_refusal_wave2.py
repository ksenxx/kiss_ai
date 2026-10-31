# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A refused implicit finish sends the veto's text to the model, not the generic retry.

After ``MAX_CONSECUTIVE_NO_TOOL_CALLS`` text-only turns, ``KISSAgent``
probes ``finish`` through the ``tool_call_hook`` and ``tool_call_guard``
before ending the run implicitly.  Sorcar's finish gates are one-shot:
the unmerged-worktree gate refuses one ``finish`` and lets the next one
through.  The probe used to spend such a gate and then send the generic
"call a tool" retry, so the model never read the refusal and its next
real ``finish`` passed the spent gate.  The refusal text is now the next
prompt.  Every test drives a real agent against a local
OpenAI-compatible HTTP server; nothing is mocked.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kiss.core.kiss_agent import KISSAgent
from kiss.core.tool_verdict import ALLOW, Verdict, refuse
from kiss.tests.core.test_audit0902_fix_core_implicit_finish import (
    _run,
    _serve,
    _text_response,
    _VetoRecorder,
)

_GENERIC_RETRY = "MUST have at least one function call"


class _ScriptedGuard:
    """Real ``tool_call_guard`` answering with scripted refusals, then allowing."""

    def __init__(self, refusals: list[str | None]) -> None:
        self._refusals = iter(refusals)

    def guard(self, name: str, args: dict[str, Any]) -> str | None:
        """Answer the next scripted refusal (``None`` once the script is spent)."""
        del name, args
        return next(self._refusals, None)


def _last_messages(
    hook: Callable[[str, dict[str, Any]], Verdict] | None,
    guard: Callable[[str, dict[str, Any]], str | None] | None,
) -> tuple[list[str], str]:
    """Run a text-only model; return each request's last message and the run's result.

    The agent appends its usage footer to every user message it sends, so
    callers compare the start of a message.
    """
    requests: list[dict[str, Any]] = []

    def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        return _text_response(f"Text {turn}.")

    server = _serve(respond)
    try:
        agent = KISSAgent("wave2-refused-implicit-finish")
        agent.tool_call_guard = guard
        result = _run(server, agent, tool_call_hook=hook)
    finally:
        server.shutdown()
    return [str(request["messages"][-1]["content"]) for request in requests], result


def test_hook_refusal_text_is_the_next_prompt() -> None:
    """Turn 2's probe is refused by the hook: turn 3 reads the refusal, then the run ends."""
    recorder = _VetoRecorder([refuse("Merge the worktree first."), ALLOW])
    last, result = _last_messages(recorder.hook, recorder.guard)
    assert _GENERIC_RETRY in last[1]
    assert last[2].startswith("Merge the worktree first.\n")
    assert _GENERIC_RETRY not in last[2]
    assert result == "Text 2."


def test_guard_refusal_text_is_the_next_prompt() -> None:
    """A refusal from the framework's guard reaches the model the same way."""
    guard = _ScriptedGuard(["A user follow-up is queued; answer it first."])
    last, result = _last_messages(None, guard.guard)
    assert last[2].startswith("A user follow-up is queued; answer it first.\n")
    assert _GENERIC_RETRY not in last[2]
    assert result == "Text 2."


def test_empty_refusal_still_vetoes_with_the_generic_retry() -> None:
    """A refusal without text still blocks the finish; the model gets the generic retry."""
    guard = _ScriptedGuard([""])
    last, result = _last_messages(None, guard.guard)
    assert _GENERIC_RETRY in last[2]
    assert result == "Text 2."
