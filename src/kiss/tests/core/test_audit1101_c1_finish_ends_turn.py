# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""An accepted ``finish`` ends the turn: the calls after it in the same reply do not run.

Audit 1101, C1 F3.  A reply of ``[finish(...), other_tool(...)]`` used to
run ``other_tool`` after the task had finished (a mutating call after
the final outcome, a second ``finish`` overwriting the first's result,
a limit error overriding it).  The calls before ``finish`` still run and
the finish's own result stands.  Real ``KISSAgent.run`` against a
loopback chat-completions server, nothing mocked.
"""

from __future__ import annotations

from typing import Any

from kiss.tests.core.test_hook_ends_run import _run, _tool_calls_response


class _Effects:
    """Tools recording their executions."""

    def __init__(self) -> None:
        self.ran: list[str] = []

    def check(self) -> str:
        """A read-only call made before ``finish``."""
        self.ran.append("check")
        return "checked"

    def mutate(self) -> str:
        """A mutating call the model put after ``finish``."""
        self.ran.append("mutate")
        return "changed something"


def test_the_calls_after_finish_in_the_same_reply_do_not_run() -> None:
    effects = _Effects()

    def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
        return _tool_calls_response(
            "Done.", [("check", {}), ("finish", {"result": "DONE"}), ("mutate", {})]
        )

    result, agent, requests = _run(respond, [effects.check, effects.mutate])
    assert result == "DONE"
    assert effects.ran == ["check"], "the call before finish ran; the one after did not"
    assert len(requests) == 1
    # The turn's record names the calls that were made and nothing else.
    results_text = agent.messages[-1]["content"]
    assert "[check]: checked" in results_text
    assert "[finish]: DONE" in results_text
    assert "[mutate]" not in results_text


def test_a_second_finish_in_the_same_reply_does_not_replace_the_first() -> None:
    effects = _Effects()

    def respond(turn: int, request: dict[str, Any]) -> dict[str, Any]:
        return _tool_calls_response(
            "Done twice.",
            [("finish", {"result": "FIRST"}), ("mutate", {}), ("finish", {"result": "SECOND"})],
        )

    result, _agent, requests = _run(respond, [effects.mutate])
    assert result == "FIRST"
    assert effects.ran == []
    assert len(requests) == 1
