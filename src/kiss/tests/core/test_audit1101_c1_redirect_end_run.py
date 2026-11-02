# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A run-ending verdict on a redirected call, or on its follow-up, ends the run (audit 1101, C1 F1).

A tool that raises :class:`~kiss.core.tool_verdict.ToolRedirect` has the
agent make the rewritten call (and the follow-up) through
``_guarded_execute_tool``, where the ``tool_call_hook`` answers for it.
When that answer is ``end_run(...)``, the run must end there: the later
calls of the same reply do not run and no further model request is
made.  Before the fix ``_execute_redirected`` turned the nested verdict
into result text and returned no verdict, so the agent carried on as if
the hook had merely refused the nested call.

Three paths, nothing mocked: the ordinary tool loop of a real
``KISSAgent.run`` against a loopback chat-completions server (the
verdict on the redirected call, then on its follow-up), and the native
helper-thread path through a real Codex ``ToolBridge`` over a socket.
"""

from __future__ import annotations

from typing import Any

from kiss.core.base import collecting_trajectories
from kiss.core.kiss_agent import KISSAgent
from kiss.core.tool_verdict import ALLOW, REFUSED_PREFIX, ToolRedirect, Verdict, end_run
from kiss.tests.core.test_audit1018_codex_bridge_stop import _call, _reply, _schema
from kiss.tests.core.test_f1_codex_bridge_findings import _bridge_of
from kiss.tests.core.test_hook_ends_run import _run, _tool_calls_response

STOP_TEXT = "Episode over: the ledger gate ended the run."


class _Chain:
    """The tools of a redirect chain, recording what ran and what the hook saw."""

    def __init__(self) -> None:
        self.effects: list[str] = []
        self.hook_calls: list[str] = []
        self.end_at = ""

    def outer(self) -> str:
        """Stand for a shell rewrite: redirect to ``write`` with ``followup`` as the rest."""
        raise ToolRedirect("write", {}, "Redirected write:", ("followup", {}))

    def write(self) -> str:
        """The rewrite the redirect stands for."""
        self.effects.append("write")
        return "written"

    def followup(self) -> str:
        """The rest of the compound command."""
        self.effects.append("followup")
        return "followed up"

    def later(self) -> str:
        """A later call of the same model reply."""
        self.effects.append("later")
        return "later ran"

    def hook(self, name: str, args: dict[str, Any]) -> Verdict:
        """End the run at the call named :attr:`end_at`; allow the rest."""
        self.hook_calls.append(name)
        return end_run(STOP_TEXT, success=False) if name == self.end_at else ALLOW

    def respond(self, turn: int, request: dict[str, Any]) -> dict[str, Any]:
        """Turn 0 asks for the redirecting call and a later one; any later turn finishes."""
        if turn == 0:
            return _tool_calls_response("Working.", [("outer", {}), ("later", {})])
        return _tool_calls_response("Done.", [("finish", {"result": "MODEL KEPT GOING"})])


def test_a_verdict_ending_the_run_on_the_redirected_call_ends_it() -> None:
    chain = _Chain()
    chain.end_at = "write"
    result, agent, requests = _run(
        chain.respond,
        [chain.outer, chain.write, chain.followup, chain.later],
        tool_call_hook=chain.hook,
    )
    assert result == STOP_TEXT
    assert chain.effects == [], "neither the refused write nor the later call ran"
    assert chain.hook_calls == ["outer", "write"], "the follow-up and the later call were not asked"
    assert len(requests) == 1, "the run ends at the verdict: no further model request"
    # The outer call's result (the note and the nested refusal) is recorded
    # as the turn's only result; the later call is not recorded as made.
    results_text = agent.messages[-1]["content"]
    assert "Redirected write:" in results_text
    assert REFUSED_PREFIX.format(name="write") in results_text
    assert "[later]" not in results_text


def test_a_verdict_ending_the_run_on_the_follow_up_ends_it() -> None:
    chain = _Chain()
    chain.end_at = "followup"
    result, agent, requests = _run(
        chain.respond,
        [chain.outer, chain.write, chain.followup, chain.later],
        tool_call_hook=chain.hook,
    )
    assert result == STOP_TEXT
    assert chain.effects == ["write"], "the write went through; the refused follow-up did not"
    assert chain.hook_calls == ["outer", "write", "followup"]
    assert len(requests) == 1
    results_text = agent.messages[-1]["content"]
    assert "written" in results_text
    assert REFUSED_PREFIX.format(name="followup") in results_text
    assert "[later]" not in results_text


def test_a_verdict_ending_the_run_on_a_redirected_call_served_natively_ends_the_turn() -> None:
    """Over the Codex bridge the verdict lands on the turn's record (``turn.ended``)."""
    chain = _Chain()
    chain.end_at = "write"
    with collecting_trajectories(save=False):
        agent = KISSAgent("native redirect end_run")
        agent._reset("codex/gpt-5.5", True, 10, 10.0, None, verbose=False)
        agent.tool_call_hook = chain.hook
        agent.function_map = {
            "outer": chain.outer,
            "write": chain.write,
            "followup": chain.followup,
            "later": chain.later,
        }
        agent._cached_tools_schema = _schema("outer", "write", "followup", "later")
        with agent._native_tools_exposed():
            bridge = _bridge_of(agent)
            first = _reply(_call(bridge, "outer"))
            second = _reply(_call(bridge, "later"))
    assert "text" in first and REFUSED_PREFIX.format(name="write") in first["text"]
    ended = agent._native_turn.ended
    assert ended is not None and ended.ends_run and ended.text == STOP_TEXT
    assert chain.effects == [], "the later call of the turn was refused, not run"
    assert "text" in second and "The run has ended" in second["text"]
