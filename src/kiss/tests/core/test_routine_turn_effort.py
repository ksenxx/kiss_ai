# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the routine-turn effort override (P14 of the 2026-10-24 audit).

With ``DEFAULT_CONFIG.routine_turn_effort`` set, the model call that
follows a ``summary`` / memory tool, a refused call, a ``bash_job`` /
``agent_job`` wait or an unchanged ``Read`` is sent at that effort and
marked ``routine`` on its ``llm_call`` event; every other call keeps
the model's own effort.  The scripted Anthropic server records the
``output_config.effort`` of each request.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent, is_routine_turn
from kiss.core.printer import Printer
from kiss.core.tool_verdict import ALLOW, Verdict, refuse
from kiss.tests.core import local_anthropic_server as anthropic_server


class CapturePrinter(Printer):
    """Records every event; no token callback, so the scripted model is not asked to stream."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.token_callback = None  # type: ignore[method-assign,assignment]

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:  # noqa: A002
        self.events.append({"type": type, "content": content, **kwargs})

    def token_callback(self, token: str) -> None:  # type: ignore[no-redef]
        pass

    def reset(self) -> None:
        pass


def summary(description: str) -> str:
    """Record a progress summary.

    Args:
        description: The summary text.
    """
    return "ok"


def note(text: str) -> str:
    """Record a note.

    Args:
        text: The note.
    """
    return "noted " + text


def test_is_routine_turn_rules() -> None:
    assert is_routine_turn("Bash", {}, "x", refused=True)
    assert is_routine_turn("summary", {"description": "d"}, "ok", refused=False)
    assert is_routine_turn("memory_search", {"query": "q"}, "hits", refused=False)
    # A wait that timed out with the job still running is a polling turn; one that
    # returned the finished job's result is real work (the model must read it).
    running = "Job j: running (pid 42, 60s)\nlog tail:\n..."
    wait, tail = {"job_id": "j", "action": "wait"}, {"job_id": "j", "action": "tail"}
    assert is_routine_turn("bash_job", wait, running, refused=False)
    assert is_routine_turn(
        "agent_job", wait, "The reviewer agent task is still running after 60s as job j.",
        refused=False,
    )
    assert not is_routine_turn("bash_job", wait, "Job j: exited with code 0\ndone", refused=False)
    # Only the status line counts: a finished job's log may say "still running".
    finished = "Job j: exited with code 0\nserver still running (pid 7) when the test ended"
    assert not is_routine_turn("bash_job", wait, finished, refused=False)
    assert not is_routine_turn("bash_job", tail, running, refused=False)
    assert is_routine_turn(
        "Read", {"file_path": "f"}, "f: content is already in your context above.", refused=False
    )
    assert not is_routine_turn("Read", {"file_path": "f"}, "1: line", refused=False)
    assert not is_routine_turn("Bash", {"command": "ls"}, "a b", refused=False)


def _run(monkeypatch: pytest.MonkeyPatch, url: str, script_len: int, printer: Printer) -> KISSAgent:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
    monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
    agent = KISSAgent("routine-effort-test")
    agent.run(
        model_name=anthropic_server.MODEL,
        prompt_template="Summarize, then take notes, then finish.",
        tools=[summary, note],
        max_steps=script_len + 2,
        max_budget=50.0,
        verbose=False,
        printer=printer,
        tool_call_hook=_refuse_secret_notes,
    )
    return agent


def _refuse_secret_notes(name: str, args: dict[str, Any]) -> Verdict:
    """Refuse ``note(text="secret")``; allow everything else."""
    if name == "note" and args.get("text") == "secret":
        return refuse("REFUSED: no notes about secrets")
    return ALLOW


_SCRIPT = [
    anthropic_server.tool_use_message("summary", {"description": "- started"}, 1000, "toolu_1"),
    anthropic_server.tool_use_message("note", {"text": "plain"}, 1100, "toolu_2"),
    anthropic_server.tool_use_message("note", {"text": "secret"}, 1200, "toolu_3"),
    anthropic_server.tool_use_message("finish", {"result": "<p>done</p>"}, 1300, "toolu_4"),
]


def test_routine_turns_run_at_the_configured_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "routine_turn_effort", "low")
    printer = CapturePrinter()
    with anthropic_server.serve(_SCRIPT) as (url, requests):
        agent = _run(monkeypatch, url, len(_SCRIPT), printer)
    steps = [r for r in requests if r.get("stream")]
    assert len(steps) == 4
    efforts = [(r.get("output_config") or {}).get("effort") for r in steps]
    # Step 2 follows the summary, step 4 the refused note; steps 1 and 3
    # (the first turn, the turn after a plain note) keep the model's effort.
    assert efforts == [None, "low", None, "low"]
    calls = [e for e in printer.events if e["type"] == "llm_call"]
    assert [(c["routine"], c["effort"]) for c in calls] == [
        (False, ""), (True, "low"), (False, ""), (True, "low"),
    ]
    assert "reasoning_effort" not in agent.model.model_config, "the override was restored"


def test_no_override_without_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "routine_turn_effort", "")
    printer = CapturePrinter()
    with anthropic_server.serve(_SCRIPT) as (url, requests):
        _run(monkeypatch, url, len(_SCRIPT), printer)
    steps = [r for r in requests if r.get("stream")]
    assert all("output_config" not in r for r in steps), steps
    calls = [e for e in printer.events if e["type"] == "llm_call"]
    # The routine turns are still marked, so cost_report can measure them.
    assert [c["routine"] for c in calls] == [False, True, False, True]
    assert all(c["effort"] == "" for c in calls)
