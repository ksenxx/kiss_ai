# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The root layer's ``summary`` cadence guardrail (``base_sea.py``).

A run that has the ``summary`` tool must call it at every 10th step:
from such a step on, ``BaseSea.tool_call_hook`` refuses every other
tool (``finish`` excepted) until ``summary`` runs.  The tests drive a
real :class:`ChatSorcarAgent` against the scripted local model with
the hooks :func:`evaluate_sea` stages for the bare base, exactly as
the daemon wires them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import (
    SUMMARY_EVERY_STEPS,
    BaseSea,
)
from kiss.agents.sorcar.sea_commands import evaluate_sea
from kiss.core.tool_verdict import REFUSED_PREFIX, REISSUE_REMINDER
from kiss.tests.agents.seas.conftest import run_sea
from kiss.tests.agents.seas.sea_contract import tool_result_text, tool_results
from kiss.tests.agents.sorcar.local_model_server import finish_body, tool_call_body


def _bash(step: int) -> bytes:
    return tool_call_body(
        "Bash", {"command": f"echo step-{step}", "description": f"step {step}"},
        prompt_tokens=500 + step,
    )


def _run(tmp_path: Path, script: list[bytes], tool_profile: str) -> tuple[Any, list[dict]]:
    run = evaluate_sea([BaseSea()], "count to twelve with Bash")
    return run_sea(run, script, tmp_path, max_steps=20, tool_profile=tool_profile)


def test_tenth_step_tool_call_is_refused_until_summary_runs(tmp_path: Path) -> None:
    """Steps 1-9 run Bash; step 10's Bash is refused; summary at 11 clears it; 12's Bash runs."""
    script = [_bash(step) for step in range(1, 11)]
    script.append(tool_call_body("summary", {"description": "- ten steps"}, prompt_tokens=700))
    script.append(_bash(12))
    script.append(finish_body("<p>done</p>", prompt_tokens=900))
    parsed, requests = _run(tmp_path, script, "shell+control")
    assert parsed["success"] is True
    results = tool_results(requests[-1])
    assert len(results) == 12, results
    for step in range(9):
        assert f"step-{step + 1}" in results[step], results[step]
    # The refusal cannot be read as a success: it opens by saying the
    # call did not run, and asks for the exact call back after the summary.
    assert results[9].startswith(REFUSED_PREFIX.format(name="Bash")), results[9]
    assert f"Step {SUMMARY_EVERY_STEPS} is a multiple of {SUMMARY_EVERY_STEPS}" in results[9]
    assert "call summary(description=...) now" in results[9]
    assert "re-issue this exact Bash call" in results[9]
    assert "step-10" not in results[9]
    # The summary's own result reminds the model of the refused call at
    # the very step it chooses what to do next; the re-issued Bash is
    # plain again.
    assert tool_result_text(results[10]).strip() == REISSUE_REMINDER.format(name="Bash").strip()
    assert "step-12" in results[11], results[11]
    assert "Reminder" not in results[11], results[11]


def test_finish_is_allowed_on_a_tenth_step(tmp_path: Path) -> None:
    """A run that finishes at step 10 is not held up by the owed summary."""
    script = [_bash(step) for step in range(1, 10)]
    script.append(finish_body("<p>finished at ten</p>", prompt_tokens=900))
    parsed, requests = _run(tmp_path, script, "shell+control")
    assert parsed["success"] is True
    assert "finished at ten" in parsed["summary"]
    assert len(requests) == 10


def test_without_the_summary_tool_nothing_is_refused(tmp_path: Path) -> None:
    """A bash-only run has no summary tool, so its tenth step runs like any other."""
    script = [_bash(step) for step in range(1, 12)]
    script.append(finish_body("<p>done</p>", prompt_tokens=900))
    parsed, requests = _run(tmp_path, script, "bash")
    assert parsed["success"] is True
    results = tool_results(requests[-1])
    assert len(results) == 11, results
    assert "step-10" in results[9], results[9]


def test_a_second_run_on_the_same_hooks_starts_its_count_afresh(tmp_path: Path) -> None:
    """The daemon runs the ``<task>`` blocks of one prompt on the same hooks, one after another.

    Nine steps of the first task must not make the second task's first
    tool call a "tenth step", and a summary owed when the first task
    finished is not carried over either.
    """
    run = evaluate_sea([BaseSea()], "two tasks")
    first = [_bash(step) for step in range(1, 10)]
    first.append(finish_body("<p>first</p>", prompt_tokens=900))
    second = [_bash(step) for step in range(1, 3)]
    second.append(finish_body("<p>second</p>", prompt_tokens=900))
    results = []
    for script in (first, second):
        _, requests = run_sea(run, script, tmp_path, max_steps=20, tool_profile="shell+control")
        results.append(tool_results(requests[-1]))
    assert len(results[0]) == 9 and "step-9" in results[0][8], results[0]
    assert len(results[1]) == 2, results[1]
    assert "step-1" in results[1][0] and "step-2" in results[1][1], results[1]


def summary(description: str) -> str:
    """Stand-in ``summary`` tool for the hooks, which find it by name."""
    return description


def test_the_summary_comes_due_again_at_step_twenty() -> None:
    """The debt recurs at every multiple of 10: after step 10's summary, step 20 owes one.

    Driven through the hooks :func:`evaluate_sea` stages; the first ten
    steps (no summary tool, steps 1-9, ``finish`` exempt) are pinned by
    ``tests/server/test_base_sea_root_layer.py``.
    """
    run = evaluate_sea([BaseSea()], "t")
    run.tools_hook([_bash, summary])
    for _ in range(SUMMARY_EVERY_STEPS):
        run.llm_call_hook([])
    assert run.tool_call_hook("summary", {"description": "x"}).allowed
    for _ in range(SUMMARY_EVERY_STEPS):
        run.llm_call_hook([])
    refused = run.tool_call_hook("Read", {"file_path": "/x"})
    assert not refused.allowed and "re-issue this exact Read call" in refused.text
    assert run.tool_call_hook("summary", {"description": "y"}).allowed
    assert run.tool_call_hook("Read", {"file_path": "/x"}).allowed
