# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The root layer's ``summary`` cadence guardrail (``base_sea.py``).

A run that has the ``summary`` tool must call it at every 10th step:
from such a step on, every other tool call (``finish`` excepted) still
runs, but its result ends with ``SUMMARY_DUE_NOTE`` until ``summary``
runs.  The step is the one the tool-result footer prints, so the rule
the model reads and the one the gate applies are the same count, and a
continuation session restarts both.  The tests drive a real
:class:`ChatSorcarAgent` against the scripted local model with the
hooks :func:`evaluate_sea` stages for the bare base, exactly as the
daemon wires them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import (
    SUMMARY_DUE_NOTE,
    SUMMARY_EVERY_STEPS,
    BaseSea,
)
from kiss.agents.sorcar.sea_commands import evaluate_sea
from kiss.core.llm_reply import LLMReply
from kiss.tests.agents.seas.conftest import run_sea
from kiss.tests.agents.seas.sea_contract import tool_results
from kiss.tests.agents.sorcar.local_model_server import finish_body, tool_call_body


def _bash(step: int) -> bytes:
    return tool_call_body(
        "Bash", {"command": f"echo step-{step}", "description": f"step {step}"},
        prompt_tokens=500 + step,
    )


def _run(tmp_path: Path, script: list[bytes], tool_profile: str) -> tuple[Any, list[dict]]:
    run = evaluate_sea([BaseSea()], "count to twelve with Bash")
    return run_sea(run, script, tmp_path, max_steps=20, tool_profile=tool_profile)


def test_tenth_step_tool_call_runs_and_asks_for_the_summary(tmp_path: Path) -> None:
    """Steps 1-9 run Bash plain; step 10's Bash runs with the note; summary clears it.

    The model that misses the summary at step 10 is not sent back a
    step: the Bash ran (``step-10`` is in its result) and the note tells
    it to summarise next.  Step 11 ignores the note too, so its result
    carries the note again; the summary at step 12 clears the debt and
    step 13's Bash is plain.
    """
    script = [_bash(step) for step in range(1, 12)]
    script.append(tool_call_body("summary", {"description": "- ten steps"}, prompt_tokens=700))
    script.append(_bash(13))
    script.append(finish_body("<p>done</p>", prompt_tokens=900))
    parsed, requests = _run(tmp_path, script, "shell+control")
    assert parsed["success"] is True
    results = tool_results(requests[-1])
    assert len(results) == 13, results
    note = SUMMARY_DUE_NOTE.format(every=SUMMARY_EVERY_STEPS).strip()
    for step in range(9):
        assert f"step-{step + 1}" in results[step], results[step]
        assert "A summary is due" not in results[step], results[step]
    assert "step-10" in results[9] and note in results[9], results[9]
    assert "REFUSED" not in results[9], results[9]
    assert "step-11" in results[10] and note in results[10], results[10]
    assert "A summary is due" not in results[11], results[11]  # the summary's own result
    assert "step-13" in results[12] and "A summary is due" not in results[12], results[12]


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


def _reply(step: int) -> LLMReply:
    return LLMReply("", [{"name": "Bash", "arguments": {}}], step=step)


def test_the_summary_comes_due_again_at_step_twenty() -> None:
    """The debt recurs at every multiple of 10: after step 10's summary, step 20 owes one.

    Driven through the hooks :func:`evaluate_sea` stages, with the
    replies' steps as the agent reports them; the first ten steps (no
    summary tool, steps 1-9, ``finish`` exempt) are pinned by
    ``tests/server/test_base_sea_root_layer.py``.
    """
    run = evaluate_sea([BaseSea()], "t")
    run.tools_hook([_bash, summary])
    for step in range(1, SUMMARY_EVERY_STEPS + 1):
        run.llm_response_hook(_reply(step))
    assert run.tool_call_hook("summary", {"description": "x"}).allowed
    for step in range(SUMMARY_EVERY_STEPS + 1, 2 * SUMMARY_EVERY_STEPS + 1):
        run.llm_response_hook(_reply(step))
    noted = run.tool_call_hook("Read", {"file_path": "/x"})
    assert noted.allowed and noted.text == SUMMARY_DUE_NOTE.format(every=SUMMARY_EVERY_STEPS)
    assert run.tool_call_hook("summary", {"description": "y"}).allowed
    assert run.tool_call_hook("Read", {"file_path": "/x"}).text == ""


def test_a_continuation_session_restarts_the_count_and_drops_the_old_debt() -> None:
    """A new session's footer starts at 1 again; a summary owed by the old session is not carried.

    The 2026-11-01 audit found 158 of 159 cadence refusals of one task
    fell on steps whose footer did not end in 9: the gate counted the
    whole run while the footer restarted with each continuation.
    """
    run = evaluate_sea([BaseSea()], "t")
    run.tools_hook([_bash, summary])
    for step in range(1, 11):
        run.llm_response_hook(_reply(step))
    assert "A summary is due" in run.tool_call_hook("Bash", {"command": "ls"}).text
    # The next session: its first reply is step 1 of a fresh footer.
    run.llm_response_hook(_reply(1))
    assert run.tool_call_hook("Bash", {"command": "ls"}).text == ""
    for step in range(2, 10):
        run.llm_response_hook(_reply(step))
        assert run.tool_call_hook("Bash", {"command": "ls"}).text == ""
    run.llm_response_hook(_reply(10))
    assert "A summary is due" in run.tool_call_hook("Bash", {"command": "ls"}).text
