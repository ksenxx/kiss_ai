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

import yaml

from kiss.agents.seas.base.base_sea import (
    SUMMARY_EVERY_STEPS,
    BaseSea,
)
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sea_commands import evaluate_sea
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)


def _bash(step: int) -> bytes:
    return tool_call_body(
        "Bash", {"command": f"echo step-{step}", "description": f"step {step}"},
        prompt_tokens=500 + step,
    )


def _run(tmp_path: Path, script: list[bytes], tool_profile: str) -> tuple[Any, list[dict]]:
    run = evaluate_sea([BaseSea()], "count to twelve with Bash")
    with serve(script) as (url, requests):
        agent = ChatSorcarAgent("summary-guardrail-test")
        result = agent.run(
            prompt_template=run.prompt,
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=20,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            system_prompt_hook=run.system_prompt_hook,
            tools_hook=run.tools_hook,
            tool_call_hook=run.tool_call_hook,
            llm_call_hook=run.llm_call_hook,
            tool_profile=tool_profile,
            web_tools=False,
            use_memory=False,
            verbose=False,
        )
    return yaml.safe_load(result), requests


def _tool_results(requests: list[dict[str, Any]]) -> list[str]:
    """The tool-result message contents of the last agentic request, in order."""
    last = [request for request in requests if request.get("tools")][-1]
    return [str(m["content"]) for m in last["messages"] if m["role"] == "tool"]


def test_tenth_step_tool_call_is_refused_until_summary_runs(tmp_path: Path) -> None:
    """Steps 1-9 run Bash; step 10's Bash is refused; summary at 11 clears it; 12's Bash runs."""
    script = [_bash(step) for step in range(1, 11)]
    script.append(tool_call_body("summary", {"description": "- ten steps"}, prompt_tokens=700))
    script.append(_bash(12))
    script.append(finish_body("<p>done</p>", prompt_tokens=900))
    parsed, requests = _run(tmp_path, script, "shell+control")
    assert parsed["success"] is True
    results = _tool_results(requests)
    assert len(results) == 12, results
    for step in range(9):
        assert f"step-{step + 1}" in results[step], results[step]
    assert f"Step {SUMMARY_EVERY_STEPS} is a multiple of {SUMMARY_EVERY_STEPS}" in results[9]
    assert "call summary(description=...) first" in results[9]
    assert "retry Bash" in results[9]
    assert "step-10" not in results[9]
    assert "step-12" in results[11], results[11]


def test_finish_is_allowed_on_a_tenth_step(tmp_path: Path) -> None:
    """A run that finishes at step 10 is not held up by the owed summary."""
    script = [_bash(step) for step in range(1, 10)]
    script.append(finish_body("<p>finished at ten</p>", prompt_tokens=900))
    parsed, requests = _run(tmp_path, script, "shell+control")
    assert parsed["success"] is True
    assert "finished at ten" in parsed["summary"]
    assert len([r for r in requests if r.get("tools")]) == 10


def test_without_the_summary_tool_nothing_is_refused(tmp_path: Path) -> None:
    """A bash-only run has no summary tool, so its tenth step runs like any other."""
    script = [_bash(step) for step in range(1, 12)]
    script.append(finish_body("<p>done</p>", prompt_tokens=900))
    parsed, requests = _run(tmp_path, script, "bash")
    assert parsed["success"] is True
    results = _tool_results(requests)
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
        with serve(script) as (url, requests):
            agent = ChatSorcarAgent("summary-guardrail-two-tasks")
            agent.run(
                prompt_template=run.prompt,
                model_name=MODEL,
                work_dir=str(tmp_path),
                max_steps=20,
                max_budget=1.0,
                model_config={"base_url": url, "api_key": "local"},
                tools_hook=run.tools_hook,
                tool_call_hook=run.tool_call_hook,
                llm_call_hook=run.llm_call_hook,
                tool_profile="shell+control",
                web_tools=False,
                use_memory=False,
                verbose=False,
            )
            results.append(_tool_results(requests))
    assert len(results[0]) == 9 and "step-9" in results[0][8], results[0]
    assert len(results[1]) == 2, results[1]
    assert "step-1" in results[1][0] and "step-2" in results[1][1], results[1]


def test_hooks_directly_a_parallel_step_with_summary_first_passes() -> None:
    """Within one 10th step, a summary call clears the debt for the calls after it."""
    run = evaluate_sea([BaseSea()], "t")
    run.tools_hook([_bash, _run])  # no summary tool: never due
    for _ in range(SUMMARY_EVERY_STEPS):
        run.llm_call_hook([])
    assert run.tool_call_hook("Bash", {}).allowed

    run = evaluate_sea([BaseSea()], "t")

    def summary(description: str) -> str:
        return description

    run.tools_hook([_bash, summary])
    for step in range(1, SUMMARY_EVERY_STEPS):
        run.llm_call_hook([])
        assert run.tool_call_hook("Bash", {}).allowed, step
    run.llm_call_hook([])
    assert run.tool_call_hook("summary", {"description": "x"}).allowed
    assert run.tool_call_hook("Bash", {}).allowed
    for _ in range(SUMMARY_EVERY_STEPS):
        run.llm_call_hook([])
    refused = run.tool_call_hook("Read", {"file_path": "/x"})
    assert not refused.allowed and "retry Read" in refused.text
    assert run.tool_call_hook("finish", {}).allowed
    assert run.tool_call_hook("summary", {"description": "y"}).allowed
    assert run.tool_call_hook("Read", {"file_path": "/x"}).allowed
