# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The framework-written progress recap (``kiss.core.step_recap``) and the no-refusal root layer.

A run that has the ``summary`` tool gets a ``summary`` event every ten
steps written by the executor from its tool log; the model is never
refused a tool call for an "owed" summary, and a ``summary`` it writes
itself restarts the cadence.  The tests drive a real
:class:`ChatSorcarAgent` against the scripted local model with the
hooks :func:`evaluate_sea` stages for the bare base, exactly as the
daemon wires them, and read the events off a capturing printer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.sorcar.sea_commands import evaluate_sea
from kiss.core.printer import Printer
from kiss.core.step_recap import SUMMARY_EVERY_STEPS, ToolRecord, describe_args, first_line, recap
from kiss.tests.agents.seas.conftest import run_sea
from kiss.tests.agents.seas.sea_contract import tool_results
from kiss.tests.agents.sorcar.local_model_server import finish_body, tool_call_body


class CapturePrinter(Printer):
    """Records every event; no token callback, so the scripted model is not asked to stream."""

    # ``None`` satisfies the abstract method and tells the agent not to stream.
    token_callback = None  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:  # noqa: A002
        self.events.append({"type": type, "content": content, **kwargs})

    def reset(self) -> None:
        pass


def _bash(step: int) -> bytes:
    return tool_call_body(
        "Bash", {"command": f"echo step-{step}", "description": f"step {step}"},
        prompt_tokens=500 + step,
    )


def _run(
    tmp_path: Path, script: list[bytes], tool_profile: str
) -> tuple[Any, list[dict], CapturePrinter]:
    run = evaluate_sea([BaseSea()], "count to twelve with Bash")
    printer = CapturePrinter()
    parsed, requests = run_sea(
        run, script, tmp_path, max_steps=30, tool_profile=tool_profile, printer=printer,
    )
    return parsed, requests, printer


def _summary_events(printer: CapturePrinter) -> list[dict[str, Any]]:
    return [
        e["tool_input"] for e in printer.events
        if e["type"] == "tool_call" and e["content"] == "summary"
    ]


def test_tenth_step_runs_and_the_framework_emits_the_recap(tmp_path: Path) -> None:
    """Twelve Bash steps all run; after step 10 a framework ``summary`` event lists steps 1-10."""
    script = [_bash(step) for step in range(1, 13)]
    script.append(finish_body("<p>done</p>", prompt_tokens=900))
    parsed, requests, printer = _run(tmp_path, script, "shell+control")
    assert parsed["success"] is True
    results = tool_results(requests[-1])
    assert len(results) == 12, results
    for step in range(12):
        assert f"step-{step + 1}" in results[step], results[step]
        assert "REFUSED" not in results[step]
    summaries = _summary_events(printer)
    assert len(summaries) == 1, summaries
    text = summaries[0]["description"]
    assert text.startswith("Steps 1-10 (framework recap")
    assert "**step 1** `Bash` step 1 → step-1" in text
    assert "**step 10** `Bash` step 10 → step-10" in text
    assert "step 11" not in text
    assert summaries[0]["source"] == "framework"
    # The recap is for the panel only: the model's conversation never saw it.
    assert "framework recap" not in str(requests[-1])
    # The footer after a step names what that step's tool output added.
    assert "This step's tool output: ~" in results[0], results[0]
    assert "This step's model call: $" in results[0], results[0]


def test_the_models_own_summary_restarts_the_cadence(tmp_path: Path) -> None:
    """A summary written at step 5 means the next framework recap covers steps 6-15."""
    script = [_bash(step) for step in range(1, 5)]
    script.append(tool_call_body("summary", {"description": "- four steps"}, prompt_tokens=700))
    script.extend(_bash(step) for step in range(6, 16))
    script.append(finish_body("<p>done</p>", prompt_tokens=900))
    parsed, _requests, printer = _run(tmp_path, script, "shell+control")
    assert parsed["success"] is True
    summaries = _summary_events(printer)
    assert [s["description"][:30] for s in summaries] == [
        "- four steps", "Steps 6-15 (framework recap of",
    ], summaries
    assert "step 5" not in summaries[1]["description"]
    assert "**step 15**" in summaries[1]["description"]


def test_without_the_summary_tool_no_recap_is_emitted(tmp_path: Path) -> None:
    """A bash-only run has no summary tool: no recap event, nothing refused."""
    script = [_bash(step) for step in range(1, 12)]
    script.append(finish_body("<p>done</p>", prompt_tokens=900))
    parsed, requests, printer = _run(tmp_path, script, "bash")
    assert parsed["success"] is True
    results = tool_results(requests[-1])
    assert len(results) == 11, results
    assert "step-10" in results[9], results[9]
    assert _summary_events(printer) == []


def test_a_second_run_on_the_same_hooks_starts_its_count_afresh(tmp_path: Path) -> None:
    """The daemon runs the ``<task>`` blocks of one prompt on the same hooks, one after another.

    Nine steps of the first task and two of the second never add up to
    a recap: each run's count starts at zero.
    """
    run = evaluate_sea([BaseSea()], "two tasks")
    first = [_bash(step) for step in range(1, 10)]
    first.append(finish_body("<p>first</p>", prompt_tokens=900))
    second = [_bash(step) for step in range(1, 3)]
    second.append(finish_body("<p>second</p>", prompt_tokens=900))
    printer = CapturePrinter()
    for script in (first, second):
        run_sea(
            run, script, tmp_path, max_steps=20, tool_profile="shell+control", printer=printer,
        )
    assert _summary_events(printer) == []


def test_hooks_directly_never_refuse() -> None:
    """The root layer's hooks count steps and allow every call, with or without ``summary``."""
    run = evaluate_sea([BaseSea()], "t")

    def summary(description: str) -> str:
        return description

    run.tools_hook([_bash, summary])
    for _ in range(2 * SUMMARY_EVERY_STEPS):
        run.llm_call_hook([])
        assert run.tool_call_hook("Bash", {}).allowed
        assert run.tool_call_hook("Read", {"file_path": "/x"}).allowed
    assert run.tool_call_hook("summary", {"description": "x"}).allowed


def test_recap_rendering_helpers() -> None:
    """``first_line`` clips, ``describe_args`` picks the title, ``recap`` filters by step."""
    assert first_line("\n\n  hello world \nsecond") == "hello world"
    assert first_line("x" * 200, limit=10) == "x" * 9 + "…"
    assert first_line(None) == ""
    assert describe_args({"command": "ls -la", "description": "list"}) == "list"
    assert describe_args({"file_path": "/a/b.py", "max_lines": 3}) == "/a/b.py"
    assert describe_args({"k": 5, "other": "value"}) == "5"
    assert describe_args({}) == ""
    records = [
        ToolRecord(1, "Read", "/a", "line one"),
        ToolRecord(2, "Edit", "/a", "Error: not read", is_error=True),
        ToolRecord(3, "Bash", "", ""),
        ToolRecord(11, "Bash", "late", "x"),
    ]
    text = recap(records, 1, 10)
    assert "**step 1** `Read` /a → line one" in text
    assert "**step 2** `Edit` /a → ERROR: Error: not read" in text
    assert "- **step 3** `Bash`\n" in text + "\n"
    assert "late" not in text
    assert recap([], 1, 10).endswith("- no tool calls")
