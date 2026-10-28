# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled task-update agent
(:mod:`kiss.agents.seas.task_update.task_update_sea`).

The ``task_transcript`` tool is exercised against tasks persisted in
the test session's real SQLite history (``KISS_HOME`` is a temporary
directory, see ``conftest.py``); the agent-level test runs a real
:class:`ChatSorcarAgent` ReAct loop against the scripted local
chat-completions server configured from the SEA's methods, so the
digest really flows through the tool result into the finish summary.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from kiss.agents.seas.task_update import task_update_sea as sea
from kiss.agents.seas.task_update.task_update_sea import TaskUpdateSea
from kiss.agents.sorcar import sea_commands
from kiss.tests.agents.seas.conftest import persist_task, run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_worker_contract,
    system_message,
    tool_names,
    tool_results,
)
from kiss.tests.agents.sorcar.local_model_server import (
    finish_body,
    tool_call_body,
)


def _persist_task(
    prompt: str,
    events: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> str:
    """Persist a still-running task row (started 90 s ago, no spend recorded yet) with
    *events* and return its id; *extra* overrides the row's columns."""
    columns: dict[str, Any] = {
        "model": "test-model",
        "startTs": int(time.time() * 1000) - 90_000,
        "endTs": 0, "cost": 0, "steps": 0, "tokens": 0,
        **(extra or {}),
    }
    return persist_task(prompt, events, **columns)


def _entries(digest: str) -> list[str]:
    """Return the numbered digest entries of a ``task_transcript`` page."""
    return [line for line in digest.splitlines() if line.startswith("[")]


def test_sea_methods_and_prompt_follow_the_contract() -> None:
    """The SEA pins its run: transcript tool, bash profile, no extras."""
    assert sea.build_prompt("abc123") == (
        "What have the task with abc123 done so far and what are the partial results?"
    )
    agent = TaskUpdateSea()
    # ``system_prompt`` replaces the assembled prompt with the fixed one.
    assert agent.system_prompt("ASSEMBLED") == sea.SYSTEM_PROMPT
    assert "task_transcript" in sea.SYSTEM_PROMPT
    assert agent.tools([]) == [sea.task_transcript]
    assert agent.tools([print]) == [print, sea.task_transcript]
    assert_worker_contract(sea, agent, {"tool_profile": "bash", "max_budget": 1.0})


def test_transcript_digests_a_running_task() -> None:
    """Every persisted event kind lands in the digest in its compact form.

    The events use the shapes the server's printer persists: a
    ``tool_call`` keeps ``path`` / ``description`` / ``command`` /
    ``content`` / ``old_string`` / ``new_string`` at the top level and
    the remaining arguments under ``extras``; the ``prompt`` event holds
    the chat-augmented prompt, which the digest skips in favour of the
    row's own task text.
    """
    long_summary = "step " * 300  # 1500 chars: kept whole (limit 4000)
    long_result = "x" * 700  # clipped to 500 chars
    events: list[dict[str, Any]] = [
        {"type": "task_settings", "settings": {"model": "test-model"}},
        {"type": "system_prompt", "text": "SYSTEM TEXT MUST NOT APPEAR"},
        {"type": "prompt", "text": "PREVIOUS CHAT CONTEXT MUST NOT APPEAR\n" * 200
         + "Count the files"},
        {"type": "thinking_start"},
        {"type": "thinking_delta", "text": "Let me "},
        {"type": "thinking_delta", "text": "look."},
        {"type": "thinking_end"},
        {"type": "text_delta", "text": "I will "},
        {"type": "text_delta", "text": "list them."},
        {"type": "text_end"},
        {"type": "usage_info", "text": "Steps: 1/100, Budget: $0.01"},
        {"type": "tool_call", "name": "Bash", "callId": 1,
         "description": "list files", "command": "ls"},
        {"type": "system_output", "text": "a.py\n"},
        {"type": "system_output", "text": "b.py\n"},
        {"type": "tool_result", "content": "", "tool_name": "Bash"},
        {"type": "tool_call", "name": "Read", "callId": 2,
         "path": "/work/dir/a.py", "lang": "python", "extras": {"max_lines": 40}},
        {"type": "tool_result", "content": long_result, "is_error": True},
        {"type": "tool_call", "name": "summary", "callId": 3,
         "description": long_summary},
        {"type": "tool_result", "content": "Summary recorded."},
        {"type": "tool_call", "name": "Edit", "callId": 4, "path": "b.py",
         "old_string": "x = 1", "new_string": ""},
        {"type": "usage_info", "text": "Steps: 4/100, Budget: $0.05"},
        {"type": "new_tab", "task_id": "child"},
        {"type": "custom_note", "text": "a note"},
    ]
    task_id = _persist_task("Count the files", events)

    digest = sea.task_transcript(task_id)
    header, _sep, body = digest.partition("\n\n")
    assert f"Task id: {task_id}" in header
    assert "Task prompt: Count the files" in header
    assert "PREVIOUS CHAT CONTEXT" not in digest
    assert "Status: running" in header
    assert "Model: test-model" in header
    assert "Work dir: /work/dir" in header
    assert "Elapsed: 1 min" in header
    assert "Spend: Steps: 4/100, Budget: $0.05" in header
    assert "SYSTEM TEXT MUST NOT APPEAR" not in digest
    assert "NEW_TAB" not in digest and "USAGE_INFO" not in digest

    entries = _entries(body)
    assert entries[0] == "[0] THOUGHT: Let me look."
    assert entries[1] == "[1] ASSISTANT: I will list them."
    assert entries[2] == (
        '[2] TOOL CALL Bash({"description": "list files", "command": "ls"})'
    )
    assert "[3] OUTPUT: a.py\nb.py\n[4] RESULT: \n" in body
    assert entries[5] == (
        '[5] TOOL CALL Read({"path": "/work/dir/a.py", "max_lines": 40})'
    )
    assert entries[6].startswith("[6] RESULT (error): " + "x" * 500)
    assert entries[6].endswith("…[200 more chars]")
    assert entries[7] == f"[7] SUMMARY: {long_summary}"
    assert entries[8] == "[8] RESULT: Summary recorded."
    assert entries[9] == (
        '[9] TOOL CALL Edit({"path": "b.py", "old_string": "x = 1", '
        '"new_string": ""})'
    )
    assert entries[10] == "[10] CUSTOM_NOTE: a note"
    assert len(entries) == 11
    assert "Transcript entries: 11" in header
    assert digest.endswith("(end of transcript)")


def test_transcript_of_a_finished_subagent_without_events() -> None:
    """A finished sub-agent row with no events digests its prompt and result."""
    start = int(time.time() * 1000) - 125_000
    parent_id = _persist_task("Parent prompt", [])
    task_id = _persist_task(
        "Sub task prompt",
        [{"type": "result", "text": "success: true\nsummary: done", "summary": "done"}],
        extra={
            "startTs": start,
            "endTs": start + 65_000,
            "subagent": {"parent_task_id": parent_id},
        },
    )
    digest = sea.task_transcript(task_id)
    assert "Status: finished" in digest
    assert "Elapsed: 1 min 5 s" in digest
    assert f"Sub-agent of task: {parent_id}" in digest
    assert "Spend: not recorded yet" in digest
    assert "[0] TASK RESULT: success: true\nsummary: done\n" in digest
    assert len(_entries(digest)) == 1

    bare = sea.task_transcript(_persist_task("Only a prompt", []))
    assert "Task prompt: Only a prompt" in bare
    assert "Transcript entries: 0" in bare
    assert _entries(bare) == []
    assert bare.endswith("(no transcript entries yet)")


def test_agent_reads_the_transcript_and_finishes_with_the_report(tmp_path: Path) -> None:
    """With the SEA's configuration the model gets Bash, finish and task_transcript.

    The scripted model calls ``task_transcript`` with the id from the
    prompt and then finishes; the test checks the tools offered, the
    system prompt, the real digest in the tool-result message, and
    that the finish summary is the run's result.
    """
    task_id = _persist_task(
        "Refactor the parser",
        [
            {"type": "prompt", "text": "Refactor the parser"},
            {"type": "tool_call", "name": "Read", "path": "parser.py", "lang": "python"},
            {"type": "tool_result", "content": "def parse(): ..."},
        ],
    )
    report = "<h4>Goal</h4><ul><li>Refactor the parser</li></ul>"
    script = [
        tool_call_body("task_transcript", {"task_id": task_id}, prompt_tokens=500),
        finish_body(report, prompt_tokens=700),
    ]
    run = sea_commands.evaluate_sea([TaskUpdateSea()], sea.build_prompt(task_id))
    assert run.prompt == sea.build_prompt(task_id)
    assert run.tools_hook is not None and run.system_prompt_hook is not None
    parsed, agentic = run_sea(run, script, tmp_path, max_steps=4)
    assert parsed["success"] is True
    assert parsed["summary"] == report

    assert len(agentic) == 2
    for request in agentic:
        names = tool_names(request)
        assert names == {"Bash", "finish", "task_transcript"}, names
        system = system_message(request)
        assert system.startswith(sea.SYSTEM_PROMPT)
    user = next(m for m in agentic[0]["messages"] if m["role"] == "user")
    assert sea.build_prompt(task_id) in str(user["content"])
    results = tool_results(agentic[1])
    assert len(results) == 1
    digest = results[0]
    assert f"Task id: {task_id}" in digest
    assert "Task prompt: Refactor the parser" in digest
    assert '[0] TOOL CALL Read({"path": "parser.py"})' in digest
    assert "[1] RESULT: def parse(): ..." in digest
