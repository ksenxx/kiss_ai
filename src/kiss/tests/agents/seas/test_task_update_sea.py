# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled task-update agent
(:mod:`kiss.agents.seas.task_update_sea`).

The ``task_transcript`` tool is exercised against tasks persisted in
the test session's real SQLite history (``KISS_HOME`` is a temporary
directory, see ``conftest.py``); the agent-level test runs a real
:class:`ChatSorcarAgent` ReAct loop against the scripted local
chat-completions server configured from the SEA's getters, so the
digest really flows through the tool result into the finish summary.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.seas import task_update_sea as sea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _append_chat_event,
    _flush_chat_events,
)
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)

_SEA_PATH = Path(sea.__file__).resolve()


def _persist_task(
    prompt: str,
    events: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
    chat_id: str = "",
) -> str:
    """Persist a task row with *events* and return its id."""
    payload: dict[str, object] = {
        "model": "test-model",
        "work_dir": "/work/dir",
        "startTs": int(time.time() * 1000) - 90_000,
    }
    payload.update(extra or {})
    task_id, _chat = _add_task(prompt, chat_id=chat_id, extra=payload)
    for ev in events:
        _append_chat_event(ev, task_id=task_id)
    _flush_chat_events(task_id)
    return task_id


def _entries(digest: str) -> list[str]:
    """Return the numbered digest entries of a ``task_transcript`` page."""
    return [line for line in digest.splitlines() if line.startswith("[")]


def test_sea_getters_and_prompt_follow_the_contract() -> None:
    """The SEA pins its run: transcript tool, bash profile, no extras."""
    assert sea.build_prompt("abc123") == (
        "What have the task with abc123 done so far and what are the partial results?"
    )
    assert sea.system_prompt() == sea.SYSTEM_PROMPT
    assert "task_transcript" in sea.SYSTEM_PROMPT
    assert sea.tools() == [sea.task_transcript]
    assert sea.tool_profile() == "bash"
    assert sea.max_budget() == 1.0
    for getter in (
        sea.is_parallel, sea.use_web_tools, sea.use_memory, sea.use_worktree,
        sea.auto_commit, sea.classify_tasks,
    ):
        assert getter() is False, getter.__name__


def test_slash_task_update_resolves_to_the_bundled_sea() -> None:
    """``/task_update <id>`` runs this SEA through ``run_agent`` in the chat."""
    assert sea_commands.get_command("task_update") == _SEA_PATH
    rewritten = sea_commands.rewrite_prompt_if_command("/task_update deadbeef")
    assert rewritten is not None
    prompt, path = rewritten
    assert path == _SEA_PATH
    assert prompt.endswith("TASK TEXT FOR run_agent:\ndeadbeef")


def test_transcript_errors_for_missing_and_unknown_ids() -> None:
    """No id and an unknown id produce one-line errors, not exceptions."""
    assert sea.task_transcript("") == "Error: no task id given."
    assert sea.task_transcript("   ") == "Error: no task id given."
    assert sea.task_transcript("no-such-task") == "Error: no task with id 'no-such-task'."


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


def test_transcript_pages_and_clips_the_page_size() -> None:
    """``start``/``count`` page through the entries; the tail says what remains."""
    events = [{"type": "prompt", "text": "p"}] + [
        {"type": "tool_call", "name": f"tool{i}", "extras": {"i": i}} for i in range(10)
    ]
    task_id = _persist_task("p", events)

    first = sea.task_transcript(task_id, 0, 4)
    assert "Entries 0..3 of 10:" in first
    assert [e[:4] for e in _entries(first)] == ["[0] ", "[1] ", "[2] ", "[3] "]
    assert first.endswith("... 6 more entries; call again with start=4.")

    last = sea.task_transcript(task_id, 4, 100)
    assert "Entries 4..9 of 10:" in last
    assert len(_entries(last)) == 6
    assert last.endswith("(end of transcript)")

    # A negative start and an oversized count are clamped.
    clamped = sea.task_transcript(task_id, -5, 10_000)
    assert "Entries 0..9 of 10:" in clamped
    assert sea._MAX_PAGE == 400


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
    with serve(script) as (url, requests):
        agent = ChatSorcarAgent("task-update-sea-test")
        result = agent.run(
            prompt_template=sea.build_prompt(task_id),
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=4,
            max_budget=sea.max_budget(),
            model_config={"base_url": url, "api_key": "local"},
            tools=sea.tools(),
            tool_profile=sea.tool_profile(),
            base_system_prompt=sea.system_prompt(),
            web_tools=sea.use_web_tools(),
            use_memory=sea.use_memory(),
            is_parallel=sea.is_parallel(),
            verbose=False,
        )
    parsed = yaml.safe_load(result)
    assert parsed["success"] is True
    assert parsed["summary"] == report

    agentic = [r for r in requests if r.get("tools")]
    assert len(agentic) == 2, [list(r) for r in requests]
    for request in agentic:
        names = {t["function"]["name"] for t in request["tools"]}
        assert names == {"Bash", "finish", "task_transcript"}, names
        system = next(m for m in request["messages"] if m["role"] == "system")
        assert str(system["content"]).startswith(sea.SYSTEM_PROMPT)
    user = next(m for m in agentic[0]["messages"] if m["role"] == "user")
    assert sea.build_prompt(task_id) in str(user["content"])
    tool_results = [m for m in agentic[1]["messages"] if m["role"] == "tool"]
    assert len(tool_results) == 1
    digest = str(tool_results[0]["content"])
    assert f"Task id: {task_id}" in digest
    assert "Task prompt: Refactor the parser" in digest
    assert '[0] TOOL CALL Read({"path": "parser.py"})' in digest
    assert "[1] RESULT: def parse(): ..." in digest
