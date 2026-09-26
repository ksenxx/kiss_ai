# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of :mod:`kiss.agents.sorcar.task_digest` through the
``/ask`` SEA's tools (:mod:`kiss.agents.third_party_agents.ask_sea`).

Tasks are persisted in the test session's real SQLite history
(``KISS_HOME`` is a temporary directory, see ``conftest.py``) and read
back through ``task_overview`` / ``task_transcript`` / ``task_step``,
so every digest branch is exercised on real rows and events.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar import task_digest
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _append_chat_event,
    _flush_chat_events,
    _save_task_result,
)
from kiss.agents.third_party_agents import ask_sea
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    finish_body,
    serve,
    tool_call_body,
)

_NOW_MS = int(time.time() * 1000)


def _persist(
    prompt: str,
    events: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
    result: str = "",
) -> str:
    """Persist a task row with *events* (and an optional final result); return its id."""
    payload: dict[str, object] = {
        "model": "test-model",
        "work_dir": "/work/dir",
        # Relative to the real clock at call time (not the frozen module-level
        # _NOW_MS): the digest measures elapsed time against time.time(), so a
        # module imported long before the test ran would report "2 min".
        "startTs": int(time.time() * 1000) - 90_000,
    }
    payload.update(extra or {})
    task_id, _chat = _add_task(prompt, extra=payload)
    for ev in events:
        _append_chat_event(ev, task_id=task_id)
    _flush_chat_events(task_id)
    if result:
        _save_task_result(result, task_id=task_id)
    return task_id


def _running_task_events() -> list[dict[str, Any]]:
    """Events of a task that took steps, got steered, and answered two /ask questions."""
    return [
        {"type": "task_settings", "settings": {"model": "test-model"}},
        {"type": "system_prompt", "text": "SYSTEM TEXT MUST NOT APPEAR"},
        {"type": "prompt",
         "text": "## Previous tasks\nOLD CONTEXT MUST NOT APPEAR\n\nRun the benchmark"},
        {"type": "thinking_delta", "text": "Plan: "},
        {"type": "thinking_delta", "text": "run it."},
        {"type": "text_delta", "text": "Starting the run."},
        {"type": "usage_info", "text": "Steps: 1/100, Budget: $0.10"},
        {"type": "tool_call", "name": "Bash", "callId": 1,
         "description": "start", "command": "python bench.py --status"},
        {"type": "system_output", "text": "trial 1 done\n"},
        {"type": "system_output", "text": "trial 2 done\n" + "x" * 900},
        {"type": "tool_result", "content": "", "tool_name": "Bash"},
        {"type": "tool_call", "name": "summary", "callId": 2,
         "description": "First summary: started the run"},
        {"type": "tool_result", "content": "Summary recorded."},
        {"type": "prompt", "text": "/ask how many trials are done?"},
        {"type": "ask_answer", "question": "how many trials are done?",
         "text": "<p>2 of 10 trials are done.</p>", "success": True},
        {"type": "prompt", "text": "  please also report the cost  "},
        {"type": "ask_answer", "question": "second question", "text": "", "success": False},
        {"type": "tool_call", "name": "Edit", "callId": 3, "path": "paper.tex",
         "old_string": "old", "new_string": ""},
        {"type": "tool_result", "content": "Edit failed", "is_error": True},
        {"type": "usage_info", "text": "Steps: 3/100, Budget: $0.30"},
        {"type": "new_tab", "task_id": "child"},
        {"type": "custom_note", "text": "a note"},
        {"type": "text_delta", "text": "   \n"},
        {"type": "custom_empty"},
    ]


def _entries(text: str) -> list[str]:
    """Return the ``[i] ...`` entry lines of a digest."""
    return [ln for ln in text.splitlines() if ln.startswith("[")]


def test_overview_of_a_running_task_with_children() -> None:
    """The overview shows status, timing, worker children, user messages,
    /ask answers, summaries and the transcript tail, in that order."""
    task_id = _persist("Run the benchmark", _running_task_events())
    worker = _persist(
        "Review the diff of module x carefully",
        [{"type": "tool_call", "name": "Read", "path": "x.py"}],
        extra={"subagent": {"parent_task_id": task_id}, "endTs": _NOW_MS - 1000,
               "steps": 4, "cost": 1.234},
        result="  All   good\nno issues  ",
    )
    running_worker = _persist(
        "Run the tests", [], extra={"subagent": {"parent_task_id": task_id}, "startTs": 0},
    )
    side = _persist(
        "how many trials are done?", [],
        extra={"subagent": {"parent_task_id": task_id, "side_channel": True}},
        result="<p>2 of 10</p>",
    )

    text = ask_sea.task_overview(task_id)
    head, _, rest = text.partition("== Sub-agent tasks")
    assert f"Task id: {task_id}" in head
    assert "Task prompt: Run the benchmark" in head
    assert "Status: running" in head
    assert "Model: test-model" in head
    assert "Work dir: /work/dir" in head
    assert "Elapsed: 1 min" in head
    assert "Spend: Steps: 3/100, Budget: $0.30" in head
    assert "Last event: " in head and " ago)" in head
    assert "Now: " in head and "UTC (local: " in head
    assert "OLD CONTEXT MUST NOT APPEAR" not in text
    assert "SYSTEM TEXT MUST NOT APPEAR" not in text
    assert "NEW_TAB" not in text and "USAGE_INFO" not in text

    children, _, rest = rest.partition("== User messages")
    assert children.startswith(" (2; plus 1 side-channel /ask or task-update runs not listed) ==")
    assert (
        f"[{worker}] steps=4, cost=$1.23, started " in children
        and ", finished: All good no issues\n    task: Review the diff of module x carefully"
        in children
    )
    # ``startTs: 0`` (a legacy row) falls back to the row's insertion time.
    row = task_digest.load_task(running_worker)
    assert row is not None and int(row["start_ts"] or 0) == 0
    created = task_digest._fmt_ts(int(float(row["timestamp"]) * 1000))
    assert (
        f"[{running_worker}] steps=0, cost=$0.00, started {created}, running\n"
        "    task: Run the tests" in children
    )
    assert task_digest._fmt_ts(0) == "unknown"
    assert side not in children

    users, _, rest = rest.partition("== Previous /ask answers")
    assert users.startswith(" after the initial prompt (1) ==")
    assert "USER: please also report the cost" in users
    assert "/ask how many" not in users

    asks, _, rest = rest.partition("== Progress summaries")
    assert asks.startswith(" (2 of 2) ==")
    assert "ASK ANSWER: Q: how many trials are done?\nA: <p>2 of 10 trials are done.</p>" in asks
    assert "ASK ANSWER: Q: second question\nA: (no answer)" in asks

    summaries, _, tail = rest.partition("== Last ")
    assert summaries.startswith(" written by the task (1) ==")
    assert "SUMMARY: First summary: started the run" in summaries
    assert tail.startswith("13 of 13 transcript entries ==")
    entries = _entries(tail)
    assert entries[0] == "[0] THOUGHT: Plan: run it."
    assert entries[1] == "[1] ASSISTANT: Starting the run."
    assert entries[2] == (
        '[2] TOOL CALL Bash({"description": "start", "command": "python bench.py --status"})'
    )
    assert "[3] OUTPUT: trial 1 done\ntrial 2 done\n" + "x" * 474 + " …[426 more chars]\n" in tail
    assert "[4] RESULT: " in tail
    assert entries[5] == "[5] SUMMARY: First summary: started the run"
    assert entries[6] == "[6] RESULT: Summary recorded."
    assert entries[7] == "[7] ASK ANSWER: Q: how many trials are done?"
    assert entries[8] == "[8] USER: please also report the cost"
    assert entries[9] == "[9] ASK ANSWER: Q: second question"
    assert entries[10] == (
        '[10] TOOL CALL Edit({"path": "paper.tex", "old_string": "old", "new_string": ""})'
    )
    assert "[11] RESULT (error): Edit failed" in tail
    assert "[12] CUSTOM_NOTE: a note" in tail
    # Whitespace-only streamed text and a text-less event leave no entry.
    assert len(entries) == 13
    assert tail.rstrip().endswith("task_step(task_id, index) returns one entry in full.")


def test_overview_of_a_finished_task_without_children_or_summaries() -> None:
    """A finished task shows its result and end time; empty sections say ``(none)``."""
    start = _NOW_MS - 3_700_000
    task_id = _persist(
        "Only a prompt", [{"type": "result", "text": "success: true\nsummary: done"}],
        extra={"startTs": start, "endTs": start + 3_661_000},
        result="Done: " + "r" * 300,
    )
    text = ask_sea.task_overview(task_id)
    assert "Status: finished (Done: " + "r" * 194 + " …[106 more chars])" in text
    finished = ask_sea.task_overview(_persist("Finish", [
        {"type": "tool_call", "name": "finish",
         "extras": {"success": True, "summary_in_html": "<p>ok</p>"}},
    ]))
    assert '[0] FINISH: {"success": true, "summary_in_html": "<p>ok</p>"}' in finished
    assert "Elapsed: 1 h 1 min 1 s" in text
    assert "Ended: " in text
    assert "Spend: not recorded yet" in text
    assert text.count("(none)") == 4
    assert (
        "== Last 1 of 1 transcript entries ==\n[0] TASK RESULT: success: true\nsummary: done"
        in text
    )

    bare = ask_sea.task_overview(_persist("Nothing yet", []))
    assert "Transcript entries: 0" in bare
    assert "Last event:" not in bare
    assert "(no transcript entries yet)" in bare

    # A legacy row without ``start_ts`` is timed from its insertion timestamp.
    legacy = ask_sea.task_overview(_persist("Legacy", [], extra={"startTs": 0}))
    assert "Started: unknown" not in legacy
    assert "Started: 20" in legacy and "Elapsed: 0 min " in legacy


def test_overview_omits_middle_summaries_and_keeps_last_three_asks() -> None:
    """Fifteen summaries print as the first 2 + last 10; only the last 3 /ask answers show."""
    events: list[dict[str, Any]] = [{"type": "prompt", "text": "p"}]
    for i in range(15):
        events.append({"type": "tool_call", "name": "summary", "description": f"summary {i}"})
    for i in range(5):
        events.append({"type": "ask_answer", "question": f"q{i}", "text": f"a{i}"})
    task_id = _persist("p", events)
    text = ask_sea.task_overview(task_id)
    section = text.partition("== Progress summaries written by the task (15) ==")[2]
    section = section.partition("== Last ")[0]
    lines = [ln for ln in section.strip().splitlines()]
    assert lines[0] == "[0] SUMMARY: summary 0"
    assert lines[1] == "[1] SUMMARY: summary 1"
    assert lines[2] == "... 3 earlier summaries omitted (see task_transcript) ..."
    assert lines[3] == "[5] SUMMARY: summary 5"
    assert lines[-1] == "[14] SUMMARY: summary 14"
    asks = text.partition("== Previous /ask answers (3 of 5) ==")[2].partition("== Progress")[0]
    assert "Q: q1\n" not in asks and "Q: q2\n" in asks and "Q: q4\n" in asks


def test_transcript_pages_filters_and_reports_errors() -> None:
    """``task_transcript`` pages, regex-filters with original indices, and errors cleanly."""
    task_id = _persist("Run the benchmark", _running_task_events())

    page = ask_sea.task_transcript(task_id, 0, 4)
    assert "Entries 0..3 of 13:" in page
    assert [e[:4] for e in _entries(page)] == ["[0] ", "[1] ", "[2] ", "[3] "]
    assert page.endswith("... 9 more entries; call again with start=4.")
    rest = ask_sea.task_transcript(task_id, 4, 400)
    assert "Entries 4..12 of 13:" in rest and rest.endswith("(end of transcript)")
    beyond = ask_sea.task_transcript(task_id, 99, 5)
    assert "Entries " not in beyond
    assert beyond.endswith("Transcript entries: 13\n\n(end of transcript)")

    hits = ask_sea.task_transcript(task_id, 0, 1, contains="TRIAL")
    assert "Entries containing 'TRIAL' from index 0: 1 shown, 1 more." in hits
    assert "\n[3] OUTPUT: trial 1 done\ntrial 2 done\n" + "x" * 474 + " …[426 more chars]\n" in hits
    assert hits.endswith("... 1 more entries; call again with start=4.")
    more = ask_sea.task_transcript(task_id, 4, 10, contains="trial")
    assert [e[:4] for e in _entries(more)] == ["[7] "]
    assert more.endswith("(end of transcript)")
    # Terms are literal, any-of, and applied to the full text, not the clipped rendering.
    deep = ask_sea.task_transcript(task_id, 0, 10, contains="|" + "x" * 900 + "|paper.tex")
    assert [e[:4] for e in _entries(deep)] == ["[3] ", "[10]"]
    regex_like = ask_sea.task_transcript(task_id, 0, 10, contains="(a+)+$")
    assert _entries(regex_like) == []
    none = ask_sea.task_transcript(task_id, 0, 10, contains="no such text anywhere")
    assert "0 shown, 0 more." in none and none.endswith("(end of transcript)")

    assert ask_sea.task_transcript("") == "Error: no task id given."
    assert ask_sea.task_transcript("no-such-task") == "Error: no task with id 'no-such-task'."
    empty = ask_sea.task_transcript(_persist("Nothing yet", []), 0, 5, contains="x")
    assert empty.endswith("(no transcript entries yet)")


def test_step_returns_one_entry_in_full_and_clamps() -> None:
    """``task_step`` returns the unclipped entry, clamps ``max_chars``, and rejects bad indices."""
    task_id = _persist("Run the benchmark", _running_task_events())
    full = ask_sea.task_step(task_id, 3)
    assert full == "[3] OUTPUT: trial 1 done\ntrial 2 done\n" + "x" * 900
    assert ask_sea.task_step(task_id, 3, 20) == (
        "[3] OUTPUT: trial 1 done\ntrial 2 …[906 more chars]"
    )
    assert ask_sea.task_step(task_id, 2, 0).startswith("[2] TOOL CALL Bash({ …[")
    assert ask_sea.task_step(task_id, 3, 10**9) == full
    assert ask_sea.task_step(task_id, 13) == "Error: entry index 13 out of range (0..12)."
    assert ask_sea.task_step(task_id, -1) == "Error: entry index -1 out of range (0..12)."
    assert ask_sea.task_step("no-such-task", 0) == "Error: no task with id 'no-such-task'."
    assert ask_sea.task_overview("  ") == "Error: no task id given."


def test_event_time_prefers_ts_and_falls_back_to_the_row_timestamp() -> None:
    """The ``Last event`` line uses the event's ``ts``; without one, the events row time."""
    stamped = _NOW_MS - 600_000
    task_id = _persist("p", [
        {"type": "prompt", "text": "p"},
        {"type": "tool_call", "name": "Bash", "ts": stamped},
    ])
    assert f"Last event: {task_digest._fmt_ts(stamped)} (10 min " in ask_sea.task_overview(task_id)
    task = task_digest.load_task(task_id)
    assert task is not None
    last = task["events"][-1]
    assert task_digest._event_ms(last) == stamped
    last.pop("ts")
    assert task_digest._event_ms(last) == int(float(last["_timestamp"]) * 1000) > 0
    assert task_digest._event_ms({}) == 0
    assert task_digest._event_ms({"ts": "12.7"}) == 12
    assert task_digest._event_ms({"ts": "not-an-int", "_timestamp": 5.0}) == 0
    assert task_digest._event_ms({"ts": None, "_timestamp": "bad"}) == 0
    unstamped = _persist("p", [{"type": "tool_call", "name": "Bash"}])
    assert "Last event: " in ask_sea.task_overview(unstamped)
    assert task_digest.load_task("missing") is None


def test_ask_agent_gets_only_the_review_tools_and_answers_from_the_overview(tmp_path: Path) -> None:
    """A real ReAct loop configured from the ``/ask`` SEA getters offers the
    three trajectory tools plus the ``review`` profile (no memory, edit,
    browser or fan-out tools), receives the playbook as the system-prompt
    suffix, gets the real overview back as a tool result and finishes with
    the answer."""
    task_id = _persist("Run the benchmark", _running_task_events())
    answer = "<p>2 of 10 trials are done; the last step failed to edit paper.tex.</p>"
    script = [
        tool_call_body("task_overview", {"task_id": task_id}, prompt_tokens=500),
        finish_body(answer, prompt_tokens=700),
    ]
    prompt = "how many trials are done?" + ask_sea.APPEND_TO_PROMPT.replace("<task_id>", task_id)
    with serve(script) as (url, requests):
        agent = ChatSorcarAgent("ask-sea-test")
        result = agent.run(
            prompt_template=prompt,
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=4,
            model_config={"base_url": url, "api_key": "local"},
            tools=ask_sea.tools(),
            tool_profile=ask_sea.tool_profile(),
            base_system_prompt=ask_sea.system_prompt(),
            system_prompt=ask_sea.append_to_system_prompt(),
            web_tools=ask_sea.use_web_tools(),
            use_memory=ask_sea.use_memory(),
            is_parallel=ask_sea.is_parallel(),
            verbose=False,
        )
    parsed = yaml.safe_load(result)
    assert parsed["success"] is True
    assert parsed["summary"] == answer

    agentic = [r for r in requests if r.get("tools")]
    assert len(agentic) == 2, [list(r) for r in requests]
    for request in agentic:
        names = {t["function"]["name"] for t in request["tools"]}
        assert {"task_overview", "task_transcript", "task_step", "Bash", "finish"} <= names
        assert not names & {
            "Edit", "Write", "memory_search", "run_agent", "run_parallel", "go_to_url",
        }
        system = str(next(m for m in request["messages"] if m["role"] == "system")["content"])
        assert system.startswith(ask_sea.system_prompt())
        assert ask_sea.append_to_system_prompt() in system
    user = next(m for m in agentic[0]["messages"] if m["role"] == "user")
    assert f"task with id {task_id}" in str(user["content"])
    tool_results = [m for m in agentic[1]["messages"] if m["role"] == "tool"]
    assert len(tool_results) == 1
    digest = str(tool_results[0]["content"])
    assert digest.startswith("== Task ==\n" + f"Task id: {task_id}")
    assert "== Previous /ask answers (2 of 2) ==" in digest
    assert "[11] RESULT (error): Edit failed" in digest
