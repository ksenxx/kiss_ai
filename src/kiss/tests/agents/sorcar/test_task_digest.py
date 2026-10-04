# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of :mod:`kiss.agents.sorcar.task_digest` and of the
``/ask`` SEA's ``task_context`` tool built on it
(:mod:`kiss.agents.seas.ask.ask_sea`).

Tasks are persisted in the test session's real SQLite history
(``KISS_HOME`` is a temporary directory, see ``conftest.py``) and read
back through ``overview`` / ``transcript_page`` / ``entry_detail`` /
``context``, so every digest branch is exercised on real rows and
events.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.seas.ask import ask_sea
from kiss.agents.sorcar import task_digest
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _append_chat_event,
    _flush_chat_events,
    _save_task_result,
)
from kiss.agents.sorcar.sea_settings import resolve_settings
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

    text = task_digest.overview(task_id)
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
    text = task_digest.overview(task_id)
    assert "Status: finished (Done: " + "r" * 194 + " …[106 more chars])" in text
    finished = task_digest.overview(_persist("Finish", [
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

    bare = task_digest.overview(_persist("Nothing yet", []))
    assert "Transcript entries: 0" in bare
    assert "Last event:" not in bare
    assert "(no transcript entries yet)" in bare

    # A legacy row without ``start_ts`` is timed from its insertion timestamp.
    legacy = task_digest.overview(_persist("Legacy", [], extra={"startTs": 0}))
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
    text = task_digest.overview(task_id)
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

    page = task_digest.transcript_page(task_id, 0, 4)
    assert "Entries 0..3 of 13:" in page
    assert [e[:4] for e in _entries(page)] == ["[0] ", "[1] ", "[2] ", "[3] "]
    assert page.endswith("... 9 more entries; call again with start=4.")
    rest = task_digest.transcript_page(task_id, 4, 400)
    assert "Entries 4..12 of 13:" in rest and rest.endswith("(end of transcript)")
    beyond = task_digest.transcript_page(task_id, 99, 5)
    assert "Entries " not in beyond
    assert beyond.endswith("Transcript entries: 13\n\n(end of transcript)")

    hits = task_digest.transcript_page(task_id, 0, 1, contains="TRIAL")
    assert "Entries containing 'TRIAL' from index 0: 1 shown, 1 more." in hits
    assert "\n[3] OUTPUT: trial 1 done\ntrial 2 done\n" + "x" * 474 + " …[426 more chars]\n" in hits
    assert hits.endswith("... 1 more entries; call again with start=4.")
    more = task_digest.transcript_page(task_id, 4, 10, contains="trial")
    assert [e[:4] for e in _entries(more)] == ["[7] "]
    assert more.endswith("(end of transcript)")
    # Terms are literal, any-of, and applied to the full text, not the clipped rendering.
    deep = task_digest.transcript_page(task_id, 0, 10, contains="|" + "x" * 900 + "|paper.tex")
    assert [e[:4] for e in _entries(deep)] == ["[3] ", "[10]"]
    regex_like = task_digest.transcript_page(task_id, 0, 10, contains="(a+)+$")
    assert _entries(regex_like) == []
    none = task_digest.transcript_page(task_id, 0, 10, contains="no such text anywhere")
    assert "0 shown, 0 more." in none and none.endswith("(end of transcript)")

    assert task_digest.transcript_page("") == "Error: no task id given."
    assert task_digest.transcript_page("no-such-task") == "Error: no task with id 'no-such-task'."
    empty = task_digest.transcript_page(_persist("Nothing yet", []), 0, 5, contains="x")
    assert empty.endswith("(no transcript entries yet)")


def test_step_returns_one_entry_in_full_and_clamps() -> None:
    """``task_step`` returns the unclipped entry, clamps ``max_chars``, and rejects bad indices."""
    task_id = _persist("Run the benchmark", _running_task_events())
    full = task_digest.entry_detail(task_id, 3)
    assert full == "[3] OUTPUT: trial 1 done\ntrial 2 done\n" + "x" * 900
    assert task_digest.entry_detail(task_id, 3, 20) == (
        "[3] OUTPUT: trial 1 done\ntrial 2 …[906 more chars]"
    )
    assert task_digest.entry_detail(task_id, 2, 0).startswith("[2] TOOL CALL Bash({ …[")
    assert task_digest.entry_detail(task_id, 3, 10**9) == full
    assert task_digest.entry_detail(task_id, 13) == "Error: entry index 13 out of range (0..12)."
    assert task_digest.entry_detail(task_id, -1) == "Error: entry index -1 out of range (0..12)."
    assert task_digest.entry_detail("no-such-task", 0) == "Error: no task with id 'no-such-task'."
    assert task_digest.overview("  ") == "Error: no task id given."


def test_event_time_prefers_ts_and_falls_back_to_the_row_timestamp() -> None:
    """The ``Last event`` line uses the event's ``ts``; without one, the events row time."""
    # Measured from now, not from the module-level _NOW_MS: the digest computes the
    # age against time.time(), and a long test session ages _NOW_MS past 10 minutes.
    stamped = int(time.time() * 1000) - 600_000
    task_id = _persist("p", [
        {"type": "prompt", "text": "p"},
        {"type": "tool_call", "name": "Bash", "ts": stamped},
    ])
    assert f"Last event: {task_digest._fmt_ts(stamped)} (10 min " in task_digest.overview(task_id)
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
    assert "Last event: " in task_digest.overview(unstamped)
    assert task_digest.load_task("missing") is None


def _write(path: Path, text: str) -> None:
    """Write *text* to *path*, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_context_of_a_running_task_with_children_and_progress_log(tmp_path: Path) -> None:
    """``task_context`` is the header, the worker sub-agents, the tail of the
    freshest progress log on disk and the whole transcript, oldest first."""
    _write(tmp_path / "PROGRESS.md", "# stale root log\nroot line\n")
    _write(tmp_path / "PROGRESS_LOG.md", "   \n")  # blank: skipped
    time.sleep(0.01)
    _write(tmp_path / "tmp" / "PROGRESS.md", "# live log\nstep one\nstep two\n")
    task_id = _persist(
        "Run the benchmark", _running_task_events(), extra={"work_dir": str(tmp_path)},
    )
    worker = _persist(
        "Run the tests", [], extra={"subagent": {"parent_task_id": task_id}},
    )
    side = _persist(
        "how many trials are done?", [],
        extra={"subagent": {"parent_task_id": task_id, "side_channel": True}},
    )
    text = ask_sea.task_context(task_id)
    head, _, rest = text.partition("== Sub-agent tasks (1) ==\n")
    assert head.startswith("== Task ==\n" + f"Task id: {task_id}\n")
    assert "Status: running" in head and f"Work dir: {tmp_path}" in head
    assert "Last event: " in head and "Now: " in head and "UTC (local: " in head
    assert "SYSTEM TEXT MUST NOT APPEAR" not in text
    children, _, rest = rest.partition("== Progress log written by the task (newest entries) ==\n")
    assert children.startswith(f"[{worker}] steps=0, cost=$0.00, started ")
    assert side not in children
    log, _, transcript = rest.partition("== Transcript (13 entries, oldest first) ==\n")
    assert log == "# live log\nstep one\nstep two\n\n"
    assert "root line" not in text
    entries = _entries(transcript)
    assert len(entries) == 13
    assert entries[0] == "[0] THOUGHT: Plan: run it."
    assert entries[7] == "[7] ASK ANSWER: Q: how many trials are done?"
    assert "[11] RESULT (error): Edit failed" in transcript
    assert transcript.endswith("[12] CUSTOM_NOTE: a note")
    assert "elided" not in text


def test_context_keeps_the_newest_entries_within_the_budget(tmp_path: Path) -> None:
    """Over budget, whole transcript entries go oldest first (marked, multi-line
    entries never split) and the progress log keeps its newest lines within its
    own limit; an unknown task or no work dir is handled."""
    _write(tmp_path / "PROGRESS.md", "\n".join(f"log line {i}" for i in range(2000)))
    events: list[dict[str, Any]] = [{"type": "prompt", "text": "p"}]
    for i in range(400):
        events.append({"type": "tool_result", "content": f"result {i}\n" + "y" * 300})
    task_id = _persist("p", events, extra={"work_dir": str(tmp_path)})

    text = task_digest.context(task_id)
    assert len(text) <= task_digest.MAX_CONTEXT_CHARS
    log = text.partition("(newest entries) ==\n")[2].partition("\n\n== Transcript")[0]
    assert log.startswith("[older progress-log entries elided]\nlog line ")
    assert log.endswith("log line 1999") and len(log) <= 8_000
    transcript = text.partition("(400 entries, oldest first) ==\n")[2]
    marker, _, kept = transcript.partition("\n")
    elided = int(marker.removeprefix("[... ").removesuffix(" older entries elided ...]"))
    assert 200 < elided < 300
    assert kept.startswith(f"[{elided}] RESULT: result {elided}\n" + "y" * 300 + "\n")
    assert kept.endswith("[399] RESULT: result 399\n" + "y" * 300)
    # Adding one more entry (and the marker it still needs) would have overflowed.
    one_more = len(f"[{elided - 1}] RESULT: result {elided - 1}\n" + "y" * 300) + 1
    assert len(text) + one_more > task_digest.MAX_CONTEXT_CHARS

    # Exact-fit boundaries: the marker line counts against the budget, and
    # entry 0 needs no marker.  The header carries the clock (Elapsed, Last
    # event, Now), so its length is re-measured from each result instead of
    # being fixed once.
    transcript_header = "== Transcript (400 entries, oldest first) ==\n"
    head_len = len(text.partition(transcript_header)[0]) + len(transcript_header)

    def transcript_within(tail_chars: int) -> str:
        """Transcript of ``context()`` given a budget of the header plus *tail_chars*.

        Retried when the clock moved the header's length between the budget
        computation and the call (e.g. ``Elapsed: 0 min 9 s`` -> ``0 min 10 s``).
        """
        nonlocal head_len
        for _ in range(10):
            result = task_digest.context(task_id, max_chars=head_len + tail_chars)
            got_head, _, tail = result.partition(transcript_header)
            if len(got_head) + len(transcript_header) == head_len:
                return tail
            head_len = len(got_head) + len(transcript_header)
        raise AssertionError("the header length never settled")

    block = len("[399] RESULT: result 399\n" + "y" * 300)
    two = len("[... 398 older entries elided ...]\n") + 2 * block + 1
    tail = transcript_within(two)
    assert tail.startswith("[... 398 older entries elided ...]\n[398] RESULT")
    assert len(tail) == two
    assert transcript_within(two - 1).startswith(
        "[... 399 older entries elided ...]\n[399] RESULT"
    )
    unlimited = task_digest.context(task_id, max_chars=10**9).partition(transcript_header)[2]
    assert unlimited.startswith("[0] RESULT: result 0\n") and "elided ...]" not in unlimited
    assert transcript_within(len(unlimited)) == unlimited
    assert transcript_within(len(unlimited) - 1).startswith(
        "[... 1 older entries elided ...]\n[1] RESULT"
    )

    # The newest entry is kept whole even when it alone overflows the budget.
    tight = task_digest.context(task_id, max_chars=10)
    assert tight.endswith(
        "[... 399 older entries elided ...]\n[399] RESULT: result 399\n" + "y" * 300
    )
    assert ask_sea.task_context("no-such-task") == "Error: no task with id 'no-such-task'."
    assert ask_sea.task_context("") == "Error: no task id given."
    bare = task_digest.context(_persist("Nothing yet", [], extra={"work_dir": ""}))
    assert "Progress log" not in bare
    assert bare.endswith("oldest first) ==\n(no transcript entries yet)")
    assert task_digest.progress_log_tail("") == ""
    assert task_digest.progress_log_tail(str(tmp_path / "missing")) == ""
    # The log tail, marker included, never exceeds its limit: the cut moves to
    # the next line start, or into the last line when no line start fits.
    marker = "[older progress-log entries elided]\n"
    assert task_digest.progress_log_tail(str(tmp_path), limit=len(marker) + 5) == marker + " 1999"
    assert task_digest.progress_log_tail(str(tmp_path), limit=len(marker) + 20) == (
        marker + "log line 1999"
    )
    assert task_digest.progress_log_tail(str(tmp_path), limit=len(marker) + 28) == (
        marker + "log line 1998\nlog line 1999"
    )
    assert task_digest._tail("aaa\nbb", 5, "M") == "M\nbb"
    assert task_digest._tail("a\nbb\n", 4, "M") == "M\nb\n"
    assert task_digest._tail("ab", 2, "M") == "ab"
    assert task_digest._tail("abc", 2, "M") == "M\n"


def test_ask_agent_has_only_task_context_and_finish_and_answers_from_it(tmp_path: Path) -> None:
    """A real ReAct loop configured from the ``/ask`` SEA ``settings()`` offers
    exactly ``task_context`` and ``finish`` (no built-in tools at all),
    receives the playbook as the system-prompt suffix, gets the real
    context back as a tool result and finishes with the answer.

    The run is configured the way the daemon configures it: the
    ``worker`` preset's flags, the ``none`` tool profile (from which
    the daemon derives ``append_basic_tools=False``), the
    ``system_prompt()`` getter as the base prompt, ``add_to_system_prompt()``
    as the suffix, ``add_to_tools()`` as the extra tools and
    ``add_to_prompt`` with ``{task_id}`` filled in appended to the task.
    """
    task_id = _persist("Run the benchmark", _running_task_events())
    answer = "<p>2 of 10 trials are done; the last step failed to edit paper.tex.</p>"
    script = [
        tool_call_body("task_context", {"task_id": task_id}, prompt_tokens=500),
        finish_body(answer, prompt_tokens=700),
    ]
    settings = resolve_settings(vars(ask_sea))
    assert settings["tool_profile"] == "none"
    assert settings["add_to_prompt"] == ask_sea.ADD_TO_PROMPT
    prompt = "how many trials are done?" + settings["add_to_prompt"].format(task_id=task_id)
    with serve(script) as (url, requests):
        agent = ChatSorcarAgent("ask-sea-test")
        result = agent.run(
            prompt_template=prompt,
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=4,
            model_config={"base_url": url, "api_key": "local"},
            tools=ask_sea.add_to_tools(),
            tool_profile=settings["tool_profile"],
            append_basic_tools=settings["tool_profile"] != "none",
            base_system_prompt=ask_sea.system_prompt(),
            system_prompt=ask_sea.add_to_system_prompt(),
            web_tools=settings["use_web_tools"],
            use_memory=settings["use_memory"],
            is_parallel=settings["is_parallel"],
            verbose=False,
        )
    parsed = yaml.safe_load(result)
    assert parsed["success"] is True
    assert parsed["summary"] == answer

    agentic = [r for r in requests if r.get("tools")]
    assert len(agentic) == 2, [list(r) for r in requests]
    for request in agentic:
        names = {t["function"]["name"] for t in request["tools"]}
        assert names == {"task_context", "finish"}
        system = str(next(m for m in request["messages"] if m["role"] == "system")["content"])
        assert system.startswith(ask_sea.system_prompt())
        assert ask_sea.add_to_system_prompt() in system
    user = next(m for m in agentic[0]["messages"] if m["role"] == "user")
    assert f"task with id {task_id}" in str(user["content"])
    tool_results = [m for m in agentic[1]["messages"] if m["role"] == "tool"]
    assert len(tool_results) == 1
    digest = str(tool_results[0]["content"])
    assert digest.startswith("== Task ==\n" + f"Task id: {task_id}")
    assert "== Transcript (13 entries, oldest first) ==" in digest
    assert "[11] RESULT (error): Edit failed" in digest
