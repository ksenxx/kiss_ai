# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A task's ``/ask`` context shows only the progress log the task itself wrote.

Work dirs are shared by every task launched in them.  The context picked
the newest ``PROGRESS_LOG.md`` / ``PROGRESS.md`` / ``tmp/PROGRESS.md`` of
the task's work dir modified during the task's run and labelled it
"Progress log written by the task".  A modification time does not name
the writer: a task overlapping another one in the same directory showed
the other task's log as its own.  The log is now the file the task's own
newest successful ``Write`` / ``Edit`` of a progress log targeted, read
only while it still holds the text that call wrote.  Real history rows
and events, real files; nothing is mocked.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import task_digest
from kiss.agents.sorcar.persistence import _add_task, _append_chat_event, _flush_chat_events
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.tool_interrupt import USER_INTERRUPTED_MESSAGE
from kiss.core.tool_verdict import ToolRedirect

_LOG_HEADER = "== Progress log written by the task (newest entries) ==\n"


def _write_call(path: Path | str, content: str) -> list[dict[str, Any]]:
    """Events of a successful ``Write`` of *content* to *path*."""
    return [
        {"type": "tool_call", "name": "Write", "path": str(path), "content": content},
        {
            "type": "tool_result",
            "content": f"Successfully wrote {len(content)} characters",
            "tool_name": "Write",
        },
    ]


def _edit_call(path: Path, extras: dict[str, Any], result: str) -> list[dict[str, Any]]:
    """Events of an ``Edit`` of *path* with the given arguments, answered by *result*."""
    return [
        {"type": "tool_call", "name": "Edit", "path": str(path), **extras},
        {"type": "tool_result", "content": result, "tool_name": "Edit"},
    ]


def _task(
    work_dir: Path,
    events: list[dict[str, Any]],
    start: float,
    end: float | None = None,
) -> str:
    """Persist a task row run in *work_dir* from *start* (to *end*) with *events*."""
    extra: dict[str, object] = {
        "model": "test-model",
        "work_dir": str(work_dir),
        "startTs": int(start * 1000),
    }
    if end is not None:
        extra["endTs"] = int(end * 1000)
    task_id, _chat = _add_task("Refactor the parser", extra=extra)
    for event in [{"type": "prompt", "text": "p"}, *events]:
        _append_chat_event(event, task_id=task_id)
    _flush_chat_events(task_id)
    return task_id


def _write(path: Path, text: str, mtime: float | None = None) -> None:
    """Write *text* to *path* (parents created), optionally setting its modification time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _log_of(task_id: str) -> str:
    """Return the progress-log section of *task_id*'s ``/ask`` context (``""``: none).

    The context shows the log with its trailing whitespace stripped.
    """
    text = task_digest.context(task_id)
    return text.partition(_LOG_HEADER)[2].partition("\n\n== Transcript")[0]


def test_an_overlapping_task_does_not_show_the_log_only_the_other_task_wrote(
    tmp_path: Path,
) -> None:
    now = time.time()
    log = tmp_path / "tmp" / "PROGRESS.md"
    bash = [
        {"type": "tool_call", "name": "Bash", "command": "make"},
        {"type": "tool_result", "content": "ok"},
    ]
    # A runs from now - 3000 and never writes a progress log; B starts later
    # in the same work dir and writes the shared one while A runs.
    running = _task(tmp_path, bash, now - 3000)
    ended = _task(tmp_path, bash, now - 3000, end=now - 500)
    _write(log, "# B's log\n- step 1\n", mtime=now - 1000)
    writer = _task(tmp_path, _write_call(log, "# B's log\n- step 1\n"), now - 1500, end=now - 900)

    assert _log_of(writer) == "# B's log\n- step 1"
    for task_id in (running, ended):
        text = task_digest.context(task_id)
        assert "Progress log" not in text and "B's log" not in text


def test_a_log_another_task_rewrote_since_the_tasks_last_write_is_not_shown(
    tmp_path: Path,
) -> None:
    now = time.time()
    log = tmp_path / "tmp" / "PROGRESS.md"
    _write(log, "# A's log\n")
    first = _task(tmp_path, _write_call(log, "# A's log\n"), now - 3000)
    assert _log_of(first) == "# A's log"

    _write(log, "# B's log\n")
    second = _task(tmp_path, _write_call(log, "# B's log\n"), now - 1000)

    assert _log_of(second) == "# B's log"
    # Each task is shown the text it wrote (rebuilt from its own calls), never B's.
    assert _log_of(first) == "# A's log"
    assert "B's log" not in task_digest.context(first)


def test_the_newest_successful_write_names_the_text_the_log_must_hold(tmp_path: Path) -> None:
    now = time.time()
    log = tmp_path / "PROGRESS_LOG.md"
    body = "# log\n- step 1\n- step 2\n"
    _write(log, body)
    events = [
        *_write_call(log, "# log\n- step 1\n"),
        *_edit_call(
            log,
            {"old_string": "- step 1\n", "new_string": "- step 1\n- step 2\n"},
            "edited",
        ),
        # Calls that wrote nothing: a failed Edit (its error is the returned
        # text), a Write that raised, and a call that only read the log.
        *_edit_call(
            log,
            {"old_string": "- step 9", "new_string": "- step 3"},
            "Error: String not found in file",
        ),
        {"type": "tool_call", "name": "Write", "path": str(log), "content": "# lost\n"},
        {"type": "tool_result", "content": "interrupted", "is_error": True},
        # A Write and an Edit the user stopped, as the executor records them
        # (no error, ``interrupted``): neither says what the file holds.
        {"type": "tool_call", "name": "Write", "path": str(log), "content": "# stopped\n"},
        {
            "type": "tool_result",
            "content": USER_INTERRUPTED_MESSAGE,
            "is_error": False,
            "interrupted": True,
            "tool_name": "Write",
        },
        {
            "type": "tool_call",
            "name": "Edit",
            "path": str(log),
            "old_string": "- step 2\n",
            "new_string": "- step 2\n- stopped\n",
        },
        {
            "type": "tool_result",
            "content": USER_INTERRUPTED_MESSAGE,
            "is_error": False,
            "interrupted": True,
            "tool_name": "Edit",
        },
        {"type": "tool_call", "name": "Read", "path": str(log)},
        {"type": "tool_result", "content": body},
    ]
    task_id = _task(tmp_path, events, now - 60)
    assert _log_of(task_id) == body.rstrip()

    # A multi-edit call: its edits apply in order to the task's own text,
    # whatever the file holds.
    edits = json.dumps(
        [
            {"old_string": "# log", "new_string": "# progress log"},
            {"old_string": "- step 2\n", "new_string": "- step 2\n- step 3\n"},
        ]
    )
    multi = _task(
        tmp_path, [*events, *_edit_call(log, {"extras": {"edits": edits}}, "ok")], now - 60
    )
    assert _log_of(multi) == "# progress log\n- step 1\n- step 2\n- step 3"
    _write(log, "# progress log\n- step 1\n- step 2\n- step 3\n")
    assert _log_of(multi) == "# progress log\n- step 1\n- step 2\n- step 3"

    # A call that cannot be replayed (a malformed edits list) or deletes
    # text the task never wrote leaves the task's text as it was.
    for extras in (
        {"extras": {"edits": "not json"}},
        {"old_string": "- step 3\n", "new_string": ""},
    ):
        unproven = _task(tmp_path, [*events, *_edit_call(log, extras, "ok")], now - 60)
        assert _log_of(unproven) == body.rstrip()


def test_whitespace_tolerant_edits_are_replayed_as_the_tool_applied_them(tmp_path: Path) -> None:
    # S9/F3: Edit accepts an old_string that matches the file whitespace
    # aside (indentation, trailing blanks, a tab for spaces) and re-indents
    # the new text to the file's lines; the replay did only exact and CRLF
    # matching, so such an edit was replayed as an append and the digest
    # kept the entry the edit replaced next to its replacement.
    log = tmp_path / "tmp" / "PROGRESS.md"
    tools = UsefulTools(work_dir=str(tmp_path))
    original = "# log\n  - still pending\n  - a\n  - b\n\tc\n"
    events = _write_call(log, original)
    assert tools.Write(str(log), original).startswith("Successfully")
    multi = json.dumps(
        [
            # Lines the file indents deeper: the new text is re-indented as
            # the file's lines are.  A tab for the blank, trailing blanks.
            {"old_string": " - a\n - b\n", "new_string": " - a\n - b done\n"},
            {"old_string": " c  ", "new_string": "\tc done"},
        ]
    )
    cases: tuple[dict[str, Any], ...] = (
        # Deeper indentation than the file's line: applied as written.
        {"old_string": "    - still pending\n", "new_string": "    - done\n"},
        {"extras": {"edits": multi}},
        # Trailing blanks the file line does not have.
        {"old_string": "# log   \n", "new_string": "# progress\n"},
    )
    for extras in cases:
        edits = extras.get("extras", {}).get("edits")
        result = (
            tools.Edit(str(log), edits=edits)
            if edits
            else tools.Edit(str(log), str(extras["old_string"]), str(extras["new_string"]))
        )
        assert result.startswith("Successfully"), result
        events += _edit_call(log, extras, result)
    expected = "# progress\n    - done\n  - a\n  - b done\n\tc done\n"
    assert log.read_text(encoding="utf-8") == expected
    task_id = _task(tmp_path, events, time.time() - 60)

    assert _log_of(task_id) == expected.rstrip()

    # The first edit on a CRLF log: the CRLF fallback and the tolerant
    # match combine as they do in the tool.
    crlf = original.replace("\n", "\r\n")
    shutil.rmtree(tmp_path / "tmp")
    events = _write_call(log, crlf)
    assert tools.Write(str(log), crlf).startswith("Successfully")
    result = tools.Edit(str(log), "    - still pending\n", "    - done\n")
    assert result.startswith("Successfully"), result
    events += _edit_call(
        log, {"old_string": "    - still pending\n", "new_string": "    - done\n"}, result
    )
    expected = "# log\r\n    - done\r\n  - a\r\n  - b\r\n\tc\r\n"
    assert log.read_bytes() == expected.encode()
    assert _log_of(_task(tmp_path, events, time.time() - 60)) == expected.rstrip()


def test_a_worktree_tasks_log_is_read_in_its_worktree_then_where_the_merge_left_it(
    tmp_path: Path,
) -> None:
    now = time.time()
    repo = tmp_path / "repo"
    worktree_log = repo / ".kiss-worktrees" / "kiss_wt-abc" / "tmp" / "PROGRESS.md"
    main_log = repo / "tmp" / "PROGRESS.md"
    text = "# worktree task's log\n- step 1\n"
    _write(main_log, "# an earlier merge's log\n")
    _write(worktree_log, text)
    # The history row records the main checkout as the task's work dir.
    task_id = _task(repo, _write_call(worktree_log, text), now - 600)
    assert _log_of(task_id) == text.rstrip()

    # Merged: the worktree is gone and the merge copied the log.
    shutil.rmtree(repo / ".kiss-worktrees")
    _write(main_log, text)
    assert _log_of(task_id) == text.rstrip()

    # A later merge replaced the copy with its own task's log: still only
    # this task's text is shown, never the later task's.
    _write(main_log, "# a later task's log\n")
    assert _log_of(task_id) == text.rstrip()
    assert "a later task's log" not in task_digest.context(task_id)


def test_a_relative_log_path_is_read_from_the_work_dir(tmp_path: Path) -> None:
    text = "# sub-agent log\n" + "".join(f"- step {i}\n" for i in range(10))
    _write(tmp_path / "tmp" / "scratch-1" / "PROGRESS.md", text)
    task_id = _task(tmp_path, _write_call("tmp/scratch-1/PROGRESS.md", text), time.time() - 60)

    assert _log_of(task_id) == text.rstrip()
    task = task_digest.load_task(task_id)
    assert task is not None
    marker = "[older progress-log entries elided]\n"
    assert task_digest.progress_log_tail(task, limit=len(marker) + 9) == marker + "- step 9\n"
    assert task_digest.progress_log_tail({"events": [], "work_dir": str(tmp_path)}) == ""


def test_a_here_document_append_replays_only_the_text_it_appended(tmp_path: Path) -> None:
    """Audit 2026-11-10 S12-F1: an append is logged as a ``Write`` of the whole file.

    Under an agent the Bash tool applies ``cat >> log <<EOF`` as
    ``Write(the file's text + the body)``, so the event's content holds
    what other tasks left in the shared log: only the body is replayed.
    A here-document that overwrites, and a later plain ``Write``, still
    put their whole content.
    """
    now = time.time()
    log = tmp_path / "tmp" / "PROGRESS.md"
    _write(log, "# another task\n- OTHER-TASK-ENTRY\n")
    tools = UsefulTools(work_dir=str(tmp_path))
    tools.redirect = True  # as under an agent
    command = "cat >> tmp/PROGRESS.md <<'EOF'\n- MY-ENTRY\nEOF"
    with pytest.raises(ToolRedirect) as redirect:
        tools.Bash(command=command, description="log")
    content = redirect.value.arguments["content"]
    assert content == "# another task\n- OTHER-TASK-ENTRY\n- MY-ENTRY\n"
    appended = [
        {"type": "tool_call", "name": "Bash", "command": command},
        *_write_call(redirect.value.arguments["file_path"], content),
        {"type": "tool_result", "content": redirect.value.note, "tool_name": "Bash"},
    ]
    assert _log_of(_task(tmp_path, appended, now - 60)) == "- MY-ENTRY"
    overwritten = [
        {"type": "tool_call", "name": "Bash",
         "command": "cat > tmp/PROGRESS.md <<'EOF'\n# mine\nEOF"},
        *_write_call(log, "# mine\n"),
        {"type": "tool_result", "content": "ok", "tool_name": "Bash"},
    ]
    assert _log_of(_task(tmp_path, [*appended, *overwritten], now - 60)) == "# mine"
    rewritten = [*appended, *_write_call(log, "# all of it\n- new\n")]
    assert _log_of(_task(tmp_path, rewritten, now - 60)) == "# all of it\n- new"
