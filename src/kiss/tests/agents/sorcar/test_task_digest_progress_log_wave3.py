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

import functools
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import task_digest
from kiss.agents.sorcar.docker_manager import _with_exit_code
from kiss.agents.sorcar.persistence import _add_task, _append_chat_event, _flush_chat_events
from kiss.agents.sorcar.shell_guards import DESTRUCTIVE_VERDICT
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.kiss_tmp import kiss_tmp_dir
from kiss.core.processes import bash_version, find_bash
from kiss.core.tool_interrupt import USER_INTERRUPTED_MESSAGE
from kiss.core.tool_verdict import ToolRedirect
from kiss.server.json_printer import JsonPrinter

_LOG_HEADER = "== Progress log written by the task (newest entries) ==\n"
_COPY = "<<'EOF'\n# Plan\n- step 1\nEOF"
# Puts a ``cat`` that copies nothing where a ``PATH`` of ``1`` finds it (with commands that are
# no reason to refuse a copy: the row ``_FAKE_CAT`` alone opens is proven).
_FAKE_CAT = "mkdir {d}/1 && echo '#!/bin/sh' > {d}/1/cat && chmod +x {d}/1/cat && cd {d} && "
# Review FX06d/r3 and its 18 variants: a ``<<-`` body whose text, its tabs stripped, is also in
# the larger body of another here-document, after a ``PATH`` assigned in an expression.
_OVERLAPPING_BODIES = tuple(
    _FAKE_CAT + f"{assigned} && : <<'OUTER' && cat > {{d}}/PROGRESS.md <<-'EOF'\n"
    f"{before}{first}{second}\n{after}OUTER\n{first}{tabs}{second}\nEOF"
    for tabs in ("\t", "\t\t", "\t\t\t")
    for before, after in (("", ""), ("# start\n", "# end\n"), ("# a\n# b\n" * 5, "# z\n" * 7))
    for assigned, first, second in (
        ("((PATH=1))", "# (\\\n", "("),
        ("x=abc && : > {d}/o${{x:PATH=1}}", "# $\\\n", "{{x:1}}"),
    )
)
# Commands that exit 0 in an empty directory ``{d}``, and whether that proves the copy made.
_EXIT_0 = (
    (True, "mkdir -p {d}/new && cat > {d}/new/PROGRESS.md " + _COPY),
    (True, "S={d}; cat >> $S/PROGRESS.md " + _COPY),
    (True, "cat > {d}/PROGRESS.md <<'EOF' && echo ok\n"
           "# Plan: exec no coproc, keep PATH, BASH_CMDS, BASH_ALIASES and POSIXLY_CORRECT\n"
           "a[1]=((2)) [[ ok ]] ${{x:1}} {{fd}}>f b=(c) $[3] LD_PRELOAD ${{PATH:=d}}\nEOF"),
    (True, "cd {d} && tee PROGRESS.md > /dev/null " + _COPY),
    (True, "mkdir {d}/coproc && tee {d}/coproc/PROGRESS.md > /dev/null " + _COPY),  # a name
    # ... a delimiter, a joined line of a body (review FX06c/6: read as commands, as the keyword).
    (True, "tee {d}/PROGRESS.md > /dev/null <<'coproc'\nxoproc\nxoproc\n# Plan\ncoproc"),
    (True, "cat > {d}/PROGRESS.md <<'EOF'\n# Plan: co\\\nproc\nEOF"),
    (True, "x\\\nx() {{ :; }}; x\\\nx && tee {d}/PROGRESS.md > /dev/null <<'EOF'\n# coproc\nEOF"),
    (True, "cat > {d}/PROGRESS.md <<'A' && tee -a {d}/PROGRESS.md 2>/dev/null <<'B'\n"
           "# Plan\nA\nError (exit code 1): retry\nB"),
    # The copy failed or did not run, and the status is another command's or none's.
    (False, "cat > {d}/none/PROGRESS.md " + _COPY + "\necho ok"),
    (False, "cat > {d}/none/PROGRESS.md <<'EOF' || true\n# Plan\nEOF"),
    (False, "cd {d}/none; cat > PROGRESS.md " + _COPY),
    (False, "trap 'exit 0' EXIT; cat > {d}/none/PROGRESS.md " + _COPY),
    (False, "command exec true; cat > {d}/PROGRESS.md " + _COPY),
    (False, "printf '#!/bin/sh\n' > {d}/cat && chmod +x {d}/cat && PATH={d}:$PATH && "
            "cat > {d}/PROGRESS.md " + _COPY),
    # ... or fills the shell's own tables of what ``cat`` runs, as an assignment or not.
    (False, "printf '#!/bin/sh\n' > {d}/no && chmod +x {d}/no && BASH_CMDS=([cat]={d}/no) && "
            "cat > {d}/PROGRESS.md " + _COPY),
    (False, "printf '#!/bin/sh\n' > {d}/no && chmod +x {d}/no && : ${{BASH_CMDS[cat]:={d}/no}}\n"
            "cat > {d}/PROGRESS.md " + _COPY),
    (False, "printf '#!/bin/sh\n' > {d}/1 && chmod +x {d}/1 && cd {d} && ((BASH_CMDS[cat]=1)) && "
            "cat > {d}/PROGRESS.md " + _COPY),
    (False, "POSIXLY_CORRECT=1\nBASH_ALIASES=([cat]=true)\ncat > {d}/PROGRESS.md " + _COPY),
    # ... or assigns ``PATH`` in an expression: its own, an integer's, a variable's value.
    (False, _FAKE_CAT + "((PATH=1)) && cat > {d}/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "declare -i n && n=PATH=1 && cat > {d}/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "t=$(printf 'PA%s=1' TH) && [[ t -eq 1 ]] && "
            "cat > {d}/PROGRESS.md " + _COPY),
    # (review FX06d/7: a reference in a value, and a ``<<-`` body that is not the text's)
    (False, _FAKE_CAT + "x=$(printf 'a%sPA%s=1]' [ TH) && : ${{!x}} && "
            "cat > {d}/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "((PATH=1)) && cat > {d}/PROGRESS.md <<-'EOF'\n# (\\\n\t(\nEOF"),
    (False, "coproc cat > {d}/none/PROGRESS.md " + _COPY),
    (False, "co\\\nproc cat > {d}/none/PROGRESS.md " + _COPY),  # the keyword, on a joined line
    (False, "coproc time cat > {d}/none/PROGRESS.md " + _COPY),  # no keyword after that one
    (False, "coproc true; cat > {d}/PROGRESS.md " + _COPY),  # the keyword of another command
    (False, "cat > {d}/PROGRESS.md <<'EOF'\n" + "coproc\n" * 1001 + "EOF"),  # too many to read
    (False, "! cat > {d}/none/PROGRESS.md " + _COPY),
    (False, "cat > {d}/none/PROGRESS.md <<'EOF' &\n# Plan\nEOF"),
    (False, "cat > {d}/none/PROGRESS.md <<'EOF' | cat\n# Plan\nEOF"),
    (False, "if cat > {d}/none/PROGRESS.md " + _COPY + "\nthen :; fi"),
    (False, "f() {{ cat > {d}/PROGRESS.md " + _COPY + "\n}}"),
    # The file took other text than the here-document's, or the command is no plain copy.
    (False, "cat -n > {d}/PROGRESS.md " + _COPY),
    (False, "cat > {d}/PROGRESS.md <<EOF\n# Plan in $PWD\nEOF"),
    (False, "> {d}/PROGRESS.md " + _COPY),
    (False, "cat " + _COPY),
    (False, "tee {d}/PROGRESS.md {d}/copy.md > /dev/null " + _COPY),
    (False, "tee {d}/PROGRESS.md > {d}/copy.md " + _COPY),
    (False, "cat > {d}/PROGRESS.md 2> {d}/errors " + _COPY),
    (False, "cat > {d}/a.md > {d}/PROGRESS.md " + _COPY),
    # FX06e: a body is credited where the scan read it, never where its text is found again.
    *((False, command) for command in _OVERLAPPING_BODIES),
    (True, _FAKE_CAT + ": <<'OUTER' && cat > {d}/PROGRESS.md <<-'EOF'\n"
           "# start\n# (\\\n(\n# end\nOUTER\n# (\\\n\t(\nEOF"),
    (True, "cd {d}\necho x\ntrue\ncat > {d}/PROGRESS.md <<'true'\necho x\ntrue"),
    (False, _FAKE_CAT + "\n((PATH=1))\ntrue\ncat > {d}/PROGRESS.md <<'true'\n((PATH=1))\ntrue"),
    (False, _FAKE_CAT + "((PATH=1)) && cat > {d}/PROGRESS.md <<'-EOF'\n# ${{x:\\\n-EOF"),
    # ... and only commands known to leave the shell alone may come before a copy: programs,
    # (one that the bash of every machine finds: Git Bash on Windows has no ``python3``)
    (True, "sort <<'PY' && cat > {d}/PROGRESS.md <<'EOF'\nprint(((1)), [[2]], '$[3]')\nPY\n"
           "# Plan\nEOF"),
    (True, "env X=1 true; ./none 2>/dev/null; \"$HOME\"/none 2>/dev/null; > {d}/x; "
           "cat > {d}/PROGRESS.md " + _COPY),
    # ... assignments of names the shell gives no meaning, functions that shadow nothing,
    (True, "X=1 Y+=2 true && for i in 1 2; do echo $i; done; f() {{ echo ${{X}}; }}; f; "
           "cat > {d}/PROGRESS.md " + _COPY),
    # ... and a few builtins, by their operands.
    (True, "set -euo pipefail; set +x; umask 022; export S={d} T; read -r x < /dev/null || :\n"
           "unset x; unset -f f; kill -0 $$; hash -r; unalias -a; wait\n"
           "while true; do break; done; cat > \"$S\"/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "RANDOM=PATH=1; cat > {d}/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "let x=PATH=1; cat > {d}/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "printf -vPATH 1; cat > {d}/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "o=-vPATH; printf $o 1; cat > {d}/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "printf 'x%n' PATH; cat > {d}/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "y='1 PATH=1'; command export x=$y; cat > {d}/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "export PA{{TH=1,}}; cat > {d}/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "read {{PATH,x}} <<< '1 2'; cat > {d}/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "x=abc && : > {d}/o${{x:\"-\"1,PATH=1}} && cat > {d}/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "for PATH in 1; do :; done; cat > {d}/PROGRESS.md " + _COPY),
    (False, ": {{x}}>/dev/null; cat > {d}/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "unset -v x 'a[PATH=1]'; cat > {d}/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "a[PATH=1]=2; cat > {d}/PROGRESS.md " + _COPY),
    (False, "history -s 'exit 0'; fc -s; cat > {d}/none/PROGRESS.md " + _COPY),
    (False, "command -p exec true; cat > {d}/none/PROGRESS.md " + _COPY),
    (True, "echo exec; cat > {d}/PROGRESS.md " + _COPY),  # a word, not the builtin
    (False, "set -n; cat > {d}/none/PROGRESS.md " + _COPY),
    (False, "cat() if true; then :; fi; cat > {d}/PROGRESS.md " + _COPY),
    (False, "function cat {{ :; }}; cat > {d}/PROGRESS.md " + _COPY),
    (False, "echo() {{ :; }}; echo x; cat > {d}/PROGRESS.md " + _COPY),  # no builtin's ``echo``
    (False, _FAKE_CAT + "test -v 'a[PATH=1]'; cat > {d}/PROGRESS.md " + _COPY),
    (False, _FAKE_CAT + "eval '(''(PATH=1))'; cat > {d}/PROGRESS.md " + _COPY),
)
# The bash a command of ``_EXIT_0`` exits 0 with, by a text only that command has (each measured
# with bash 3.2 to 5.3): the ``coproc`` keyword and an ``fc -s`` that finds the ``history -s``
# line came with 4.0, quotes removed from the offset of a substring with 5.2.  An older bash
# (3.2 on macOS) fails these commands; the digest is then given the record a bash that runs
# them leaves (an empty result), so that the shape of the command still decides its answer.
_SINCE = (
    ("coproc cat", (4, 0)), ("co\\\nproc cat", (4, 0)), ("coproc time", (4, 0)),
    ("fc -s", (4, 0)), ('${x:"-"1', (5, 2)),
)


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


def _bash_call(command: str, result: str, **fields: Any) -> list[dict[str, Any]]:
    """Events of a ``Bash`` call of *command* answered by *result* (*fields*: the result's)."""
    return [
        {"type": "tool_call", "name": "Bash", "command": command},
        {"type": "tool_result", "content": result, "tool_name": "Bash", **fields},
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


def test_a_tilde_path_and_its_absolute_spelling_name_one_log(tmp_path: Path) -> None:
    """Audit 2026-11-21 S15-F2: the file tools expand ``~``, the replay did not.

    A ``Write`` of ``~/PROGRESS.md`` and an ``Edit`` of that file by its absolute
    path were two logs, and the newest held the edit's new text alone.
    """
    tilde, absolute = Path("~/PROGRESS.md"), Path.home() / "PROGRESS.md"
    plan = "# Plan\n- keep this\n- status: pending\n"
    copied = _bash_call(f"cat > ~/PROGRESS.md <<'EOF'\n{plan}EOF", "")  # as the shell names it
    done = {"old_string": "pending", "new_string": "done"}
    for written, edited in (
        (_write_call(tilde, plan), absolute),
        (_write_call(absolute, plan), tilde),
        (copied, absolute),
    ):
        task_id = _task(tmp_path, [*written, *_edit_call(edited, done, "ok")], time.time() - 60)
        assert _log_of(task_id) == "# Plan\n- keep this\n- status: done"


def test_a_here_document_append_replays_only_the_text_it_appended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit 2026-11-10 S12-F1: an append is logged as a ``Write`` of the whole file.

    Under an agent the Bash tool applies ``cat >> log <<EOF`` as
    ``Write(the file's text + the body)``, so the event's content holds
    what other tasks left in the shared log: only the body is replayed.
    A here-document that overwrites, and a later plain ``Write``, still
    put their whole content.
    """
    now = time.time()
    log = Path(tempfile.mkdtemp(dir=kiss_tmp_dir())) / "PROGRESS.md"  # a scratch dir's log
    _write(log, "# another task\n- OTHER-TASK-ENTRY\n")
    tools = UsefulTools(work_dir=str(tmp_path))
    tools.redirect = True  # as under an agent
    # as_posix: bash eats the backslashes of a Windows path.
    command = f"cat >> {log.as_posix()} <<'EOF'\n- MY-ENTRY\nEOF"
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
         "command": f"cat > {log.as_posix()} <<'EOF'\n# mine\nEOF"},
        *_write_call(log, "# mine\n"),
        {"type": "tool_result", "content": "ok", "tool_name": "Bash"},
    ]
    assert _log_of(_task(tmp_path, [*appended, *overwritten], now - 60)) == "# mine"
    rewritten = [*appended, *_write_call(log, "# all of it\n- new\n")]
    assert _log_of(_task(tmp_path, rewritten, now - 60)) == "# all of it\n- new"
    # The same append after a ``cd``, digested where a ``CDPATH`` is exported (audit 2026-11-21
    # LG/3): the tool made the ``Write`` reading that ``cd`` as written, and so its record is read.
    _write(tmp_path / "d" / "PROGRESS.md", "# another task\n- OTHER-TASK-ENTRY\n")
    tools.Read(str(tmp_path / "d" / "PROGRESS.md"))  # shown: the append is applied as a Write
    command = "cd d && cat >> PROGRESS.md <<'EOF'\n- MY-ENTRY\nEOF"
    with pytest.raises(ToolRedirect) as redirect:
        tools.Bash(command=command, description="log")
    assert redirect.value.arguments["content"] == content
    entered = [
        {"type": "tool_call", "name": "Bash", "command": command},
        *_write_call(redirect.value.arguments["file_path"], content),
        {"type": "tool_result", "content": redirect.value.note, "tool_name": "Bash"},
    ]
    monkeypatch.setenv("CDPATH", "elsewhere")
    assert _log_of(_task(tmp_path, entered, now - 60)) == "- MY-ENTRY"


def test_a_here_document_that_creates_the_log_is_replayed_from_its_bash_call(
    tmp_path: Path,
) -> None:
    """Audit 2026-11-21 S15-F1: the shell runs a copy that creates its file as written.

    No ``Write`` is logged for it, so the plan a task opened its log with was
    missing from the digest and a later ``Edit`` was replayed onto nothing.  The
    ``Bash`` call is replayed unless its result is one of the tools' reports of
    a command that did not exit 0 (an error, a refusal, a Stop, a job started or
    left running) or the call polled its command or handed it to a job.  The
    ``Write`` a copy was applied as replaces it.  What a bare ``tee`` prints is
    the command's output and no report, however it opens, was cut or had its
    line ends changed (reviews of S15-F1): a log that opens with "Error" was
    dropped as a failed copy.
    """
    now = time.time()
    scratch = Path(tempfile.mkdtemp(dir=kiss_tmp_dir()))
    log = scratch / "PROGRESS.md"
    tools = UsefulTools(work_dir=str(tmp_path))
    tools.redirect = True  # as under an agent
    create = f"cat > {log.as_posix()} <<'EOF'\n# Plan\n- step 1\nEOF"
    created = _bash_call(create, tools.Bash(command=create, description="log"))
    assert created[1]["content"] == ""
    assert log.read_text(encoding="utf-8") == "# Plan\n- step 1\n"
    edited = _edit_call(log, {"old_string": "- step 1\n", "new_string": "- step 1: done\n"}, "ok")
    assert _log_of(_task(tmp_path, [*created, *edited], now - 60)) == "# Plan\n- step 1: done"

    # An append: as the shell ran it, and as the Write it is applied as once the
    # file exists (the call that follows the Bash call): the body is added once.
    append = f"cat >> {log.as_posix()} <<'EOF'\n- step 2\nEOF"
    with pytest.raises(ToolRedirect) as redirect:
        tools.Bash(command=append, description="log")
    applied = [
        {"type": "tool_call", "name": "Bash", "command": append},
        *_write_call(log, redirect.value.arguments["content"]),
        {"type": "tool_result", "content": redirect.value.note, "tool_name": "Bash"},
    ]
    for appended in (_bash_call(append, ""), applied):
        task_id = _task(tmp_path, [*created, *appended], now - 60)
        assert _log_of(task_id) == "# Plan\n- step 1\n- step 2"
    # In a directory the command enters, named from the task's work dir.
    entered = _bash_call("cd tmp && tee -a PROGRESS.md <<'EOF' > /dev/null\n- step 2\nEOF", "")
    mine = _write_call("tmp/PROGRESS.md", "# Plan\n")
    assert _log_of(_task(tmp_path, [*mine, *entered], now - 60)) == "# Plan\n- step 2"
    # ... and so is the Write it is applied as: the body alone is the task's.
    shared = _write_call("tmp/PROGRESS.md", "# another task's\n- step 2\n")
    redirected = [*mine, entered[0], *shared, entered[1]]
    assert _log_of(_task(tmp_path, redirected, now - 60)) == "# Plan\n- step 2"
    # ... which a task without one cannot name: only a path written in full is a log's.
    nowhere = [*created, *entered, *_bash_call("cat > PROGRESS.md <<'EOF'\n# lost\nEOF", "")]
    assert task_digest.progress_log_tail({"events": nowhere}) == "# Plan\n- step 1\n"
    # Nor is ``$PWD`` after a ``cd`` that may have failed (review FX06e r2: the shell then wrote
    # in the directory it was in), or a variable that a here-document body the shell expands
    # assigns before the copy's words are read (r3).  The host's tool refuses to run either.
    for command in (
        'cd tmp; tee -a "$PWD"/PROGRESS.md <<\'EOF\' > /dev/null\n- lost\nEOF',
        f"S=; : <<X; tee -a {tmp_path.as_posix()}/tmp/$S/PROGRESS.md <<'EOF' > /dev/null\n"
        "${S:=a}\nX\n- lost\nEOF",
    ):
        stayed = _bash_call(command, "")
        assert _log_of(_task(tmp_path, [*mine, *stayed], now - 60)) == "# Plan", command
    # A bare ``tee`` answers with the body it echoes: output, and no report of the tool's,
    # whole, cut to the call's budget (to 100: without a counted marker) or with its CRLF
    # line ends changed, and whatever it opens with or quotes.
    quoted = "Error recovery plan\n\n... [truncated 7 chars] ...\n\n"
    plan = quoted + "".join(f"- step {i} of the plan\n" for i in range(40))
    bodies = [(plan, len(plan)), (plan, 600), (plan, 100), ("Error plan\r\n- keep\r\n", 600)]
    for opening in ("REFUSED", "Started background job", "Still running after", "[exit code: 1]"):
        bodies.append((f"{opening}\n- keep\n", 600))
    for body, budget in bodies:
        shutil.rmtree(scratch / "tee", ignore_errors=True)
        (scratch / "tee").mkdir()
        echo = f"tee {scratch.as_posix()}/tee/PROGRESS.md <<'EOF'\n{body}EOF"
        result = tools.Bash(command=echo, description="log", max_output_chars=budget)
        assert result.startswith(body.splitlines()[0]) and (result == plan) == (budget == len(plan))
        assert _log_of(_task(tmp_path, _bash_call(echo, result), now - 60)) == body.rstrip()

    # The container's failure line inside the text closes nothing: the copy was made.
    inner = "# Plan\n[exit code: 1]\n[note: the sleep of 1s]\n- keep going\n"
    (scratch / "inner").mkdir()
    echo = f"tee {scratch.as_posix()}/inner/PROGRESS.md <<'EOF'\n{inner}EOF"
    assert tools.Bash(command=echo, description="log") == inner
    assert _log_of(_task(tmp_path, _bash_call(echo, inner), now - 60)) == inner.rstrip()

    # Results that do not say the copy was made: the log stays as it was.
    lost = f"cat > {log.as_posix()} <<'EOF'\n# lost\nEOF"
    _write(tmp_path / "PROGRESS.md", "# another task's, not shown to this one\n")
    unshown = f"cat > {tmp_path.as_posix()}/PROGRESS.md <<'EOF'\n# lost\nEOF"
    refused = tools.Bash(command=unshown, description="log")
    assert refused.startswith("REFUSED: the command did not run")
    missing = f"cat > {scratch.as_posix()}/none/PROGRESS.md <<'EOF'\n# lost\nEOF"
    failed = tools.Bash(command=missing, description="log")
    assert failed.startswith("Error (exit code 1)")
    more = f"cat > {scratch.as_posix()}/PROGRESS_LOG.md <<'EOF'\n# lost\nEOF\necho ok"
    assert tools.Bash(command=more, description="log") == "ok\n"
    # A call that polls its command or hands it to a job says nothing of the command's end:
    # a container's record of one is blank.
    unsettled = [_bash_call(lost, "") for _ in range(3)]
    for call, extras in zip(
        unsettled, ({"until": "x"}, {"background": "True"}, {"on_timeout": "background"}),
    ):
        call[0]["extras"] = extras
    unmade = f"tee {scratch.as_posix()}/none/PROGRESS.md <<'EOF'\n{plan}EOF"
    # A failed copy of the very text its failure prints: no echo, ``cat`` prints nothing.
    quoting = missing.replace("# lost\n", failed)
    assert tools.Bash(command=quoting, description="log") == failed
    # A failed ``tee`` of lines like the one its result opens with, cut: longer by that line.
    lines = "Error (exit code 1):\n" * 60
    repeated = f"tee {scratch.as_posix()}/none/PROGRESS.md 2>/dev/null <<'EOF'\n{lines}EOF"
    for unproven in (
        _bash_call(unshown, refused),
        _bash_call(missing, failed),
        _bash_call(quoting, failed),
        _bash_call(unmade, tools.Bash(command=unmade, description="log")),
        _bash_call(repeated, tools.Bash(command=repeated, description="log", max_output_chars=600)),
        _bash_call(more, "ok\n"),
        _bash_call(lost, USER_INTERRUPTED_MESSAGE, interrupted=True),
        _bash_call(lost, "Started background job j1 (pid 1).\n"),
        _bash_call(lost, "Still running after 120s; left running as background job j1 (pid 1)."),
        *unsettled,
        # As the container tool reports a failure: after the output, before its notes.
        _bash_call(lost, "cat: write error\n[exit code: 1]"),
        _bash_call(lost, "[exit code: 1]\n[note: the sleep of 60 s ran as written]"),
        # The note of a sleep spent on a job: on a line of its own, and glued on in older records.
        _bash_call(lost, "[exit code: 1]\n[until 'x': not matched]\nWaited on job j1 (running).\n"),
        _bash_call(lost, "[exit code: 1]Waited on job j1 for up to 60 s instead of sleeping.\n"),
        _bash_call(lost.replace("PROGRESS.md", "notes.md"), ""),  # no progress log
        # Shapes the host tool refuses to run: a record of one names no file, or hides commands.
        _bash_call("X=$(pwd); cat > $X/PROGRESS.md <<'EOF'\n# lost\nEOF", ""),
        _bash_call(f"HOME={scratch.as_posix()}; cat > ~/PROGRESS.md <<'EOF'\n# lost\nEOF", ""),
        _bash_call("{ " * 60 + ":; " + "}; " * 60 + lost, ""),
    ):
        task_id = _task(tmp_path, [*created, *unproven], now - 60)
        assert _log_of(task_id) == "# Plan\n- step 1"
    shutil.rmtree(scratch)


def test_a_copy_after_a_cd_is_replayed_only_where_the_inherited_cdpath_leaves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit 2026-11-21 LG/3: a tool's shell inherits the ``CDPATH`` of the process that starts it.

    With ``sub`` there bash takes ``cd d`` to ``sub/d``.  The digest read that ``cd``
    without it: a copy after it, run as written (by a tool whose guard does not refuse it:
    a container's, the host's before it read ``CDPATH`` too), was replayed into
    ``d/PROGRESS.md``, and the log shown after the task's ``Write`` of that file was one no
    file holds.  Bash is the oracle: each command exits 0, and the log is the text of
    ``d/PROGRESS.md``, with the entry only where bash put it there.
    """
    monkeypatch.setenv("CDPATH", "sub")
    entry = "<<'EOF'\n- step `2`\nEOF"
    for number, (command, appended) in enumerate((
        ("cd d && cat >> PROGRESS.md " + entry, False),  # into sub/d/PROGRESS.md
        ("cd d && tee -a PROGRESS.md > /dev/null " + entry, False),
        ("cd d && cd .. && cat >> d/PROGRESS.md " + entry, False),  # into sub/d/PROGRESS.md
        # What bash does not look up in ``CDPATH`` is where the text says ...
        ("cd ./d && cat >> PROGRESS.md " + entry, True),
        ("CDPATH= cd d && cat >> PROGRESS.md " + entry, True),
        # ... as is a file named in full, wherever the ``cd`` led.
        ("cd d && cat >> {d}/d/PROGRESS.md " + entry, True),
    )):
        work = tmp_path / str(number)
        (work / "sub" / "d").mkdir(parents=True)
        _write(work / "d" / "PROGRESS.md", "# Plan\n")
        command = command.format(d=work.as_posix())
        ran = subprocess.run(
            [find_bash() or "bash", "-c", command], cwd=work, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, check=False, timeout=30)
        assert ran.returncode == 0, (command, ran.stderr)
        held = (work / "d" / "PROGRESS.md").read_text(encoding="utf-8")
        assert held == "# Plan\n" + "- step `2`\n" * appended, command
        events = [*_write_call("d/PROGRESS.md", "# Plan\n"), *_bash_call(command, ran.stdout)]
        shown = task_digest.progress_log_tail({"work_dir": str(work), "events": events})
        assert shown == held, command


def test_a_copy_inside_a_longer_command_is_replayed_when_exit_status_0_proves_it(
    tmp_path: Path,
) -> None:
    """Reviews of S15-F1: a copy after ``mkdir -p D &&``, by a variable or before ``&& echo``.

    Only a command that was the copy and nothing else was replayed.  The commands
    of :data:`_EXIT_0` are run by the real tool and recorded as a daemon records
    them (the output streamed, the result the tool's own words).  Each exits 0, so
    the shape decides: a copy that status vouches for is the log, to the letter
    of the file; one that failed, did not run or wrote other text is not.

    A command that assigns bash's own tables of what a name runs (``BASH_CMDS``,
    ``BASH_ALIASES``) is refused by today's tool, unrun; a daemon before that guard ran
    it, and its record (bash's own run of it here) is digested like the others.
    """
    printer = JsonPrinter()
    printer.adopt_thread_binding("digest")
    stream = functools.partial(printer.print, type="bash_stream")
    tools = UsefulTools(work_dir=str(tmp_path), stream_callback=stream)
    for number, (proven, template) in enumerate(_EXIT_0):
        folder = tmp_path / str(number)
        folder.mkdir()
        command = template.format(d=folder.as_posix())
        call = {"command": command, "description": "log"}
        printer.start_recording()
        printer.print("Bash", type="tool_call", tool_input=call)
        result = tools.Bash(command=command, description="log")
        if result == DESTRUCTIVE_VERDICT:
            assert not proven and not list(folder.iterdir()), command
            result = subprocess.run(
                [find_bash() or "bash", "-c", command], cwd=tmp_path, capture_output=True,
                text=True, check=True,
            ).stdout
        if any(bash_version() < since for mark, since in _SINCE if mark in command):
            assert not proven and result, command
            result = ""
        printer.print(result, type="tool_result", tool_name="Bash", is_error=False)
        events = printer.stop_recording()
        assert events[-1]["content"] == "", command
        held = "".join(path.read_text(encoding="utf-8") for path in folder.rglob("PROGRESS.md"))
        shown = task_digest.progress_log_tail({"work_dir": str(tmp_path), "events": events})
        assert shown == (held if proven else "") and bool(shown) == proven, command


def test_a_copy_the_bash_tool_made_as_a_write_itself_is_replayed_from_its_result(
    tmp_path: Path,
) -> None:
    """A run without ``Write`` / ``Edit`` tools: the ``Bash`` tool applies the copy and says so.

    No call follows the ``Bash`` call then.  Its result opens with the ``Write``
    that was made, whatever the rest of the command did after it, or says that
    the ``Write`` was not applied.  A command can print those words itself and
    exit 0 (review FX06c/1), so such a result proves the tool's copy only when
    an exit status 0 would prove it too: not before more commands, whose status
    is further down the result.
    """
    scratch = Path(tempfile.mkdtemp(dir=kiss_tmp_dir()))
    log, notes = (scratch / "PROGRESS.md").as_posix(), (scratch / "notes.md").as_posix()
    (scratch / "folder" / "PROGRESS.md").mkdir(parents=True)
    tools = UsefulTools(work_dir=str(tmp_path))
    events: list[dict[str, Any]] = []
    for command, opening in (
        (f"cat > {log} <<'EOF'\n# Plan\nEOF", ""),  # creates its file: the shell runs it
        (f"cat >> {log} <<'EOF'\n- step 2\nEOF", "Appended through Write("),
        (f"cat >> {log} <<'EOF'\n- step 3\nEOF\nfalse", "Appended through Write("),
        (
            f"cat 2>/dev/null > {scratch.as_posix()}/none/PROGRESS.md <<'EOF'\n# lost\nEOF\n"
            "printf 'Applied as Write(file_path=x): ok\\n'",
            "Applied as Write(file_path=x): ok",
        ),
        (f"cat > {scratch.as_posix()}/folder/PROGRESS.md <<'EOF'\n# lost\nEOF", "Not applied: "),
        (f"cat > {notes} <<'EOF'\nnotes\nEOF", ""),
        (f"cat > {notes} <<'EOF'\n# no log\nEOF", "Applied as Write("),
    ):
        result = tools.Bash(command=command, description="log")
        assert result.startswith(opening) and bool(result) == bool(opening), result
        events += _bash_call(command, result)
    # As the tool reports a rewrite it made as an ``Edit`` and the rest it ran after it.
    rest = f"cat > {scratch.as_posix()}/none/PROGRESS.md <<'EOF'\n# lost\nEOF"
    edited = (
        f'Applied as Edit(file_path="{notes}", old_string="no", new_string="a"): Successfully '
        "replaced 1 occurrence(s)\nThen made Bash(command=...), whose result follows:\n"
        "Error (exit code 1):\n"
    )
    events += _bash_call(f"sed -i 's/no/a/' {notes} && {rest}", edited)
    # As the container tool reports a failed ``tee`` of a text that opens with those words
    # (review FX06c/1 of round 2): the status that closes the result is read first.
    echo = 'Applied as Write(file_path="x"): ok\n'
    missed = f"tee {scratch.as_posix()}/none/PROGRESS.md 2>/dev/null <<'EOF'\n{echo}EOF"
    assert tools.Bash(command=missed, description="log") == "Error (exit code 1):\n" + echo
    events += _bash_call(missed, _with_exit_code(echo, 1))
    assert task_digest.progress_log_tail({"events": events}) == "# Plan\n- step 2\n"
    shutil.rmtree(scratch)
