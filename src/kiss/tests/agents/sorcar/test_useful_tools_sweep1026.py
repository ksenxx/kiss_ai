# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Sweep 2026-10-26 (sorcar-tools F3/F5): ``sleep_guard`` and the batch ``Edit`` report.

F3: ``sleep_guard`` matched a ``sleep`` quoted inside a heredoc body (a
Markdown file that *mentions* the idiom), missed the second of ``sleep 5;
sleep 60`` because the first match consumed the ``;`` the second needed
as its prefix, and missed a ``sleep`` on its own line.  A ``sleep N &``
stays refused: the backgrounded child inherits the call's output pipe,
so ``_bash_streaming`` waits for it anyway (round 2, R3-sorcar-b P2).
A heredoc body is only skipped when the shell will not execute it: a
quoted tag, or no ``$(``/backtick command substitution in the body.

F5: the per-edit reports of ``Edit(edits=...)`` were rendered right after
each edit, so a later edit higher in the file shifted the line numbers of
the earlier reports.  Every report is now rendered from the final content.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import UsefulTools, sleep_guard


@pytest.mark.parametrize(
    "command",
    [
        "cat > notes.md <<'EOF'\nRun `(sleep 60; curl http://x) &` to delay it.\nEOF",
        "cat > notes.md <<\"EOF\"\nx=$(sleep 60)\nEOF",
        "cat <<EOF > notes.md\nsleep 120; echo done\nEOF\necho written",
        "cat <<-EOF\n\tsleep 120 &\n\tEOF",
        "echo sleep 100",
        "sleep 30",
    ],
)
def test_sleep_guard_allows_quoted_sleeps(command: str) -> None:
    """A sleep in a heredoc body the shell only copies is not a polling turn."""
    assert sleep_guard(command) is None, command


@pytest.mark.parametrize(
    "command",
    [
        "sleep 60",
        "make && sleep 60",
        "sleep 60 && echo",
        "(sleep 60) && echo",
        "(sleep 60; curl http://x); echo",
        "sleep 5; sleep 60",
        "(sleep 60 | cat) &&\necho",
        "cat <<EOF > notes.md\nhello\nEOF\nsleep 60",
        "sleep 30.5",
        "sleep 60 &&",
        "echo hi\nsleep 60",
        "sleep 60\necho",
        "x=$(sleep 60)",
        "x=`sleep 60`",
        # Backgrounded: the child keeps the output pipe, the call blocks anyway.
        "sleep 60 &",
        "(sleep 60; curl http://x) &",
        "(sleep 60) & wait",
        "sleep 60 & tail -f log",
        "sleep 60 | cat &",
        "sleep 60 && echo done &",
        'sleep 60; echo ")" &',
        # An unquoted heredoc body runs its command substitutions.
        "cat <<EOF\n$(sleep 60)\nEOF",
        "cat <<EOF\nnow `sleep 60` later\nEOF",
        "cat <<EOF\nmention sleep 60\n$(sleep 60)\nEOF",
    ],
)
def test_sleep_guard_refuses_every_sleep_the_call_waits_for(command: str) -> None:
    """A sleep the shell runs is refused, foreground or backgrounded, wherever it sits."""
    refusal = sleep_guard(command)
    assert refusal is not None and refusal.startswith("REFUSED:"), command


def test_bash_refuses_a_backgrounded_sleep_without_running_it(tmp_path: Path) -> None:
    """The real ``Bash`` path refuses ``sleep N &`` up front instead of blocking on it."""
    tools = UsefulTools(work_dir=str(tmp_path))
    started = time.monotonic()
    out = tools.Bash("sleep 60 & touch ran", "poll", timeout_seconds=5)
    assert time.monotonic() - started < 2
    assert out.startswith("REFUSED: the Bash call did NOT run: a foreground sleep of 60s")
    assert not (tmp_path / "ran").exists()


def _reports(result: str) -> list[str]:
    """The per-edit reports of a batch ``Edit`` result, without the summary line."""
    reports = result.split("\n\n")
    reports[0] = reports[0].split("\n", 1)[1]
    return reports


def test_batch_edit_reports_use_final_line_numbers(tmp_path: Path) -> None:
    """A later edit higher in the file does not stale the earlier report's lines."""
    target = tmp_path / "doc.txt"
    target.write_text("".join(f"line {n}\n" for n in range(1, 21)), encoding="utf-8")
    tools = UsefulTools(work_dir=str(tmp_path))
    tools.Read(str(target))
    result = tools.Edit(
        str(target),
        edits=json.dumps([
            {"old_string": "line 15\n", "new_string": "LINE 15\nLINE 15b\n"},
            {"old_string": "line 3\n", "new_string": "l3a\nl3b\nl3c\n"},
            {"old_string": "line 1\n", "new_string": ""},
            {"old_string": "line 20\n", "new_string": "LAST\n"},
        ]),
    )
    assert result.startswith("Successfully replaced 4 occurrence(s)"), result
    final = target.read_text(encoding="utf-8").splitlines()
    reports = _reports(result)
    assert len(reports) == 4
    # Edit 1 sat at lines 15-16 right after it; edits 2 (+2) and 3 (-1) moved it to 16-17.
    assert reports[0].startswith("edit 1 (1 occurrence(s)); lines 13-20 after the edit:\n")
    assert f"{16:5d} | LINE 15\n{17:5d} | LINE 15b" in reports[0]
    assert final[15] == "LINE 15" and final[16] == "LINE 15b"
    # Edit 2 was at lines 3-5 right after it; the deletion above moved it to 2-4.
    assert reports[1].startswith("edit 2 (1 occurrence(s)); lines 1-7 after the edit:\n")
    assert f"{2:5d} | l3a\n{3:5d} | l3b\n{4:5d} | l3c" in reports[1]
    assert final[1:4] == ["l3a", "l3b", "l3c"]
    assert reports[2].startswith("edit 3 (1 occurrence(s)); lines 1-4 after the edit:\n")
    assert reports[3].startswith("edit 4 (1 occurrence(s)); lines 19-25 after the edit:\n")
    assert f"{22:5d} | LAST" in reports[3]
    assert final[21] == "LAST" and len(final) == 22


def test_batch_edit_swallowed_by_a_later_edit_reports_the_replacement(tmp_path: Path) -> None:
    """An earlier edit inside a later replacement's region reports that region, not the tail.

    Reviewer's example (R3-sorcar-b P2): subtracting the later edit's
    full length delta made the first anchor ``7 - 9 = -2``, which
    ``line_of`` read as an end-relative offset: edit 1 reported lines
    28-34 of a 33-line file although it touched line 2.
    """
    target = tmp_path / "doc.txt"
    target.write_text(
        "prefix\nold\nsuffix\n" + "".join(f"{n}\n" for n in range(1, 31)), encoding="utf-8",
    )
    tools = UsefulTools(work_dir=str(tmp_path))
    tools.Read(str(target))
    result = tools.Edit(
        str(target),
        edits=json.dumps([
            {"old_string": "old", "new_string": "NEW"},
            {"old_string": "prefix\nNEW\nsuffix\n", "new_string": "REPLACED\n"},
        ]),
    )
    assert result.startswith("Successfully replaced 2 occurrence(s)"), result
    reports = _reports(result)
    assert reports[0].startswith("edit 1 (1 occurrence(s)); lines 1-4 after the edit:\n")
    assert f"{1:5d} | REPLACED\n{2:5d} | 1\n" in reports[0]
    assert reports[1].startswith("edit 2 (1 occurrence(s)); lines 1-4 after the edit:\n")
    assert target.read_text(encoding="utf-8").startswith("REPLACED\n1\n2\n")
    # Three occurrences of a later replace_all: one above edit 1, one inside its
    # text (a partial overlap keeps edit 1's anchor), one below.
    target.write_text("ab\nX\nab\nX\nab\n", encoding="utf-8")
    tools.Read(str(target))
    result = tools.Edit(
        str(target),
        edits=json.dumps([
            {"old_string": "X\nab\nX", "new_string": "Y\nab\nY"},
            {"old_string": "ab\n", "new_string": "c\nd\ne\n", "replace_all": True},
        ]),
    )
    assert result.startswith("Successfully replaced 4 occurrence(s)"), result
    reports = _reports(result)
    assert target.read_text(encoding="utf-8") == "c\nd\ne\nY\nc\nd\ne\nY\nc\nd\ne\n"
    # Edit 1 began at the old line 2 ("X"), now line 4 after the first occurrence grew.
    assert reports[0].startswith("edit 1 (1 occurrence(s)); lines 1-9 after the edit:\n")
    assert f"{4:5d} | Y\n" in reports[0]


def test_batch_edit_replace_all_shifts_reports_once_per_occurrence_above(
    tmp_path: Path,
) -> None:
    """A ``replace_all`` edit shifts an earlier report by the occurrences above it only."""
    target = tmp_path / "doc.txt"
    target.write_text("x\nmid\nx\nTARGET\nx\n", encoding="utf-8")
    tools = UsefulTools(work_dir=str(tmp_path))
    tools.Read(str(target))
    result = tools.Edit(
        str(target),
        edits=json.dumps([
            {"old_string": "TARGET\n", "new_string": "hit\n"},
            {"old_string": "x\n", "new_string": "y\ny\n", "replace_all": True},
        ]),
    )
    assert result.startswith("Successfully replaced 4 occurrence(s)"), result
    reports = _reports(result)
    # TARGET was line 4; the two occurrences above it each added a line: now line 6.
    assert reports[0].startswith("edit 1 (1 occurrence(s)); lines 3-9 after the edit:\n")
    assert f"{6:5d} | hit" in reports[0]
    assert reports[1].startswith("edit 2 (3 occurrence(s)); lines 1-5 after the edit:\n")
    assert target.read_text(encoding="utf-8") == "y\ny\nmid\ny\ny\nhit\ny\ny\n"
