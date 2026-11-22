# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of ``src/kiss/scripts/sync_kiss_notes.py``.

The merge rules run on real note texts; the two-way sync runs the real
script against a sandbox "remote": a fake ``ssh`` on ``PATH`` executes the
remote half locally with ``HOME`` pointed at another directory, as the
sync-memory tests do.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import IO

import pytest

from kiss.agents.seas.autorouter.autorouter_sea import log_decision
from kiss.core.file_lock import lock_exclusive, unlock
from kiss.core.processes import kill_process_tree, popen_process_group
from kiss.scripts import sync_kiss_notes as notes
from kiss.tests.conftest import is_root, posix_only

SCRIPT = Path(notes.__file__).resolve()

pytestmark = posix_only("drives the script with a bash ssh stand-in")

renames_over_a_note = pytest.mark.skipif(
    sys.platform != "linux",
    reason="off Linux Python cannot tell whether a new file can stand in for a note that is "
    "there (no os.listxattr): a merge rewrites the note in place instead of renaming over it",
)
"""Marks a test of what only the rename gives: ``merge_into_file`` says where it is to be had."""

FAKE_SSH = """#!/bin/bash
while [ $# -gt 0 ]; do
    case "$1" in
        -o|-p|-i|-l|-F|-c) shift 2 ;;
        -*) shift ;;
        *) break ;;
    esac
done
shift                       # drop the user@host argument
export HOME="$REMOTE_HOME"
unset KISS_HOME             # a real ssh session does not inherit it either
exec bash -c "$*"
"""

DEAD_SSH = "#!/bin/bash\necho 'ssh: connect to host fakehost: Connection refused' >&2\nexit 255\n"

HEADER = (
    "# Model routing decisions\n\n"
    "| time (UTC) | task_id | unit | tier | model | reason | outcome |\n"
    "|---|---|---|---|---|---|---|\n"
)
ROW_A = "| 2026-09-27 10:00 | t1 | grep | small | m-small | tier small 0.9 | pending |\n"
ROW_B = "| 2026-09-27 10:02 | t2 | fix bug | medium | m-med | tier medium 0.7 | pending |\n"
ROW_C = "| 2026-09-27 10:05 | t1 | grep | small | m-small | tier small 0.9 | passed |\n"

EVIDENCE = """_Observed in the task history, refreshed 2026-09-27 by /rsi7d._

Window: 2026-09-20 to 2026-09-27 UTC, 2,684 tasks.

| model | tasks | fail | $/step |
|---|---|---|---|
| big-model | 1,809 | 96 | 0.1015 |
| mid-model | 396 | 10 | 0.0378 |
| small-model | 22 | 4 | 0.0822 |
| others (7 models) | 26 | | insufficient data |

Others: tiny-a 3, tiny-b 1.

- mid-model is the best-measured frontier pick: 396 tasks, 10 failures.
- big-model carries the most work (1,809 tasks) and the most failures (96);
  it costs 2.7x mid-model per step.
- small-model is superseded: 4 failures in 22 tasks.
"""


def test_ledger_rows_are_united_in_time_order() -> None:
    """Distinct rows of both copies survive once, ordered by their time cell."""
    ours = HEADER + ROW_A + ROW_C
    theirs = HEADER + ROW_B + ROW_A
    merged = notes.merge_ledger(ours, theirs)
    assert merged == HEADER + ROW_A + ROW_B + ROW_C
    assert notes.merge_ledger(theirs, ours) == merged
    assert notes.merge_ledger(merged, ours) == merged
    assert notes.merge_ledger(ours, "") == ours and notes.merge_ledger("", theirs) == theirs
    assert notes.merge_ledger("", "  \n") == ""


def test_ledger_prose_and_header_survive_odd_copies() -> None:
    """Prose outside the table is united, ours first; a copy without a header borrows one."""
    ours = HEADER + ROW_A + "\nNote: rows before 10:00 were lost.\n"
    theirs = "Hand-made ledger\n" + ROW_B
    merged = notes.merge_ledger(ours, theirs)
    assert merged == (
        "# Model routing decisions\nNote: rows before 10:00 were lost.\nHand-made ledger\n\n"
        + HEADER.split("\n\n", 1)[1]
        + ROW_A
        + ROW_B
    )
    assert notes.merge_ledger(theirs, "Hand-made ledger\n" + ROW_A) == (
        "Hand-made ledger\n\n" + ROW_A + ROW_B
    )
    assert notes.merge_ledger("Note A\n", "Note B\n\nNote A\n") == "Note A\nNote B\n"


def test_evidence_with_the_later_stamp_wins_outright() -> None:
    """A newer refresh replaces an older one; an unstamped copy loses to a stamped one."""
    newer = EVIDENCE.replace("2026-09-27", "2026-10-04").replace("| 396 |", "| 500 |")
    assert notes.merge_evidence(EVIDENCE, newer) == newer
    assert notes.merge_evidence(newer, EVIDENCE) == newer
    unstamped = "| model | tasks |\n|---|---|\n| x | 1 |\n"
    assert notes.merge_evidence(EVIDENCE, unstamped) == EVIDENCE
    assert notes.merge_evidence(unstamped, EVIDENCE) == EVIDENCE
    assert notes.merge_evidence(unstamped, "| model | tasks |\n|---|---|\n| y | 2 |\n") == (
        "| model | tasks |\n|---|---|\n| y | 2 |\n| x | 1 |\n"
    )  # both unstamped: combined like two same-day copies
    assert notes.merge_evidence(EVIDENCE, "") == EVIDENCE
    assert notes.merge_evidence("", EVIDENCE) == EVIDENCE
    assert notes.merge_evidence(EVIDENCE, EVIDENCE + "\n") == EVIDENCE  # merged like any pair
    assert notes.merge_evidence(EVIDENCE + "\n", EVIDENCE) == EVIDENCE
    assert notes.merge_evidence(EVIDENCE + "\n", EVIDENCE + "\n") == EVIDENCE + "\n"  # identical
    assert notes.stamp_of(EVIDENCE) == "2026-09-27" and notes.stamp_of(unstamped) == ""


def test_same_day_evidence_is_combined_row_by_row_and_bullet_by_bullet() -> None:
    """Same stamp: rows united by model (larger task count wins), bullets united, prose from
    the base -- the copy with more rows -- and the same result from either side."""
    theirs = (
        EVIDENCE.replace("| mid-model | 396 | 10 | 0.0378 |", "| mid-model | 410 | 11 | 0.0380 |")
        .replace("| big-model | 1,809 | 96 | 0.1015 |", "| big-model | 1,700 | 90 | 0.1000 |")
        .replace(
            "| small-model | 22 | 4 | 0.0822 |",
            "| small-model | 22 | 4 | 0.0822 |\n| new-model | 30 | 0 | 0.0100 |",
        )
        .replace("2,684 tasks", "2,700 tasks")
        + "* new-model: 30 tasks, no failures.\n- mid-model is the best-measured frontier pick:  "
        "396 tasks, 10 failures.\n"
    )
    merged = notes.merge_evidence(EVIDENCE, theirs)
    assert notes.merge_evidence(theirs, EVIDENCE) == merged
    lines = merged.splitlines()
    rows = [line for line in lines if line.startswith("| ") and not line.startswith("| model")]
    assert rows == [
        "| big-model | 1,809 | 96 | 0.1015 |",
        "| mid-model | 410 | 11 | 0.0380 |",
        "| new-model | 30 | 0 | 0.0100 |",
        "| small-model | 22 | 4 | 0.0822 |",
        "| others (7 models) | 26 | | insufficient data |",
    ]
    # Theirs has one row more, so it is the base: its prose and bullet order.
    assert "2,700 tasks" in merged and "2,684 tasks" not in merged
    assert merged.count("_Observed") == 1 and merged.count("Others: tiny-a") == 1
    bullets = [line for line in lines if line[:2] in ("- ", "* ")]
    assert bullets == [
        "- mid-model is the best-measured frontier pick: 396 tasks, 10 failures.",
        "- big-model carries the most work (1,809 tasks) and the most failures (96);",
        "- small-model is superseded: 4 failures in 22 tasks.",
        "* new-model: 30 tasks, no failures.",
    ]
    assert "  it costs 2.7x mid-model per step." in lines
    assert merged.endswith("* new-model: 30 tasks, no failures.\n")
    assert notes.merge_evidence(merged, theirs) == merged
    assert notes.merge_evidence(merged, EVIDENCE) == merged


def test_same_day_evidence_conflicts_resolve_the_same_way_on_both_machines() -> None:
    """Equal counts, or different headers: the base (more rows, then the greater layout) wins
    outright, so ``merge(a, b) == merge(b, a)`` and a sync converges."""
    stamp = "_Observed in the task history, refreshed 2026-09-27 by /rsi7d._\n"
    a = stamp + "\n| model | tasks | fail |\n|---|---|---|\n| a | 10 | 1 |\n"
    b = stamp + "\n| model | tasks | fail |\n|---|---|---|\n| a | 10 | 9 |\n"
    assert notes.merge_evidence(a, b) == notes.merge_evidence(b, a) == b
    wide = (
        stamp + "\n| model | tasks | $/step | s/step |\n|---|---|---|---|\n| a | 20 | 0.05 | 7 |\n"
    )
    assert notes.merge_evidence(a, wide) == notes.merge_evidence(wide, a) == a  # the greater header
    two = a.replace("| a | 10 | 1 |", "| a | 10 | 1 |\n| b | 3 | 0 |")
    assert notes.merge_evidence(two, wide) == notes.merge_evidence(wide, two) == two
    # Rows tied on the count keep the greater text whichever copy is the base, so a
    # merge result merged again with either input stays as it is.
    c = stamp + "\n| model | tasks | fail |\n|---|---|---|\n| a | 20 | 1 |\n| z | 10 | 9 |\n"
    d = stamp + "\n| model | tasks | fail |\n|---|---|---|\n| z | 10 | 1 |\n| a | 5 | 1 |\n"
    m = notes.merge_evidence(c, d)
    assert m == notes.merge_evidence(d, c)
    assert "| a | 20 | 1 |\n| z | 10 | 9 |" in m
    assert notes.merge_evidence(m, c) == notes.merge_evidence(d, m) == m


def test_a_same_day_merge_merged_again_with_either_copy_stays_as_it_is() -> None:
    """One sync leaves one text: the remote merges what this machine pushed into its own copy.

    The base was picked on the raw text and the raw counts, which a merge changes: 10 sorts
    below 9, rows are re-sorted by count, a repeated bullet is united, bullet groups are moved
    together, a final blank line is dropped.  A copy that then outranked the merge result set
    the layout on its own machine, and the two machines held two texts.
    """
    table = f"{EVIDENCE.splitlines()[0]}\n\n| model | tasks |\n|---|---|\n"
    pairs = [
        (table + "| x | 9 |\n\n- a-rule\n- z-rule\n", table + "| x | 10 |\n\n- z-rule\n- a-rule\n"),
        (table + "| x | 9 |\n\nUse the cheap one.\n", table + "| x | 10 |\n\nUse the large one.\n"),
        (
            table + "| zeta | 2 |\n| alpha | 9 |\n\n- alpha is cheap\n- zeta is slow\n",
            table + "| zeta | 1 |\n| alpha | 0 |\n\n- zeta is slow\n- alpha is cheap\n",
        ),
        (table + "| m | 5 |\n\n- b1\n- b1 \n- b2\nP\n", table + "| m | 5 |\n\n- b2\n- b1\nQ\n"),
        (table + "| m | 5 |\n\n- b1\nZzz\n- b2\n", table + "| m | 5 |\n\n- b1\nA\n- b2\n"),
        (table + "| m | 5 |\n\n- b1\n\n", table),
    ]
    for ours, theirs in pairs:
        merged = notes.merge_evidence(ours, theirs)
        assert notes.merge_evidence(theirs, ours) == merged
        for copy in (ours, theirs):
            assert notes.merge_evidence(copy, merged) == merged, (ours, theirs)
            assert notes.merge_evidence(merged, copy) == merged, (ours, theirs)


def test_same_day_evidence_without_a_table_or_bullets_borrows_them() -> None:
    """The base lacking bullets gets the other's appended; the base is the copy with more
    rows, then more bullets; count-less rows stay last."""
    stamp = "_Observed in the task history, refreshed 2026-09-27 by /rsi7d._\n"
    ours = stamp + "\nNothing measured.\n"
    theirs = (
        stamp + "\n| model | tasks |\n|---|---|\n| a | 5 |\n| b | n/a |\n| c | 9 |\n\n- a: fine.\n"
    )
    # Theirs has the rows, so it is the base and our prose is dropped with the rest of ours.
    assert notes.merge_evidence(ours, theirs) == (
        stamp + "\n| model | tasks |\n|---|---|\n| c | 9 |\n| a | 5 |\n| b | n/a |\n\n- a: fine.\n"
    )
    # The base with rows but no bullets borrows the other's bullets.
    plain = stamp + "\n| model | tasks |\n|---|---|\n| a | 5 |\n| c | 9 |\n"
    assert notes.merge_evidence(plain, stamp + "\n- a: fine.\n") == (
        stamp + "\n| model | tasks |\n|---|---|\n| c | 9 |\n| a | 5 |\n\n- a: fine.\n"
    )
    # No rows on either side: the copy with more bullets is the base.
    empty = stamp + "\nNothing measured yet.\n\n| model | tasks |\n|---|---|\n"
    assert notes.merge_evidence(empty, stamp + "\n- one.\n") == stamp + "\n- one.\n"
    header_only = stamp + "\n| model | tasks |\n|---|---|\n"
    with_row = stamp + "\n| model | tasks |\n|---|---|\n| model-a | 10 |\n"
    assert notes.merge_evidence(header_only, with_row) == with_row
    # A copy with rows but no header row goes with any header; the other is the base
    # (one row each, greater layout: its header), so the result keeps the header.
    assert notes.merge_evidence(with_row, stamp + "\n| model-b | 20 |\n") == (
        stamp + "\n| model | tasks |\n|---|---|\n| model-b | 20 |\n| model-a | 10 |\n"
    )
    # Bullet groups split by prose: the later group moves up into the first
    # (ours is the base: no rows on either side, more bullets).
    ours = stamp + "\n- one.\n\nProse.\n\n- two.\n"
    assert notes.merge_evidence(ours, stamp + "\n- a-three.\n") == (
        stamp + "\n- one.\n- two.\n- a-three.\n\nProse.\n"
    )


def test_merge_into_file_reports_what_it_did(tmp_path: Path) -> None:
    """created / updated / unchanged; nothing is written for two missing or blank copies."""
    path = tmp_path / "MODEL_DECISIONS.md"
    assert notes.merge_into_file("MODEL_DECISIONS.md", str(path), None) == "unchanged"
    assert notes.merge_into_file("MODEL_DECISIONS.md", str(path), "\n") == "unchanged"
    assert not path.exists()
    assert notes.merge_into_file("MODEL_DECISIONS.md", str(path), HEADER + ROW_A) == "created"
    assert path.read_text() == HEADER + ROW_A
    assert notes.merge_into_file("MODEL_DECISIONS.md", str(path), HEADER + ROW_A) == "unchanged"
    assert notes.merge_into_file("MODEL_DECISIONS.md", str(path), HEADER + ROW_B) == "updated"
    assert path.read_text() == HEADER + ROW_A + ROW_B
    assert sorted(p.name for p in tmp_path.iterdir()) == ["MODEL_DECISIONS.md"]
    with pytest.raises(notes.SyncError, match="not a synced note"):
        notes.merge_note("AGENTS.md", "a", "b")


def _merge_rows_in_thread(path: Path, theirs: str, verdicts: list[str]) -> threading.Thread:
    """Merge the ledger text *theirs* into *path* in a thread; the verdict goes to *verdicts*."""
    thread = threading.Thread(
        target=lambda: verdicts.append(
            notes.merge_into_file("MODEL_DECISIONS.md", str(path), theirs)
        )
    )
    thread.start()
    return thread


def test_a_row_appended_while_the_ledger_is_merged_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The merge waits for the lock the agents append under and rewrites the file they hold open.

    An autorouter task that is appending a row when the sync merges the
    ledger opened the ledger before the merge; replacing the file under
    it would leave its row in an orphaned copy.  With the ledger locked
    by a third party, the appender waits, the merge waits too, and once
    the lock is released every row -- ours, the remote's and the
    appended one -- is in the one ledger.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path))
    path = tmp_path / "MODEL_DECISIONS.md"
    path.write_text(HEADER + ROW_A)
    inode = path.stat().st_ino
    verdicts: list[str] = []
    with path.open("a") as held:
        lock_exclusive(held)
        appender = threading.Thread(
            target=log_decision, args=("grep logs", "small", "m-small", "tier small 0.9")
        )
        appender.start()
        merger = _merge_rows_in_thread(path, HEADER + ROW_B, verdicts)
        merger.join(timeout=1.0)
        assert merger.is_alive(), "the merge went ahead while an agent held the ledger"
        unlock(held)
    appender.join(timeout=30)
    merger.join(timeout=30)
    assert not appender.is_alive() and not merger.is_alive()
    text = path.read_text()
    assert verdicts == ["updated"]
    assert text.startswith(HEADER + ROW_A + ROW_B)
    assert text.count("| - | grep logs | small | m-small | tier small 0.9 | pending |") == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["MODEL_DECISIONS.md"]
    assert path.stat().st_ino == inode  # whichever of the two had the lock first


def test_two_merges_of_the_same_ledger_at_once_lose_nothing(tmp_path: Path) -> None:
    """Two syncs merging at once (two deploys, a deploy and a remote) take turns."""
    path = tmp_path / "MODEL_DECISIONS.md"
    path.write_text(HEADER + ROW_A)
    verdicts: list[str] = []
    with path.open("a") as held:
        lock_exclusive(held)
        first = _merge_rows_in_thread(path, HEADER + ROW_B, verdicts)
        second = _merge_rows_in_thread(path, HEADER + ROW_C, verdicts)
        first.join(timeout=1.0)
        assert first.is_alive() and second.is_alive()
        unlock(held)
    first.join(timeout=30)
    second.join(timeout=30)
    assert verdicts == ["updated", "updated"]
    assert path.read_text() == HEADER + ROW_A + ROW_B + ROW_C


def _append_result(results: list[object], function: Callable[..., object], *args: object) -> None:
    """Thread target: append what ``function(*args)`` returns, or the SyncError it raises."""
    try:
        results.append(function(*args))
    except notes.SyncError as exc:
        results.append(exc)


def _refresh(path: Path, text: str) -> IO[str]:
    """Replace the note at *path* by a rename, with no lock held; return the new file, locked."""
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(text)
    held = partial.open("r+")
    lock_exclusive(held)
    partial.replace(path)
    return held


def test_a_note_replaced_while_it_is_merged_is_merged_again(tmp_path: Path) -> None:
    """A merge that finds its note replaced by a rename merges into the file now at the path.

    A writer that takes none of the note's locks (an editor, a daemon from before ``/rsi7d``
    took the lock file) renames a fresh AUTOROUTER.md over the old one; a merge holding the
    old file open wrote the incoming rows into that orphan and said "updated".
    Here every file at the path stays locked until the next one has replaced it, so each
    merge attempt meets a replaced note: the merge lands once the refreshes stop, and gives
    up, rather than report success, when they do not.
    """
    path = tmp_path / "AUTOROUTER.md"
    table = f"{EVIDENCE.splitlines()[0]}\n\n| model | tasks |\n|---|---|\n"
    local, remote = table + "| local-model | 110 |\n", table + "| remote-model | 90 |\n"
    merge = (notes.merge_into_file, "AUTOROUTER.md", str(path), remote)
    files = [_refresh(path, local)]
    outcomes: list[object] = []
    merger = threading.Thread(target=_append_result, args=(outcomes, *merge))
    merger.start()
    while merger.is_alive() and not any(
        b"remote-model" in os.pread(orphan.fileno(), 4096, 0) for orphan in files[:-1]
    ):
        files.append(_refresh(path, local))
        unlock(files[-2])
        merger.join(timeout=0.05)
    unlock(files[-1])  # the refreshes stop: the note now at the path is free
    merger.join(timeout=30)
    assert outcomes == ["updated"]
    assert path.read_text() == table + "| local-model | 110 |\n| remote-model | 90 |\n"
    files.append(_refresh(path, local))
    merger = threading.Thread(target=_append_result, args=(outcomes, *merge))
    merger.start()
    while merger.is_alive():  # the refreshes never stop
        files.append(_refresh(path, local))
        unlock(files[-2])
        merger.join(timeout=0.05)
    for held in files:
        held.close()
    assert isinstance(outcomes[-1], notes.SyncError) and "replaced" in str(outcomes[-1])
    assert path.read_text() == local


def _start_at(
    started: list[tuple[threading.Thread, bool]],
    call: str,
    action: Callable[[], object],
    _frame: FrameType,
    event: str,
    arg: object,
) -> None:
    """Profile function of a merge: a thread runs *action* as the merge first calls *call*.

    *call* is the qualified name of a C function (``replace`` is ``os.replace``).  *started*
    gets the thread and whether it was still at work a second later.
    """
    if event == "c_call" and getattr(arg, "__qualname__", "") == call and not started:
        thread = threading.Thread(target=action)
        thread.start()
        thread.join(timeout=1.0)
        started.append((thread, thread.is_alive()))


def _merge_at_each_lock(
    competitors: list[threading.Thread],
    path: Path,
    stamp: str,
    frame: FrameType,
    event: str,
    _arg: object,
) -> None:
    """Profile function of a merge: each time it goes for a lock, another merge gets a second.

    Four times at most; merge *n* brings bullet *n* to the evidence at *path*.
    """
    if event == "call" and frame.f_code is notes.lock_note.__code__ and len(competitors) < 4:
        merge = ("AUTOROUTER.md", str(path), f"{stamp}- bullet {len(competitors)}.\n")
        competitors.append(threading.Thread(target=notes.merge_into_file, args=merge))
        competitors[-1].start()
        competitors[-1].join(timeout=1.0)


def test_merges_of_the_evidence_take_turns_and_lose_nothing(tmp_path: Path) -> None:
    """Merges of AUTOROUTER.md take turns, whichever name each was given for the note.

    A merge renames the merged evidence over the file the note's path leads to (a rename
    over a symlink would leave a regular file in its place and the old text in the file it
    named).  Merges that only locked the file each had opened found it replaced by the merge
    they had waited for: one that renamed its text all the same lost that merge's bullet, and
    one that merged again gave up after three replacements.  Here a merge of the file itself
    starts each time a merge through a symlink goes for a lock: for the link's lock file, for
    the file's, for the note.  The last one waits, no merge gives up, and the note holds the
    bullet of each.
    """
    real, link = tmp_path / "dotfiles" / "AUTOROUTER.md", tmp_path / "home" / "AUTOROUTER.md"
    real.parent.mkdir()
    link.parent.mkdir()
    stamp = EVIDENCE.splitlines()[0] + "\n\n"
    real.write_text(stamp + "- seed.\n")
    link.symlink_to(real)
    competitors: list[threading.Thread] = []
    sys.setprofile(functools.partial(_merge_at_each_lock, competitors, real, stamp))
    try:
        verdict = notes.merge_into_file("AUTOROUTER.md", str(link), stamp + "- first.\n")
    finally:
        sys.setprofile(None)
    for competitor in competitors:
        competitor.join(timeout=30)
    assert verdict == "updated" and len(competitors) == 3
    assert sorted(real.read_text().splitlines()[2:]) == [
        "- bullet 0.",
        "- bullet 1.",
        "- bullet 2.",
        "- first.",
        "- seed.",
    ]
    assert link.is_symlink()
    for folder in (link.parent, real.parent):
        assert sorted(os.listdir(folder)) == ["AUTOROUTER.md", "AUTOROUTER.md.lock"]


def _save_with_a_backup(path: Path, text: str) -> None:
    """Move the note at *path* aside and write *text* as a new note, as an editor may save."""
    path.rename(path.with_name(path.name + "~"))
    path.write_text(text)


def test_evidence_is_renamed_over_the_file_that_was_merged_and_no_other(tmp_path: Path) -> None:
    """A merge that finds another file at the note's path than the one it read merges again.

    An editor that keeps a backup moves the note aside and writes a new one, with none of the
    note's locks.  A merge that had read the old note and renamed its text over the path all
    the same put the old evidence back over what the editor had saved.  Here the editor saves
    when the merge has staged its text: the merge renames nothing, and merges into the new note.
    """
    path = tmp_path / "AUTOROUTER.md"
    stamp = EVIDENCE.splitlines()[0] + "\n\n"
    path.write_text(stamp + "- old.\n")
    save = functools.partial(_save_with_a_backup, path, stamp + "- saved.\n")
    sys.setprofile(functools.partial(_start_at, [], "fchown", save))
    try:
        verdict = notes.merge_into_file("AUTOROUTER.md", str(path), stamp + "- incoming.\n")
    finally:
        sys.setprofile(None)
    assert verdict == "updated"
    assert sorted(path.read_text().splitlines()[2:]) == ["- incoming.", "- saved."]


def _read_into(texts: list[str | None], path: Path) -> None:
    """Thread body: append to *texts* what ``read_note`` returns for the note at *path*."""
    texts.append(notes.read_note(str(path)))


@pytest.mark.skipif(is_root(), reason="root ignores directory permission bits")
@pytest.mark.parametrize("lock_file", [False, True])
def test_evidence_in_a_folder_that_takes_no_new_file_is_merged_in_place(
    tmp_path: Path, lock_file: bool
) -> None:
    """A writable AUTOROUTER.md in a folder its user cannot write to is merged all the same.

    Nothing can be renamed into such a folder, so the merged text is written into the note
    itself, as the ledger's is -- also when the note is reached through a symlink in a folder
    that would take a new file, and whether or not the folder holds the note's lock file from
    the time it took one.  The merge failed there instead: "Permission denied".  Between the
    write and the truncate the note is the start of the new text and the end of the old one;
    a reader that takes the note's lock, as ``read_note`` does, waits and reads the new text.
    """
    path, link = tmp_path / "dotfiles" / "AUTOROUTER.md", tmp_path / "home" / "AUTOROUTER.md"
    path.parent.mkdir()
    link.parent.mkdir()
    path.write_text(EVIDENCE)
    link.symlink_to(path)
    if lock_file:
        path.with_name("AUTOROUTER.md.lock").touch()
    beside = sorted(os.listdir(path.parent))
    stamp = EVIDENCE.splitlines()[0]
    newer = stamp.replace("2026-09-27", "2026-10-04") + "\n\n" + "- a newer bullet.\n" * 3000
    newest = stamp.replace("2026-09-27", "2026-10-11") + "\n\n" + "- the newest bullet.\n" * 1000
    inode = path.stat().st_ino
    texts: list[str | None] = []
    started: list[tuple[threading.Thread, bool]] = []
    read = functools.partial(_read_into, texts, path)
    path.parent.chmod(0o500)
    try:
        assert notes.merge_into_file("AUTOROUTER.md", str(path), newer) == "updated"
        assert path.read_text() == newer
        sys.setprofile(functools.partial(_start_at, started, "TextIOWrapper.truncate", read))
        try:
            assert notes.merge_into_file("AUTOROUTER.md", str(link), newest) == "updated"
        finally:
            sys.setprofile(None)
    finally:
        path.parent.chmod(0o700)
    [(reader, waited)] = started
    reader.join(timeout=30)
    assert waited and texts == [newest]
    assert path.read_text() == newest and path.stat().st_ino == inode and link.is_symlink()
    assert sorted(os.listdir(path.parent)) == beside
    assert sorted(os.listdir(link.parent)) == ["AUTOROUTER.md", "AUTOROUTER.md.lock"]


@pytest.mark.skipif(is_root(), reason="root ignores directory permission bits")
@pytest.mark.parametrize(
    "why",
    ["attribute", "second name", "lock symlink", "lock hard link", "lock is note", "folder closes"],
)
def test_evidence_no_new_file_can_stand_in_for_is_merged_in_place(tmp_path: Path, why: str) -> None:
    """Where a new file would not be what the note was, the evidence is written into the note.

    A new file lacks the note's extended attributes (an ACL is one: whoever it let read the
    note could not read the new file) and is not the file the note's second name, a hard link,
    leads to.  A lock file that is a link is no lock to take (here a link to the lock file of
    the file the note, a symlink, leads to), nor is one the note is a link to: the merge waited
    for a lock it held itself until it gave up.  And a folder
    that stops taking files once the text is staged allows neither the rename nor the removal
    of the staged file: that second error ended the merge.  Each of these notes is merged as
    it always was, in place.
    """
    path = tmp_path / "AUTOROUTER.md"
    path.write_text(EVIDENCE)
    newer = EVIDENCE.replace("2026-09-27", "2026-10-04")
    inode = path.stat().st_ino
    hook = None
    if why == "attribute":
        try:
            if sys.platform != "linux":
                raise OSError("os.setxattr is Linux's")
            os.setxattr(path, "user.kiss", b"kept")
        except OSError:
            pytest.skip("no user extended attributes here")
    elif why == "second name":
        os.link(path, tmp_path / "twin.md")
    elif why in ("lock symlink", "lock hard link"):
        path.rename(tmp_path / "real.md")
        path.symlink_to("real.md")
        (tmp_path / "real.md.lock").touch()
        link = os.symlink if why == "lock symlink" else os.link
        link(tmp_path / "real.md.lock", path.with_name("AUTOROUTER.md.lock"))
    elif why == "lock is note":
        path.rename(path.with_name("AUTOROUTER.md.lock"))
        path.symlink_to("AUTOROUTER.md.lock")
    else:
        close = functools.partial(tmp_path.chmod, 0o500)
        hook = functools.partial(_start_at, [], "replace", close)
    sys.setprofile(hook)
    try:
        verdict = notes.merge_into_file("AUTOROUTER.md", str(path), newer)
    finally:
        sys.setprofile(None)
        tmp_path.chmod(0o700)
    assert verdict == "updated" and path.read_text() == newer and path.stat().st_ino == inode


@pytest.mark.parametrize("target", ["gone/../target.md", "target.md/"])
def test_evidence_behind_a_link_the_kernel_does_not_open_is_not_made_elsewhere(
    tmp_path: Path, target: str
) -> None:
    """A note that is a symlink the kernel refuses to open fails its merge, as it always did.

    ``os.path.realpath`` reads both targets as ``target.md``: a merge that opened what it
    returns made that file and said "created", and the note's path led nowhere, as before.

    Not reached by any test: the bound on the links a merge follows (``_rename_locks``).  A
    chain the kernel gives up on fails the merge's ``os.stat`` before any link is followed,
    so the bound only ends a loop of links made while the merge runs, which no real file
    system makes on demand.
    """
    path = tmp_path / "AUTOROUTER.md"
    os.symlink(f"{tmp_path}/{target}", path)
    with pytest.raises(OSError):
        notes.merge_into_file("AUTOROUTER.md", str(path), EVIDENCE)
    assert sorted(os.listdir(tmp_path)) == ["AUTOROUTER.md", "AUTOROUTER.md.lock"]


def test_evidence_that_cannot_be_written_leaves_the_note_and_nothing_beside_it(
    tmp_path: Path,
) -> None:
    """A text UTF-8 cannot encode fails the merge: the note is whole, no staged file is left."""
    path = tmp_path / "AUTOROUTER.md"
    path.write_text(EVIDENCE)
    newer = EVIDENCE.replace("2026-09-27", "2026-10-04") + "- \udcff.\n"
    with pytest.raises(UnicodeEncodeError):
        notes.merge_into_file("AUTOROUTER.md", str(path), newer)
    assert path.read_text() == EVIDENCE
    assert sorted(os.listdir(tmp_path)) == ["AUTOROUTER.md", "AUTOROUTER.md.lock"]


def _merge_under(profile: Callable[..., object], path: Path, text: str) -> None:
    """Thread body: merge *text* into the evidence at *path* with *profile* as profile function."""
    sys.setprofile(profile)
    try:
        notes.merge_into_file("AUTOROUTER.md", str(path), text)
    finally:
        sys.setprofile(None)


def _replace_under_lock(path: Path, text: str) -> None:
    """Thread body: put *text* at *path* the way ``/rsi7d`` refreshes the evidence."""
    with notes.rename_lock(str(path)):
        notes.replace_note(str(path), text)


@renames_over_a_note
def test_a_lock_file_renamed_over_while_a_merge_waits_for_it_is_not_its_lock(
    tmp_path: Path,
) -> None:
    """A merge that waited for a lock file goes on with the file at that name, not an old one.

    A note that is a link to another note's lock file is merged like any note: the new text
    is renamed over that lock file.  A merge of the other note that had waited for the old
    file went on with it, a file nothing led to any more: a refresh took the new lock file at
    once, and the merge renamed its older text over the refreshed note.  Here a merge of the
    note starts as its lock file is renamed over, and a refresh as that merge renames: both
    wait, and the note is the refreshed one.
    """
    note, alias = tmp_path / "home" / "AUTOROUTER.md", tmp_path / "alias" / "AUTOROUTER.md"
    note.parent.mkdir()
    alias.parent.mkdir()
    stamp = EVIDENCE.splitlines()[0] + "\n\n"
    note.write_text(stamp + "- seed.\n")
    lock_file = note.with_name("AUTOROUTER.md.lock")
    lock_file.write_text(stamp + "- elsewhere.\n")
    alias.symlink_to(lock_file)
    newer = EVIDENCE.replace("2026-09-27", "2026-10-04")
    merges: list[tuple[threading.Thread, bool]] = []
    refreshes: list[tuple[threading.Thread, bool]] = []
    refresh = functools.partial(_replace_under_lock, note, newer)
    as_it_renames = functools.partial(_start_at, refreshes, "replace", refresh)
    merge = functools.partial(_merge_under, as_it_renames, note, stamp + "- incoming.\n")
    sys.setprofile(functools.partial(_start_at, merges, "replace", merge))
    try:
        notes.merge_into_file("AUTOROUTER.md", str(alias), stamp + "- via the link.\n")
    finally:
        sys.setprofile(None)
    merges[0][0].join(timeout=30)
    refreshes[0][0].join(timeout=30)
    assert merges[0][1] and refreshes[0][1] and note.read_text() == newer


def _read_at_each_lock(
    texts: list[str | None], path: Path, frame: FrameType, event: str, _arg: object
) -> None:
    """Profile function of a merge: each time it goes for a lock, a reader reads *path*."""
    if event == "call" and frame.f_code is notes.lock_note.__code__:
        reader = threading.Thread(target=_read_into, args=(texts, path))
        reader.start()
        reader.join(timeout=1.0)


@pytest.mark.parametrize("linked", [False, True])
def test_evidence_that_is_not_there_yet_is_never_read_empty(tmp_path: Path, linked: bool) -> None:
    """The first evidence appears whole: a reader finds no note, or all of it.

    A merge made the missing note, empty, and then went for its lock: a reader that got the
    lock first read an empty note.  Here a reader reads each time the merge goes for a lock;
    the note is made by a rename, private as ever, also where a dangling link leads to it.
    """
    path = tmp_path / "AUTOROUTER.md"
    target = tmp_path / "dotfiles.md" if linked else path
    if linked:
        path.symlink_to(target)
    texts: list[str | None] = []
    sys.setprofile(functools.partial(_read_at_each_lock, texts, path))
    try:
        verdict = notes.merge_into_file("AUTOROUTER.md", str(path), EVIDENCE)
    finally:
        sys.setprofile(None)
    assert verdict == "created" and texts and set(texts) == {None}
    assert notes.read_note(str(path)) == notes.merge_note("AUTOROUTER.md", "", EVIDENCE)
    assert path.is_symlink() == linked and stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(is_root(), reason="root ignores directory permission bits")
def test_evidence_that_cannot_be_made_fails_as_it_always_did(tmp_path: Path) -> None:
    """Where the first evidence cannot be staged, the merge fails on the note itself, as before.

    The folder takes no new file and the lock file is there from better days: the error is
    that of making the note, not of a file staged beside it, and nothing is left behind.
    """
    path = tmp_path / "home" / "AUTOROUTER.md"
    path.parent.mkdir()
    path.with_name("AUTOROUTER.md.lock").touch()
    path.parent.chmod(0o500)
    try:
        with pytest.raises(PermissionError) as failure:
            notes.merge_into_file("AUTOROUTER.md", str(path), EVIDENCE)
    finally:
        path.parent.chmod(0o700)
    assert failure.value.filename == str(path)
    assert os.listdir(path.parent) == ["AUTOROUTER.md.lock"]


def test_command_line_merges_two_files_and_explains_itself(tmp_path: Path) -> None:
    """``merge NAME INTO FROM`` merges on disk; anything else prints the usage.

    A FROM that is no regular file is refused: the command waited for ever to open a named
    pipe nobody writes to (it is ended at the timeout here when it does).
    """
    into, other = tmp_path / "into.md", tmp_path / "from.md"
    into.write_text(EVIDENCE)
    other.write_text(EVIDENCE.replace("2026-09-27", "2026-10-04"))
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "merge", "AUTOROUTER.md", str(into), str(other)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0 and done.stdout.strip() == "updated"
    assert into.read_text() == other.read_text()
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "merge", "AGENTS.md", str(into), str(other)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1 and "not a synced note" in done.stderr
    pipe = tmp_path / "pipe.md"
    os.mkfifo(pipe)
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "merge", "AUTOROUTER.md", str(into), str(pipe)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert done.returncode == 1 and "pipe.md is not a regular file" in done.stderr
    assert into.read_text() == other.read_text()
    for args in ([], ["--help"], ["merge"], ["_merge"]):
        done = subprocess.run(
            [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
        )
        assert done.returncode == 2 and done.stderr.startswith("usage:"), args


class Sandbox:
    """Two fake machines and the ssh stub that connects them.

    With *brand_home* the script runs from a copy of a checkout whose
    ``.brand/brand.json`` overlay names that state directory, and with
    no ``KISS_HOME`` on either machine, as ``rsorcar`` runs it.
    """

    def __init__(self, tmp: Path, brand_home: str | None = None) -> None:
        self.brand_home = brand_home
        self.local = tmp / "home" / (brand_home or ".kiss")
        self.remote = tmp / "rhome" / (brand_home or ".kiss")
        self.bindir = tmp / "bin"
        for directory in (self.local, self.remote, self.bindir):
            directory.mkdir(parents=True)
        self.install("ssh", FAKE_SSH)
        self.script = SCRIPT
        if brand_home:
            checkout = tmp / "checkout"
            self.script = checkout / "src" / "kiss" / "scripts" / SCRIPT.name
            self.script.parent.mkdir(parents=True)
            shutil.copy(SCRIPT, self.script)
            (checkout / ".brand").mkdir()
            (checkout / ".brand" / "brand.json").write_text(json.dumps({"home_dir": brand_home}))

    def install(self, name: str, text: str) -> None:
        """Put an executable stub named *name* on the sandbox PATH."""
        path = self.bindir / name
        path.write_text(text)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def environment(self) -> dict[str, str]:
        """Return the environment of a run: this machine's home, the stub ssh first on PATH."""
        env = dict(os.environ)
        env.pop("KISS_HOME", None)
        env.update(
            {
                "HOME": str(self.local.parent),
                "REMOTE_HOME": str(self.remote.parent),
                "PATH": f"{self.bindir}:{env['PATH']}",
            }
        )
        if not self.brand_home:
            env["KISS_HOME"] = str(self.local)
        return env

    def run(self) -> subprocess.CompletedProcess[str]:
        """Run the real script against the sandbox remote."""
        return subprocess.run(
            [sys.executable, str(self.script), "me@fakehost"],
            env=self.environment(),
            capture_output=True,
            text=True,
            check=False,
        )


def test_sync_brings_both_machines_to_the_same_merged_notes(tmp_path: Path) -> None:
    """Both notes end up identical on both machines, holding what either had; a rerun is a no-op."""
    box = Sandbox(tmp_path)
    (box.local / "MODEL_DECISIONS.md").write_text(HEADER + ROW_A + ROW_C)
    (box.remote / "MODEL_DECISIONS.md").write_text(HEADER + ROW_B)
    (box.remote / "AUTOROUTER.md").write_text(EVIDENCE)
    result = box.run()
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "AUTOROUTER.md: here created, me@fakehost unchanged",
        "MODEL_DECISIONS.md: here updated, me@fakehost updated",
    ]
    for name in notes.NOTES:
        assert (box.local / name).read_text() == (box.remote / name).read_text()
    assert (box.local / "MODEL_DECISIONS.md").read_text() == HEADER + ROW_A + ROW_B + ROW_C
    assert (box.local / "AUTOROUTER.md").read_text() == EVIDENCE
    assert sorted(p.name for p in box.remote.iterdir()) == [
        "AUTOROUTER.md",
        "AUTOROUTER.md.lock",  # of whoever replaces the evidence: left for the next one
        "MODEL_DECISIONS.md",
    ]
    again = box.run()
    assert again.returncode == 0 and again.stdout.splitlines() == [
        "AUTOROUTER.md: here unchanged, me@fakehost unchanged",
        "MODEL_DECISIONS.md: here unchanged, me@fakehost unchanged",
    ]


def test_sync_leaves_one_evidence_text_on_both_machines(tmp_path: Path) -> None:
    """One sync of two same-day copies writes one text on both machines; the next, nothing."""
    box = Sandbox(tmp_path)
    table = f"{EVIDENCE.splitlines()[0]}\n\n| model | tasks |\n|---|---|\n"
    (box.local / "AUTOROUTER.md").write_text(table + "| x | 9 |\n\nUse the cheaper one.\n")
    (box.remote / "AUTOROUTER.md").write_text(table + "| x | 10 |\n\nUse the larger one.\n")
    result = box.run()
    assert result.returncode == 0, result.stderr
    merged = (box.local / "AUTOROUTER.md").read_text()
    assert merged == (box.remote / "AUTOROUTER.md").read_text()
    assert merged == table + "| x | 10 |\n\nUse the larger one.\n"
    assert box.run().stdout.startswith("AUTOROUTER.md: here unchanged, me@fakehost unchanged\n")


def test_a_remote_python_with_another_stdin_encoding_keeps_the_text(tmp_path: Path) -> None:
    """The pushed note is written as the UTF-8 it was sent in, whatever the remote locale is."""
    box = Sandbox(tmp_path)
    box.install("ssh", FAKE_SSH.replace("exec bash", "export PYTHONIOENCODING=latin-1\nexec bash"))
    row = "| 2026-09-27 10:00 | t1 | café → thé | small | m-small | tier small 0.9 | pending |\n"
    (box.local / "MODEL_DECISIONS.md").write_text(HEADER + row, encoding="utf-8")
    result = box.run()
    assert result.returncode == 0, result.stderr
    assert (box.remote / "MODEL_DECISIONS.md").read_text(encoding="utf-8") == HEADER + row
    assert "MODEL_DECISIONS.md: here unchanged, me@fakehost unchanged" in box.run().stdout


@pytest.mark.parametrize("corrupt", ["here", "there"])
def test_a_note_that_is_not_utf8_fails_its_own_sync_only(tmp_path: Path, corrupt: str) -> None:
    """A note this script cannot decode is reported, on whichever machine; the other note syncs.

    Both copies stay as they were.  The remote's copy used to be decoded with replacement
    characters: newer evidence holding one stray byte replaced this machine's valid copy with
    U+FFFD text before the remote's own merge failed on that byte.
    """
    box = Sandbox(tmp_path)
    newer = EVIDENCE.replace("2026-09-27", "2026-10-04").encode() + b"\xff stray byte\n"
    here, there = (newer, EVIDENCE.encode()) if corrupt == "here" else (EVIDENCE.encode(), newer)
    (box.local / "AUTOROUTER.md").write_bytes(here)
    (box.remote / "AUTOROUTER.md").write_bytes(there)
    (box.remote / "MODEL_DECISIONS.md").write_text(HEADER + ROW_B)
    result = box.run()
    assert result.returncode == 1
    assert (box.local / "AUTOROUTER.md").read_bytes() == here
    assert (box.remote / "AUTOROUTER.md").read_bytes() == there
    assert result.stderr.startswith("warning: AUTOROUTER.md: ") and "Traceback" not in result.stderr
    assert ("AUTOROUTER.md on me@fakehost is not UTF-8" in result.stderr) == (corrupt == "there")
    assert result.stdout == "MODEL_DECISIONS.md: here created, me@fakehost unchanged\n"
    assert (box.local / "MODEL_DECISIONS.md").read_text() == HEADER + ROW_B


@pytest.mark.timeout(120)  # bounds a sync that hangs again; the four below end by themselves
def test_a_note_that_stays_locked_fails_its_own_sync_only(tmp_path: Path) -> None:
    """A writer stalled with a note's lock costs each waiter ``LOCK_SECONDS``, not the deploy.

    The waits had no deadline: this machine's merge and its read blocked in ``flock``, and so
    did the remote's read and its merge, under an ssh whose keep-alives sshd goes on answering.
    Each now gives up at the deadline; the sync reports that note, syncs the other one, and
    leaves no waiter behind, here or on the remote.
    """
    boxes = [Sandbox(tmp_path / side) for side in ("here", "read", "merged")]
    for box in boxes:
        (box.local / "MODEL_DECISIONS.md").write_text(HEADER + ROW_A)
        (box.remote / "MODEL_DECISIONS.md").write_text(HEADER + ROW_B)
    stalled = [boxes[0].local, boxes[1].remote, boxes[2].remote]
    into = tmp_path / "into.md"
    merge = [str(SCRIPT), "merge", "AUTOROUTER.md", str(into), str(stalled[0] / "AUTOROUTER.md")]
    commands = [([str(box.script), "me@fakehost"], box.environment()) for box in boxes]
    commands.append((merge, dict(os.environ)))
    held = []
    for folder in stalled:
        (folder / "AUTOROUTER.md").write_text(EVIDENCE)
        held.append((folder / "AUTOROUTER.md").open("a"))
        # A reader's lock on the third note lets the remote's read through and stalls its merge.
        notes.lock_note(held[-1], shared=folder is stalled[2])
    pipes = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
    runs = [popen_process_group([sys.executable, *cmd], env=env, **pipes) for cmd, env in commands]
    try:
        outputs = [run.communicate(timeout=60) for run in runs]
        for run in runs:  # the notes are still locked: nothing is left waiting for them
            with pytest.raises(ProcessLookupError):
                os.killpg(run.pid, 0)
    finally:
        for run in runs:
            if run.poll() is None:
                kill_process_tree(run)
                run.wait()
        for handle in held:
            handle.close()
    locked = f"still locked by another process after {notes.LOCK_SECONDS} s\n"
    synced = "MODEL_DECISIONS.md: here updated, me@fakehost updated\n"
    unread = "could not read AUTOROUTER.md on me@fakehost (ssh exited 1): "
    unmerged = "could not merge AUTOROUTER.md into me@fakehost's copy (exit 1): sync_kiss_notes: "
    assert outputs == [
        (synced, "warning: AUTOROUTER.md: " + locked),
        (synced, "warning: AUTOROUTER.md: " + unread + locked),
        (synced, "warning: AUTOROUTER.md: " + unmerged + locked),
        ("", "sync_kiss_notes: " + locked),
    ]
    assert [run.returncode for run in runs] == [1, 1, 1, 1] and not into.exists()


def test_sync_with_nothing_anywhere_and_with_a_dead_connection(tmp_path: Path) -> None:
    """No notes on either side: nothing is created.  A dead ssh: a warning per note, exit 1."""
    box = Sandbox(tmp_path)
    result = box.run()
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "AUTOROUTER.md: neither machine has it",
        "MODEL_DECISIONS.md: neither machine has it",
    ]
    assert list(box.local.iterdir()) == [] and list(box.remote.iterdir()) == []
    (box.local / "AUTOROUTER.md").write_text(EVIDENCE)
    box.install("ssh", DEAD_SSH)
    result = box.run()
    assert result.returncode == 1 and result.stdout == ""
    warnings = result.stderr.splitlines()
    assert len(warnings) == 2 and all("Connection refused" in line for line in warnings)
    assert warnings[0].startswith(
        "warning: AUTOROUTER.md: could not read AUTOROUTER.md on me@fakehost"
    )
    assert (box.local / "AUTOROUTER.md").read_text() == EVIDENCE


def test_a_branded_installation_syncs_the_notes_its_agents_write(tmp_path: Path) -> None:
    """Without ``KISS_HOME`` the notes are the brand's (``~/.s10s``), on both machines.

    The agents write the notes of the brand's home directory, so stale
    copies left in ``~/.kiss`` by a stock install are neither read nor
    touched, here or on the remote.
    """
    box = Sandbox(tmp_path, brand_home=".s10s")
    (box.local / "MODEL_DECISIONS.md").write_text(HEADER + ROW_A)
    (box.remote / "MODEL_DECISIONS.md").write_text(HEADER + ROW_B)
    for stock in (box.local.parent / ".kiss", box.remote.parent / ".kiss"):
        stock.mkdir()
        (stock / "MODEL_DECISIONS.md").write_text(HEADER + ROW_C)
        (stock / "AUTOROUTER.md").write_text(EVIDENCE)
    result = box.run()
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "AUTOROUTER.md: neither machine has it",
        "MODEL_DECISIONS.md: here updated, me@fakehost updated",
    ]
    for side in (box.local, box.remote):
        assert (side / "MODEL_DECISIONS.md").read_text() == HEADER + ROW_A + ROW_B
        assert sorted(p.name for p in side.iterdir()) == ["MODEL_DECISIONS.md"]
        assert (side.parent / ".kiss" / "MODEL_DECISIONS.md").read_text() == HEADER + ROW_C
        assert (side.parent / ".kiss" / "AUTOROUTER.md").read_text() == EVIDENCE


REPORT_LOCK = """import fcntl, os, sys
def report(event, args):
    "Audit hook of a reader process: say lock on stdout just before it first locks $NOTE."
    if event == "fcntl.flock" and not report.said and not args[1] & fcntl.LOCK_UN:
        if os.path.samefile(args[0], os.environ["NOTE"]):
            report.said = os.write(1, b"lock")
report.said = 0
sys.addaudithook(report)
"""


def _stop_at_ssh(frame: FrameType, event: str, _arg: object) -> None:
    """Profile function: leave a call of ``notes._ssh`` at its entry, raising its command."""
    if event == "call" and frame.f_code is notes._ssh.__code__:
        raise SystemExit(frame.f_locals["command"])


@pytest.mark.timeout(60)  # the bound of every wait below: a reader that never reports fails it
def test_a_note_being_rewritten_is_never_read_half_written(tmp_path: Path) -> None:
    """A snapshot taken while a merge rewrites the note waits for the merge to finish.

    The test writes as ``merge_into_file`` rewrites a note in place (the ledger, and the evidence
    where it cannot be renamed over): lock, write the shorter, newer text over the start, flush --
    and truncate only later.  Until then the file holds the new stamp and the old tail, which a
    sync would push back as bullets of the new evidence.  Both readers wait for the lock and print
    the new text alone: ``read_note``, and what ``fetch_remote`` hands to ssh.
    """
    path = tmp_path / ".kiss" / "AUTOROUTER.md"
    path.parent.mkdir()
    path.write_text(EVIDENCE)
    (tmp_path / "sitecustomize.py").write_text(REPORT_LOCK)
    newer = EVIDENCE.splitlines()[0].replace("2026-09-27", "2026-10-04") + "\n\nNothing yet.\n"
    sys.setprofile(_stop_at_ssh)
    try:
        with pytest.raises(SystemExit) as sent:
            notes.fetch_remote("me@fakehost", "AUTOROUTER.md", ".kiss")
    finally:
        sys.setprofile(None)
    local = f"import runpy; print(end=runpy.run_path({str(SCRIPT)!r})['read_note']({str(path)!r}))"
    commands = ([sys.executable, "-c", local], ["bash", "-c", sent.value.args[0]])
    env = dict(os.environ, HOME=str(tmp_path), PYTHONPATH=str(tmp_path), NOTE=str(path))
    env.pop("KISS_HOME", None)  # a real ssh session does not inherit it either
    with path.open("r+") as writer:
        lock_exclusive(writer)
        writer.write(newer)
        writer.flush()
        readers = [popen_process_group(argv, env=env, stdout=subprocess.PIPE) for argv in commands]
        try:  # each reader says lock (REPORT_LOCK), or prints what it read without the lock
            reports = [reader.stdout.read(4) for reader in readers if reader.stdout]
            writer.truncate()
            unlock(writer)
            snapshots = [said + reader.communicate()[0] for said, reader in zip(reports, readers)]
        finally:
            for reader in readers:
                if reader.poll() is None:
                    kill_process_tree(reader)
                    reader.wait()
    assert snapshots == [b"lock" + newer.encode()] * 2
    assert [reader.returncode for reader in readers] == [0, 0]


def _read_unlocked(
    path: Path, seen: list[str], _frame: FrameType, event: str, _arg: object
) -> None:
    """Profile function: at each call into C, read the note as a reader without its lock does."""
    if event.startswith("c_"):
        seen.append(path.read_text(encoding="utf-8"))


@renames_over_a_note
def test_evidence_read_without_its_lock_during_a_merge_is_one_whole_version(tmp_path: Path) -> None:
    """Whoever reads AUTOROUTER.md without the note's lock never gets a mix of two texts.

    ``observed_evidence`` spliced ``read_text()`` of the note into every routing prompt.  A merge
    that rewrote the note in place showed it the start of the newer evidence followed by the tail
    of the longer, older one, from the merge's write until its truncate.  Here the note is read
    that way at every call a real merge makes into C: each read is the old text or the new one.
    The file that replaces the note has its permission bits and its group (here one of the
    user's other groups, where it has any), as the note rewritten in place kept them.

    Where the note is still rewritten in place, the mix is there to be read: these reads get it
    under Python 3.13 and miss it under 3.14, whose buffer holds the text back until the truncate.
    """
    path = tmp_path / "AUTOROUTER.md"
    stamp = EVIDENCE.splitlines()[0]
    old = stamp + "\n\n" + "- an old bullet.\n" * 3000
    new = stamp.replace("2026-09-27", "2026-10-04") + "\n\n" + "- a new bullet.\n" * 1000
    path.write_text(old)
    path.chmod(0o640)
    group = next((gid for gid in os.getgroups() if gid != os.getegid()), os.getegid())
    os.chown(path, -1, group)
    seen: list[str] = []
    sys.setprofile(functools.partial(_read_unlocked, path, seen))
    try:
        verdict = notes.merge_into_file("AUTOROUTER.md", str(path), new)
    finally:
        sys.setprofile(None)
    assert verdict == "updated"
    assert set(seen) == {old, new}
    assert path.read_text() == new and stat.S_IMODE(path.stat().st_mode) == 0o640
    assert path.stat().st_gid == group
    assert sorted(os.listdir(tmp_path)) == ["AUTOROUTER.md", "AUTOROUTER.md.lock"]


def test_a_host_that_never_answers_the_ssh_handshake_fails_the_fetch() -> None:
    """The real ``ssh`` gives up on a host that takes the connection and then says nothing.

    Nothing accepts on the listener, so ssh connects, sends its banner and waits for the
    host's: for ever before it had a connect deadline, and a deploy with it.  (The deadline
    for a connection that dies later, ServerAliveInterval, needs a real sshd to test.)
    """
    with socket.create_server(("127.0.0.1", 0)) as silent:
        host = f"ssh://127.0.0.1:{silent.getsockname()[1]}"
        with pytest.raises(notes.SyncError, match="timed out"):
            notes.fetch_remote(host, "AUTOROUTER.md", ".kiss")
