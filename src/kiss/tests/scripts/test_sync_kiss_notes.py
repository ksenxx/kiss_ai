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

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.scripts import sync_kiss_notes as notes
from kiss.tests.conftest import posix_only

SCRIPT = Path(notes.__file__).resolve()

pytestmark = posix_only("drives the script with a bash ssh stand-in")

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
    assert notes.merge_evidence(EVIDENCE, EVIDENCE + "\n") == EVIDENCE + "\n"  # same text: max
    assert notes.merge_evidence(EVIDENCE + "\n", EVIDENCE) == EVIDENCE + "\n"
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
    """Equal counts, or different headers: the base (more rows, then the greater text) wins
    outright, so ``merge(a, b) == merge(b, a)`` and a sync converges."""
    stamp = "_Observed in the task history, refreshed 2026-09-27 by /rsi7d._\n"
    a = stamp + "\n| model | tasks | fail |\n|---|---|---|\n| a | 10 | 1 |\n"
    b = stamp + "\n| model | tasks | fail |\n|---|---|---|\n| a | 10 | 9 |\n"
    assert notes.merge_evidence(a, b) == notes.merge_evidence(b, a) == b
    wide = (
        stamp + "\n| model | tasks | $/step | s/step |\n|---|---|---|---|\n| a | 20 | 0.05 | 7 |\n"
    )
    assert notes.merge_evidence(a, wide) == notes.merge_evidence(wide, a) == a  # a > wide as text
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
    # A copy with rows but no header row goes with any header; here it is the base
    # (one row each, greater text), so the result has its layout: no header.
    assert notes.merge_evidence(with_row, stamp + "\n| model-b | 20 |\n") == (
        stamp + "\n| model-b | 20 |\n| model-a | 10 |\n"
    )
    # Bullet groups split by prose: the later group moves up into the first
    # (ours is the base: no rows on either side, greater text).
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
        notes.merge_note("SORCAR.md", "a", "b")


def test_command_line_merges_two_files_and_explains_itself(tmp_path: Path) -> None:
    """``merge NAME INTO FROM`` merges on disk; anything else prints the usage."""
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
        [sys.executable, str(SCRIPT), "merge", "SORCAR.md", str(into), str(other)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1 and "not a synced note" in done.stderr
    for args in ([], ["--help"], ["merge"], ["_merge"]):
        done = subprocess.run(
            [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
        )
        assert done.returncode == 2 and done.stderr.startswith("usage:"), args


class Sandbox:
    """Two fake machines and the ssh stub that connects them."""

    def __init__(self, tmp: Path) -> None:
        self.local = tmp / "home" / ".kiss"
        self.remote = tmp / "rhome" / ".kiss"
        self.bindir = tmp / "bin"
        for directory in (self.local, self.remote, self.bindir):
            directory.mkdir(parents=True)
        self.install("ssh", FAKE_SSH)

    def install(self, name: str, text: str) -> None:
        """Put an executable stub named *name* on the sandbox PATH."""
        path = self.bindir / name
        path.write_text(text)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def run(self) -> subprocess.CompletedProcess[str]:
        """Run the real script against the sandbox remote."""
        env = dict(os.environ)
        env.update(
            {
                "HOME": str(self.local.parent),
                "REMOTE_HOME": str(self.remote.parent),
                "PATH": f"{self.bindir}:{env['PATH']}",
                "KISS_HOME": str(self.local),
            }
        )
        return subprocess.run(
            [sys.executable, str(SCRIPT), "me@fakehost"],
            env=env,
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
    assert sorted(p.name for p in box.remote.iterdir()) == ["AUTOROUTER.md", "MODEL_DECISIONS.md"]
    again = box.run()
    assert again.returncode == 0 and again.stdout.splitlines() == [
        "AUTOROUTER.md: here unchanged, me@fakehost unchanged",
        "MODEL_DECISIONS.md: here unchanged, me@fakehost unchanged",
    ]


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
