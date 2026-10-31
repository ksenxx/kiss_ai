# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-30, slice S7: the host Bash tools.

* F1: a truncated output was spilled under the vanished worktree,
  recreating its directory as an empty husk.
* F2: the here-document guard looked its targets up under the work dir
  while the command ran in the parent repo the vanished worktree falls
  back to; the Bash view of a file there did not count as a read.
* F3: ``sleep_guard`` missed ``sleep 5m``, ``sleep infinity`` and loop
  sleeps, and refused ``sleep`` inside quoted text.
* F5: the here-document guard of a ``machine=`` job asked the local
  disk about a file on the remote machine.
* F8: ``bash_job(wait, until=...)`` reread the whole job log once a second.
* F14: a foreground command's whole output was held in memory (a list
  of lines plus their join) before it was cut to the budget, and a
  single huge line was read in one piece.

Not covered, as no real command can make them happen: a failing
``close`` of the stdout pipe in ``_drain_stdout`` and an exception out
of ``_bash_streaming`` in ``_run_one_command`` (see
test_run_commands_parallel.py).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import tracemalloc
import uuid
from pathlib import Path

import pytest

from kiss.agents.sorcar.edit_guards import HEREDOC_VERDICT
from kiss.agents.sorcar.useful_tools import UsefulTools, sleep_guard
from kiss.tests.conftest import posix_only

pytestmark = posix_only("the host Bash tool runs bash")


def _vanished_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """Return ``(repo, worktree)``: a repo with ``notes.md`` and a worktree path that is gone."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "notes.md").write_text("old\n")
    return repo, repo / ".kiss-worktrees" / "kiss_wt-gone"


def test_spill_of_a_vanished_worktree_goes_where_the_command_ran(tmp_path: Path) -> None:
    """F1: the full output lands in the parent repo; the worktree is not recreated."""
    repo, wt = _vanished_worktree(tmp_path)
    tools = UsefulTools(work_dir=str(wt))
    out = tools.Bash("python3 -c \"print('x' * 5000)\"", "big", max_output_chars=300)
    match = re.search(r" in (tmp/tool-out/[^\]\s]+)\]", out)
    assert match is not None, out
    assert not wt.exists()
    assert (repo / match[1]).read_text() == "x" * 5000 + "\n"
    # With no directory to fall back to there is no spill and nothing is created.
    lost = tmp_path / "lost" / ".kiss-worktrees" / "kiss_wt-gone"
    out = UsefulTools(work_dir=str(lost)).Bash("python3 -c \"print('y' * 5000)\"", "big",
                                               max_output_chars=300)
    assert "[truncated" in out and "tool-out" not in out
    assert not (tmp_path / "lost").exists()


def test_heredoc_guard_checks_the_directory_a_vanished_worktree_falls_back_to(
    tmp_path: Path,
) -> None:
    """F2: Bash, run_commands_parallel and the read bookkeeping use where the command runs."""
    repo, wt = _vanished_worktree(tmp_path)
    notes = repo / "notes.md"
    refusal = HEREDOC_VERDICT.format(path=notes)
    tools = UsefulTools(work_dir=str(wt))
    assert tools.Bash("cat > notes.md <<'EOF'\nnew\nEOF", "rewrite") == refusal
    report = tools.run_commands_parallel(json.dumps(["cat >> notes.md <<EOF\nx\nEOF"]))
    assert refusal in report, report
    assert notes.read_text() == "old\n"
    assert tools.Bash("cat notes.md", "view") == "old\n"
    assert tools.Edit(str(notes), "old", "new").startswith("Successfully")
    assert notes.read_text() == "new\n"


def test_remote_background_job_is_not_judged_by_local_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F5: an existing local ``notes.md`` says nothing about the machine's files."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home"))
    work = tmp_path / "work"
    work.mkdir()
    (work / "notes.md").write_text("old\n")
    tools = UsefulTools(work_dir=str(work))
    command = "cat > notes.md <<EOF\nx\nEOF"
    out = tools.Bash(command, "remote", background=True, machine="bench")
    assert out.startswith("Error: no registered machine named 'bench'"), out
    assert tools.Bash(command, "local", background=True) == HEREDOC_VERDICT.format(
        path=work / "notes.md")
    assert (work / "notes.md").read_text() == "old\n"


@pytest.mark.parametrize(
    ("command", "shown"),
    [
        ("sleep 5m", "300s"),
        ("sleep 1m 30s", "90s"),
        ("sleep 2h", "7200s"),
        ("sleep 1d", "86400s"),
        ("sleep 31", "31s"),
        ("sleep infinity", "infinity"),
        ("sleep 60 # wait for the server", "60s"),
        ("while true; do sleep 60; done", "60s"),
        ("for i in 1 2; do curl -s x; sleep 45; done", "45s"),
        ("if true; then sleep 40; fi", "40s"),
        ("/bin/sleep 60", "60s"),
        ("command sleep 60", "60s"),
        ("nohup sleep 60 &", "60s"),
        ("x=$((1<<4)); sleep 60", "60s"),
    ],
)
def test_sleep_guard_refuses_every_spelling_of_a_long_sleep(command: str, shown: str) -> None:
    """F3: GNU suffixes and sums count, and a sleep in a loop or behind a wrapper is found."""
    refusal = sleep_guard(command)
    assert refusal is not None, command
    assert refusal.startswith(f"REFUSED: the Bash call did NOT run: a foreground sleep of {shown} ")


@pytest.mark.parametrize(
    "command",
    [
        "printf '%s\\n' '(sleep 60)'",
        "git commit -m \"fix; sleep 60 && retry\"",
        "echo 'sleep 5m'",
        "echo \"x; sleep 60\"",
        "sleep 20 10",
        "sleep 0.5m",
        "sleep \"$WAIT\"",
        "sleep --help",
    ],
)
def test_sleep_guard_allows_quoted_and_short_sleeps(command: str) -> None:
    """F3: text that mentions a sleep, and a sleep of 30 s or less, run."""
    assert sleep_guard(command) is None, command


def test_bash_refuses_a_minute_sleep_without_running_it(tmp_path: Path) -> None:
    """F3 end to end: ``sleep 1m`` is refused before anything runs."""
    tools = UsefulTools(work_dir=str(tmp_path))
    out = tools.Bash("touch ran; sleep 1m", "poll")
    assert out.startswith("REFUSED: the Bash call did NOT run: a foreground sleep of 60s")
    assert not (tmp_path / "ran").exists()
    assert tools.Bash("echo '(sleep 60)'", "text") == "(sleep 60)\n"


def _job_id(started: str) -> str:
    assert started.startswith("Started background job "), started
    return started.split()[3]


def test_wait_until_reads_a_bounded_tail_of_a_big_log(tmp_path: Path) -> None:
    """F8: the once-a-second scan of a 64 MB log does not read all of it."""
    tools = UsefulTools(work_dir=str(tmp_path))
    job = _job_id(tools.Bash(
        "head -c 64000000 /dev/zero; echo; echo READY; sleep 20", "job", background=True))
    tracemalloc.start()
    try:
        report = tools.bash_job(job, action="wait", until="READY", timeout_seconds=30)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        tools.bash_job(job, action="kill")
    assert "running (pid" in report and "READY" in report, report[-300:]
    assert peak < 40_000_000, peak


def _traced_peak(tools: UsefulTools, method: str, *args: object, **kwargs: object) -> tuple:
    """Call ``tools.<method>(*args, **kwargs)``; return its result and the traced memory peak."""
    tracemalloc.start()
    try:
        result = getattr(tools, method)(*args, **kwargs)
        return result, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


_FLOOD = 64_000_000


@pytest.mark.parametrize(
    "flood",
    [
        f"head -c {_FLOOD} /dev/zero | tr '\\0' x | fold -w 9999",  # 10 kB lines
        f"head -c {_FLOOD} /dev/zero | tr '\\0' x",  # one 64 MB line
    ],
)
def test_a_huge_foreground_output_is_held_in_bounded_memory(tmp_path: Path, flood: str) -> None:
    """F14: memory stays bounded, the result keeps head and tail, the spill file has it all."""
    tools = UsefulTools(work_dir=str(tmp_path))
    out, peak = _traced_peak(tools, "Bash", f"echo FIRST; {flood}; echo; echo LAST", "flood",
                             timeout_seconds=120)
    assert peak < 48_000_000, peak
    match = re.search(r"full output \((\d+) chars\) in (tmp/tool-out/[^\]\s]+)\]", out)
    assert match is not None, out[:500]
    spill = tmp_path / match[2]
    assert int(match[1]) == spill.stat().st_size > _FLOOD
    assert out.startswith("FIRST\nxxxx") and out.endswith("x\nLAST\n")
    with spill.open() as file:
        assert file.readline() == "FIRST\n"
    assert spill.read_bytes()[-6:] == b"\nLAST\n"


def test_a_huge_parallel_output_is_held_in_bounded_memory(tmp_path: Path) -> None:
    """F14: a run_commands_parallel command too; the report still counts what it cut."""
    tools = UsefulTools(work_dir=str(tmp_path))
    command = f"echo FIRST; head -c {_FLOOD} /dev/zero | tr '\\0' x; echo; echo LAST"
    report, peak = _traced_peak(tools, "run_commands_parallel", json.dumps([command]),
                                max_output_chars=1000)
    assert peak < 48_000_000, peak
    assert report.startswith("1 commands: 1 succeeded"), report[:300]
    match = re.search(r"\[truncated (\d+) chars\]", report)
    # What was cut: the whole output (the flood and 12 more characters)
    # less the ~1000 the report keeps, not just what the capture held.
    assert match is not None and _FLOOD - 1000 < int(match[1]) < _FLOOD, report[:300]
    assert "FIRST\nxxx" in report and report.rstrip().endswith("x\nLAST")


def test_a_huge_output_of_a_command_that_finishes_before_its_deadline(tmp_path: Path) -> None:
    """F14: ``on_timeout="background"`` reads the job log in pieces, not whole."""
    tools = UsefulTools(work_dir=str(tmp_path))
    out, peak = _traced_peak(
        tools, "Bash", f"echo FIRST; head -c {_FLOOD} /dev/zero | tr '\\0' x; echo; echo LAST",
        "flood", timeout_seconds=120, on_timeout="background",
    )
    assert peak < 48_000_000, peak
    match = re.search(r"full output \((\d+) chars\) in (tmp/tool-out/[^\]\s]+)\]", out)
    assert match is not None, out[:500]
    assert int(match[1]) == (tmp_path / match[2]).stat().st_size > _FLOOD
    assert out.startswith("FIRST\nxxxx") and out.endswith("x\nLAST\n")


def test_a_big_output_before_a_timeout_keeps_its_ends_and_its_spill(tmp_path: Path) -> None:
    """F14: a command killed at its deadline reports both ends of what it printed."""
    tools = UsefulTools(work_dir=str(tmp_path))
    out = tools.Bash(
        "echo FIRST; head -c 5000000 /dev/zero | tr '\\0' x; echo; echo LAST; sleep 30",
        "flood", timeout_seconds=3,
    )
    assert out.startswith(
        "Error: Command execution timeout after 3s. Output before the timeout:\nFIRST\nx"
    ), out[:200]
    assert out.endswith("x\nLAST\n")
    match = re.search(r"full output \(\d+ chars\) in (tmp/tool-out/[^\]\s]+)\]", out)
    assert match is not None, out[:500]
    spill = (tmp_path / match[1]).read_bytes()
    assert len(spill) == 5_000_012
    assert spill.startswith(b"FIRST\nx") and spill.endswith(b"x\nLAST\n")


@pytest.mark.parametrize("blocker", ["tmp is a file", "tool-out is read-only"])
def test_no_spill_file_is_named_when_none_can_be_made(tmp_path: Path, blocker: str) -> None:
    """F14: when the spill file cannot be created the result is cut without naming one."""
    if blocker == "tmp is a file":
        (tmp_path / "tmp").write_text("")
    else:
        if os.geteuid() == 0:
            pytest.skip("root writes into a read-only directory")
        (tmp_path / "tmp" / "tool-out").mkdir(parents=True, mode=0o500)
    try:
        out = UsefulTools(work_dir=str(tmp_path)).Bash("seq 1 20000", "count")
    finally:
        if blocker != "tmp is a file":
            (tmp_path / "tmp" / "tool-out").chmod(0o700)
    assert out.startswith("1\n2\n") and out.rstrip().endswith("20000")
    assert "[truncated" in out and "tool-out" not in out, out[-300:]


@pytest.mark.skipif(not os.path.exists("/dev/full"), reason="needs /dev/full")
@pytest.mark.parametrize("lines", [30, 100_000])
def test_a_spill_file_that_cannot_be_written_is_not_named(tmp_path: Path, lines: int) -> None:
    """F14: a spill file whose writes fail (it is /dev/full) is given up, not named.

    30 lines fit the file's write buffer, so the failure comes when it
    is closed; 100000 lines (589 kB) fail on a write.
    """
    tools = UsefulTools(work_dir=str(tmp_path))
    spill_dir = tmp_path / "tmp" / "tool-out"
    spill_dir.mkdir(parents=True)
    (spill_dir / f"{tools._spill_tag}-1-bash.txt").symlink_to("/dev/full")
    out = tools.Bash(f"seq 1 {lines}", "count", max_output_chars=60)
    assert "[truncated" in out and "tool-out" not in out, out


def _slow_sink(piece: str) -> None:
    """A ``stream_callback`` that takes 10 ms per piece of output."""
    del piece
    time.sleep(0.01)


def test_a_flood_read_slower_than_it_is_written_is_held_in_bounded_memory(
    tmp_path: Path,
) -> None:
    """F14: the reader thread waits for room instead of queueing the whole flood.

    The shell exits at once; a background child keeps the pipe and
    prints without end.  Bash returns after its deadline and the short
    drain, as before; the queue between the reader thread and the slow
    stream callback used to grow by millions of lines meanwhile.
    """
    tag = f"flood-{uuid.uuid4().hex}"
    flooder = f"python3 -c 'import itertools; [print(i) for i in itertools.count()]' {tag}"
    tools = UsefulTools(work_dir=str(tmp_path), stream_callback=_slow_sink)
    started = time.monotonic()
    try:
        out, peak = _traced_peak(
            tools, "Bash", f"({flooder} &); echo started", "flood", timeout_seconds=1,
        )
    finally:
        subprocess.run(["pkill", "-f", tag], check=False)
    assert time.monotonic() - started < 20
    assert peak < 48_000_000, peak
    assert "started\n" in out, out[-300:]
