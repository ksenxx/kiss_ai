# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01, report S3 (useful_tools): the wait of a foreground call run as a job.

Through a real :class:`UsefulTools` with real shells in ``tmp_path``,
nothing mocked; ``DEFAULT_CONFIG.bash_detach_seconds`` is set with
``monkeypatch``:

* F3: a command writing its log faster than the tool panel takes the
  pieces cannot hold the wait past the grace or past a Stop;
* F4: a stream callback that raises during the grace kills the command
  and leaves neither a job nor its log behind (as a foreground shell's
  does); on ``on_timeout="background"`` the job is left running;
* F5: a command that writes its last output and exits between the
  wait's last read and its end keeps its job and log, so that output
  is not lost with the deleted log.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.useful_tools import BackgroundJob, UsefulTools
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.processes import pid_alive
from kiss.tests.conftest import posix_only

pytestmark = posix_only("the host Bash tool runs bash loops and sleeps")

# Writes 64 KB pieces several times faster than the sinks below take them.
FLOOD = "while :; do head -c 65536 /dev/zero | tr '\\0' x; sleep 0.01; done"


def _bash_in_thread(tools: UsefulTools, command: str, **kwargs: object) -> tuple[
    threading.Thread, list[str],
]:
    outcome: list[str] = []
    thread = threading.Thread(
        target=lambda: outcome.append(tools.Bash(command, "flood", **kwargs)),  # type: ignore[arg-type]
        daemon=True,
    )
    thread.start()
    return thread, outcome


def _kill_jobs(tools: UsefulTools) -> None:
    for job_id in list(tools._jobs):
        tools.bash_job(job_id, action="kill")


def _wait_exit(job: BackgroundJob) -> int | None:
    deadline = time.monotonic() + 10
    while job.returncode() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    return job.returncode()


# --- F3: a log growing faster than it is read ---------------------------------


def test_grace_holds_against_a_job_writing_faster_than_its_output_is_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 1)
    taken: list[int] = []

    def slow_sink(piece: str) -> None:
        taken.append(len(piece))
        time.sleep(0.2)

    tools = UsefulTools(stream_callback=slow_sink, work_dir=str(tmp_path))
    before = time.monotonic()
    thread, outcome = _bash_in_thread(tools, FLOOD, timeout_seconds=60)
    thread.join(timeout=8)
    try:
        assert not thread.is_alive(), f"Bash still draining after 8 s; {sum(taken)} chars taken"
    finally:
        _kill_jobs(tools)  # stops the flood, so a wait stuck on it ends too
        thread.join(timeout=60)
    (out,) = outcome
    assert time.monotonic() - before < 8
    assert out.startswith("Still running after 1s; left running as background job j"), out
    assert "Output so far:\n" in out and taken and all(n <= 1 << 16 for n in taken)


def test_stop_ends_the_wait_on_a_job_writing_faster_than_its_output_is_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 30)
    stop = threading.Event()

    def slow_sink(piece: str) -> None:
        time.sleep(0.2)
        stop.set()  # a Stop lands once the first piece is on screen

    tools = UsefulTools(stream_callback=slow_sink, stop_event=stop, work_dir=str(tmp_path))
    before = time.monotonic()
    thread, outcome = _bash_in_thread(tools, FLOOD, timeout_seconds=60)
    thread.join(timeout=8)
    try:
        assert not thread.is_alive(), "Bash still draining 8 s after the Stop"
    finally:
        _kill_jobs(tools)
        thread.join(timeout=60)
    assert time.monotonic() - before < 8
    assert outcome[0].startswith("Still running after 30s; left running as background job j")


# --- F4: a stream callback that raises during the grace ------------------------


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_callback_failure_during_the_grace_kills_the_command_and_its_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exception: type[BaseException],
) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 1)
    seen: list[BackgroundJob] = []

    def boom(piece: str) -> None:
        seen.extend(tools._jobs.values())
        raise exception("callback exploded")

    tools = UsefulTools(stream_callback=boom, work_dir=str(tmp_path))
    with pytest.raises(exception, match="callback exploded"):
        tools.Bash("echo ready; sleep 20", "long", timeout_seconds=10)
    (job,) = seen
    assert _wait_exit(job) is not None, "the command runs on after Bash raised"
    assert not tools._jobs and not job.log_path.exists()
    # Explicitly detached (on_timeout="background"), the job is left running as a
    # bash_job Stop leaves it, still addressable by its id.
    seen.clear()
    with pytest.raises(exception, match="callback exploded"):
        tools.Bash("echo ready; sleep 20", "long", timeout_seconds=10, on_timeout="background")
    (job,) = seen
    assert job.returncode() is None and list(tools._jobs) == [job.job_id]
    tools.bash_job(job.job_id, action="kill")
    assert _wait_exit(job) is not None


def test_a_foreground_result_leaves_its_detached_child_and_no_watcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command that finished within the grace is a foreground result: nothing watches it on.

    Its job leaves the registry, but the job's watcher went on polling
    for as long as a child the command detached (``nohup server &``)
    lived, and killed that child when the task was stopped later; run
    with a shorter timeout, as a foreground shell, the same command
    leaves the child running (audit 2026-11-10, C/C13-F2).
    """
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 120)
    stop = threading.Event()
    tools = UsefulTools(work_dir=str(tmp_path), stop_event=stop)
    before = set(threading.enumerate())
    out = tools.Bash("nohup tail -f /dev/null >/dev/null 2>&1 & echo pid=$!", "serve")
    child = int(out.split("pid=")[1].split()[0])
    try:
        assert not tools._jobs
        deadline = time.monotonic() + 10
        while set(threading.enumerate()) - before and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not set(threading.enumerate()) - before, "the job's watcher is still polling"
        stop.set()
        time.sleep(1)  # twice the poll of a watcher, were one left
        assert pid_alive(child), "the Stop killed the child the command detached"
    finally:
        os.kill(child, signal.SIGKILL)


# --- F5: an exit between the last read and the end of the wait -----------------


class _StopAsTheWaitLooks(threading.Event):
    """A task Stop that lands exactly when the wait next looks for one.

    Once armed (the first output piece reached the tool panel), the
    first ``is_set`` releases the gated child and returns only after
    the child has written its last line and exited: the wait then ends
    on a Stop with that line still unread in the log, the situation a
    process exiting right after the wait's last read creates.
    """

    def __init__(self, release: Path, jobs: dict[str, BackgroundJob]) -> None:
        super().__init__()
        self.release = release
        self.jobs = jobs
        self.armed = False

    def is_set(self) -> bool:
        if self.armed and not super().is_set():
            self.release.touch()
            for job in self.jobs.values():
                _wait_exit(job)
            self.set()
        return super().is_set()


def test_exit_after_the_last_read_keeps_the_job_and_its_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "bash_detach_seconds", 5)
    jobs: dict[str, BackgroundJob] = {}
    stop = _StopAsTheWaitLooks(tmp_path / "release", jobs)
    pieces: list[str] = []

    def on_piece(piece: str) -> None:
        pieces.append(piece)
        stop.armed = True

    tools = UsefulTools(
        stream_callback=on_piece, stop_event=stop, work_dir=str(tmp_path), jobs=jobs,
    )
    command = "echo early; while [ ! -e release ]; do sleep 0.01; done; echo late"
    out = tools.Bash(command, "gated", timeout_seconds=30)
    assert pieces == ["early\n"]
    # The wait ended before reading "late": the job and its log stay, nothing is lost.
    assert "left running as background job " in out and out.endswith("Output so far:\nearly\n")
    (job,) = jobs.values()
    assert job.log_path.read_text() == "early\nlate\n"
    report = tools.bash_job(job.job_id, action="wait", timeout_seconds=10)
    assert "exited with code 0" in report and report.endswith("early\nlate\n")


# --- A10-1 (fable): a character the job writes in two pieces -------------------


class _ReleaseOnceTheFirstPieceWasRead(threading.Event):
    """A task Stop that never fires but, looked at by the wait once the
    first piece reached the tool panel, releases the gated child to write
    the rest of the character and exit before the wait reads again."""

    def __init__(self, release: Path, jobs: dict[str, BackgroundJob]) -> None:
        super().__init__()
        self.release = release
        self.jobs = jobs
        self.armed = False
        self.released = False

    def is_set(self) -> bool:
        if self.armed and not self.released:
            self.released = True
            self.release.touch()
            for job in self.jobs.values():
                _wait_exit(job)
        return False


def test_a_character_the_job_writes_in_two_pieces_is_read_whole(tmp_path: Path) -> None:
    """The wait decodes the growing log incrementally: U+65E5 split over two
    writes of the command (its lead bytes, then its last byte) reaches the
    tool panel and the result as that character, not as two U+FFFD."""
    jobs: dict[str, BackgroundJob] = {}
    gate = _ReleaseOnceTheFirstPieceWasRead(tmp_path / "release", jobs)
    pieces: list[str] = []

    def on_piece(piece: str) -> None:
        pieces.append(piece)
        gate.armed = True

    tools = UsefulTools(
        stream_callback=on_piece, stop_event=gate, work_dir=str(tmp_path), jobs=jobs,
    )
    command = (
        "printf 'abc\\346\\227'; while [ ! -e release ]; do sleep 0.01; done; printf '\\245\\n'"
    )
    out = tools.Bash(command, "split character", timeout_seconds=30, on_timeout="background")
    assert gate.released, "the wait never took the first piece"
    assert "".join(pieces) == "abc\u65e5\n", pieces
    assert out == "abc\u65e5\n", out


def test_crlf_output_reads_as_the_foreground_pipe_reads_it(tmp_path: Path) -> None:
    """A job's CRLF line endings come back as ``\\n``, as a foreground run's
    text-mode pipe returns them (Windows tools print CRLF), even when the
    CR and the LF land in two pieces of the growing log."""
    jobs: dict[str, BackgroundJob] = {}
    gate = _ReleaseOnceTheFirstPieceWasRead(tmp_path / "release", jobs)
    pieces: list[str] = []

    def on_piece(piece: str) -> None:
        pieces.append(piece)
        gate.armed = True

    tools = UsefulTools(
        stream_callback=on_piece, stop_event=gate, work_dir=str(tmp_path), jobs=jobs,
    )
    command = "printf 'one\\r\\ntwo\\r'; while [ ! -e release ]; do sleep 0.01; done; printf '\\n'"
    out = tools.Bash(command, "crlf output", timeout_seconds=30, on_timeout="background")
    assert gate.released, "the wait never took the first piece"
    assert "".join(pieces) == "one\ntwo\n", pieces
    assert out == "one\ntwo\n", out
