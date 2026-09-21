# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests for ``Bash(background=True)`` and the ``bash_job`` primitive.

A background job is started detached (own session, stdin from
``/dev/null``, stdout and stderr into ``<work_dir>/tmp/bash_jobs/<id>.log``)
and ``Bash`` returns at once with a job id.  ``bash_job`` then reports the
job's status and log tail (``tail``), blocks until it exits (``wait``), or
kills its whole process group (``kill``).  Every test drives the real
tool objects against real shell processes; nothing is mocked.

Not covered here, because neither can be produced without a test double:

* the ``except BaseException`` guard around the watcher ``Thread.start()``
  in ``UsefulTools._start_background_job`` (kill the just-spawned shell's
  group, re-raise) fires only on thread exhaustion or a stop injected
  inside ``Thread.start`` (``RLIMIT_NPROC`` would break the ``Popen`` fork
  first);
* the ``_kill_job_group`` branch that skips the orphan-group kill because
  the exited shell's pid is alive again (pid recycled by an unrelated
  process) needs the kernel to hand that exact pid out once more.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core.processes import pid_alive
from kiss.core.tool_interrupt import (
    ToolCallInterrupted,
    begin_tool_call,
    end_tool_call,
    interrupt_tool_call,
    unregister_tool_call,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX shell/process-group semantics"
)

_JOB_ID_RE = re.compile(r"Started background job (j[0-9a-f]{8}) \(pid (\d+)\)")


def _start(tools: UsefulTools, command: str) -> tuple[str, int, Path]:
    """Start *command* in the background and return ``(job_id, pid, log_path)``."""
    result = tools.Bash(command, "start a background job", background=True)
    match = _JOB_ID_RE.search(result)
    assert match, result
    log_line = next(line for line in result.splitlines() if line.startswith("Log: "))
    return match.group(1), int(match.group(2)), Path(log_line[len("Log: "):])


def _wait_pid_dead(pid: int, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not pid_alive(pid), f"pid {pid} still alive"


def _wait_for_text(path: Path, text: str, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and text in path.read_text():
            return
        time.sleep(0.05)
    pytest.fail(f"{text!r} never appeared in {path}")


class TestStart:
    def test_returns_at_once_with_job_id_and_log_under_work_dir(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        started = time.monotonic()
        job_id, pid, log_path = _start(tools, "echo begin; sleep 20; echo never")
        assert time.monotonic() - started < 5
        assert log_path == tmp_path / "tmp" / "bash_jobs" / f"{job_id}.log"
        assert pid_alive(pid)
        _wait_for_text(log_path, "begin")
        # The job runs in the agent's work_dir with a cleaned environment.
        tools.bash_job(job_id, action="kill")
        _wait_pid_dead(pid)

    def test_job_runs_in_work_dir_and_survives_the_call(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        job_id, _pid, _log = _start(tools, "sleep 0.5; pwd > where.txt; echo done")
        report = tools.bash_job(job_id, action="wait", timeout_seconds=20)
        assert "exited with code 0" in report
        assert (tmp_path / "where.txt").read_text().strip() == str(tmp_path.resolve())

    def test_stderr_and_stdout_share_the_log(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        job_id, _pid, log_path = _start(tools, "echo out; echo err >&2; exit 3")
        report = tools.bash_job(job_id, action="wait", timeout_seconds=20)
        assert "exited with code 3" in report
        assert "out\n" in report and "err\n" in report
        assert log_path.read_text() == "out\nerr\n"

    def test_stdin_is_devnull(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        job_id, _pid, _log = _start(tools, "cat; echo eof-reached")
        report = tools.bash_job(job_id, action="wait", timeout_seconds=20)
        assert "exited with code 0" in report and "eof-reached" in report

    def test_timeout_seconds_does_not_apply_to_background(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        result = tools.Bash("sleep 2; echo late", "bg", timeout_seconds=0.1, background=True)
        job_id = _JOB_ID_RE.search(result).group(1)  # type: ignore[union-attr]
        report = tools.bash_job(job_id, action="wait", timeout_seconds=20)
        assert "exited with code 0" in report and "late" in report

    def test_without_work_dir_logs_under_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        tools = UsefulTools()
        job_id, _pid, log_path = _start(tools, "echo hi")
        assert log_path == tmp_path / "tmp" / "bash_jobs" / f"{job_id}.log"
        assert "exited with code 0" in tools.bash_job(job_id, action="wait", timeout_seconds=20)

    def test_unwritable_log_dir_is_an_error(self, tmp_path: Path) -> None:
        (tmp_path / "tmp").write_text("a file where the tmp dir should be")
        tools = UsefulTools(work_dir=str(tmp_path))
        result = tools.Bash("echo hi", "bg", background=True)
        assert result.startswith("Error: could not start background job"), result
        assert tools._jobs == {}

    def test_worktree_guard_runs_before_the_job_starts(self, tmp_path: Path) -> None:
        # A command that names the parent checkout of a worktree is refused
        # by the same guard as a foreground command, so nothing is launched.
        parent = tmp_path / "repo"
        worktree = parent / ".kiss-worktrees" / "kiss_wt-1-abc"
        worktree.mkdir(parents=True)
        marker = parent / "leaked"
        tools = UsefulTools(work_dir=str(worktree))
        result = tools.Bash(f"touch {marker}", "escape", background=True)
        assert result.startswith("Error:"), result
        assert tools._jobs == {}
        time.sleep(0.3)
        assert not marker.exists()


class TestBashJob:
    def test_tail_while_running_then_wait(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        job_id, pid, log_path = _start(
            tools, "for i in 1 2 3; do echo line$i; done; sleep 1.5; echo final",
        )
        _wait_for_text(log_path, "line3")
        tail = tools.bash_job(job_id)  # default action is "tail"
        assert f"Job {job_id}: running (pid {pid}, " in tail
        assert "Command: for i in 1 2 3" in tail
        assert f"Log: {log_path}" in tail
        assert tail.endswith("--- last 50 log lines ---\nline1\nline2\nline3\n")

        started = time.monotonic()
        report = tools.bash_job(job_id, action="wait", timeout_seconds=30)
        assert time.monotonic() - started < 15
        assert f"Job {job_id}: exited with code 0" in report
        assert report.endswith("line1\nline2\nline3\nfinal\n")

    def test_wait_on_finished_job_returns_immediately(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        job_id, pid, _log = _start(tools, "echo quick")
        _wait_pid_dead(pid)
        started = time.monotonic()
        report = tools.bash_job(job_id, action="wait", timeout_seconds=60)
        assert time.monotonic() - started < 2
        assert "exited with code 0" in report

    def test_wait_times_out_and_leaves_the_job_running(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        job_id, pid, _log = _start(tools, "sleep 30")
        started = time.monotonic()
        report = tools.bash_job(job_id, action="wait", timeout_seconds=1)
        elapsed = time.monotonic() - started
        assert 0.9 <= elapsed < 5, elapsed
        assert "running (pid" in report
        assert pid_alive(pid)
        tools.bash_job(job_id, action="kill")
        _wait_pid_dead(pid)

    def test_kill_takes_the_whole_process_group(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        child_pid_file = tmp_path / "child.pid"
        job_id, pid, _log = _start(
            tools, f"sleep 60 & echo $! > {child_pid_file}; wait",
        )
        _wait_for_text(child_pid_file, "\n")
        child_pid = int(child_pid_file.read_text().strip())
        assert pid_alive(child_pid)
        report = tools.bash_job(job_id, action="kill")
        assert f"Job {job_id}: killed by signal 9" in report
        _wait_pid_dead(pid)
        _wait_pid_dead(child_pid)
        # Killing again is a no-op that still reports the final status.
        assert "killed by signal 9" in tools.bash_job(job_id, action="kill")

    def test_kill_reaches_daemonized_descendants_of_an_exited_shell(
        self, tmp_path: Path,
    ) -> None:
        # ``sh -c "server &"`` exits at once and leaves the server in the
        # job's process group; kill must still reach it.
        tools = UsefulTools(work_dir=str(tmp_path))
        child_pid_file = tmp_path / "daemon.pid"
        job_id, pid, _log = _start(tools, f"sleep 60 & echo $! > {child_pid_file}")
        _wait_for_text(child_pid_file, "\n")
        _wait_pid_dead(pid)
        child_pid = int(child_pid_file.read_text().strip())
        assert pid_alive(child_pid)
        report = tools.bash_job(job_id, action="kill")
        assert "exited with code 0" in report
        _wait_pid_dead(child_pid)
        # With the whole group gone, a second kill finds nothing (ESRCH).
        assert "exited with code 0" in tools.bash_job(job_id, action="kill")

    def test_tail_reads_only_the_end_of_a_large_log(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        job_id, _pid, log_path = _start(tools, "seq 1 200000")
        tools.bash_job(job_id, action="wait", timeout_seconds=30)
        assert log_path.stat().st_size > 1_000_000
        report = tools.bash_job(job_id, tail_lines=3, max_output_chars=300)
        assert report.endswith("199998\n199999\n200000\n"), report
        assert len(report) <= 300
        # The partial line at the read boundary is dropped, never shown.
        wide = tools.bash_job(job_id, tail_lines=1000, max_output_chars=600)
        assert len(wide) <= 600 and "truncated" not in wide
        lines = wide.split("--- last 1000 log lines ---\n", 1)[1].splitlines()
        assert lines and all(line.isdigit() for line in lines), lines
        assert int(lines[0]) == 200000 - len(lines) + 1

    def test_tail_window_on_a_line_boundary_keeps_the_whole_line(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        job_id, _pid, _log = _start(tools, "printf 'x\\nabc\\n'")
        tools.bash_job(job_id, action="wait", timeout_seconds=20)
        marker = "--- last 10 log lines ---\n"
        full = tools.bash_job(job_id, tail_lines=10)
        header_len = full.index(marker) + len(marker)
        assert full == full[:header_len] + "x\nabc\n"
        # A 4-byte window starts right after the first "\n": "abc\n" is whole.
        aligned = tools.bash_job(job_id, tail_lines=10, max_output_chars=header_len + 4)
        assert aligned.endswith(marker + "abc\n"), aligned
        # A 3-byte window starts inside "abc": the cut line is dropped.
        cut = tools.bash_job(job_id, tail_lines=10, max_output_chars=header_len + 3)
        assert cut.endswith(marker), cut

    def test_tail_lines_and_truncation(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        job_id, _pid, log_path = _start(tools, "seq 1 100")
        tools.bash_job(job_id, action="wait", timeout_seconds=20)
        report = tools.bash_job(job_id, tail_lines=3)
        assert report.endswith("--- last 3 log lines ---\n98\n99\n100\n")
        assert "\n97\n" not in report
        assert tools.bash_job(job_id, tail_lines=0).endswith("--- last 0 log lines ---\n")
        # The tail is cut to what fits under max_output_chars, whole lines only.
        narrow = tools.bash_job(job_id, tail_lines=100, max_output_chars=400)
        assert len(narrow) <= 400 and narrow.endswith("\n99\n100\n") and "\n1\n" not in narrow
        # A cap smaller than the header itself leaves only a truncated header.
        tiny = tools.bash_job(job_id, tail_lines=100, max_output_chars=60)
        assert len(tiny) <= 60 and "truncated" in tiny
        log_path.unlink()
        assert "(log unreadable:" in tools.bash_job(job_id)

    def test_unknown_job_and_bad_action(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        assert tools.bash_job("j00000000") == (
            "Error: unknown job id 'j00000000'. Known jobs: none."
        )
        job_id, _pid, _log = _start(tools, "echo hi")
        assert tools.bash_job("nope") == f"Error: unknown job id 'nope'. Known jobs: {job_id}."
        assert tools.bash_job(job_id, action="restart") == (
            "Error: action must be one of wait, tail, kill; got 'restart'."
        )
        tools.bash_job(job_id, action="wait", timeout_seconds=20)


class TestStop:
    def test_task_stop_kills_running_jobs(self, tmp_path: Path) -> None:
        stop = threading.Event()
        tools = UsefulTools(stop_event=stop, work_dir=str(tmp_path))
        job_id, pid, _log = _start(tools, "sleep 60")
        finished_id, finished_pid, _log = _start(tools, "echo done")
        _wait_pid_dead(finished_pid)
        stop.set()
        _wait_pid_dead(pid)
        assert "killed by signal 9" in tools.bash_job(job_id)
        assert "exited with code 0" in tools.bash_job(finished_id)

    def test_wait_ends_early_on_task_stop(self, tmp_path: Path) -> None:
        stop = threading.Event()
        tools = UsefulTools(stop_event=stop, work_dir=str(tmp_path))
        job_id, pid, _log = _start(tools, "sleep 60")
        threading.Timer(0.7, stop.set).start()
        started = time.monotonic()
        tools.bash_job(job_id, action="wait", timeout_seconds=60)
        assert time.monotonic() - started < 10
        _wait_pid_dead(pid)

    def test_tool_panel_interrupt_raises_from_wait(self, tmp_path: Path) -> None:
        tools = UsefulTools(work_dir=str(tmp_path))
        job_id, pid, _log = _start(tools, "sleep 60")
        outcome: dict[str, Any] = {}

        def run() -> None:
            token = begin_tool_call("bash_job")
            try:
                try:
                    outcome["result"] = tools.bash_job(job_id, action="wait", timeout_seconds=60)
                    end_tool_call(token)
                except ToolCallInterrupted:
                    outcome["interrupted"] = True
            finally:
                unregister_tool_call(token)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        time.sleep(0.7)
        assert thread.ident is not None
        assert interrupt_tool_call(thread.ident, "bash_job") is True
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert outcome == {"interrupted": True}, outcome
        # Interrupting the WAIT does not kill the job itself.
        assert pid_alive(pid)
        tools.bash_job(job_id, action="kill")
        _wait_pid_dead(pid)


class TestRegistration:
    def test_bash_job_is_offered_next_to_bash(self, tmp_path: Path) -> None:
        agent = SorcarAgent("bg-tool-list")
        agent.work_dir = str(tmp_path)
        agent._use_web_tools = False
        tools = agent._get_tools()
        names = [t.__name__ for t in tools]
        assert names.index("Bash") + 1 == names.index("bash_job")
        bash = tools[names.index("Bash")]
        assert "background" in bash.__code__.co_varnames  # part of the model's schema
        job_id, _pid, _log = _start(bash.__self__, "echo via-agent")  # type: ignore[attr-defined]
        report = tools[names.index("bash_job")](job_id, action="wait", timeout_seconds=20)
        assert "exited with code 0" in report and "via-agent" in report
        assert os.path.isdir(tmp_path / "tmp" / "bash_jobs")

    def test_jobs_survive_a_rebuilt_tool_list(self, tmp_path: Path) -> None:
        # The server reuses one agent per chat tab, and every prompt
        # rebuilds the tool list: a job started by the previous prompt
        # must still be addressable from the next one.
        agent = SorcarAgent("bg-registry")
        agent.work_dir = str(tmp_path)
        agent._use_web_tools = False
        first = {t.__name__: t for t in agent._get_tools()}
        job_id, pid, _log = _start(first["Bash"].__self__, "sleep 30")  # type: ignore[attr-defined]
        second = {t.__name__: t for t in agent._get_tools()}
        assert second["bash_job"] is not first["bash_job"]
        assert "running (pid" in second["bash_job"](job_id)
        assert "killed by signal 9" in second["bash_job"](job_id, action="kill")
        _wait_pid_dead(pid)

    def test_private_registry_without_a_host_agent(self, tmp_path: Path) -> None:
        a, b = UsefulTools(work_dir=str(tmp_path)), UsefulTools(work_dir=str(tmp_path))
        job_id, _pid, _log = _start(a, "echo private")
        assert b.bash_job(job_id).startswith("Error: unknown job id")
        assert "exited with code 0" in a.bash_job(job_id, action="wait", timeout_seconds=20)
