# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression test: ``git_worktree._git`` must time out.

Finding #1 (findings-6): ``git_worktree._git`` had NO timeout — a git
process wedged in a hook or local repository operation blocked the
agent thread forever.  This test pins the bounded-wait guarantee of
the shared git runner.

The hang is reproduced with a REAL stub ``git`` executable placed
first on ``PATH`` (a shell script that sleeps far longer than the
timeout), so the actual ``subprocess.run(..., timeout=...)`` code path
is exercised — no kiss code is mocked.  Before the fix this test hung
until the outer pytest timeout instead of returning.
"""

import _thread
import ctypes
import os
import stat
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import git_worktree
from kiss.agents.sorcar.git_worktree import _git
from kiss.core.processes import (
    _group_exited,
    kill_process_tree,
    start_process_group,
    terminate_process_group,
)
from kiss.tests.conftest import posix_only


def _install_hanging_git(tmp_path: Path) -> Path:
    """Create a stub ``git`` that hangs, in a fresh bin dir on disk."""
    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir()
    stub = bin_dir / "git"
    stub.write_text("#!/bin/sh\nsleep 300\n", encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


class _StopAt:
    """Profile hook raising the runner's Stop at an *event* ("call" or "return") of *function*.

    An injected ``KeyboardInterrupt`` that arrives during a call is raised
    as it returns: after the spawn, git runs and no name is bound to it yet.
    Set as the trace hook as well, a second one raises a second Stop, and
    sets *then* (the ``"c_return"`` of a C *function*) as the profile hook
    of a third.
    """

    def __init__(self, event: str, function: Any, then: Any = None) -> None:
        self.git: subprocess.Popen[bytes] | None = None
        self.event, self.code = event, getattr(function, "__code__", function)
        self.then = then and _StopAt(*then)

    def __call__(self, frame: Any, event: str, arg: Any) -> Any:
        """Keep the git that starts; raise the Stop at the event (raising unsets the hook)."""
        if event == "return" and frame.f_code is start_process_group.__code__:
            self.git = frame.f_locals["proc"]
        if event == self.event and (frame.f_code is self.code or arg is self.code):
            if self.then:
                sys.setprofile(self.then)
            raise KeyboardInterrupt
        return self  # a trace hook is told of a frame's return only when it returns itself


def _stop_before_spawn(frame: Any, event: str, arg: Any) -> None:
    """Profile hook raising that Stop as ``start_process_group`` is called: git never starts."""
    if event == "call" and frame.f_code is start_process_group.__code__:
        raise KeyboardInterrupt


#: A shell alias whose background child outlives one TERM: git and the alias shell exit on
#: it, a member of their group runs on.
_LEAVES_A_MEMBER = "!(trap 'trap - TERM' TERM; sleep 300; sleep 300) & sleep 300"


def _git_stopped_by(hook: Any, cwd: Path, alias: str = "!sleep 300", again: Any = None) -> None:
    """Run real git on a shell *alias* that waits, under profile *hook*: its Stop must come out.

    Trace hook *again* raises one more Stop.
    """
    previous, timeout = sys.getprofile(), git_worktree._GIT_TIMEOUT_SECONDS
    traced = sys.gettrace()
    git_worktree._GIT_TIMEOUT_SECONDS = 0.5  # for the Stops raised while it is handled
    sys.setprofile(hook)
    sys.settrace(again)
    try:
        with pytest.raises(KeyboardInterrupt):
            _git("-c", f"alias.wait={alias}", "wait", cwd=cwd, input="x")
    finally:
        sys.settrace(traced)
        sys.setprofile(previous)
        git_worktree._GIT_TIMEOUT_SECONDS = timeout


def _running_in_group(pgid: int) -> list[str]:
    """The pids of the processes of group *pgid* that still run (zombies left out)."""
    listing = subprocess.run(
        ["ps", "-A", "-o", "pgid=,stat=,pid="], capture_output=True, text=True, check=True,
    ).stdout
    rows = [line.split() for line in listing.splitlines()]
    return [row[2] for row in rows if row[0] == str(pgid) and not row[1].startswith("Z")]


class TestGitTimeout:
    """``_git`` returns a synthesized failure instead of hanging."""

    @posix_only("shebang git shim on PATH cannot shadow git.exe on Windows")
    def test_hanging_git_returns_timeout_completedprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hung git yields returncode 124 with a timeout message."""
        bin_dir = _install_hanging_git(tmp_path)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        saved_timeout = git_worktree._GIT_TIMEOUT_SECONDS
        git_worktree._GIT_TIMEOUT_SECONDS = 1.0
        try:
            start = time.monotonic()
            result = _git("status", cwd=tmp_path)
            elapsed = time.monotonic() - start
        finally:
            git_worktree._GIT_TIMEOUT_SECONDS = saved_timeout
        assert isinstance(result, subprocess.CompletedProcess)
        assert result.returncode == 124, (
            "expected the synthesized timeout returncode (124), got "
            f"{result.returncode}; stderr={result.stderr!r}"
        )
        assert "timed out" in result.stderr
        assert elapsed < 30, (
            f"_git took {elapsed:.1f}s — the 1s timeout did not fire"
        )

    @posix_only("a sleeping clean filter needs a POSIX shell")
    def test_timed_out_git_removes_its_index_lock(self, tmp_path: Path) -> None:
        """A git killed on the deadline must get the chance to remove
        ``index.lock`` (SIGTERM first, as the interrupt path already
        does); a SIGKILL left the lock behind and every later index
        write failed with "index.lock: File exists" (audit A09)."""
        repo = tmp_path / "repo"
        repo.mkdir()
        assert _git("init", cwd=repo).returncode == 0
        assert _git("config", "filter.slow.clean", "sleep 300", cwd=repo).returncode == 0
        (repo / ".gitattributes").write_text("*.txt filter=slow\n")
        (repo / "data.txt").write_text("payload\n")
        saved_timeout = git_worktree._GIT_TIMEOUT_SECONDS
        git_worktree._GIT_TIMEOUT_SECONDS = 1.0
        try:
            result = _git("add", "-A", cwd=repo)
        finally:
            git_worktree._GIT_TIMEOUT_SECONDS = saved_timeout
        assert result.returncode == 124
        assert not (repo / ".git" / "index.lock").exists(), (
            "the timed-out git left a stale index.lock behind"
        )
        assert _git("config", "--unset", "filter.slow.clean", cwd=repo).returncode == 0
        retry = _git("add", "-A", cwd=repo)
        assert retry.returncode == 0, retry.stderr

    @posix_only("the process group is listed with ps")
    @pytest.mark.parametrize(
        ("event", "function", "alias", "again"),
        [
            pytest.param("return", start_process_group, "!sleep 300", None, id="spawn-return"),
            pytest.param(
                "call", terminate_process_group, "!sleep 300", None, id="timeout-before-term",
            ),
            pytest.param(
                "return", terminate_process_group, "!sleep 300", None, id="timeout-after-term",
            ),
            pytest.param("call", subprocess.run, _LEAVES_A_MEMBER, None, id="timeout-member-left"),
            pytest.param(
                "return", start_process_group, "!sleep 300", ("call", terminate_process_group),
                id="second-stop-before-term",
            ),
            pytest.param(
                "call", subprocess.run, _LEAVES_A_MEMBER, ("call", _group_exited),
                id="second-stop-in-grace",
            ),
            pytest.param(
                "return", start_process_group, "!sleep 300", ("return", terminate_process_group),
                id="second-stop-before-pipes",
            ),
            pytest.param(
                "return", start_process_group, "!sleep 300",
                ("call", terminate_process_group, ("c_return", _thread.start_new_thread)),
                id="third-stop-as-the-cleanup-thread-starts",
            ),
        ],
    )
    def test_stop_as_git_starts_ends_git_and_its_group(
        self, tmp_path: Path, event: str, function: Any, alias: str, again: Any
    ) -> None:
        """A Stop as git starts, or as its timeout is handled, ends git, its group and its pipes.

        S5-F3: the spawn stood before the ``try`` whose handler stops git: the
        waiting git (real git, on a shell alias) and its ``sleep`` outlived
        the interrupted call.  The handler then left git's three pipes open
        for as long as the caller kept the error.  A Stop raised before the
        spawn has no git to end.  One raised while the timeout was handled
        missed that handler: git ran on (before the ``SIGTERM``), two pipes
        stayed open (after it), and a member of the group that outlived git
        was spared (as ``ps`` lists the group: git has exited, unreaped).
        L11-F7: a second Stop (*again*) raised inside that cleanup ended it
        early: before its ``SIGTERM`` git ran on, during the grace the
        member was spared and git left unreaped, after it the pipes stayed
        open.  (In the grace case the trace hook fires first, in the timeout
        handler, and the profile hook inside the cleanup, at its ``ps``.)
        A third one, raised as the call that hands the cleanup to a thread
        returns, must not end the wait for that thread.
        Not exercised here: the cleanup on Windows, which leaves the pipes
        to ``communicate`` once that has begun, a close that fails (a
        descriptor closed behind its file object), and two branches no real
        process reaches: a git that outlives its ``SIGKILL`` (L01-F3: the
        wait for it had no bound) and a cleanup thread that cannot start.
        """
        _git_stopped_by(_stop_before_spawn, tmp_path)
        hook = _StopAt(event, function)
        _git_stopped_by(hook, tmp_path, alias, again and _StopAt(*again))
        git = hook.git
        assert git is not None and git.stdout and git.stderr
        pipes = (git.stdout, git.stderr)  # its stdin is no pipe
        try:
            assert git.returncode is not None, "the interrupted call left git running"
            assert [pipe.closed for pipe in pipes] == [True] * 2, "git's pipes were left open"
            deadline = time.monotonic() + 10
            while _running_in_group(git.pid):  # a killed straggler takes a moment to go
                assert time.monotonic() < deadline, "git's group outlived the interrupted call"
                time.sleep(0.05)
        finally:  # what the product left behind is not the next test's
            if git.returncode is None or _running_in_group(git.pid):
                kill_process_tree(git)
                git.wait(timeout=10)
            for pipe in pipes:
                pipe.close()

    @posix_only("the process group is listed with ps")
    @pytest.mark.parametrize("given", [None, "a commit message"])
    def test_stop_reaches_a_git_that_writes_nothing(
        self, tmp_path: Path, given: str | None,
    ) -> None:
        """A Stop injected while git runs silently is raised at once, and ends git.

        ``_git`` waited for git in one ``communicate``: the thread sat in its
        ``poll`` and the injected exception was raised only when git wrote,
        exited (here after 20 s) or timed out (300 s).  A git that was *given*
        its stdin, as the ``commit -F -`` that runs the hooks is, still was.
        The file git reads it from is closed by the cleanup that ends git.  (Where
        that matters, no test reaches: on Windows the file's ``close`` is Python
        code a Stop can cut, here it is one C call.)
        """
        seen: dict[str, Any] = {}

        def task() -> None:
            start = time.monotonic()
            try:
                _git("-c", "alias.wait=!sleep 20", "wait", cwd=tmp_path, input=given)
            except KeyboardInterrupt as stop:
                seen["seconds"] = time.monotonic() - start
                held = next(
                    frame.f_locals for frame, _ in traceback.walk_tb(stop.__traceback__)
                    if frame.f_code is _git.__code__
                )
                seen["git"], seen["stdin"] = held["proc"], held["source"]

        thread = threading.Thread(target=task)
        thread.start()
        time.sleep(1.0)  # git runs, the thread waits for it
        ident = thread.ident
        assert ident is not None
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(ident), ctypes.py_object(KeyboardInterrupt),
        )
        thread.join(60)
        assert seen, "the Stop never came out of _git"
        assert seen["seconds"] < 5, f"the Stop waited {seen['seconds']:.1f}s for git"
        assert seen["git"].returncode is not None, "the interrupted call left git running"
        assert not _running_in_group(seen["git"].pid)
        assert (seen["stdin"] is None) == (given is None)
        assert given is None or seen["stdin"].closed, "git's stdin file was left open"

    @posix_only("the reader is a shell alias")
    def test_stdin_larger_than_a_pipe_reaches_a_slow_reader_whole(self, tmp_path: Path) -> None:
        """Input that outlasts several slices of the wait is written once, to its end."""
        result = _git(
            "-c", "alias.count=!sleep 0.5; wc -c", "count", cwd=tmp_path, input="x" * 2_000_000,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "2000000"

    def test_real_git_still_works_within_timeout(self, tmp_path: Path) -> None:
        """The normal (non-hanging) path is unchanged by the timeout."""
        repo = tmp_path / "repo"
        repo.mkdir()
        assert _git("init", cwd=repo).returncode == 0
        result = _git("status", "--porcelain", cwd=repo)
        assert result.returncode == 0
        assert result.stdout == ""
