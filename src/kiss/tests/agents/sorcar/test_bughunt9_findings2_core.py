# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 9 (findings-2 audit): persistence, git_worktree, skills,
custom_commands, cli_helpers regressions.

Covers, end-to-end with real files/processes/DBs (no mocks/patches):

* S2-01 — ``_stop_event_writer`` must never strand late-enqueued events.
* S2-03 — prefix autocomplete must return distinct older matches even
  when many newer duplicates exist.
* S2-04 — ``has_uncommitted_changes`` must treat a failing ``git
  status`` as dirty (never report clean on error).
* S2-05 — ``copy_dirty_state`` must raise on a failing ``git status``
  instead of silently omitting the user's dirty state.
* S2-30 — ``copy_dirty_state`` must mirror dirty submodule content.
* S2-07 — one invalid-UTF-8 SKILL.md/command file must not abort
  discovery, and activation reports a readable error.
* S2-08 — a timed-out shell injection must kill descendant processes.
* S2-09 — binary shell-injection output must not raise.
* S2-29 — an invalid-UTF-8 ``@{file}`` injects the error marker.
* S2-22 — ``--max_budget`` rejects nan/inf/zero/negative values.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

import kiss.agents.sorcar.custom_commands as cc
import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.cli_helpers import _parse_budget_value
from kiss.agents.sorcar.custom_commands import CustomCommand, expand_command
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.persistence import _add_task, _prefix_match_tasks
from kiss.agents.sorcar.skills import parse_frontmatter


def _run_git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
    )


def _make_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _run_git("init", "-q", cwd=repo)
    _run_git("config", "user.email", "t@t", cwd=repo)
    _run_git("config", "user.name", "t", cwd=repo)
    (repo / "a.txt").write_text("hello\n")
    _run_git("add", "-A", cwd=repo)
    _run_git("commit", "-q", "-m", "init", cwd=repo)
    return repo


def _install_failing_status_git(shim_dir: Path) -> dict[str, str]:
    """Put a ``git`` shim on PATH whose ``status`` fails (rc 124, no output).

    Every other subcommand is delegated to the real git, so the repo
    behaves normally except for the synthesized status failure — the
    exact shape ``_git`` produces for a timed-out command.
    """
    shim_dir.mkdir(parents=True, exist_ok=True)
    real_git = shutil.which("git")
    shim = shim_dir / "git"
    shim.write_text(
        "#!/bin/sh\n"
        "for arg in \"$@\"; do\n"
        "  if [ \"$arg\" = status ]; then exit 124; fi\n"
        "done\n"
        f"exec {real_git} \"$@\"\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ)
    env["PATH"] = f"{shim_dir}:{env['PATH']}"
    return env


class TestGitStatusFailureIsNeverClean:
    """S2-04 / S2-05: a failing ``git status`` must never look clean."""

    def setup_method(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = _make_repo(self.tmp)
        (self.repo / "dirty.txt").write_text("uncommitted work\n")
        self.saved_path = os.environ["PATH"]
        env = _install_failing_status_git(self.tmp / "bin")
        os.environ["PATH"] = env["PATH"]

    def teardown_method(self) -> None:
        os.environ["PATH"] = self.saved_path
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_has_uncommitted_changes_reports_dirty_on_status_failure(self) -> None:
        # With the shim, `git status` exits 124 with empty stdout.  The
        # old code returned False (clean) and the caller then removed
        # the worktree, losing the uncommitted file.
        assert GitWorktreeOps.has_uncommitted_changes(self.repo) is True

    def test_copy_dirty_state_raises_on_status_failure(self) -> None:
        wt = self.tmp / "wt"
        wt.mkdir()
        with pytest.raises(OSError):
            GitWorktreeOps.copy_dirty_state(self.repo, wt)


class TestCopyDirtySubmodule:
    """S2-30: dirty submodule content must be mirrored into the worktree."""

    def setup_method(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dirty_submodule_file_is_copied(self) -> None:
        sub = _make_repo(self.tmp)  # creates self.tmp / "repo"
        sub = sub.rename(self.tmp / "sub")
        repo = _make_repo(self.tmp)
        _run_git(
            "-c", "protocol.file.allow=always",
            "submodule", "add", str(sub), "mod", cwd=repo,
        )
        _run_git("commit", "-q", "-m", "add submodule", cwd=repo)
        # Dirty a tracked file inside the submodule working tree.
        (repo / "mod" / "a.txt").write_text("submodule dirty content\n")

        wt = self.tmp / "wt"
        (wt / "mod").mkdir(parents=True)
        copied = GitWorktreeOps.copy_dirty_state(repo, wt)

        assert copied is True
        assert (wt / "mod" / "a.txt").read_text() == "submodule dirty content\n"
        assert not (wt / "mod" / ".git").exists()

    def test_untracked_embedded_repo_is_not_mirrored(self) -> None:
        """``?? dir/`` entries (foreign repos/worktrees under the repo)
        must NOT be treated as dirty submodules: mirroring an agent
        worktree located inside the repo into itself recurses until
        "File name too long"."""
        repo = _make_repo(self.tmp)
        embedded = repo / ".kiss-worktrees" / "kiss_wt-test-1"
        embedded.mkdir(parents=True)
        _run_git("init", "-q", cwd=embedded)
        (embedded / "inner.txt").write_text("agent worktree file\n")

        wt = self.tmp / "wt"
        wt.mkdir()
        copied = GitWorktreeOps.copy_dirty_state(repo, wt)

        assert copied is False, "clean repo (plus embedded repo) is not dirty"
        assert not (wt / ".kiss-worktrees").exists(), (
            "untracked embedded repo was mirrored into the new worktree"
        )


class TestUnicodeRobustness:
    """S2-07 / S2-09 / S2-29: invalid UTF-8 must not abort parsing."""

    def setup_method(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parse_frontmatter_skips_invalid_utf8_file(self) -> None:
        bad = self.tmp / "SKILL.md"
        bad.write_bytes(b"---\nname: x\n---\n\xff\xfe body")
        assert parse_frontmatter(bad) is None  # skipped, not raised

    def test_shell_injection_binary_output_does_not_raise(self) -> None:
        cmd = CustomCommand(
            name="c", description="", argument_hint="",
            template="out: !`printf 'a\\377b'`", source="user", path="c.md",
        )
        expanded = expand_command(cmd, "", str(self.tmp))
        assert expanded.startswith("out: a")
        assert expanded.endswith("b")

    def test_file_injection_invalid_utf8_injects_marker(self) -> None:
        (self.tmp / "blob.bin").write_bytes(b"\xff\xfe\x00")
        cmd = CustomCommand(
            name="c", description="", argument_hint="",
            template="data: @{blob.bin}", source="user", path="c.md",
        )
        expanded = expand_command(cmd, "", str(self.tmp))
        assert expanded == "data: [could not read file: blob.bin]"


class TestShellInjectionTimeoutKillsChildren:
    """S2-08: a timed-out injection must not orphan descendant processes."""

    def setup_method(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.saved_timeout = cc._SHELL_TIMEOUT_SECONDS
        cc._SHELL_TIMEOUT_SECONDS = 0.3

    def teardown_method(self) -> None:
        cc._SHELL_TIMEOUT_SECONDS = self.saved_timeout
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_background_child_is_killed_on_timeout(self) -> None:
        pid_file = self.tmp / "child.pid"
        cmd = CustomCommand(
            name="c", description="", argument_hint="",
            template=f"!`sleep 30 & echo $! > {pid_file}; wait`",
            source="user", path="c.md",
        )
        expanded = expand_command(cmd, "", str(self.tmp))
        assert "[shell command timed out:" in expanded
        child_pid = int(pid_file.read_text().strip())
        # Give the SIGKILL a moment to be delivered and reaped.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            os.kill(child_pid, 9)
            pytest.fail(f"background child {child_pid} survived the timeout")


class TestBudgetValidation:
    """S2-22: nan/inf/zero/negative budgets must be rejected."""

    def test_rejects_non_finite_and_non_positive(self) -> None:
        import argparse

        for bad in ("nan", "inf", "-inf", "0", "-3", "NaN"):
            with pytest.raises(argparse.ArgumentTypeError):
                _parse_budget_value(bad)

    def test_accepts_positive_finite(self) -> None:
        assert _parse_budget_value("12.5") == 12.5


class _TempDbTestBase:
    """Fresh temp SQLite DB per test, fully restored after."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        th._invalidate_chat_context_cache("")

    def teardown_method(self) -> None:
        th._close_db()
        th._invalidate_chat_context_cache("")
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestPrefixMatchDistinct(_TempDbTestBase):
    """S2-03: duplicates must not crowd out older distinct matches."""

    def test_distinct_matches_beyond_duplicate_window(self) -> None:
        for i in range(7):
            _add_task(f"fix bug variant {i}")
            time.sleep(0.002)
        for _ in range(40):
            _add_task("fix bug again")
            time.sleep(0.001)

        out = _prefix_match_tasks("fix bug", limit=8)

        assert len(out) == 8
        assert len(set(out)) == 8
        assert out[0] == "fix bug again"
        assert {f"fix bug variant {i}" for i in range(7)} <= set(out)


class TestStopEventWriterDrainsLateEnqueues(_TempDbTestBase):
    """S2-01: events enqueued during stop must still be persisted."""

    def test_no_stranded_events_after_stop(self) -> None:
        task_id, _ = _add_task("event target")
        n_producers = 4
        events_per_producer = 25
        start = threading.Event()

        def produce(worker: int) -> None:
            start.wait()
            for i in range(events_per_producer):
                time.sleep(0.001)
                th._queue_chat_event(
                    {"type": "test", "worker": worker, "i": i}, task_id,
                )

        producers = [
            threading.Thread(target=produce, args=(w,))
            for w in range(n_producers)
        ]
        for p in producers:
            p.start()
        start.set()
        time.sleep(0.01)  # let some events flow before stopping
        th._stop_event_writer()
        for p in producers:
            p.join()
        # Producers may have enqueued after the first stop completed.
        th._stop_event_writer()

        assert th._event_queue.unfinished_tasks == 0, (
            "stop returned with unfinished queue items (stranded events)"
        )
        db = th._get_db()
        row = db.execute(
            "SELECT COUNT(*) AS n FROM events WHERE task_id = ?", (task_id,),
        ).fetchone()
        assert row["n"] == n_producers * events_per_producer
