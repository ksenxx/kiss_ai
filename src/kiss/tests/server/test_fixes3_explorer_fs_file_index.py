# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the cluster-3 server fixes (audit G1, G2, G3, G8).

* ``explorer.git_action`` mutates the main tree's HEAD/index, so it
  must queue behind :func:`kiss.agents.sorcar.git_worktree.repo_lock`
  like every other main-tree mutation (auto-commit, the worktree merge
  transaction) instead of interleaving with them.
* ``fs_actions._run_capped`` must enforce its timeout even when a
  grandchild of the command inherited the stdout pipe: the whole
  process group is killed, and ``compare_files`` disables
  ``diff.external`` / ``GIT_EXTERNAL_DIFF`` so no such grandchild is
  spawned in the first place.
* The file-index worker must survive a root whose name is not valid
  UTF-8 (``str.encode`` raised on the surrogate-escaped path; the
  exception escaped ``_build`` and killed the only worker thread), and
  ``_save_listings`` must not leak its temp file when the final
  ``os.replace`` fails.

Everything is real: temp git repositories, real threads, real
subprocesses, no mocks.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.git_worktree import repo_lock
from kiss.server import fs_actions
from kiss.server.explorer import git_action
from kiss.server.file_index import FileIndexRegistry, _save_listings
from kiss.tests.conftest import posix_only


def _init_repo(path: Path) -> str:
    """Create a repository with one commit at *path*; return its HEAD sha."""
    path.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    (path / "a.txt").write_text("a\n")
    subprocess.run(["git", "add", "a.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True, env=env)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True,
    )
    return head.stdout.strip()


class TestGitActionTakesRepoLock:
    """G1: the commit-graph actions serialise with the merge transaction."""

    def test_git_action_blocks_while_repo_lock_is_held(self, tmp_path: Path) -> None:
        """A ``createTag`` waits for ``repo_lock`` and runs once it is released."""
        repo = tmp_path / "repo"
        sha = _init_repo(repo)
        lock = repo_lock(repo)
        holding = threading.Event()
        release = threading.Event()

        def hold_lock() -> None:
            with lock:
                holding.set()
                release.wait(timeout=30)

        holder = threading.Thread(target=hold_lock, daemon=True)
        holder.start()
        assert holding.wait(timeout=10)

        results: list[dict[str, object]] = []
        actor = threading.Thread(
            target=lambda: results.append(
                git_action(str(repo), "createTag", sha, name="locked-tag"),
            ),
            daemon=True,
        )
        actor.start()
        actor.join(timeout=1.0)
        try:
            assert actor.is_alive(), (
                f"git_action completed while repo_lock was held: {results}"
            )
            assert not results
            tags = subprocess.run(
                ["git", "tag", "--list"], cwd=repo, capture_output=True, text=True,
            ).stdout
            assert "locked-tag" not in tags, "the tag was created under another holder's lock"
        finally:
            release.set()
        actor.join(timeout=30)
        holder.join(timeout=5)
        assert not actor.is_alive(), "git_action never ran after repo_lock was released"
        assert results and results[0].get("ok") is True, results
        tags = subprocess.run(
            ["git", "tag", "--list"], cwd=repo, capture_output=True, text=True,
        ).stdout
        assert "locked-tag" in tags

    def test_git_action_without_contention_still_works(self, tmp_path: Path) -> None:
        """The lock is uncontended in the common case: the action just runs."""
        repo = tmp_path / "repo"
        sha = _init_repo(repo)
        result = git_action(str(repo), "checkoutDetached", sha)
        assert result.get("ok") is True, result
        head = subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"], cwd=repo, capture_output=True, text=True,
        )
        assert head.returncode != 0, "HEAD should be detached"


class TestRunCappedKillsProcessGroup:
    """G2/G9: a timed-out command cannot wedge the caller through a grandchild."""

    @posix_only("the grandchild-spawning command is a /bin/sh script")
    def test_timeout_kills_grandchild_holding_the_pipe(self, tmp_path: Path) -> None:
        """A grandchild that inherited stdout dies with the timed-out command."""
        marker = tmp_path / "grandchild-survived"
        cmd = [
            "sh", "-c",
            f"sh -c 'sleep 2; touch \"{marker}\"' & sleep 30",
        ]
        start = time.monotonic()
        out = fs_actions._run_capped(cmd, None, 1 << 20, 0.5)
        elapsed = time.monotonic() - start
        time.sleep(3)
        assert out.timed_out is True
        assert elapsed < 2.0, (
            f"_run_capped blocked for {elapsed:.1f}s past a 0.5s cap: the grandchild "
            "kept the stdout pipe open"
        )
        assert not marker.exists(), "the grandchild outlived the timeout"

    @posix_only("the grandchild-spawning command is a /bin/sh script")
    def test_timeout_applies_after_the_leader_exited(self, tmp_path: Path) -> None:
        """A descendant holding stdout after ``exit 0`` is still killed at the cap."""
        marker = tmp_path / "orphan-survived"
        cmd = ["sh", "-c", f"sh -c 'sleep 2; touch \"{marker}\"' & exit 0"]
        start = time.monotonic()
        out = fs_actions._run_capped(cmd, None, 1 << 20, 0.3)
        elapsed = time.monotonic() - start
        time.sleep(3)
        assert out.timed_out is True
        assert elapsed < 1.5, f"_run_capped waited {elapsed:.1f}s for an orphan to exit"
        assert not marker.exists(), "the orphan outlived the timeout"

    @posix_only("the grandchild-spawning command is a /bin/sh script")
    def test_stray_stderr_holder_is_reaped_after_completion(self, tmp_path: Path) -> None:
        """A finished command whose orphan only holds stderr returns promptly."""
        marker = tmp_path / "stderr-holder-survived"
        cmd = [
            "sh", "-c",
            f"sh -c 'sleep 4; touch \"{marker}\"' >/dev/null & echo hi; exit 0",
        ]
        start = time.monotonic()
        out = fs_actions._run_capped(cmd, None, 1 << 20, 30)
        elapsed = time.monotonic() - start
        time.sleep(5)
        assert out.timed_out is False
        assert out.returncode == 0
        assert out.stdout == b"hi\n"
        assert elapsed < 3.5, f"_run_capped waited {elapsed:.1f}s on a stray stderr holder"
        assert not marker.exists(), "the stray stderr holder was not reaped"

    def test_fast_command_is_not_stamped_timed_out(self) -> None:
        """A command that finishes within the cap reports ``timed_out=False``."""
        out = fs_actions._run_capped(["git", "--version"], None, 1 << 20, 5.0)
        assert out.timed_out is False
        assert out.returncode == 0
        assert out.stdout.startswith(b"git version")

    def test_deadline_outlives_stdout_eof(self) -> None:
        """A process that closes stdout and runs past the cap is a timeout.

        Regression: ``finished`` was set at stdout EOF, which disarmed
        the timer while the process was still alive; a fixed 5 s wait
        then replaced the caller's deadline and reported success.
        """
        cmd = [sys.executable, "-c", "import os, time; os.close(1); time.sleep(10)"]
        start = time.monotonic()
        out = fs_actions._run_capped(cmd, None, 1024, 0.3)
        elapsed = time.monotonic() - start
        assert out.timed_out is True
        assert out.returncode != 0
        assert elapsed < 2.0, f"_run_capped returned after {elapsed:.1f}s on a 0.3s cap"

    def test_exit_before_deadline_after_stdout_eof_is_not_a_timeout(self) -> None:
        """A process that closes stdout, then exits within the cap, is complete.

        The deadline, not a fixed wait, bounds the time after EOF: the
        process gets its whole allowance and is reported as exited.
        """
        cmd = [sys.executable, "-c", "import os, time; os.close(1); time.sleep(0.5)"]
        start = time.monotonic()
        out = fs_actions._run_capped(cmd, None, 1024, 10.0)
        elapsed = time.monotonic() - start
        assert out.timed_out is False
        assert out.returncode == 0
        assert 0.4 < elapsed < 5.0, f"_run_capped took {elapsed:.1f}s"

    @posix_only("the external diff stub is a /bin/sh script")
    def test_compare_files_ignores_a_slow_external_diff(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``compare_files`` returns a real unified diff despite GIT_EXTERNAL_DIFF."""
        stub = tmp_path / "slow-diff"
        stub.write_text("#!/bin/sh\nsleep 40\n", encoding="utf-8")
        stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        monkeypatch.setenv("GIT_EXTERNAL_DIFF", str(stub))
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("one\n")
        b.write_text("two\n")
        start = time.monotonic()
        result = fs_actions.compare_files(str(a), str(b))
        elapsed = time.monotonic() - start
        assert elapsed < 10, f"compare_files took {elapsed:.1f}s: the external diff ran"
        assert result.get("ok") is True, result
        assert "-one" in result["text"] and "+two" in result["text"], result["text"]


class TestFileIndexWorkerSurvives:
    """G3/G8: one bad root or one failed save must not disable the picker."""

    @posix_only("a non-UTF-8 directory name needs a bytes path on a POSIX filesystem")
    def test_non_utf8_root_is_indexed_and_the_worker_lives_on(self, tmp_path: Path) -> None:
        """A surrogate-escaped root builds, persists, and later roots still build."""
        raw = os.fsencode(str(tmp_path)) + b"/bad-\xff-root"
        os.mkdir(raw)
        root = os.fsdecode(raw)
        Path(root, "x.py").write_text("x\n")
        other = tmp_path / "other"
        other.mkdir()
        (other / "y.py").write_text("y\n")
        reg = FileIndexRegistry(home=str(tmp_path / "nohome"), cache_dir=tmp_path / "cache")
        try:
            # Cache path must be computable for the undecodable name.
            cache_file = reg._cache_path(root)
            for _ in range(2):  # first scan, then a refresh that relists
                done = threading.Event()
                assert reg.ensure(root, on_ready=done.set)
                assert done.wait(timeout=30), "build of the non-UTF-8 root never finished"
            assert cache_file.exists(), "listings of the non-UTF-8 root were not persisted"
            assert reg._worker is not None and reg._worker.is_alive(), "worker died"
            done = threading.Event()
            assert reg.ensure(str(other), on_ready=done.set)
            assert done.wait(timeout=30), "the worker no longer serves builds"
            picker = reg.picker_for(str(other))
            assert picker is not None
        finally:
            reg.stop()

    def test_save_listings_removes_its_temp_file_when_replace_fails(
        self, tmp_path: Path,
    ) -> None:
        """A failing ``os.replace`` (target is a directory) leaves no ``.tmp`` behind."""
        target = tmp_path / "cache" / "abc.json"
        target.mkdir(parents=True)  # os.replace(file -> existing dir) raises OSError
        _save_listings(target, "/some/root", {"": (0, ["a.py"], [])})
        leftovers = [p.name for p in target.parent.iterdir() if p.name != target.name]
        assert leftovers == [], f"temp files leaked: {leftovers}"
