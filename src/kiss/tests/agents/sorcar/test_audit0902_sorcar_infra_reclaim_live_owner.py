# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cross-process reclaim must never destroy a LIVE process's worktree.

Race reproduced here (audit 2026-09-02, sorcar-infra):

``GitWorktreeOps.reclaim_orphaned_worktrees`` runs in every Sorcar
process (a ``kiss-web`` daemon and a ``kiss`` CLI run share the same
repository).  It protects live worktrees of *other* processes only via
the ``branch.<name>.kiss-owner-pid`` config — but

* ``GitWorktreeOps.create`` never recorded that owner pid: it was
  written later by ``ChatSorcarAgent._try_setup_worktree``, so a
  worktree that another process had just created (and was about to
  run a task in) looked like a legacy orphan and was squash-merged
  (a no-op) and DELETED under its owner;
* the owner check must run BEFORE every destructive branch, so a
  worktree whose recorded owner is another LIVE process is skipped
  until that process exits (and a recycled pid is not mistaken for
  the owner).

Every test drives real git repositories and a real second Python
process; nothing is mocked.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from kiss.agents.sorcar.git_worktree import (
    _WORKTREE_SUBDIR,
    GitWorktreeOps,
)
from kiss.core.processes import pid_alive, process_identity


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)], capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        capture_output=True, check=True,
    )
    (path / "README.md").write_text("# repo\n")
    subprocess.run(["git", "-C", str(path), "add", "."], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True, check=True,
    )
    return path


def _run_in_other_process(code: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh interpreter (a second Sorcar process)."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, check=True, cwd=str(cwd), timeout=120,
    )


_RECLAIM_IN_OTHER_PROCESS = """
    from pathlib import Path
    from kiss.agents.sorcar.git_worktree import GitWorktreeOps
    print(GitWorktreeOps.reclaim_orphaned_worktrees(Path({repo!r})))
"""


class TestCreateRecordsOwner:
    def setup_method(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmp) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_stamps_this_process_as_owner(self) -> None:
        branch = GitWorktreeOps.new_task_branch(self.repo)
        wt_dir = self.repo / _WORKTREE_SUBDIR / branch.replace("/", "_")
        assert GitWorktreeOps.create(self.repo, branch, wt_dir)
        assert GitWorktreeOps.load_owner_pid(self.repo, branch) == os.getpid()

    def test_failed_create_stamps_nothing(self) -> None:
        """``git worktree add`` refusing (branch already exists) returns
        False and leaves no owner-pid config behind for the branch."""
        assert not GitWorktreeOps.create(
            self.repo, "main", self.repo / _WORKTREE_SUBDIR / "dup",
        )
        assert GitWorktreeOps.load_owner_pid(self.repo, "main") is None
        assert not (self.repo / _WORKTREE_SUBDIR / "dup").exists()

    def test_other_process_does_not_reclaim_fresh_worktree(self) -> None:
        """The window between ``create`` and the agent's own owner-pid
        write must not let a second process merge-and-delete the
        worktree the first process is about to run a task in."""
        branch = GitWorktreeOps.new_task_branch(self.repo)
        wt_dir = self.repo / _WORKTREE_SUBDIR / branch.replace("/", "_")
        assert GitWorktreeOps.create(self.repo, branch, wt_dir)

        proc = _run_in_other_process(
            _RECLAIM_IN_OTHER_PROCESS.format(repo=str(self.repo)), self.repo,
        )

        assert proc.stdout.strip() == "0", proc.stdout + proc.stderr
        assert wt_dir.is_dir()
        assert GitWorktreeOps.branch_exists(self.repo, branch)

    def test_create_waits_for_another_process_reclaim(self) -> None:
        """``create`` runs under the cross-process reclaim lock, so it can
        never interleave with a reclaim sweep that is already enumerating
        worktrees in another process."""
        hold_s = 1.5
        holder = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(f"""
                import sys, time
                from pathlib import Path
                from kiss.agents.sorcar.git_worktree import _reclaim_process_lock
                with _reclaim_process_lock(Path({str(self.repo)!r})):
                    print("held", flush=True)
                    time.sleep({hold_s})
            """)],
            stdout=subprocess.PIPE, text=True, cwd=str(self.repo),
        )
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline().strip() == "held"
            started = time.monotonic()
            branch = GitWorktreeOps.new_task_branch(self.repo)
            wt_dir = self.repo / _WORKTREE_SUBDIR / branch.replace("/", "_")
            assert GitWorktreeOps.create(self.repo, branch, wt_dir)
            elapsed = time.monotonic() - started
        finally:
            holder.wait(timeout=30)
        assert elapsed >= hold_s * 0.8, elapsed


class TestOwnerLiveness:
    def setup_method(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmp) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _create_worktree(self) -> tuple[str, Path]:
        branch = GitWorktreeOps.new_task_branch(self.repo)
        wt_dir = self.repo / _WORKTREE_SUBDIR / branch.replace("/", "_")
        assert GitWorktreeOps.create(self.repo, branch, wt_dir)
        return branch, wt_dir

    def test_other_process_keeps_live_worktree(self) -> None:
        """A worktree owned by THIS live process survives another
        process's reclaim pass."""
        branch, wt_dir = self._create_worktree()
        assert GitWorktreeOps.load_owner_pid(self.repo, branch) == os.getpid()

        proc = _run_in_other_process(
            _RECLAIM_IN_OTHER_PROCESS.format(repo=str(self.repo)), self.repo,
        )

        assert proc.stdout.strip() == "0", proc.stdout + proc.stderr
        assert wt_dir.is_dir()
        assert GitWorktreeOps.branch_exists(self.repo, branch)

    def test_worktree_of_dead_process_is_reclaimed(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A worktree whose owning process has exited is an orphan and
        is reclaimed by the next pass."""
        proc = _run_in_other_process(
            f"""
            from pathlib import Path
            from kiss.agents.sorcar.git_worktree import _WORKTREE_SUBDIR, GitWorktreeOps
            repo = Path({str(self.repo)!r})
            branch = GitWorktreeOps.new_task_branch(repo)
            wt_dir = repo / _WORKTREE_SUBDIR / branch.replace("/", "_")
            assert GitWorktreeOps.create(repo, branch, wt_dir)
            print(branch)
            print(wt_dir)
            """,
            self.repo,
        )
        branch, wt_dir_s = proc.stdout.strip().splitlines()[-2:]
        wt_dir = Path(wt_dir_s)
        assert wt_dir.is_dir()
        dead_pid = GitWorktreeOps.load_owner_pid(self.repo, branch)
        assert dead_pid is not None and dead_pid != os.getpid()
        assert not pid_alive(dead_pid)

        with caplog.at_level(logging.INFO, logger="kiss.agents.sorcar.git_worktree"):
            reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)
        assert reclaimed == 1, (
            f"reclaim skipped the dead owner's worktree; main status="
            f"{GitWorktreeOps.status_porcelain(self.repo)!r} worktree status="
            f"{GitWorktreeOps.status_porcelain(wt_dir)!r} log="
            f"{[r.getMessage() for r in caplog.records]}"
        )
        assert not wt_dir.exists()
        assert not GitWorktreeOps.branch_exists(self.repo, branch)

    def test_worktree_owned_by_live_foreign_pid_is_skipped_until_it_dies(self) -> None:
        """In-process view of the same rule: a worktree whose recorded
        owner is another LIVE process is skipped; once that process
        exits the very same worktree is reclaimed."""
        branch, wt_dir = self._create_worktree()
        owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            subprocess.run(
                ["git", "-C", str(self.repo), "config",
                 f"branch.{branch}.kiss-owner-pid", str(owner.pid)],
                capture_output=True, check=True,
            )
            assert GitWorktreeOps.reclaim_orphaned_worktrees(self.repo) == 0
            assert wt_dir.is_dir()
            assert GitWorktreeOps.branch_exists(self.repo, branch)
        finally:
            owner.kill()
            owner.wait(timeout=30)
        assert GitWorktreeOps.reclaim_orphaned_worktrees(self.repo) == 1
        assert not wt_dir.exists()
        assert not GitWorktreeOps.branch_exists(self.repo, branch)

    def test_recycled_owner_pid_is_not_mistaken_for_the_owner(self) -> None:
        """A live pid whose process is NOT the stamping process is a
        recycled pid: the worktree's owner is dead and it is reclaimed.

        The OS recycling of a dead owner's pid cannot be forced, so the
        stamp is built the way a recycled pid presents itself: the pid of
        a live stranger, with the identity of the real (other) owner.
        """
        branch, wt_dir = self._create_worktree()
        stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            real_owner_identity = process_identity(os.getpid())
            assert real_owner_identity is not None
            assert real_owner_identity != process_identity(stranger.pid)
            for key, value in (
                ("kiss-owner-pid", str(stranger.pid)),
                ("kiss-owner-identity", f"{stranger.pid}:{real_owner_identity}"),
            ):
                subprocess.run(
                    ["git", "-C", str(self.repo), "config",
                     f"branch.{branch}.{key}", value],
                    capture_output=True, check=True,
                )
            assert GitWorktreeOps.reclaim_orphaned_worktrees(self.repo) == 1
            assert not wt_dir.exists()
            assert not GitWorktreeOps.branch_exists(self.repo, branch)
        finally:
            stranger.kill()
            stranger.wait(timeout=30)

    def test_matching_owner_identity_keeps_the_worktree(self) -> None:
        """The stamp written by ``save_owner_pid`` (pid + identity of the
        live stamping process) protects the worktree while that process
        runs."""
        branch, wt_dir = self._create_worktree()
        owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            identity = process_identity(owner.pid)
            assert identity is not None
            for key, value in (
                ("kiss-owner-pid", str(owner.pid)),
                ("kiss-owner-identity", f"{owner.pid}:{identity}"),
            ):
                subprocess.run(
                    ["git", "-C", str(self.repo), "config",
                     f"branch.{branch}.{key}", value],
                    capture_output=True, check=True,
                )
            assert GitWorktreeOps.reclaim_orphaned_worktrees(self.repo) == 0
            assert wt_dir.is_dir()
        finally:
            owner.kill()
            owner.wait(timeout=30)
        assert GitWorktreeOps.reclaim_orphaned_worktrees(self.repo) == 1

    def test_own_worktree_without_exclusion_is_still_reclaimed(self) -> None:
        """Within one process the exclusion set is authoritative: a
        worktree this process created but no longer tracks (agent state
        dropped) is reclaimed by its own reclaim pass."""
        branch, wt_dir = self._create_worktree()
        assert GitWorktreeOps.reclaim_orphaned_worktrees(self.repo) == 1
        assert not wt_dir.exists()
        assert not GitWorktreeOps.branch_exists(self.repo, branch)
