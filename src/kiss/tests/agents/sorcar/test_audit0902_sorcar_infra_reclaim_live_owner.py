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
  written later by ``WorktreeSorcarAgent._try_setup_worktree``, so a
  worktree that another process had just created (and was about to
  run a task in) looked like a legacy orphan and was squash-merged
  (a no-op) and DELETED under its owner;
* pooled spares (``worktree_pool.prewarm``) never got an owner pid at
  all, and the reclaim's spare-marker branch ran BEFORE the owner
  check, so another process discarded a spare that this process's
  pool was still holding and about to hand to a task.

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

from kiss.agents.sorcar import worktree_pool
from kiss.agents.sorcar.git_worktree import (
    _WORKTREE_SUBDIR,
    GitWorktreeOps,
)


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
        branch = worktree_pool.new_task_branch(self.repo)
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
        branch = worktree_pool.new_task_branch(self.repo)
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
            branch = worktree_pool.new_task_branch(self.repo)
            wt_dir = self.repo / _WORKTREE_SUBDIR / branch.replace("/", "_")
            assert GitWorktreeOps.create(self.repo, branch, wt_dir)
            elapsed = time.monotonic() - started
        finally:
            holder.wait(timeout=30)
        assert elapsed >= hold_s * 0.8, elapsed


class TestSpareOwnerLiveness:
    def setup_method(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.repo = _make_repo(Path(self.tmp) / "repo")
        worktree_pool.discard_all()

    def teardown_method(self) -> None:
        worktree_pool.discard_all()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_other_process_keeps_live_spare(self) -> None:
        """A spare held by THIS process's pool survives another
        process's reclaim pass."""
        assert worktree_pool.prewarm(self.repo)
        spare_branch, spare_dir = worktree_pool._spares[
            worktree_pool._repo_key(self.repo)
        ]
        assert GitWorktreeOps.load_owner_pid(self.repo, spare_branch) == os.getpid()

        proc = _run_in_other_process(
            _RECLAIM_IN_OTHER_PROCESS.format(repo=str(self.repo)), self.repo,
        )

        assert proc.stdout.strip() == "0", proc.stdout + proc.stderr
        assert spare_dir.is_dir()
        assert GitWorktreeOps.branch_exists(self.repo, spare_branch)
        assert worktree_pool.take_spare(self.repo) == (spare_branch, spare_dir)

    def test_spare_of_dead_process_is_discarded(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A spare whose owning process has exited is plumbing debris and
        is still discarded by the next reclaim pass."""
        proc = _run_in_other_process(
            f"""
            from pathlib import Path
            from kiss.agents.sorcar import worktree_pool
            assert worktree_pool.prewarm(Path({str(self.repo)!r}))
            branch, wt_dir = worktree_pool._spares[
                worktree_pool._repo_key(Path({str(self.repo)!r}))
            ]
            print(branch)
            print(wt_dir)
            """,
            self.repo,
        )
        branch, wt_dir_s = proc.stdout.strip().splitlines()[-2:]
        wt_dir = Path(wt_dir_s)
        assert wt_dir.is_dir()
        assert GitWorktreeOps.load_spare_marker(self.repo, branch)
        dead_pid = GitWorktreeOps.load_owner_pid(self.repo, branch)
        assert dead_pid is not None and dead_pid != os.getpid()
        assert not GitWorktreeOps._pid_alive(dead_pid)

        with caplog.at_level(logging.INFO, logger="kiss.agents.sorcar.git_worktree"):
            reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)
        assert reclaimed == 1, (
            f"reclaim skipped the dead owner's spare; main status="
            f"{GitWorktreeOps.status_porcelain(self.repo)!r} spare status="
            f"{GitWorktreeOps.status_porcelain(wt_dir)!r} log="
            f"{[r.getMessage() for r in caplog.records]}"
        )
        assert not wt_dir.exists()
        assert not GitWorktreeOps.branch_exists(self.repo, branch)

    def test_spare_owned_by_live_foreign_pid_is_skipped_until_it_dies(self) -> None:
        """In-process view of the same rule: a spare whose recorded owner
        is another LIVE process is skipped; once that process exits the
        very same spare is discarded."""
        assert worktree_pool.prewarm(self.repo)
        spare_branch, spare_dir = worktree_pool._spares.pop(
            worktree_pool._repo_key(self.repo)
        )
        owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            subprocess.run(
                ["git", "-C", str(self.repo), "config",
                 f"branch.{spare_branch}.kiss-owner-pid", str(owner.pid)],
                capture_output=True, check=True,
            )
            assert GitWorktreeOps.reclaim_orphaned_worktrees(self.repo) == 0
            assert spare_dir.is_dir()
            assert GitWorktreeOps.branch_exists(self.repo, spare_branch)
        finally:
            owner.kill()
            owner.wait(timeout=30)
        assert GitWorktreeOps.reclaim_orphaned_worktrees(self.repo) == 1
        assert not spare_dir.exists()
        assert not GitWorktreeOps.branch_exists(self.repo, spare_branch)

    def test_own_spare_without_exclusion_is_still_discarded(self) -> None:
        """Within one process the exclusion set is authoritative: a spare
        this process created but no longer tracks (pool state dropped)
        is discarded by its own reclaim pass, exactly as before."""
        assert worktree_pool.prewarm(self.repo)
        spare_branch, spare_dir = worktree_pool._spares.pop(
            worktree_pool._repo_key(self.repo)
        )
        assert GitWorktreeOps.reclaim_orphaned_worktrees(self.repo) == 1
        assert not spare_dir.exists()
        assert not GitWorktreeOps.branch_exists(self.repo, spare_branch)
