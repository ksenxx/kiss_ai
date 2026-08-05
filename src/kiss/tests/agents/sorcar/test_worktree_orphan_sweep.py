# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: leftover ``kiss/wt-*`` state must be swept away.

Bug report: a repository that has run many agent tasks accumulates
orphaned ``kiss/wt-*`` branches and stale ``branch.kiss/wt-*.*`` git
config sections.  None of it shows up in ``git status``, so it grows
silently — the reporting repo had reached 83 dead branches and 174
dead config entries.

Fixing the *leak* (see ``test_worktree_leak_when_main_tree_busy.py``)
stops new debris appearing but cannot remove what previous crashes,
kills and refused cleanups already left behind.  A worktree can also
be orphaned in ways no lifecycle fix can prevent — ``kill -9`` on the
agent, or a user deleting ``.kiss-worktrees/`` by hand.  Cleanup
therefore has to be *convergent*: every worktree task sweeps the
debris of the ones before it.

:meth:`GitWorktreeOps.sweep_orphaned_state` is that sweep.  These
tests pin both halves of its contract:

* it removes branches, worktree registrations and config sections that
  are provably dead, and
* it never touches a branch holding unreachable commits, a branch some
  worktree still has checked out, or anything outside ``kiss/wt-*``.

Everything runs against a real git repository created in ``tmp_path``
— real ``git init``, real commits, real ``git worktree add``.  Nothing
is mocked, so a regression in the underlying git plumbing fails these
tests rather than hiding behind a stub.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from kiss.agents.sorcar.git_worktree import GitWorktreeOps


def _git(*args: str, cwd: Path) -> str:
    """Run git in *cwd*, asserting success, and return its stdout."""
    proc = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"git {args} failed: {proc.stderr}"
    return proc.stdout


def _make_repo(tmp_path: Path) -> Path:
    """Create a real git repo with one commit on ``main``."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-b", "main", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-m", "initial", cwd=repo)
    return repo


def _branches(repo: Path) -> set[str]:
    """Return every local branch name in *repo*."""
    out = _git("for-each-ref", "--format=%(refname:short)", "refs/heads", cwd=repo)
    return set(out.split())


def _config_sections(repo: Path) -> set[str]:
    """Return branch names owning a ``branch.kiss/wt-*.*`` config section."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "config", "--get-regexp", r"^branch\.kiss/wt-"],
        capture_output=True,
        text=True,
        check=False,
    )
    names: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split(" ", 1)[0].split(".")
        names.add(".".join(parts[1:-1]))
    return names


def _add_orphan_branch(repo: Path, name: str) -> None:
    """Create a ``kiss/wt-*`` branch at HEAD plus its config section.

    This is exactly the state ``_try_setup_worktree`` leaves behind
    after ``git worktree add -b`` + ``save_original_branch`` when the
    worktree directory is later removed without deleting the branch.
    """
    _git("branch", name, cwd=repo)
    _git("config", f"branch.{name}.kiss-original", "main", cwd=repo)


def _add_unmerged_branch(repo: Path, name: str) -> str:
    """Create a ``kiss/wt-*`` branch carrying one unique commit.

    Returns:
        The SHA of the unique commit, so tests can prove it survives.
    """
    _add_orphan_branch(repo, name)
    worktree = repo / ".kiss-worktrees" / name.replace("/", "_")
    _git("worktree", "add", str(worktree), name, cwd=repo)
    (worktree / "work.txt").write_text("agent output\n", encoding="utf-8")
    _git("add", "-A", cwd=worktree)
    _git("commit", "-m", "agent work", cwd=worktree)
    sha = _git("rev-parse", "HEAD", cwd=worktree).strip()
    _git("worktree", "remove", str(worktree), "--force", cwd=repo)
    return sha


class TestSweepRemovesDeadState:
    """Debris that is provably dead is removed."""

    def test_merged_orphan_branch_is_deleted(self, tmp_path: Path) -> None:
        """A ``kiss/wt-*`` branch with no unique commits is reaped."""
        repo = _make_repo(tmp_path)
        _add_orphan_branch(repo, "kiss/wt-1700000000-deadbeef")
        assert "kiss/wt-1700000000-deadbeef" in _branches(repo)

        deleted = GitWorktreeOps.sweep_orphaned_state(repo)

        assert deleted == 1, f"expected 1 branch reaped, got {deleted}"
        assert _branches(repo) == {"main"}, (
            "the orphaned worktree branch survived the sweep: "
            f"{sorted(_branches(repo))}"
        )

    def test_config_section_of_deleted_branch_is_purged(
        self, tmp_path: Path
    ) -> None:
        """``branch.kiss/wt-*.*`` config dies with its branch."""
        repo = _make_repo(tmp_path)
        _add_orphan_branch(repo, "kiss/wt-1700000000-deadbeef")
        assert _config_sections(repo) == {"kiss/wt-1700000000-deadbeef"}

        GitWorktreeOps.sweep_orphaned_state(repo)

        assert _config_sections(repo) == set(), (
            "stale git config survived: " f"{sorted(_config_sections(repo))}"
        )

    def test_config_section_without_branch_is_purged(
        self, tmp_path: Path
    ) -> None:
        """Config left behind by an already-deleted branch is purged.

        This is the shape most of the reporting repo's 174 stale
        entries had: the branch was deleted at some point but its
        config section was not, so there is nothing for the branch
        loop to reap — only the config pass can clean it.
        """
        repo = _make_repo(tmp_path)
        _git(
            "config", "branch.kiss/wt-1699999999-cafe0001.kiss-original",
            "main", cwd=repo,
        )
        _git(
            "config", "branch.kiss/wt-1699999999-cafe0001.vscode-merge-base",
            "origin/main", cwd=repo,
        )
        assert _config_sections(repo) == {"kiss/wt-1699999999-cafe0001"}

        deleted = GitWorktreeOps.sweep_orphaned_state(repo)

        assert deleted == 0, "no branch existed, so none should be reported"
        assert _config_sections(repo) == set()

    def test_squash_merged_branch_is_deleted(self, tmp_path: Path) -> None:
        """A branch whose commits landed on main via merge is reaped.

        ``git branch -d`` alone would refuse here after a *squash*
        merge; the sweep asks whether any commit is unreachable from
        every other ref, which is the question that actually matters.
        """
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-1700000001-feedface"
        _add_unmerged_branch(repo, branch)
        _git("merge", "--no-ff", "-m", "merge agent work", branch, cwd=repo)

        deleted = GitWorktreeOps.sweep_orphaned_state(repo)

        assert deleted == 1
        assert _branches(repo) == {"main"}

    def test_many_orphans_are_all_reaped(self, tmp_path: Path) -> None:
        """The sweep scales to a large accumulated backlog."""
        repo = _make_repo(tmp_path)
        for index in range(12):
            _add_orphan_branch(repo, f"kiss/wt-17000001{index:02d}-000000{index:02d}")

        deleted = GitWorktreeOps.sweep_orphaned_state(repo)

        assert deleted == 12
        assert _branches(repo) == {"main"}
        assert _config_sections(repo) == set()

    def test_dead_worktree_registration_is_pruned(self, tmp_path: Path) -> None:
        """A registration whose directory vanished no longer protects it.

        Deleting ``.kiss-worktrees/<slug>`` with ``rm -rf`` (or losing
        it to a crash) leaves git still listing the worktree.  Without
        the prune, ``checked_out_branches`` would report the branch as
        in use and the sweep would skip it forever.
        """
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-1700000002-0badcafe"
        worktree = repo / ".kiss-worktrees" / branch.replace("/", "_")
        _git("worktree", "add", "-b", branch, str(worktree), cwd=repo)
        # Simulate a crash: the directory is gone, the registration is not.
        subprocess.run(["rm", "-rf", str(worktree)], check=True)
        assert branch in GitWorktreeOps.checked_out_branches(repo)

        deleted = GitWorktreeOps.sweep_orphaned_state(repo)

        assert deleted == 1, "the branch of a vanished worktree was not reaped"
        assert _branches(repo) == {"main"}
        assert GitWorktreeOps.checked_out_branches(repo) == {"main"}


class TestSweepPreservesLiveState:
    """Anything that might still be wanted is left strictly alone."""

    def test_unmerged_branch_survives(self, tmp_path: Path) -> None:
        """A branch with unique commits is never deleted."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-1700000003-abcd1234"
        sha = _add_unmerged_branch(repo, branch)

        deleted = GitWorktreeOps.sweep_orphaned_state(repo)

        assert deleted == 0
        assert branch in _branches(repo), (
            "the sweep destroyed unmerged agent work"
        )
        assert _git("rev-parse", branch, cwd=repo).strip() == sha
        assert _config_sections(repo) == {branch}, (
            "config of a surviving branch must survive with it"
        )

    def test_checked_out_branch_survives(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A branch a live worktree has checked out is left untouched.

        This is the running-task case: the branch has *no* unique
        commits, so it looks exactly like reapable debris.

        The sweep must skip it rather than lean on git refusing the
        delete.  Git does refuse — ``branch -d`` and ``-D`` both fail
        with "cannot delete branch ... used by worktree at ..." — but
        relying on that would make :meth:`delete_branch` log a warning
        for every running task on every sweep.  Asserting the log is
        silent is what distinguishes "skipped it" from "tried and was
        blocked", so this test fails if the in-use check is dropped.
        """
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-1700000004-11112222"
        worktree = repo / ".kiss-worktrees" / branch.replace("/", "_")
        _git("worktree", "add", "-b", branch, str(worktree), cwd=repo)
        _git("config", f"branch.{branch}.kiss-original", "main", cwd=repo)

        with caplog.at_level(logging.WARNING, logger="kiss.agents.sorcar.git_worktree"):
            deleted = GitWorktreeOps.sweep_orphaned_state(repo)

        assert deleted == 0, "a running task's branch was reaped"
        assert branch in _branches(repo)
        assert worktree.exists(), "a live worktree directory was removed"
        assert _config_sections(repo) == {branch}
        assert caplog.records == [], (
            "the sweep attempted to delete a live worktree's branch instead "
            f"of skipping it: {[r.getMessage() for r in caplog.records]}"
        )

    def test_user_branches_are_never_touched(self, tmp_path: Path) -> None:
        """Only the ``kiss/wt-`` namespace is in scope."""
        repo = _make_repo(tmp_path)
        for name in ("feature/login", "kiss-notes", "wt-1700000000-x", "main"):
            if name != "main":
                _git("branch", name, cwd=repo)
        _git("config", "branch.feature/login.description", "mine", cwd=repo)

        deleted = GitWorktreeOps.sweep_orphaned_state(repo)

        assert deleted == 0
        assert _branches(repo) == {
            "main", "feature/login", "kiss-notes", "wt-1700000000-x",
        }
        assert (
            _git("config", "branch.feature/login.description", cwd=repo).strip()
            == "mine"
        )

    def test_config_purge_skips_sections_of_surviving_branches(
        self, tmp_path: Path
    ) -> None:
        """A dead section is purged while a live one beside it stays.

        Both sections are visited in the same pass, so this pins that
        the purge decides per branch rather than clearing the lot.
        """
        repo = _make_repo(tmp_path)
        alive = "kiss/wt-1700000012-aaaa0001"
        _add_unmerged_branch(repo, alive)
        _git(
            "config", "branch.kiss/wt-1700000013-bbbb0002.kiss-original",
            "main", cwd=repo,
        )
        assert _config_sections(repo) == {alive, "kiss/wt-1700000013-bbbb0002"}

        GitWorktreeOps.sweep_orphaned_state(repo)

        assert _config_sections(repo) == {alive}, (
            "the purge must keep config whose branch still exists"
        )

    def test_sweep_is_idempotent_on_a_clean_repo(self, tmp_path: Path) -> None:
        """Running the sweep with nothing to do changes nothing."""
        repo = _make_repo(tmp_path)
        before = _branches(repo)

        assert GitWorktreeOps.sweep_orphaned_state(repo) == 0
        assert GitWorktreeOps.sweep_orphaned_state(repo) == 0
        assert _branches(repo) == before

    def test_mixed_repo_reaps_only_the_dead(self, tmp_path: Path) -> None:
        """Live and dead state side by side: only the dead is removed."""
        repo = _make_repo(tmp_path)
        dead = "kiss/wt-1700000005-dead0001"
        unmerged = "kiss/wt-1700000006-live0001"
        running = "kiss/wt-1700000007-live0002"
        _add_orphan_branch(repo, dead)
        _add_unmerged_branch(repo, unmerged)
        worktree = repo / ".kiss-worktrees" / running.replace("/", "_")
        _git("worktree", "add", "-b", running, str(worktree), cwd=repo)
        _git("branch", "feature/keep", cwd=repo)

        deleted = GitWorktreeOps.sweep_orphaned_state(repo)

        assert deleted == 1
        assert _branches(repo) == {"main", unmerged, running, "feature/keep"}
        assert _config_sections(repo) == {unmerged}


class TestSweepHelpers:
    """The building blocks report the truth on a real repo."""

    def test_checked_out_branches_lists_every_worktree(
        self, tmp_path: Path
    ) -> None:
        """Both the main tree and added worktrees are reported."""
        repo = _make_repo(tmp_path)
        branch = "kiss/wt-1700000008-33334444"
        worktree = repo / ".kiss-worktrees" / branch.replace("/", "_")
        _git("worktree", "add", "-b", branch, str(worktree), cwd=repo)

        assert GitWorktreeOps.checked_out_branches(repo) == {"main", branch}

    def test_checked_out_branches_ignores_detached_worktrees(
        self, tmp_path: Path
    ) -> None:
        """A detached worktree contributes no branch name.

        ``git worktree list --porcelain`` emits ``detached`` instead of
        a ``branch`` line, which the parser must simply skip rather
        than mis-read as a branch called ``detached``.
        """
        repo = _make_repo(tmp_path)
        head = _git("rev-parse", "HEAD", cwd=repo).strip()
        detached = repo / ".kiss-worktrees" / "kiss_wt-detached"
        _git("worktree", "add", "--detach", str(detached), head, cwd=repo)

        assert GitWorktreeOps.checked_out_branches(repo) == {"main"}

    def test_branch_is_expendable_matches_reachability(
        self, tmp_path: Path
    ) -> None:
        """Expendable exactly when no commit is unique to the branch."""
        repo = _make_repo(tmp_path)
        empty = "kiss/wt-1700000009-55556666"
        _git("branch", empty, cwd=repo)
        unique = "kiss/wt-1700000010-77778888"
        _add_unmerged_branch(repo, unique)

        assert GitWorktreeOps._branch_is_expendable(repo, empty) is True
        assert GitWorktreeOps._branch_is_expendable(repo, unique) is False

    def test_config_sections_reports_only_kiss_worktree_branches(
        self, tmp_path: Path
    ) -> None:
        """The config scan is scoped to the agent's own namespace."""
        repo = _make_repo(tmp_path)
        _git("config", "branch.kiss/wt-1700000011-99990000.kiss-original",
             "main", cwd=repo)
        _git("config", "branch.main.remote", "origin", cwd=repo)
        _git("config", "branch.feature/x.description", "d", cwd=repo)

        assert GitWorktreeOps._config_branch_sections(repo) == {
            "kiss/wt-1700000011-99990000"
        }

    def test_config_sections_empty_when_none_present(
        self, tmp_path: Path
    ) -> None:
        """``git config --get-regexp`` exits non-zero on no match."""
        repo = _make_repo(tmp_path)

        assert GitWorktreeOps._config_branch_sections(repo) == set()
