# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""BUGHUNT-5: repo-scoped ``GIT_*`` environment variables must not leak
into the worktree machinery's git subprocesses.

``git_worktree._git`` runs ``git -C <cwd> ...`` with the parent
process's full environment.  When KISS is launched from a context that
exports ``GIT_DIR`` / ``GIT_WORK_TREE`` / ``GIT_INDEX_FILE`` — e.g. a
git hook (``post-commit`` launching an agent), ``git rebase --exec``,
or a user shell export — those variables OVERRIDE git's ``-C``-based
repository discovery.  Every single worktree operation then silently
targets the WRONG repository: ``discover_repo`` returns the hook's
repo, ``has_uncommitted_changes`` reports the hook repo's state,
``stage_all``/``commit`` mutate the hook repo's index, and worktree
branches are created in the wrong repo.

These integration tests use real on-disk git repos and a real
environment mutation (restored in ``finally``) — no mocks.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from kiss.agents.sorcar.git_worktree import GitWorktreeOps, _git
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _make_repo(path: Path) -> Path:
    """Create a git repo on branch ``main`` with one initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True,
        check=True,
    )
    for key, val in (("user.email", "t@t.com"), ("user.name", "T")):
        subprocess.run(
            ["git", "-C", str(path), "config", key, val],
            capture_output=True,
            check=True,
        )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
    )
    return path


@contextmanager
def _hook_env(repo: Path) -> Iterator[None]:
    """Simulate running inside a git hook of *repo* (real env vars)."""
    saved = {
        k: os.environ.get(k)
        for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")
    }
    os.environ["GIT_DIR"] = str(repo / ".git")
    os.environ["GIT_WORK_TREE"] = str(repo)
    os.environ["GIT_INDEX_FILE"] = str(repo / ".git" / "index")
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _kiss_branches(repo: Path) -> list[str]:
    """All kiss/wt-* branch names in *repo* (read with a clean env)."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "for-each-ref",
            "--format=%(refname:short)",
            "refs/heads/kiss/wt-*",
        ],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
    )
    return [b for b in result.stdout.strip().splitlines() if b]


class TestGitEnvLeak:
    """GIT_DIR/GIT_WORK_TREE/GIT_INDEX_FILE must not redirect _git calls."""

    def test_discover_repo_ignores_git_dir_env(self) -> None:
        """discover_repo(B) must return B even when GIT_DIR points at A."""
        with tempfile.TemporaryDirectory() as tmp:
            repo_a = _make_repo(Path(tmp) / "A")
            repo_b = _make_repo(Path(tmp) / "B")
            with _hook_env(repo_a):
                found = GitWorktreeOps.discover_repo(repo_b)
            assert found is not None
            assert found.resolve() == repo_b.resolve(), (
                f"GIT_DIR env leaked: discover_repo({repo_b}) "
                f"returned {found}"
            )

    def test_dirty_state_checked_in_target_repo_not_env_repo(self) -> None:
        """has_uncommitted_changes(B) must reflect B's state, not A's."""
        with tempfile.TemporaryDirectory() as tmp:
            repo_a = _make_repo(Path(tmp) / "A")
            repo_b = _make_repo(Path(tmp) / "B")
            (repo_b / "README.md").write_text("# Test\ndirty in B\n")
            with _hook_env(repo_a):
                assert GitWorktreeOps.has_uncommitted_changes(repo_b), (
                    "GIT_DIR env leaked: dirty repo B reported clean"
                )
                assert not GitWorktreeOps.has_uncommitted_changes(repo_a), (
                    "clean repo A reported dirty"
                )

    def test_worktree_setup_targets_correct_repo(self) -> None:
        """_try_setup_worktree(B) must create branch+worktree in B and
        copy B's dirty file — never touch repo A from the environment."""
        with tempfile.TemporaryDirectory() as tmp:
            repo_a = _make_repo(Path(tmp) / "A")
            repo_b = _make_repo(Path(tmp) / "B")
            (repo_b / "README.md").write_text("# Test\nuser edit in B\n")

            agent = WorktreeSorcarAgent("bh5-env-leak")
            with _hook_env(repo_a):
                wt_work_dir = agent._try_setup_worktree(repo_b, None)

            assert wt_work_dir is not None, (
                "worktree setup failed under hook env"
            )
            assert agent._wt is not None
            assert agent._wt.repo_root.resolve() == repo_b.resolve()
            assert _kiss_branches(repo_b), "no kiss branch created in B"
            assert _kiss_branches(repo_a) == [], (
                "GIT_DIR env leaked: kiss branch created in repo A"
            )
            # B's dirty edit was mirrored into the worktree.
            copied = (wt_work_dir / "README.md").read_text()
            assert copied == "# Test\nuser edit in B\n"
            # A's tree was never mutated.
            status_a = subprocess.run(
                ["git", "-C", str(repo_a), "status", "--porcelain"],
                capture_output=True,
                text=True,
                env={
                    k: v
                    for k, v in os.environ.items()
                    if not k.startswith("GIT_")
                },
            ).stdout.strip()
            assert status_a == "", f"repo A was mutated: {status_a}"
            agent.discard()

    def test_git_helper_scrubs_repo_scoped_env(self) -> None:
        """_git itself must not let GIT_INDEX_FILE redirect staging."""
        with tempfile.TemporaryDirectory() as tmp:
            repo_a = _make_repo(Path(tmp) / "A")
            repo_b = _make_repo(Path(tmp) / "B")
            (repo_b / "new.txt").write_text("new in B\n")
            with _hook_env(repo_a):
                GitWorktreeOps.stage_all(repo_b)
                staged = _git(
                    "diff", "--cached", "--name-only", cwd=repo_b
                ).stdout.split()
            assert "new.txt" in staged, (
                "GIT_INDEX_FILE env leaked: new.txt staged into A's index"
            )
