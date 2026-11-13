# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A worktree's git-ignored files are removed with it.

``git add -A`` skips files matched by ``.gitignore``, so the commit a
worktree teardown makes never holds them, and nothing copies them into
the main checkout either: a merge, a retirement, a discard and the
orphan reclaim remove the worktree directory with every ignored file
in it, whatever those files are, and leave the checkout's own ignored
files as they are.

These tests drive the real ``ChatSorcarAgent`` against a real git
repository (the parent class' ``run`` is replaced with a deterministic
stub that calls the agent's real ``create_worktree`` tool and writes
files, the same no-mock pattern the rest of the worktree suite uses).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.relentless_agent import resolve_work_dir
from kiss.agents.sorcar.sorcar_agent import SorcarAgent

_IGNORED_REL = "data/out.csv"


def _redirect_db(tmpdir: str) -> tuple:
    """Redirect the persistence DB to a temp dir; return prior state."""
    old = (th._DB_PATH, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    return old


def _restore_db(saved: tuple) -> None:
    """Undo :func:`_redirect_db`."""
    th._close_thread_db()
    (th._DB_PATH, th._KISS_DIR) = saved


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in *cwd* capturing output."""
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, check=False,
    )


def _make_repo(path: Path) -> Path:
    """Create a git repo with a seed commit and a committed .gitignore."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q").check_returncode()
    _git(path, "config", "user.email", "test@test.com").check_returncode()
    _git(path, "config", "user.name", "Test").check_returncode()
    _git(path, "config", "commit.gpgsign", "false").check_returncode()
    (path / "README.md").write_text("# Test\n")
    (path / ".gitignore").write_text("data/\n.env\n")
    _git(path, "add", ".").check_returncode()
    _git(path, "commit", "-q", "-m", "initial").check_returncode()
    return path


def _add_ignored_nested_repo(wt_dir: Path) -> None:
    """Put a git repository under the ignored ``data/`` of *wt_dir*.

    git lists such a repository as one directory entry and never
    descends into it: the kind of ignored content no file copy handles.
    """
    nested = wt_dir / "data" / "clone"
    nested.mkdir(parents=True)
    _git(nested, "init", "-q").check_returncode()
    (nested / "payload.txt").write_text("payload\n")


def _worktree_dirs(repo: Path) -> list[Path]:
    """The task worktree directories still on disk under *repo*."""
    return sorted((repo / ".kiss-worktrees").glob("*"))


def _stub_parent_run(files: dict[str, str]) -> Any:
    """Replace the parent class' ``run`` with a file-writing stub.

    The stub does what a model-driven run does before its first edit:
    it enters the run's ``work_dir`` and calls the agent's real
    ``create_worktree`` tool, then writes *files* into the worktree.

    Args:
        files: Relative path -> content, written into the per-run
            ``work_dir`` (the worktree) by the stub.

    Returns:
        The original ``run`` for restoration in teardown.
    """
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def stub_run(self_agent: Any, **kwargs: Any) -> str:
        self_agent.work_dir = resolve_work_dir(kwargs.get("work_dir"))
        self_agent.create_worktree()
        for rel, content in files.items():
            target = Path(self_agent.work_dir) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        return "success: true\nsummary: stub\n"

    parent_class.run = stub_run
    return original


class TestIgnoredFilesAgent:
    """Merge, retirement and discard of an agent's worktree."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-ignored-")
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self._original_run: Any = None

    def teardown_method(self) -> None:
        if self._original_run is not None:
            cast(Any, SorcarAgent.__mro__[1]).run = self._original_run
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_task(self, files: dict[str, str]) -> ChatSorcarAgent:
        """Run one stubbed worktree task that writes *files*."""
        self._original_run = _stub_parent_run(files)
        agent = ChatSorcarAgent("ignored-files-test")
        agent.run("task", work_dir=str(self.repo))
        assert agent._wt_pending, "worktree should be pending after run"
        return agent

    def test_merge_copies_no_ignored_file_into_the_checkout(self) -> None:
        """The tracked work is merged; the ignored files go with the worktree."""
        user_env = self.repo / ".env"
        user_env.write_text("SECRET=users-own\n")
        agent = self._run_task({
            "tracked.txt": "tracked\n",
            _IGNORED_REL: "col\n1\n",
            ".env": "SECRET=agents-version\n",
        })
        assert agent._wt is not None
        branch = agent._wt.branch
        assert agent.merge() == f"Successfully merged branch '{branch}'."
        assert (self.repo / "tracked.txt").read_text() == "tracked\n"
        assert not (self.repo / "data").exists()
        assert user_env.read_text() == "SECRET=users-own\n"
        assert sorted(p.name for p in self.repo.iterdir()) == [
            ".env", ".git", ".gitignore", ".kiss-worktrees", "README.md", "tracked.txt",
        ]
        assert _worktree_dirs(self.repo) == []

    def test_merge_is_not_held_back_by_an_ignored_nested_repository(self) -> None:
        """No ignored content keeps the worktree of a merged branch on disk."""
        agent = self._run_task({"tracked.txt": "tracked\n"})
        assert agent._wt is not None
        _add_ignored_nested_repo(agent._wt.wt_dir)
        msg = agent.merge()
        assert "Successfully merged" in msg, msg
        assert not agent._wt_pending
        assert (self.repo / "tracked.txt").read_text() == "tracked\n"
        assert not (self.repo / "data").exists()
        assert _worktree_dirs(self.repo) == []

    def test_retirement_copies_no_ignored_file_into_the_checkout(self) -> None:
        """A retired worktree leaves its tracked work on the branch only."""
        agent = self._run_task({
            "tracked.txt": "tracked\n",
            _IGNORED_REL: "col\n1\n",
        })
        assert agent._wt is not None
        branch = agent._wt.branch
        _add_ignored_nested_repo(agent._wt.wt_dir)
        assert agent._retire_previous_worktree()
        assert not agent._wt_pending
        assert _git(self.repo, "show", f"{branch}:tracked.txt").stdout == "tracked\n"
        assert not (self.repo / "tracked.txt").exists()
        assert not (self.repo / "data").exists()
        assert _worktree_dirs(self.repo) == []

    def test_discard_copies_no_ignored_file_into_the_checkout(self) -> None:
        """discard() throws the ignored files away with everything else."""
        agent = self._run_task({
            "tracked.txt": "tracked\n",
            _IGNORED_REL: "col\n1\n",
        })
        assert agent._wt is not None
        branch = agent._wt.branch
        assert agent.discard() == f"Discarded branch '{branch}'."
        assert not (self.repo / "data").exists()
        assert _worktree_dirs(self.repo) == []


class TestIgnoredFilesReclaim:
    """The orphan-worktree reclaim pass."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-ignored-reclaim-")
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reclaim_copies_no_ignored_file_into_the_checkout(self) -> None:
        """A crashed task's tracked work is merged; its ignored files are not kept."""
        branch = "kiss/wt-reclaim-ignored"
        wt_dir = self.repo / ".kiss-worktrees" / "kiss_wt-reclaim-ignored"
        GitWorktreeOps.ensure_excluded(self.repo)
        assert GitWorktreeOps.create(self.repo, branch, wt_dir)
        assert GitWorktreeOps.save_original_branch(
            self.repo, branch, GitWorktreeOps.current_branch(self.repo) or "",
        )
        (wt_dir / "work.txt").write_text("committed work\n")
        _add_ignored_nested_repo(wt_dir)
        (wt_dir / _IGNORED_REL).write_text("col\n1\n")
        assert GitWorktreeOps.reclaim_orphaned_worktrees(self.repo) == 1
        assert (self.repo / "work.txt").read_text() == "committed work\n"
        assert not (self.repo / "data").exists()
        assert _worktree_dirs(self.repo) == []
