# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Work-loss audit: git-ignored task output must survive worktree teardown.

A worktree task that creates files matched by ``.gitignore`` (a
downloaded dataset in an ignored ``data/`` directory, a generated
``*.csv`` report, ...) used to lose them silently on EVERY teardown
path: ``git add -A`` skips ignored files, so the auto-commit cannot
capture them, and ``git worktree remove --force`` then deletes the
directory.  Had the same task run without a worktree, those files
would still be on disk.

These tests drive the real ``WorktreeSorcarAgent`` against a real git
repository (the parent class' ``run`` is replaced with a deterministic
stub that writes files — the same no-mock pattern the rest of the
worktree suite uses) and assert the rescue behavior:

* merge / release rescues ignored files into the main repository;
* an existing main-tree file is NEVER overwritten by a rescue;
* regenerable cache directories (``__pycache__``, ``.venv``, ...) are
  not rescued;
* the automatic discard paths rescue too (``rescue_ignored=True``),
  while a user-explicit ``discard()`` still throws everything away;
* the orphan-worktree reclaim pass rescues before it removes;
* the post-task auto path in the server rescues a task whose ONLY
  output was ignored files (the changed-files probe reports the
  worktree as empty and picks "discard").
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.git_worktree import GitWorktreeOps
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

_IGNORED_REL = "data/out.csv"


_IGNORED_CONTENT = "col\n1\n"


def _redirect_db(tmpdir: str) -> tuple:
    """Redirect the persistence DB to a temp dir; return prior state."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return old


def _restore_db(saved: tuple) -> None:
    """Undo :func:`_redirect_db`."""
    if th._db_conn is not None:
        th._db_conn.close()
        th._db_conn = None
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in *cwd* capturing output."""
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, check=False,
    )


def _make_repo(path: Path) -> Path:
    """Create a git repo with a seed commit and a committed .gitignore."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@test.com")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "commit.gpgsign", "false")
    (path / "README.md").write_text("# Test\n")
    (path / ".gitignore").write_text(
        "data/\n.env\n__pycache__/\n.venv/\n*.log\n"
        "*.vsix\n.coverage*\n*.pyc\n*.pyo\n*.tsbuildinfo\n"
    )
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "initial")
    return path


def _stub_parent_run(files: dict[str, str]) -> Any:
    """Replace the parent class' ``run`` with a file-writing stub.

    Args:
        files: Relative path -> content, written into the per-run
            ``work_dir`` (the worktree) by the stub.

    Returns:
        The original ``run`` for restoration in teardown.
    """
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def stub_run(self_agent: object, **kwargs: object) -> str:
        work_dir = kwargs.get("work_dir")
        if isinstance(work_dir, str) and work_dir:
            for rel, content in files.items():
                target = Path(work_dir) / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
        return "success: true\nsummary: stub\n"

    parent_class.run = stub_run
    return original


class TestIgnoredFileRescueAgent:
    """Agent-level rescue behavior across merge/release/discard."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-rescue-")
        self.db_saved = _redirect_db(self.tmpdir)
        self.repo = _make_repo(Path(self.tmpdir) / "repo")
        self._original_run: Any = None

    def teardown_method(self) -> None:
        if self._original_run is not None:
            cast(Any, SorcarAgent.__mro__[1]).run = self._original_run
        _restore_db(self.db_saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_task(self, files: dict[str, str]) -> WorktreeSorcarAgent:
        """Run one stubbed worktree task that writes *files*."""
        self._original_run = _stub_parent_run(files)
        agent = WorktreeSorcarAgent("rescue-test")
        agent.run("task", work_dir=str(self.repo), auto_commit=True)
        assert agent._wt_pending, "worktree should be pending after run"
        return agent

    def test_merge_rescues_ignored_file(self) -> None:
        """merge() copies task-created ignored files into the repo."""
        agent = self._run_task({
            "tracked.txt": "tracked\n",
            _IGNORED_REL: _IGNORED_CONTENT,
        })
        msg = agent.merge()
        assert "Successfully merged" in msg, msg
        rescued = self.repo / _IGNORED_REL
        assert rescued.is_file(), (
            "ignored task output was destroyed by the worktree merge"
        )
        assert rescued.read_text() == _IGNORED_CONTENT
        # The rescued file must stay ignored (not committed).
        show = _git(self.repo, "ls-files", "--", _IGNORED_REL)
        assert show.stdout.strip() == ""

    def test_merge_never_overwrites_existing_ignored_file(self) -> None:
        """A main-tree file (e.g. the user's .env) is never clobbered."""
        user_env = self.repo / ".env"
        user_env.write_text("SECRET=users-own\n")
        agent = self._run_task({
            "tracked.txt": "tracked\n",
            ".env": "SECRET=agents-version\n",
        })
        msg = agent.merge()
        assert "Successfully merged" in msg, msg
        assert user_env.read_text() == "SECRET=users-own\n", (
            "rescue overwrote the user's own ignored file"
        )

    def test_merge_skips_regenerable_cache_dirs(self) -> None:
        """Cache/venv junk created in the worktree is not copied over."""
        agent = self._run_task({
            "tracked.txt": "tracked\n",
            "__pycache__/mod.cpython-312.pyc": "bytecode",
            ".venv/bin/python": "#!/usr/bin/env python\n",
            _IGNORED_REL: _IGNORED_CONTENT,
        })
        msg = agent.merge()
        assert "Successfully merged" in msg, msg
        assert not (self.repo / "__pycache__").exists()
        assert not (self.repo / ".venv").exists()
        assert (self.repo / _IGNORED_REL).is_file()

    def test_merge_skips_regenerable_artifact_patterns(self) -> None:
        """Build artifacts matched by name are neither rescued nor sibling'd.

        A rebuilt ``*.vsix``, pytest-cov's ``.coverage`` data files and
        stray ``*.pyc`` outside ``__pycache__`` differ from the main
        tree on every worktree run; before the basename skip list each
        teardown landed a ``<stem>.kiss-rescued-<ns><ext>`` sibling.
        """
        main_vsix = self.repo / "ext" / "kiss-sorcar.vsix"
        main_vsix.parent.mkdir()
        main_vsix.write_bytes(b"real build")
        main_cov = self.repo / ".coverage"
        main_cov.write_bytes(b"main coverage")
        agent = self._run_task({
            "tracked.txt": "tracked\n",
            "ext/kiss-sorcar.vsix": "worktree build",
            ".coverage": "worktree coverage",
            ".coverage.host.123.456": "parallel coverage shard",
            "pkg/mod.pyc": "bytecode",
            "pkg/old.pyo": "bytecode",
            "ext/tsconfig.tsbuildinfo": "{}",
            _IGNORED_REL: _IGNORED_CONTENT,
        })
        msg = agent.merge()
        assert "Successfully merged" in msg, msg
        siblings = sorted(
            p.relative_to(self.repo)
            for p in self.repo.rglob("*.kiss-rescued-*")
        )
        assert siblings == [], f"artifacts rescued as siblings: {siblings}"
        assert main_vsix.read_bytes() == b"real build"
        assert main_cov.read_bytes() == b"main coverage"
        for rel in (
            ".coverage.host.123.456", "pkg/mod.pyc", "pkg/old.pyo",
            "ext/tsconfig.tsbuildinfo",
        ):
            assert not (self.repo / rel).exists(), f"{rel} was rescued"
        # Genuine ignored output next to the artifacts is still rescued.
        assert (self.repo / _IGNORED_REL).read_text() == _IGNORED_CONTENT

    def test_rescue_reports_ok_when_only_artifacts_are_skipped(self) -> None:
        """Skipped artifacts count as safe: the fail-closed flag stays True."""
        wt = Path(self.tmpdir) / "wt"
        _git(self.repo, "worktree", "add", "-q", str(wt), "-b", "wt-branch")
        (wt / "kiss-sorcar.vsix").write_bytes(b"build")
        (wt / ".coverage").write_bytes(b"cov")
        (wt / "mod.pyc").write_bytes(b"pyc")
        (wt / "keep.log").write_text("task log\n")
        rescued, ok = GitWorktreeOps.rescue_ignored_files(wt, self.repo)
        assert (rescued, ok) == (1, True)
        assert (self.repo / "keep.log").read_text() == "task log\n"
        assert not (self.repo / "kiss-sorcar.vsix").exists()
        assert not (self.repo / ".coverage").exists()
        assert not (self.repo / "mod.pyc").exists()

    def test_release_on_next_run_rescues_ignored_file(self) -> None:
        """The auto-release before a new task rescues ignored output."""
        agent = self._run_task({
            "tracked.txt": "tracked\n",
            _IGNORED_REL: _IGNORED_CONTENT,
        })
        # Second run retires (auto-merges) the first worktree.
        agent.run("task 2", work_dir=str(self.repo), auto_commit=True)
        assert (self.repo / _IGNORED_REL).is_file(), (
            "ignored output of the released worktree was destroyed"
        )
        # Retire the second worktree so teardown is clean.
        agent.merge()

    def test_explicit_discard_does_not_rescue(self) -> None:
        """User-explicit discard() throws ignored output away too."""
        agent = self._run_task({
            "tracked.txt": "tracked\n",
            _IGNORED_REL: _IGNORED_CONTENT,
        })
        msg = agent.discard()
        assert "Discarded" in msg, msg
        assert not (self.repo / _IGNORED_REL).exists()

    def test_auto_discard_rescues_ignored_file(self) -> None:
        """Automatic discard paths pass rescue_ignored=True."""
        agent = self._run_task({_IGNORED_REL: _IGNORED_CONTENT})
        msg = agent.discard(rescue_ignored=True)
        assert "Discarded" in msg, msg
        rescued = self.repo / _IGNORED_REL
        assert rescued.is_file(), (
            "ignored-only task output was destroyed by the automatic "
            "empty-worktree discard"
        )
        assert rescued.read_text() == _IGNORED_CONTENT


class TestIgnoredFileRescueReclaim:
    """The orphan-worktree reclaim pass rescues before removing."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-wt-rescue-reclaim-")
        self.repo = _make_repo(Path(self.tmpdir) / "repo")

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reclaim_rescues_ignored_file(self) -> None:
        """A crashed task's ignored output survives the reclaim merge."""
        branch = "kiss/wt-reclaim-rescue"
        wt_dir = self.repo / ".kiss-worktrees" / "kiss_wt-reclaim-rescue"
        GitWorktreeOps.ensure_excluded(self.repo)
        assert GitWorktreeOps.create(self.repo, branch, wt_dir)
        assert GitWorktreeOps.save_original_branch(
            self.repo, branch, GitWorktreeOps.current_branch(self.repo) or "",
        )
        (wt_dir / "work.txt").write_text("committed work\n")
        (wt_dir / _IGNORED_REL).parent.mkdir(parents=True, exist_ok=True)
        (wt_dir / _IGNORED_REL).write_text(_IGNORED_CONTENT)
        reclaimed = GitWorktreeOps.reclaim_orphaned_worktrees(self.repo)
        assert reclaimed == 1
        assert (self.repo / "work.txt").is_file()
        assert (self.repo / _IGNORED_REL).is_file(), (
            "ignored output of the orphan worktree was destroyed by reclaim"
        )
