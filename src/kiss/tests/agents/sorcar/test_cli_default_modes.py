# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for ``sorcar`` CLI default modes.

The ``sorcar`` CLI must default to:

* **worktree mode ON** — every interactive task runs on an isolated
  ``git worktree`` branch (the user's main working tree is never
  touched).
* **parallel mode ON** — sub-agents are dispatched concurrently.
* **auto-commit ON** — when a task finishes in a worktree, any
  uncommitted changes are auto-committed before the worktree is
  cleaned up / merged.

Each toggle has a ``--no-...`` opt-out via
:class:`argparse.BooleanOptionalAction`.

These tests drive the real CLI plumbing:

* :func:`~kiss.agents.sorcar.cli_helpers._build_arg_parser` for the
  default flag values and ``--no-...`` opt-outs.
* :func:`~kiss.agents.sorcar.cli_helpers._build_run_kwargs` for the
  ``is_parallel`` propagation into ``agent.run`` kwargs.
* :meth:`~kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent._auto_commit_worktree`
  end-to-end against a real git worktree to confirm the
  ``--no-auto-commit`` opt-out actually suppresses the commit.

The non-interactive (``-t``/``-f``) CLI path always uses a bare
:class:`SorcarAgent` and never touches worktrees / chats; that
contract is verified separately in ``test_cli_only_sorcar_agent.py``.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from kiss.agents.sorcar.cli_helpers import _build_arg_parser, _build_run_kwargs
from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _make_repo(path: Path) -> Path:
    """Create a git repo with one initial commit at *path*.

    Mirrors the helper in ``test_autocommit_user_prompt.py`` so the
    behavioural auto-commit test exercises the same on-disk setup the
    real CLI sees.
    """
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        check=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        check=True,
    )
    return path


def _head_count(repo: Path) -> int:
    """Return the number of commits reachable from HEAD."""
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-list", "--count", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


class TestArgParserDefaults:
    """``_build_arg_parser`` must default worktree, parallel and
    auto-commit ALL to True so the bare ``sorcar`` invocation runs in
    the new safe + concurrent configuration without flags.
    """

    def test_default_parallel_is_on(self) -> None:
        args = _build_arg_parser().parse_args([])
        assert args.parallel is True

    def test_default_worktree_is_on(self) -> None:
        args = _build_arg_parser().parse_args([])
        assert args.worktree is True

    def test_default_auto_commit_is_on(self) -> None:
        args = _build_arg_parser().parse_args([])
        assert args.auto_commit is True


class TestArgParserOptOuts:
    """Each default-on toggle must support a ``--no-...`` opt-out."""

    def test_no_parallel_disables(self) -> None:
        args = _build_arg_parser().parse_args(["--no-parallel"])
        assert args.parallel is False

    def test_no_worktree_disables(self) -> None:
        args = _build_arg_parser().parse_args(["--no-worktree"])
        assert args.worktree is False

    def test_no_auto_commit_disables(self) -> None:
        args = _build_arg_parser().parse_args(["--no-auto-commit"])
        assert args.auto_commit is False


class TestRunKwargsPropagation:
    """``_build_run_kwargs`` must carry the parser's ``parallel``
    default through to ``agent.run`` as ``is_parallel``.
    """

    def test_default_run_kwargs_is_parallel_true(self) -> None:
        args = _build_arg_parser().parse_args([])
        run_kwargs = _build_run_kwargs(args)
        assert run_kwargs["is_parallel"] is True

    def test_no_parallel_propagates_to_run_kwargs(self) -> None:
        args = _build_arg_parser().parse_args(["--no-parallel"])
        run_kwargs = _build_run_kwargs(args)
        assert run_kwargs["is_parallel"] is False


class TestAutoCommitEnabledGate:
    """Behavioural test: ``auto_commit_enabled`` must actually gate
    :meth:`WorktreeSorcarAgent._auto_commit_worktree`.

    Drives the real method against an on-disk git worktree and
    asserts no commit is created when the flag is off, but a commit
    IS created when the flag is on.
    """

    def _setup_worktree(
        self, tmp: str, slug_suffix: str,
    ) -> tuple[Path, Path, WorktreeSorcarAgent]:
        repo = _make_repo(Path(tmp) / "repo")
        branch = f"kiss/wt-default-modes-{slug_suffix}"
        slug = branch.replace("/", "_")
        wt_dir = repo / ".kiss-worktrees" / slug
        assert GitWorktreeOps.create(repo, branch, wt_dir)
        subprocess.run(
            ["git", "-C", str(wt_dir), "config", "user.email", "t@t.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(wt_dir), "config", "user.name", "T"],
            check=True,
        )
        agent = WorktreeSorcarAgent("test")
        agent._wt = GitWorktree(
            repo_root=repo,
            branch=branch,
            original_branch="main",
            wt_dir=wt_dir,
            baseline_commit=None,
        )
        return repo, wt_dir, agent

    def test_auto_commit_off_skips_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, wt_dir, agent = self._setup_worktree(tmp, "off")
            (wt_dir / "new.txt").write_text("hello\n")

            before = _head_count(wt_dir)
            agent.auto_commit_enabled = False
            assert agent._auto_commit_worktree() is False
            after = _head_count(wt_dir)

            # No new commit on the worktree branch.
            assert after == before
            # Dirty file is preserved (not committed away).
            assert GitWorktreeOps.has_uncommitted_changes(wt_dir)

    def test_auto_commit_on_creates_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, wt_dir, agent = self._setup_worktree(tmp, "on")
            (wt_dir / "new.txt").write_text("hello\n")

            before = _head_count(wt_dir)
            agent.auto_commit_enabled = True
            # Force the message-generation LLM call through its
            # exception fallback so the test does not need a real
            # model — same pattern as ``test_autocommit_user_prompt``.
            import kiss.core.kiss_agent as kiss_agent_mod

            saved = kiss_agent_mod.KISSAgent

            class _RaisingAgent:
                def __init__(self, *_a: object, **_kw: object) -> None:
                    pass

                def run(self, *_a: object, **_kw: object) -> str:
                    raise RuntimeError("no LLM in test")

            kiss_agent_mod.KISSAgent = _RaisingAgent  # type: ignore[misc, assignment]
            try:
                assert agent._auto_commit_worktree() is True
            finally:
                kiss_agent_mod.KISSAgent = saved  # type: ignore[misc]
            after = _head_count(wt_dir)

            assert after == before + 1
            # Working tree is clean after the auto-commit.
            assert not GitWorktreeOps.has_uncommitted_changes(wt_dir)
