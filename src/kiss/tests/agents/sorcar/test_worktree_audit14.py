"""Audit 14: Integration tests for bugs/redundancies found in audit round 14.

BUG-64: ``WorktreeSorcarAgent.run()`` silently drops
    ``_stash_pop_warning`` and ``_merge_conflict_warning`` when the
    call falls back to direct (non-worktree) execution — the warnings
    are only broadcast on the success path.  The three fallback
    points are:

      A. ``use_worktree=False`` kwarg
      B. ``work_dir`` is not inside a git repo
      C. ``_try_setup_worktree`` returns ``None`` (detached HEAD or
         any setup failure after ``_release_worktree`` has already
         set a warning)

    In fallback C, ``_release_worktree`` runs *before* the setup
    failure and may set a warning describing a previous task's
    auto-merge outcome — that warning is then lost.  In all three
    fallbacks, the BUG-B handler in ``_run_task_inner`` may also
    have set ``_merge_conflict_warning`` just before calling
    ``run()``, and that warning is lost too.

BUG-65: ``VSCodeServer._new_chat`` does not block when a merge
    review is in progress.  ``_run_task_inner`` refuses new tasks
    while ``tab.is_merging`` is True, but ``_new_chat`` calls
    ``tab.agent.new_chat()`` which triggers ``_release_worktree``
    that auto-commits + squash-merges — destroying the user's
    hunk-picking intent and leaving the VS Code merge view stale.
    Symmetric with the ``_run_task_inner`` guard.

RED-8: ``GitWorktreeOps.manual_merge_branch`` and
    ``ManualMergeResult`` are dead code — no production caller in
    ``src/kiss/agents`` or ``src/kiss/agents/vscode``.  Only tests
    reference them.  Dead code expands attack surface and
    maintenance burden.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar.git_worktree import (
    GitWorktreeOps,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


class _RecordingPrinter:
    """Minimal concrete printer that records every broadcast event.

    Not a mock/fake — a real recording object used in place of the
    production ``_Printer`` to capture broadcast events for
    assertions.  Production code interacts with it via the exact
    same ``broadcast(event)`` method contract.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        capture_output=True, check=True,
    )
    (path / "init.txt").write_text("init\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        capture_output=True, check=True,
    )
    return path


def _patch_super_run(return_value: str = "success: true\nsummary: test\n") -> Any:
    """Replace ``SorcarAgent``'s parent ``run`` so no LLM is invoked.

    Returns the original method so callers can restore it.
    """
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    original = parent_class.run

    def fake_run(self_agent: object, **kwargs: object) -> str:
        return return_value

    parent_class.run = fake_run
    return original


def _unpatch_super_run(original: Any) -> None:
    parent_class = cast(Any, SorcarAgent.__mro__[1])
    parent_class.run = original


class TestBug64WarningsDroppedOnFallback:
    """``WorktreeSorcarAgent.run()`` must flush pending warnings to
    the printer on *every* fallback path, not only the success path.

    Without the fix, a warning set by ``_release_worktree`` or by the
    BUG-B handler in ``_run_task_inner`` is silently kept on the
    agent and the user is never informed.
    """

    def setup_method(self) -> None:
        self.orig = _patch_super_run()

    def teardown_method(self) -> None:
        _unpatch_super_run(self.orig)

    def test_warning_flushed_when_use_worktree_false(
        self, tmp_path: Path,
    ) -> None:
        """Fallback A: ``use_worktree=False`` kwarg must still
        broadcast a pending warning."""
        agent = WorktreeSorcarAgent("t")
        agent._merge_conflict_warning = "A pending branch needs attention."
        printer = _RecordingPrinter()

        agent.run(
            prompt_template="x",
            work_dir=str(tmp_path),
            printer=printer,
            use_worktree=False,
        )

        warnings = [e for e in printer.events if e.get("type") == "warning"]
        assert any(
            "A pending branch needs attention." in e.get("message", "")
            for e in warnings
        ), (
            "BUG-64: warning not broadcast on use_worktree=False "
            f"fallback.  Events were: {printer.events}"
        )
        assert agent._merge_conflict_warning is None

    def test_warning_flushed_when_not_a_git_repo(
        self, tmp_path: Path,
    ) -> None:
        """Fallback B: ``discover_repo`` returns None — warning
        must still be broadcast."""
        agent = WorktreeSorcarAgent("t")
        agent._stash_pop_warning = "Stash pop failed — run git stash pop."
        printer = _RecordingPrinter()

        agent.run(
            prompt_template="x",
            work_dir=str(tmp_path),
            printer=printer,
        )

        warnings = [e for e in printer.events if e.get("type") == "warning"]
        assert any(
            "Stash pop failed" in e.get("message", "") for e in warnings
        ), (
            "BUG-64: warning not broadcast on not-a-repo fallback.  "
            f"Events were: {printer.events}"
        )
        assert agent._stash_pop_warning is None

    def test_warning_flushed_on_detached_head(self, tmp_path: Path) -> None:
        """Fallback C: ``_try_setup_worktree`` returns None
        (detached HEAD) — warning must still be broadcast."""
        repo = _make_repo(tmp_path / "repo")
        head_sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "-C", str(repo), "checkout", head_sha],
            capture_output=True, check=True,
        )

        agent = WorktreeSorcarAgent("t")
        agent._merge_conflict_warning = "Prior wt branch unresolved."
        printer = _RecordingPrinter()

        agent.run(
            prompt_template="x",
            work_dir=str(repo),
            printer=printer,
        )

        warnings = [e for e in printer.events if e.get("type") == "warning"]
        assert any(
            "Prior wt branch unresolved." in e.get("message", "")
            for e in warnings
        ), (
            "BUG-64: warning not broadcast on detached-HEAD fallback.  "
            f"Events were: {printer.events}"
        )
        assert agent._merge_conflict_warning is None

    def test_success_path_still_flushes_warnings(
        self, tmp_path: Path,
    ) -> None:
        """Regression: the existing success-path broadcast must keep
        working identically."""
        repo = _make_repo(tmp_path / "repo")
        agent = WorktreeSorcarAgent("t")
        agent._stash_pop_warning = "success-path warning"
        printer = _RecordingPrinter()
        try:
            agent.run(
                prompt_template="x",
                work_dir=str(repo),
                printer=printer,
            )
        finally:
            if agent._wt is not None:
                agent.discard()

        warnings = [e for e in printer.events if e.get("type") == "warning"]
        assert any(
            "success-path warning" in e.get("message", "")
            for e in warnings
        ), f"Regression: success-path warning lost.  Events: {printer.events}"


class TestRed8ManualMergeBranchDeadCode:
    """``manual_merge_branch`` and ``ManualMergeResult`` are not
    referenced by any module under ``src/kiss/agents`` (production
    code).  They exist only for tests — dead code.
    """

    def test_no_production_callers(self) -> None:
        import re

        src = Path(__file__).resolve().parents[4] / "agents"
        offenders: list[str] = []
        for py in src.rglob("*.py"):
            if py.name == "git_worktree.py":
                continue
            text = py.read_text()
            if re.search(r"\bmanual_merge_branch\b", text) or re.search(
                r"\bManualMergeResult\b", text
            ):
                offenders.append(str(py))
        assert not offenders, (
            "RED-8 sanity: non-test production file references dead "
            f"symbol: {offenders}"
        )

    def test_manual_merge_branch_removed(self) -> None:
        assert not hasattr(GitWorktreeOps, "manual_merge_branch"), (
            "RED-8: manual_merge_branch is dead code — no production "
            "caller.  Remove it."
        )

    def test_manual_merge_result_removed(self) -> None:
        import kiss.agents.sorcar.git_worktree as gw

        assert not hasattr(gw, "ManualMergeResult"), (
            "RED-8: ManualMergeResult is dead code — no production "
            "caller.  Remove it."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
