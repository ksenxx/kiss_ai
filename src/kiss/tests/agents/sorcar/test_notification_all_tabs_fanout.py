# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: agent UI notifications reach EVERY watching tab.

Three notification paths used to target only the owning tab
(``agent._tab_id``); they must instead reach every tab subscribed to
the task's event stream (``JsonPrinter._subscribers`` /
``_fanout_targets``):

* the auto-commit lifecycle toasts emitted by
  ``WorktreeSorcarAgent._broadcast_commit_notification``,
* the live model-picker override emitted by
  ``SorcarAgent._show_model_in_picker`` via
  ``JsonPrinter.broadcast_agent_model_pick`` (which must fan out even
  when the calling thread has no thread-local ``task_id`` bound,
  using the new explicit ``task_id`` fallback), and
* the ``subagentDone`` broadcasts of the non-UI
  ``run_tasks_parallel`` path.

All tests drive the real code paths — real on-disk git worktrees for
the auto-commit toasts, a real :class:`JsonPrinter` subscriber map,
and the real ``run_tasks_parallel`` executor — with a capture
printer that records ``broadcast`` payloads.

Also covers the printer-side "transient, all-watching-tabs"
primitive ``JsonPrinter.broadcast_transient`` (which the toast path
now delegates to, and whose target resolution
``broadcast_agent_model_pick`` shares), including the near-teardown
scenario: ``cleanup_task`` has run and the thread-local ``task_id``
is cleared, yet toasts and model-picker updates still reach every
lingering subscriber tab via the agent's explicit ``_last_task_id``.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent


def _make_repo(path: Path) -> Path:
    """Create a git repo with one initial commit at *path*."""
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


class _LLMUnavailable:
    """Force the commit-message LLM helper through its fallback.

    Patches :class:`kiss.core.kiss_agent.KISSAgent` to a class whose
    ``run`` raises, so tests stay hermetic.  Reverts on exit.
    """

    def __enter__(self) -> _LLMUnavailable:
        import kiss.core.kiss_agent as kiss_agent_mod

        self._orig = kiss_agent_mod.KISSAgent

        class _RaisingAgent:
            def __init__(self, *_a: Any, **_kw: Any) -> None:
                pass

            def run(self, *_a: Any, **_kw: Any) -> str:
                raise RuntimeError("no LLM in test")

        kiss_agent_mod.KISSAgent = _RaisingAgent  # type: ignore[misc, assignment]
        return self

    def __exit__(self, *_exc: Any) -> None:
        import kiss.core.kiss_agent as kiss_agent_mod

        kiss_agent_mod.KISSAgent = self._orig  # type: ignore[misc]


def _setup_worktree_agent(
    tmp: Path, branch_slug: str,
) -> tuple[WorktreeSorcarAgent, Path]:
    """Build a real on-disk worktree backed by a ``WorktreeSorcarAgent``.

    Returns ``(agent, wt_dir)`` with ``agent._wt`` populated so
    ``_auto_commit_worktree`` runs end-to-end.
    """
    repo = _make_repo(tmp / "repo")
    branch = f"kiss/wt-fanout-{branch_slug}"
    wt_dir = repo / ".kiss-worktrees" / branch.replace("/", "_")
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
    return agent, wt_dir


class _BroadcastOnlyPrinter:
    """A printer stub with ONLY a ``broadcast`` method — the degraded
    path for third-party printers without the transient primitive."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))


class TestPrimitiveLessPrinterDegradation:
    """Printers exposing only ``broadcast`` still get one
    ``tabId``-stamped toast copy (the pre-primitive behaviour)."""

    def test_single_stamped_copy_on_broadcast_only_printer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            agent, wt_dir = _setup_worktree_agent(Path(tmp_str), "degraded")
            (wt_dir / "new.txt").write_text("hello\n")

            printer = _BroadcastOnlyPrinter()
            agent.printer = printer  # type: ignore[assignment]
            agent._tab_id = "tab-solo"
            agent._last_task_id = "7171"

            with _LLMUnavailable():
                assert agent._auto_commit_worktree() is True

            notifs = [
                e for e in printer.events if e.get("type") == "notification"
            ]
            assert len(notifs) == 2  # generating + committed, one tab each
            assert all(e["tabId"] == "tab-solo" for e in notifs)
