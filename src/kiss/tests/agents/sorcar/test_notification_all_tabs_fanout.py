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
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.git_worktree import GitWorktree, GitWorktreeOps
from kiss.agents.sorcar.sorcar_agent import run_tasks_parallel
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter


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


class _CapturePrinter(JsonPrinter):
    """Real :class:`JsonPrinter` (real subscriber map / fan-out
    lookups) that additionally records every ``broadcast`` payload,
    including the ``tabId``-stamped transient events the base class
    would forward without recording.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record the payload, then run the base recording path."""
        self.events.append(dict(event))
        super().broadcast(event)

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        """Return recorded events whose ``type`` equals *event_type*."""
        return [e for e in self.events if e.get("type") == event_type]


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


class TestAutoCommitToastReachesAllTabs:
    """The auto-commit toasts fan out to every tab watching the task."""

    def setup_method(self) -> None:
        agent_state.agent_states.clear()

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()

    def test_toasts_fan_out_to_owner_and_viewer_tabs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            agent, wt_dir = _setup_worktree_agent(Path(tmp_str), "viewers")
            (wt_dir / "new.txt").write_text("hello\n")

            printer = _CapturePrinter()
            agent.printer = printer  # type: ignore[assignment]
            agent._tab_id = "tab-owner"
            agent._last_task_id = "4242"
            # The owner tab is also in the subscriber map (as in the
            # real server flow) — it must NOT receive duplicates.
            printer.subscribe_tab("4242", "tab-owner")
            printer.subscribe_tab("4242", "tab-viewer-a")
            printer.subscribe_tab("4242", "tab-viewer-b")

            with _LLMUnavailable():
                assert agent._auto_commit_worktree() is True

            notifs = printer.of_type("notification")
            expected_tabs = {"tab-owner", "tab-viewer-a", "tab-viewer-b"}
            generating = [
                e for e in notifs
                if e["message"] == "Generating commit message"
            ]
            committed = [
                e for e in notifs if str(e["message"]).startswith("Committed ")
            ]
            assert {e["tabId"] for e in generating} == expected_tabs
            assert {e["tabId"] for e in committed} == expected_tabs
            # Exactly one copy per tab per stage (owner deduplicated).
            assert len(generating) == 3
            assert len(committed) == 3
            # Every copy of both stages shares ONE notification id so
            # each tab updates its toast in place.
            assert len({e["id"] for e in notifs}) == 1

    def test_no_subscribers_falls_back_to_owner_tab_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            agent, wt_dir = _setup_worktree_agent(Path(tmp_str), "solo")
            (wt_dir / "new.txt").write_text("hello\n")

            printer = _CapturePrinter()
            agent.printer = printer  # type: ignore[assignment]
            agent._tab_id = "tab-owner"
            agent._last_task_id = "4343"

            with _LLMUnavailable():
                assert agent._auto_commit_worktree() is True

            notifs = printer.of_type("notification")
            assert len(notifs) == 2
            assert all(e["tabId"] == "tab-owner" for e in notifs)

    def test_viewer_tabs_reached_even_without_owner_tab(self) -> None:
        """A task watched only through viewer tabs (e.g. the launching
        tab was closed) still shows the toasts on those viewers."""
        with tempfile.TemporaryDirectory() as tmp_str:
            agent, wt_dir = _setup_worktree_agent(Path(tmp_str), "orphan")
            (wt_dir / "new.txt").write_text("hello\n")

            printer = _CapturePrinter()
            agent.printer = printer  # type: ignore[assignment]
            agent._tab_id = ""
            agent._last_task_id = "4444"
            printer.subscribe_tab("4444", "tab-viewer")

            with _LLMUnavailable():
                assert agent._auto_commit_worktree() is True

            notifs = printer.of_type("notification")
            assert len(notifs) == 2
            assert all(e["tabId"] == "tab-viewer" for e in notifs)


class TestModelPickReachesAllTabs:
    """``_show_model_in_picker`` reaches every tab watching the task,
    even when the calling thread has no thread-local ``task_id``.
    """

    def setup_method(self) -> None:
        agent_state.agent_states.clear()

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()

    def test_explicit_task_id_fallback_reaches_viewers(self) -> None:
        printer = _CapturePrinter()
        printer.subscribe_tab("7777", "tab-viewer-a")
        printer.subscribe_tab("7777", "tab-viewer-b")
        assert printer._task_key() == ""  # thread-local unset

        printer.broadcast_agent_model_pick("model-x", "tab-launch", "7777")

        picks = printer.of_type("modelPick")
        assert {e["tabId"] for e in picks} == {
            "tab-launch", "tab-viewer-a", "tab-viewer-b",
        }
        assert all(e["model"] == "model-x" for e in picks)
        assert all(e["source"] == "agent" for e in picks)

    def test_thread_local_task_id_takes_precedence(self) -> None:
        printer = _CapturePrinter()
        printer.subscribe_tab("100", "tab-of-100")
        printer.subscribe_tab("200", "tab-of-200")
        printer._thread_local.task_id = "100"
        try:
            printer.broadcast_agent_model_pick("model-y", "", "200")
        finally:
            printer._thread_local.task_id = None

        picks = printer.of_type("modelPick")
        assert {e["tabId"] for e in picks} == {"tab-of-100"}

    def test_show_model_in_picker_end_to_end(self) -> None:
        printer = _CapturePrinter()
        printer.subscribe_tab("8888", "tab-viewer")

        agent = ChatSorcarAgent("picker-test")
        agent.printer = printer  # type: ignore[assignment]
        agent._tab_id = "tab-launch"  # type: ignore[attr-defined]
        agent._last_task_id = "8888"

        agent._show_model_in_picker("model-z")

        picks = printer.of_type("modelPick")
        assert {e["tabId"] for e in picks} == {"tab-launch", "tab-viewer"}


_SUB_TASK_ID = "313131"
_VIEWER_TAB = "frontend-viewer-tab"


def _patched_run(self: ChatSorcarAgent, **kwargs: Any) -> str:
    """Simulate the sub-agent lifecycle: allocate ``_last_task_id``
    and subscribe a frontend viewer tab to its event stream."""
    self._last_task_id = _SUB_TASK_ID
    printer: Any = kwargs.get("printer") or self.printer
    if printer is not None and hasattr(printer, "subscribe_tab"):
        printer.subscribe_tab(_SUB_TASK_ID, _VIEWER_TAB)
    return "success: true\nsummary: done"


class TestNonUiSubagentDoneReachesAllTabs:
    """The non-UI ``run_tasks_parallel`` path broadcasts
    ``subagentDone`` to every tab watching the sub-agent, plus the
    synthetic ``task-{parent}__sub_{idx}`` tab.
    """

    def setup_method(self) -> None:
        agent_state.agent_states.clear()

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()

    def _run(self, printer: _CapturePrinter, parent_key: str) -> None:
        printer._thread_local.task_id = parent_key or None
        original_run = ChatSorcarAgent.run
        ChatSorcarAgent.run = _patched_run  # type: ignore[assignment, method-assign]
        try:
            results = run_tasks_parallel(
                ["compute 1+1"], max_workers=1, printer=printer,
            )
        finally:
            ChatSorcarAgent.run = original_run  # type: ignore[method-assign]
            printer._thread_local.task_id = None
        assert len(results) == 1

    def test_viewer_and_synthetic_tabs_notified(self) -> None:
        printer = _CapturePrinter()
        self._run(printer, "9090")

        done_tabs = {
            e.get("tab_id") for e in printer.of_type("subagentDone")
        }
        assert _VIEWER_TAB in done_tabs
        assert "task-9090__sub_0" in done_tabs

    def test_viewer_notified_even_without_parent_task(self) -> None:
        """With no parent task id (no synthetic tab derivable), the
        subscribed viewer tab must still be told the sub-agent is
        done — previously nothing was broadcast at all."""
        printer = _CapturePrinter()
        self._run(printer, "")

        done_tabs = {
            e.get("tab_id") for e in printer.of_type("subagentDone")
        }
        assert done_tabs == {_VIEWER_TAB}
