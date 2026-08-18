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

import tempfile
from pathlib import Path
from typing import Any

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import run_tasks_parallel
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.tests.agents.sorcar.test_notification_all_tabs_fanout import (  # noqa: F401
    _LLMUnavailable,
    _make_repo,
    _setup_worktree_agent,
)


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
        """With no parent task id, the subscribed viewer tab must still
        be told the sub-agent is done — previously nothing was
        broadcast at all.

        The sub-agent's own synthetic tab is notified too: the single
        fan-out engine always names its children ``task-{key}__sub_{n}``
        (falling back to a generated key when the parent has no
        persisted task), assigns that id to the child as its
        ``_tab_id``, and the child registers under it — so it is a real
        tab, not the phantom the old base-only copy signalled.
        """
        printer = _CapturePrinter()
        self._run(printer, "")

        done_tabs = {
            e.get("tab_id") for e in printer.of_type("subagentDone")
        }
        assert _VIEWER_TAB in done_tabs
        synthetic = done_tabs - {_VIEWER_TAB}
        assert len(synthetic) == 1
        synthetic_tab = synthetic.pop()
        assert synthetic_tab is not None
        assert synthetic_tab.endswith("__sub_0")


class TestBroadcastTransientPrimitive:
    """``JsonPrinter.broadcast_transient`` — the printer-side
    "transient, all-watching-tabs" primitive: callers pass a plain
    event plus their task id; the printer resolves the watching tabs
    itself and stamps one ``tabId`` copy per tab, uniformly (no
    owner/viewer distinction).
    """

    def setup_method(self) -> None:
        agent_state.agent_states.clear()

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()

    def test_one_stamped_copy_per_watching_tab(self) -> None:
        printer = _CapturePrinter()
        printer.subscribe_tab("5151", "tab-a")
        printer.subscribe_tab("5151", "tab-b")
        assert printer._task_key() == ""  # thread-local unset

        printer.broadcast_transient(
            {"type": "notification", "id": "n1", "message": "hi"},
            task_id="5151",
        )

        notifs = printer.of_type("notification")
        assert {e["tabId"] for e in notifs} == {"tab-a", "tab-b"}
        assert len(notifs) == 2
        assert all(e["message"] == "hi" for e in notifs)

    def test_seed_tab_is_one_more_uniform_target_deduped(self) -> None:
        printer = _CapturePrinter()
        printer.subscribe_tab("5252", "tab-a")

        printer.broadcast_transient(
            {"type": "notification", "id": "n1", "message": "hi"},
            task_id="5252",
            tab_id="tab-a",  # already subscribed: no duplicate
        )
        assert len(printer.of_type("notification")) == 1

        printer.broadcast_transient(
            {"type": "notification", "id": "n2", "message": "yo"},
            task_id="5252",
            tab_id="tab-new",  # unknown to the registry: still reached
        )
        second = [e for e in printer.of_type("notification") if e["id"] == "n2"]
        assert {e["tabId"] for e in second} == {"tab-a", "tab-new"}

    def test_thread_local_task_id_takes_precedence(self) -> None:
        printer = _CapturePrinter()
        printer.subscribe_tab("100", "tab-of-100")
        printer.subscribe_tab("200", "tab-of-200")
        printer._thread_local.task_id = "100"
        try:
            printer.broadcast_transient(
                {"type": "notification", "id": "n1", "message": "hi"},
                task_id="200",
            )
        finally:
            printer._thread_local.task_id = None

        notifs = printer.of_type("notification")
        assert {e["tabId"] for e in notifs} == {"tab-of-100"}

    def test_fallback_single_copy_when_nothing_resolvable(self) -> None:
        """With no subscribers and no seed tab, exactly ONE copy with
        ``tabId: ""`` still goes out — the stamp keeps the transient
        semantics and printers that render locally (CLI console
        toasts) still show it."""
        printer = _CapturePrinter()
        printer.broadcast_transient(
            {"type": "notification", "id": "n1", "message": "hi"},
        )
        notifs = printer.of_type("notification")
        assert len(notifs) == 1
        assert notifs[0]["tabId"] == ""

    def test_transient_copies_are_never_recorded(self) -> None:
        printer = _CapturePrinter()
        printer.subscribe_tab("5353", "tab-a")
        printer.broadcast_transient(
            {"type": "notification", "id": "n1", "message": "hi"},
            task_id="5353",
        )
        assert printer._recordings == {}


class TestTransientBroadcastNearTeardown:
    """Auto-commit toasts and model-picker updates still reach every
    watching tab when the printer's thread-local ``task_id`` has been
    cleared near teardown (``cleanup_task`` already ran: ``_task_ui``
    dropped, subscriber set lingering) — the agent's explicit
    ``_last_task_id`` is the only remaining link.
    """

    def setup_method(self) -> None:
        agent_state.agent_states.clear()

    def teardown_method(self) -> None:
        agent_state.agent_states.clear()

    def _teardown_printer(self, task_id: str) -> _CapturePrinter:
        """A printer in the near-teardown state for *task_id*."""
        printer = _CapturePrinter()
        printer.register_task_ui(task_id, "tab-launch", "conn-1")
        printer.subscribe_tab(task_id, "tab-viewer")
        printer._thread_local.task_id = task_id
        printer.cleanup_task(task_id)  # drops _task_ui, subscribers linger
        printer._thread_local.task_id = None  # run thread unbound
        assert printer._task_key() == ""
        assert printer.task_ui(task_id) == ("", "")
        return printer

    def test_autocommit_toasts_after_thread_local_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            agent, wt_dir = _setup_worktree_agent(Path(tmp_str), "teardown")
            (wt_dir / "new.txt").write_text("hello\n")

            printer = self._teardown_printer("6161")
            agent.printer = printer  # type: ignore[assignment]
            agent._tab_id = "tab-launch"
            agent._last_task_id = "6161"

            with _LLMUnavailable():
                assert agent._auto_commit_worktree() is True

            notifs = printer.of_type("notification")
            expected = {"tab-launch", "tab-viewer"}
            assert {e["tabId"] for e in notifs} == expected
            assert len(notifs) == 4  # two stages x two tabs
            assert len({e["id"] for e in notifs}) == 1
            assert printer._recordings == {}  # transient: nothing recorded

    def test_model_pick_after_thread_local_cleared(self) -> None:
        printer = self._teardown_printer("6262")

        agent = ChatSorcarAgent("teardown-picker")
        agent.printer = printer  # type: ignore[assignment]
        agent._tab_id = "tab-launch"  # type: ignore[attr-defined]
        agent._last_task_id = "6262"

        agent._show_model_in_picker("model-t")

        picks = printer.of_type("modelPick")
        assert {e["tabId"] for e in picks} == {"tab-launch", "tab-viewer"}
        assert all(e["model"] == "model-t" for e in picks)
        # Both tabs are remembered so restore_model_pick hands the
        # picker back when the task ends.
        assert {"tab-launch", "tab-viewer"} <= printer._model_override_tabs
