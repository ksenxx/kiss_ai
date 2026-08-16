# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Branch coverage for the single parallel fan-out engine.

The behaviour tests live in ``test_parallel_engine_shared.py``
and ``test_abandoned_subagent_worktree.py``; this module drives
the engine's remaining paths — no printer at all, a printer whose
viewer registry already lists the child's tab, and a printer whose
broadcast raises — with real agents, real threads and the local
stand-in model.  The parentless-abandon branch, which depends only
on ``kiss.core`` and ``kiss.agents.sorcar``, lives in
``kiss.tests.agents.sorcar.test_parallel_engine_branches``.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.sorcar_agent import run_tasks_parallel
from kiss.core.print_to_console import ConsolePrinter
from kiss.tests.sorcar.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-f1b-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


def test_engine_runs_without_a_printer(env: IsolatedKissHome) -> None:
    """A printer-less caller (CLI, channel agent) still gets results."""
    server = StandInModelServer(lambda request: finish_response("no printer"))
    try:
        results = run_tasks_parallel(
            ["headless task"],
            max_workers=1,
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(env.repo),
        )
    finally:
        server.stop()

    assert len(results) == 1
    assert "no printer" in results[0]


def test_subagent_done_is_not_duplicated_for_a_subscribed_tab(
    env: IsolatedKissHome,
) -> None:
    """A viewer already watching the child is notified exactly once."""
    server = StandInModelServer(lambda request: finish_response("subscribed"))
    printer = _PreSubscribedPrinter()
    totals: dict[str, float] = {}
    try:
        run_tasks_parallel(
            ["subscribed task"],
            max_workers=1,
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(env.repo),
            printer=printer,
            totals_out=totals,
        )
    finally:
        server.stop()

    tabs = [str(e.get("tab_id", "")) for e in printer.events_of_type("subagentDone")]
    assert sorted(tabs) == sorted(set(tabs)), f"duplicate notifications: {tabs}"
    assert printer.claimed_tab in tabs
    assert totals["total_tokens_used"] > 0


def test_a_failing_broadcast_does_not_break_the_fanout(
    env: IsolatedKissHome,
) -> None:
    """A dead broadcast channel must not escape the fan-out.

    Every broadcast this printer attempts raises, as a closed
    WebSocket would.  The child's own run cannot survive that, but the
    fan-out must still return one result per task instead of
    propagating the error into the parent's tool call — and the
    ``subagentDone`` attempt in the worker's ``finally`` must be
    swallowed rather than masking the child's real outcome.
    """
    server = StandInModelServer(lambda request: finish_response("survived"))
    printer = _ExplodingPrinter()
    try:
        results = run_tasks_parallel(
            ["broadcast task"],
            max_workers=1,
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(env.repo),
            printer=printer,
        )
    finally:
        server.stop()

    assert len(results) == 1
    assert "broadcast channel closed" in results[0]
    assert printer.events_of_type("subagentDone"), (
        "the worker never attempted its subagentDone broadcast"
    )


def test_engine_runs_with_a_printer_that_has_no_viewer_registry(
    env: IsolatedKissHome,
) -> None:
    """A real ``ConsolePrinter`` has no ``_fanout_targets`` registry.

    That is the production configuration of every non-server caller
    (CLI, channel agents): the sub-agent notification is then simply a
    no-op, and the child's result must be unaffected.
    """
    server = StandInModelServer(lambda request: finish_response("plain"))
    printer = ConsolePrinter(file=io.StringIO())
    try:
        results = run_tasks_parallel(
            ["plain printer task"],
            max_workers=1,
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(env.repo),
            printer=printer,
        )
    finally:
        server.stop()

    assert "plain" in results[0]


@pytest.mark.parametrize("printer_factory", [
    lambda: _BrokenRegistryPrinter(),
    lambda: _NonListRegistryPrinter(),
])
def test_a_broken_viewer_registry_never_reaches_the_caller(
    env: IsolatedKissHome, printer_factory: Any,
) -> None:
    """Viewer-registry misbehaviour is contained by the worker.

    Whether the registry raises or answers with something that is not
    a list of tab ids, the child's result must come back intact — the
    notification is best-effort, the result is not.
    """
    server = StandInModelServer(lambda request: finish_response("intact"))
    printer = printer_factory()
    try:
        results = run_tasks_parallel(
            ["registry task"],
            max_workers=1,
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(env.repo),
            printer=printer,
        )
    finally:
        server.stop()

    assert "intact" in results[0]


def test_subagent_done_fires_once_when_the_registry_already_has_the_tab(
    env: IsolatedKissHome,
) -> None:
    """The child's own synthetic tab is not added to the list twice."""
    parent = ChatSorcarAgent("f1b-parent")
    parent._last_task_id = "aaaabbbbccccdddd"
    sub_tab_id = f"task-{parent.last_task_id}__sub_0"
    server = StandInModelServer(lambda request: finish_response("once"))
    printer = _FixedRegistryPrinter([sub_tab_id])
    try:
        run_tasks_parallel(
            ["registered task"],
            max_workers=1,
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(env.repo),
            printer=printer,
            parent_agent=parent,
        )
    finally:
        server.stop()

    tabs = [str(e.get("tab_id", "")) for e in printer.events_of_type("subagentDone")]
    assert tabs == [sub_tab_id], tabs


def test_only_the_live_child_is_tracked_after_an_abandon(
    env: IsolatedKissHome,
) -> None:
    """A sibling that finished before the abandon is not tracked.

    ``_register_abandoned`` must skip completed futures, or the parent
    would wait on — and re-count — children that are already done.
    """
    finished = threading.Event()
    wedged = threading.Event()
    release = threading.Event()

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        from kiss.tests.sorcar.parallel_agent_harness import request_text

        if "WEDGED" in request_text(request):
            wedged.set()
            release.wait(120)
        else:
            finished.set()
        return finish_response("branch child")

    server = StandInModelServer(responder)
    printer = CapturePrinter()
    parent = ChatSorcarAgent("f1b-abandon")
    parent.printer = printer
    parent.model_name = STANDIN_MODEL
    parent.model_config = server.model_config
    parent.work_dir = str(env.repo)

    def run_fanout() -> None:
        try:
            parent._run_tasks_parallel(
                ["QUICK task", "WEDGED task"], max_workers=2,
            )
        except BaseException:  # noqa: BLE001, S110 — expected interrupt
            pass

    thread = threading.Thread(target=run_fanout, daemon=True)
    thread.start()
    try:
        assert finished.wait(60) and wedged.wait(60)
        _interrupt(thread)
        thread.join(timeout=60)
        assert not thread.is_alive()
        assert len(parent._abandoned_subagents) == 1, (
            "the finished sibling was tracked as if it were abandoned"
        )
    finally:
        release.set()
        parent.reclaim_abandoned_subagents(timeout=120)
        thread.join(timeout=30)
        server.stop()


def _interrupt(thread: threading.Thread) -> None:
    """Raise ``KeyboardInterrupt`` inside *thread*, as the server does."""
    import ctypes

    assert thread.ident is not None
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread.ident), ctypes.py_object(KeyboardInterrupt),
    )


class _FixedRegistryPrinter(CapturePrinter):
    """Printer whose viewer registry returns a fixed tab list."""

    def __init__(self, tabs: list[str]) -> None:
        """Create the printer with the tab list to hand back."""
        super().__init__()
        self.tabs = tabs

    def _fanout_targets(self, task_id: Any) -> list[str]:
        """Return the configured tab list for any task."""
        return list(self.tabs)


class _BrokenRegistryPrinter(CapturePrinter):
    """Printer whose viewer registry raises, as a torn state would."""

    def _fanout_targets(self, task_id: Any) -> list[str]:
        """Fail the lookup."""
        raise RuntimeError("viewer registry unavailable")


class _NonListRegistryPrinter(CapturePrinter):
    """Printer whose viewer registry answers with the wrong type."""

    def _fanout_targets(self, task_id: Any) -> list[str]:
        """Return something that is not a list of tab ids."""
        return None  # type: ignore[return-value]


class _PreSubscribedPrinter(CapturePrinter):
    """Printer whose viewer registry already claims every sub-task."""

    def __init__(self) -> None:
        """Create the printer with one pre-claimed viewer tab."""
        super().__init__()
        self.claimed_tab = "viewer-tab-1"

    def _fanout_targets(self, task_id: Any) -> list[str]:
        """Claim the child's task for one viewer tab."""
        return [self.claimed_tab]


class _ExplodingPrinter(CapturePrinter):
    """Printer whose broadcast always fails, as a dead socket would."""

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record the attempt, then fail like a closed connection."""
        super().broadcast(event)
        raise RuntimeError("broadcast channel closed")


def test_engine_binds_a_fresh_stop_event_per_worker(
    env: IsolatedKissHome,
) -> None:
    """Workers are reused, so each run must rebind and then clear."""
    server = StandInModelServer(lambda request: finish_response("two tasks"))
    printer = CapturePrinter()
    printer._thread_local.stop_event = threading.Event()
    try:
        results = run_tasks_parallel(
            ["task one", "task two"],
            max_workers=1,
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(env.repo),
            printer=printer,
        )
    finally:
        server.stop()

    assert len(results) == 2
