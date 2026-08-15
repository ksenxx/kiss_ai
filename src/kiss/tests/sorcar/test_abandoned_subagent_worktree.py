# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""An abandoned sub-agent outlives the fan-out.

``_await_subagents`` deliberately abandons a child that ignores its
stop event for 15 seconds, and Python cannot kill the thread — it just
stops being waited on.  Two things then go wrong:

* the child's ``work_dir`` is the parent's git worktree, and the
  parent's cleanup deleted that directory out from under the live
  thread; and
* the child's spend after the abandonment was attributed to nobody,
  while the parent's budget checks used the frozen figure.

The test drives the real thing: a real worktree on a real repo, a real
``ThreadPoolExecutor`` fan-out, a real child agent wedged inside a real
(local) model call, the real 15-second grace period, and the real
cleanup path.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.worktree_sorcar_agent import (
    WorktreeSorcarAgent,
    _WorktreeCleanupOutcome,
)
from kiss.core import stop_signal
from kiss.tests.sorcar.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-f5-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


class _WedgedModel:
    """Stand-in model that parks the sub-agent inside its model call.

    A child wedged in a model call polls nothing, which is exactly the
    condition ``_await_subagents`` abandons a child for.
    """

    def __init__(self) -> None:
        """Create the two events the test drives the child with."""
        self.wedged = threading.Event()
        self.release = threading.Event()

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        """Park the sub-agent's request; answer the parent's normally."""
        if "WEDGED" in request_text(request):
            self.wedged.set()
            self.release.wait(120)
        return finish_response("f5 done")


def test_abandoned_subagent_blocks_worktree_deletion_and_is_accounted(
    env: IsolatedKissHome,
) -> None:
    """The worktree survives the live child, and its spend is banked."""
    model = _WedgedModel()
    server = StandInModelServer(model)
    printer = CapturePrinter()
    parent = WorktreeSorcarAgent("f5-parent")
    parent.printer = printer
    parent.model_name = STANDIN_MODEL
    parent.model_config = server.model_config

    wt_work_dir = parent._try_setup_worktree(env.repo, str(env.repo))
    assert wt_work_dir is not None
    wt = parent._wt
    assert wt is not None
    parent.work_dir = str(wt_work_dir)

    parent_stop = threading.Event()
    fanout: dict[str, Any] = {}

    def run_fanout() -> None:
        stop_signal.set_thread_stop_event(parent_stop)
        printer._thread_local.task_id = "f5parenttask"
        try:
            fanout["results"] = parent._run_tasks_parallel(
                ["WEDGED child task"], max_workers=1,
            )
        except BaseException as exc:  # noqa: BLE001 — recorded for assertions
            fanout["error"] = exc
        finally:
            stop_signal.set_thread_stop_event(None)

    thread = threading.Thread(target=run_fanout, daemon=True)
    thread.start()
    try:
        assert model.wedged.wait(60), "the sub-agent never reached the model"

        # Stop the parent.  The child ignores it (it is inside the
        # model call), so after the grace period the parent gives up
        # waiting and unwinds — leaving the child running.
        parent_stop.set()
        thread.join(timeout=120)
        assert not thread.is_alive(), "the parent never abandoned the child"
        assert isinstance(fanout.get("error"), KeyboardInterrupt), fanout

        banked_before = parent.total_tokens_used
        outcome, _ = parent._commit_and_clean_worktree(wt)

        assert outcome is _WorktreeCleanupOutcome.PRESERVED_SUBAGENT_ACTIVE, (
            "the parent removed a worktree an abandoned sub-agent thread "
            f"is still running in (outcome={outcome})"
        )
        assert wt.wt_dir.exists(), (
            "the live sub-agent's working directory was deleted"
        )
    finally:
        model.release.set()
        child_finished = parent.reclaim_abandoned_subagents(timeout=120)
        thread.join(timeout=30)
        server.stop()

    assert child_finished, "the abandoned sub-agent never finished"
    assert parent.total_tokens_used >= banked_before

    # With the child gone the same cleanup now completes normally.
    outcome, _ = parent._commit_and_clean_worktree(wt)
    assert outcome is _WorktreeCleanupOutcome.COMMITTED_AND_REMOVED, outcome
    assert not wt.wt_dir.exists()


def test_post_abandonment_spend_is_banked_into_the_parent(
    env: IsolatedKissHome,
) -> None:
    """A child that keeps working after the parent unwinds is counted.

    The parent's thread is interrupted the way the server interrupts
    it (``PyThreadState_SetAsyncExc``, as
    ``task_runner._force_stop_thread`` does) while the child is still
    working and its own stop event is NOT set — a parent-only stop, or
    any parent-side failure, produces exactly this.  The child then
    runs to completion at full speed, and everything it spends after
    the parent froze its totals used to be attributed to nobody.
    """
    model = _WedgedModel()
    server = StandInModelServer(model)
    printer = CapturePrinter()
    parent = WorktreeSorcarAgent("f5-accounting")
    parent.printer = printer
    parent.model_name = STANDIN_MODEL
    parent.model_config = server.model_config
    parent.work_dir = str(env.repo)

    fanout: dict[str, Any] = {}

    def run_fanout() -> None:
        printer._thread_local.task_id = "f5accounting"
        try:
            fanout["results"] = parent._run_tasks_parallel(
                ["WEDGED child task"], max_workers=1,
            )
        except BaseException as exc:  # noqa: BLE001 — recorded for assertions
            fanout["error"] = exc

    thread = threading.Thread(target=run_fanout, daemon=True)
    thread.start()
    try:
        assert model.wedged.wait(60), "the sub-agent never reached the model"
        _interrupt_thread(thread)
        thread.join(timeout=60)
        assert not thread.is_alive(), "the parent thread never unwound"
        assert isinstance(fanout.get("error"), KeyboardInterrupt), fanout

        frozen_tokens = parent.total_tokens_used
        assert parent._abandoned_subagents, (
            "the parent forgot the child it abandoned, so nothing can "
            "ever account for the rest of its spend"
        )
        model.release.set()
        assert parent.reclaim_abandoned_subagents(timeout=120)
    finally:
        model.release.set()
        thread.join(timeout=30)
        server.stop()

    assert parent.total_tokens_used > frozen_tokens, (
        "the abandoned sub-agent's spend after the parent stopped "
        "waiting was attributed to nobody"
    )


def _interrupt_thread(thread: threading.Thread) -> None:
    """Raise ``KeyboardInterrupt`` inside *thread*, as the server does."""
    import ctypes

    assert thread.ident is not None
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread.ident),
        ctypes.py_object(KeyboardInterrupt),
    )


def test_normal_fanout_registers_nothing_to_reclaim(
    env: IsolatedKissHome,
) -> None:
    """A fan-out whose children all finish leaves no residue.

    The abandon bookkeeping must not fire on the happy path, or every
    worktree cleanup would start paying the reclaim wait.
    """
    server = StandInModelServer(lambda request: finish_response("f5 quick"))
    printer = CapturePrinter()
    parent = WorktreeSorcarAgent("f5-clean")
    parent.printer = printer
    parent.model_name = STANDIN_MODEL
    parent.model_config = server.model_config
    parent.work_dir = str(env.repo)
    try:
        results = parent._run_tasks_parallel(["quick task"], max_workers=1)
    finally:
        server.stop()

    assert len(results) == 1
    assert parent._abandoned_subagents == []
    assert parent.reclaim_abandoned_subagents() is True


def test_reclaim_is_safe_without_a_worktree(env: IsolatedKissHome) -> None:
    """``reclaim_abandoned_subagents`` is a no-op on a fresh agent."""
    agent = WorktreeSorcarAgent("f5-fresh")
    assert agent.reclaim_abandoned_subagents(timeout=1.0) is True
    assert Path(env.repo).exists()
