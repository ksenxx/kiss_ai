# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The merge-conflict resolver bills the parent's epoch that launched it.

``run_merge_sea`` attributes the resolver's spend to the parent agent
when the child finishes.  An interactive merge can be resolved after
the parent has already started its NEXT task (``reset_usage`` swapped
the ledger epoch); like every other side channel (the task-update
child, dispatch fan-out) the attribution must be bound to the epoch
captured when the resolver started, so the old task's spend settles
in the discarded ledger instead of inflating the new task's budget.
"""

from __future__ import annotations

import functools
import os
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import CodeType, FrameType
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.merge_conflict_resolver import run_merge_sea
from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage, _live_agent_usage
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
)

_DISABLE_ENV = "KISS_DISABLE_TASK_CLASSIFIER"


@pytest.fixture
def home() -> Iterator[IsolatedKissHome]:
    """Isolated ``KISS_HOME`` with the task classifier off."""
    saved = os.environ.get(_DISABLE_ENV)
    os.environ[_DISABLE_ENV] = "1"
    isolated = IsolatedKissHome("kiss-merge-epoch-")
    isolated.write_config()
    try:
        yield isolated
    finally:
        if saved is None:
            os.environ.pop(_DISABLE_ENV, None)
        else:
            os.environ[_DISABLE_ENV] = saved
        isolated.cleanup()


def _parent_at(standin: StandInModelServer, repo: Path) -> ChatSorcarAgent:
    parent = ChatSorcarAgent("merge-parent")
    parent.work_dir = str(repo)
    parent.model_name = STANDIN_MODEL
    parent.model_config = standin.model_config
    return parent


def test_spend_lands_in_the_launching_epoch(home: IsolatedKissHome) -> None:
    """The parent resets for its next task mid-merge: that task is not billed."""
    parent: ChatSorcarAgent | None = None

    def respond(_request: dict[str, Any]) -> dict[str, Any]:
        # The resolver's one model call: meanwhile the parent starts
        # its next task, swapping its usage epoch.
        assert parent is not None
        parent.reset_usage()
        return finish_response("conflict resolved")

    standin = StandInModelServer(respond)
    try:
        parent = _parent_at(standin, home.repo)
        run_merge_sea(parent, "resolve f.txt", home.repo)
    finally:
        standin.stop()
    assert parent.usage_snapshot() == (0.0, 0, 0)


def test_spend_is_attributed_without_a_reset(home: IsolatedKissHome) -> None:
    """Without an intervening reset the resolver's steps reach the parent."""
    standin = StandInModelServer(lambda _req: finish_response("conflict resolved"))
    try:
        parent = _parent_at(standin, home.repo)
        run_merge_sea(parent, "resolve f.txt", home.repo)
    finally:
        standin.stop()
    _budget, tokens, steps = parent.usage_snapshot()
    assert steps >= 1 and tokens > 0


def _stop_as_the_resolver_calls(code: CodeType, frame: FrameType, event: str, arg: Any) -> None:
    """Profile hook: inject the task Stop as ``run_merge_sea`` calls the function of *code*."""
    if (
        event == "call" and frame.f_code is code
        and frame.f_back is not None and frame.f_back.f_code is run_merge_sea.__code__
    ):
        sys.setprofile(None)
        inject_async_exception(threading.get_ident(), KeyboardInterrupt)


@pytest.mark.parametrize("step", [_live_agent_usage, _attribute_sub_usage])
def test_a_stop_during_the_accounting_still_closes_the_resolver_tab(
    home: IsolatedKissHome, step: Callable[..., Any],
) -> None:
    """A Stop landing in the spend accounting must not skip the ``subagentDone`` broadcast.

    The resolver's ``finally`` read the child's spend, attributed it and
    only then told the watching surfaces the child was done.  A Stop
    delivered during either *step* (the attribution's shared offset-lock
    wait is a check point, the reading a Python call) unwound past the
    broadcast, and the finished merge tab stayed open with its spinner
    on every surface.
    """
    standin = StandInModelServer(lambda _req: finish_response("conflict resolved"))
    try:
        parent = _parent_at(standin, home.repo)
        parent.printer = CapturePrinter()
        sys.setprofile(functools.partial(_stop_as_the_resolver_calls, step.__code__))
        try:
            with pytest.raises(KeyboardInterrupt):
                run_merge_sea(parent, "resolve f.txt", home.repo)
        finally:
            sys.setprofile(None)
    finally:
        standin.stop()
    done = parent.printer.events_of_type("subagentDone")
    assert any(e["tab_id"].endswith("__merge") for e in done), parent.printer.captured
