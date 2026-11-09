# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop must land on a task blocked in ``enter_workspace`` within seconds.

``task_runner._apply_sea`` runs ``enter_workspace`` on the task thread
with a 900 s timeout, and the daemon stops a task by injecting
``KeyboardInterrupt`` into that thread.  CPython raises an injected
exception only when the thread next runs bytecode; a single
``Condition.wait(900)`` is one C-level sleep, so before the fix the
Stop took effect only when the conflicting workspace exited or the
whole timeout expired.  The wait is now sliced so the injection lands
within one slice, and the condition's mutex is taken through
``locked_call`` so an injection at the ``__enter__`` boundary cannot
strand it.

Real threads against the real registry — no mocks.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator

import pytest

from kiss.agents.sorcar import channel_workspace
from kiss.agents.sorcar.channel_workspace import (
    WORKSPACE_ENV_VAR,
    enter_workspace,
    exit_workspace,
    held_workspace,
)
from kiss.core.tool_interrupt import inject_async_exception


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Fail fast on cross-test registry leakage and clean the env var."""
    assert channel_workspace._ACTIVE_WORKSPACES == {}
    os.environ.pop(WORKSPACE_ENV_VAR, None)
    yield
    assert channel_workspace._ACTIVE_WORKSPACES == {}, "a test leaked an active workspace"
    os.environ.pop(WORKSPACE_ENV_VAR, None)


class _BlockedTask(threading.Thread):
    """A task thread waiting for a workspace another task holds."""

    def __init__(self) -> None:
        super().__init__(name="blocked-channel-task")
        self.waiting = threading.Event()
        self.entered: bool | None = None
        self.interrupted_after: float | None = None
        self.held = "unset"

    def run(self) -> None:
        self.waiting.set()
        started = time.monotonic()
        try:
            self.entered = enter_workspace("account-b", timeout=60.0)
        except KeyboardInterrupt:
            self.interrupted_after = time.monotonic() - started
        self.held = held_workspace()


def test_injected_stop_lands_while_waiting_for_a_conflicting_workspace() -> None:
    assert enter_workspace("account-a") is True
    thread = _BlockedTask()
    try:
        thread.start()
        assert thread.waiting.wait(5)
        time.sleep(0.3)  # let the thread reach the condition wait
        assert inject_async_exception(thread.ident or 0, KeyboardInterrupt) == 1
        thread.join(10)
        assert not thread.is_alive(), "the Stop did not land within 10 s of a 60 s wait"
    finally:
        exit_workspace("account-a")
    assert thread.entered is None
    assert thread.interrupted_after is not None and thread.interrupted_after < 5.0
    assert thread.held == ""
    assert channel_workspace._ACTIVE_WORKSPACES == {}
    # The condition's mutex is free: a later launch is not wedged.
    assert enter_workspace("account-c", timeout=1.0) is True
    exit_workspace("account-c")


def _wait_for_conflicting_workspace() -> None:
    """One task's wait for a workspace another task holds (always times out)."""
    assert enter_workspace("hammered", timeout=0.01) is False
    assert held_workspace() == ""


def run_hammer() -> None:
    """Inject hundreds of Stops into a thread waiting in ``enter_workspace``; nothing leaks.

    Another workspace stays active for the whole run, so the hammered
    thread only ever waits and times out: every interrupt lands in the
    sliced wait or at the mutex boundary, the two places a leak could
    strand ``_WORKSPACE_COND`` or corrupt the registry.
    """
    from kiss.tests.lock_injection import hammer

    assert enter_workspace("other") is True
    hammer(
        _wait_for_conflicting_workspace,
        {"_WORKSPACE_COND": channel_workspace._WORKSPACE_COND},
        contend=channel_workspace._WORKSPACE_COND,
    )
    assert channel_workspace._ACTIVE_WORKSPACES == {"other": 1}, (
        channel_workspace._ACTIVE_WORKSPACES
    )
    exit_workspace("other")
    assert channel_workspace._ACTIVE_WORKSPACES == {}
    assert WORKSPACE_ENV_VAR not in os.environ
    assert enter_workspace("after", timeout=1.0) is True
    exit_workspace("after")


def test_workspace_lock_survives_injection() -> None:
    """The condition's mutex and the registry survive injected Stops (child process)."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_hammer()"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]


def _exit_stopped_at(nth: int) -> tuple[int, dict[str, object]]:
    """On a task thread: enter ``account-a``, exit it with a Stop at its *nth* check, clean up.

    An injected ``KeyboardInterrupt`` lands where the thread next checks
    for it: right after a C call returned, or on entering a Python
    callee.  A profile hook raises at the *nth* such point inside
    ``_exit_workspace_locked``'s own body (none when *nth* is past them),
    then the task runner's final cleanup runs: release whatever the
    thread still records as held.

    Returns:
        The number of check points the exit body has, and the outcome.
    """
    top = channel_workspace._exit_workspace_locked.__code__
    seen = 0
    outcome: dict[str, object] = {}

    def stop_at(frame: object, event: str, arg: object) -> None:
        nonlocal seen
        code = frame.f_code  # type: ignore[attr-defined]
        back = frame.f_back  # type: ignore[attr-defined]
        if (event == "c_return" and code is top) or (
            event == "call" and back is not None and back.f_code is top
        ):
            seen += 1
            if seen == nth:
                raise KeyboardInterrupt

    def task() -> None:
        assert enter_workspace("account-a") is True
        sys.setprofile(stop_at)
        try:
            exit_workspace("account-a")
        except KeyboardInterrupt:
            outcome["stopped"] = True
        finally:
            sys.setprofile(None)
        held = held_workspace()
        outcome["held_after_stop"] = held
        if held:  # task_runner's cleanup: release exactly what this thread holds
            exit_workspace(held)

    thread = threading.Thread(target=task, name="stopped-channel-task")
    thread.start()
    thread.join(10)
    assert not thread.is_alive()
    return seen, outcome


def test_a_stop_inside_the_exit_leaks_no_hold_and_releases_none_twice() -> None:
    """Wherever a Stop lands in ``exit_workspace``, the runner's cleanup releases the hold once.

    The exit cleared the thread's record first and decremented the count
    after: a Stop in between left a count nobody could release, and every
    later launch of another account waited out its whole 900 s timeout.
    A second run of the same account holds the workspace too, so a
    release counted twice shows as well as one never made.
    """
    assert enter_workspace("account-a") is True  # the other run of the same account
    try:
        nth = 1
        while True:
            points, outcome = _exit_stopped_at(nth)
            assert channel_workspace._ACTIVE_WORKSPACES == {"account-a": 1}, (nth, outcome)
            assert os.environ.get(WORKSPACE_ENV_VAR) == "account-a"
            if nth > points:  # no Stop raised any more: the plain exit
                assert not outcome.get("stopped")
                break
            assert outcome.get("stopped") is True, (nth, outcome)
            nth += 1
        assert nth > 2  # the exit has check points: the Stop was really exercised
    finally:
        exit_workspace("account-a")
    assert channel_workspace._ACTIVE_WORKSPACES == {}
    assert WORKSPACE_ENV_VAR not in os.environ
    # The last hold of the account: its exit also withdraws the env var.
    assert enter_workspace("account-a") is True
    exit_workspace("account-a")
    assert WORKSPACE_ENV_VAR not in os.environ
    assert enter_workspace("account-b", timeout=1.0) is True  # nothing left to wait for
    exit_workspace("account-b")
