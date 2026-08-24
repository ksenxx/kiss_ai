# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the channel-workspace no-leak property (F-RC3).

``KISS_CHANNEL_WORKSPACE`` is process-global and read by a channel
module's ``get_tools()`` while a launch is in flight.  Before the fix,
``enter_workspace`` overwrote it last-writer-wins, so two overlapping
dispatches with DIFFERENT workspaces (each holding it for up to ~15
minutes) made the first task's channel import read the second task's
workspace — the wrong account's credentials.  Now a conflicting
``enter_workspace`` blocks until the other workspace exits (or its
timeout expires, entering nothing), while same-workspace launches
still overlap freely through the reference count.

Real threads against the real registry — no mocks.  The dispatch-level
behavior on top of this registry is covered end-to-end in
``kiss.tests.agents.third_party_agents.test_agent_dispatch`` (a
conflicting dispatch fails loudly) and ``test_kiss_web_launch`` (a
conflicting launcher launch waits).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator

import pytest

from kiss.agents.sorcar import channel_workspace
from kiss.agents.sorcar.channel_workspace import (
    WORKSPACE_ENV_VAR,
    enter_workspace,
    exit_workspace,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Fail fast on cross-test registry leakage and clean the env var."""
    assert channel_workspace._ACTIVE_WORKSPACES == {}
    os.environ.pop(WORKSPACE_ENV_VAR, None)
    yield
    assert channel_workspace._ACTIVE_WORKSPACES == {}, (
        "a test leaked an active workspace"
    )
    os.environ.pop(WORKSPACE_ENV_VAR, None)


def test_same_workspace_launches_overlap_freely() -> None:
    entered = threading.Barrier(3, timeout=30)
    release = threading.Event()

    def launch() -> None:
        assert enter_workspace("shared") is True
        try:
            entered.wait()
            assert release.wait(timeout=30)
        finally:
            exit_workspace("shared")

    threads = [threading.Thread(target=launch, daemon=True) for _ in range(2)]
    for t in threads:
        t.start()
    # Both entered concurrently (the barrier passed) — same-workspace
    # concurrency is preserved.
    entered.wait()
    assert os.environ[WORKSPACE_ENV_VAR] == "shared"
    release.set()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive()
    assert WORKSPACE_ENV_VAR not in os.environ


def test_conflicting_enter_times_out_without_entering() -> None:
    assert enter_workspace("account-a") is True
    try:
        start = time.monotonic()
        assert enter_workspace("account-b", timeout=0.2) is False
        assert time.monotonic() - start >= 0.2
        # Nothing was entered and, crucially, nothing was overwritten:
        # the running launch's get_tools() still reads ITS workspace.
        assert os.environ[WORKSPACE_ENV_VAR] == "account-a"
        assert channel_workspace._ACTIVE_WORKSPACES == {"account-a": 1}
    finally:
        exit_workspace("account-a")
    assert WORKSPACE_ENV_VAR not in os.environ


def test_conflicting_enter_waits_for_active_workspace_to_exit() -> None:
    # Two overlapping dispatch contexts with different workspaces: the
    # second observes the FIRST's workspace the whole time the first
    # is active, and publishes its own only after the first exits.
    assert enter_workspace("account-a") is True
    entered_b = threading.Event()
    seen_while_waiting: list[str | None] = []

    def launch_b() -> None:
        assert enter_workspace("account-b", timeout=30) is True
        entered_b.set()
        try:
            assert os.environ[WORKSPACE_ENV_VAR] == "account-b"
        finally:
            exit_workspace("account-b")

    thread = threading.Thread(target=launch_b, daemon=True)
    thread.start()
    try:
        assert not entered_b.wait(timeout=0.3), (
            "a conflicting workspace must not be entered while another "
            "is active"
        )
        seen_while_waiting.append(os.environ.get(WORKSPACE_ENV_VAR))
    finally:
        exit_workspace("account-a")
    assert entered_b.wait(timeout=30)
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert seen_while_waiting == ["account-a"], (
        "the waiting launch overwrote the active launch's workspace"
    )
    assert WORKSPACE_ENV_VAR not in os.environ


def test_refcounted_exit_keeps_env_until_last_exit() -> None:
    assert enter_workspace("ws") is True
    assert enter_workspace("ws") is True
    exit_workspace("ws")
    assert os.environ[WORKSPACE_ENV_VAR] == "ws"
    exit_workspace("ws")
    assert WORKSPACE_ENV_VAR not in os.environ
