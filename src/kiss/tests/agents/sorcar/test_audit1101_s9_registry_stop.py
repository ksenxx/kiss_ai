# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop landing in a registry subscriber must not lose the notification for good.

``_swap_registry`` advances ``_last_broadcast`` before ``_publish``
runs the subscribers.  A task thread rescans too (``get_command`` on a
miss), so the runner's Stop (an injected ``KeyboardInterrupt``) can
land inside a subscriber before it delivered the snapshot.  The Stop
propagated correctly, but the advanced ``_last_broadcast`` made every
later rescan of the same names a no-op: the daemon's ``seaCommands``
broadcaster never learned about the new command until the registry
changed again.  A publication that a ``BaseException`` interrupts now
forgets its broadcast so the next rescan delivers the snapshot.

Real threads, a real injected ``KeyboardInterrupt``, the real
subscriber mechanism; nothing is mocked.
"""

from __future__ import annotations

import inspect
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar import sea_commands
from kiss.core.config import kiss_home
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.agents.sorcar.conftest import isolated_sea_registry


@pytest.fixture(autouse=True)
def _reset_sea_commands() -> Iterator[None]:
    """Isolate the registry state and the session-wide ``SEAS.md`` per test."""
    with isolated_sea_registry():
        yield


def test_a_stopped_publication_is_repeated_by_the_next_rescan(tmp_path: Path) -> None:
    """The subscriber a Stop interrupted receives the current snapshot on the next rescan."""
    folder = tmp_path / "seas"
    name = "stoppedpublish"
    script = folder / name / f"{name}_sea.py"
    script.parent.mkdir(parents=True)
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    sea_commands.refresh_registry()

    entered = threading.Event()
    release = threading.Event()
    deliveries: list[list[str]] = []

    def subscriber(snapshot: list[str]) -> None:
        entered.set()
        assert release.wait(10)
        deliveries.append(snapshot)

    sea_commands.subscribe(subscriber)
    script.write_text("# only the registry needs this file\n", encoding="utf-8")
    outcome: list[str] = []

    def rescan() -> None:
        try:
            sea_commands.refresh_registry()
        except KeyboardInterrupt:
            outcome.append("KeyboardInterrupt")

    worker = threading.Thread(target=rescan)
    worker.start()
    try:
        assert entered.wait(10), "the subscriber never ran"
        assert inject_async_exception(worker.ident or 0, KeyboardInterrupt) == 1
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    assert outcome == ["KeyboardInterrupt"], "the Stop must propagate out of the rescan"
    assert deliveries == [], "the interrupted subscriber must not have delivered"
    assert name in sea_commands.list_commands()

    # The correcting rescan: nothing changed on disk, yet the snapshot
    # the Stop swallowed must reach the subscriber now.
    sea_commands.refresh_registry()
    assert len(deliveries) == 1 and name in deliveries[0], deliveries
    # And only once: the dedupe is back in force after the delivery.
    sea_commands.refresh_registry()
    assert len(deliveries) == 1, deliveries


def _swap_return_line() -> int:
    """The line of ``_swap_registry`` that returns the callbacks, after the stamp advanced."""
    source, start = inspect.getsourcelines(sea_commands._swap_registry)
    offset = next(
        index for index, text in enumerate(source)
        if text.strip() == "return snapshot, list(_subscribers)"
    )
    return start + offset


def test_a_stop_before_the_swap_returns_is_repeated_by_the_next_rescan(tmp_path: Path) -> None:
    """A Stop after ``_last_broadcast`` advanced, before any subscriber ran, is not final either."""
    folder = tmp_path / "seas"
    name = "stoppedswap"
    script = folder / name / f"{name}_sea.py"
    script.parent.mkdir(parents=True)
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    sea_commands.refresh_registry()
    deliveries: list[list[str]] = []
    sea_commands.subscribe(deliveries.append)
    script.write_text("# only the registry needs this file\n", encoding="utf-8")

    # The worker pauses on the swap's return statement (a line hook, the
    # debugger's mechanism): the stamp is advanced, the callbacks are not
    # yet handed out.  The Stop lands there.
    paused = threading.Event()
    release = threading.Event()
    return_line = _swap_return_line()
    swap_code = sea_commands._swap_registry.__code__

    def pause_at_return(frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code is swap_code and event == "line" and frame.f_lineno == return_line:
            paused.set()
            assert release.wait(10)
        return pause_at_return

    outcome: list[str] = []

    def rescan() -> None:
        sys.settrace(pause_at_return)
        try:
            sea_commands.refresh_registry()
        except KeyboardInterrupt:
            outcome.append("KeyboardInterrupt")
        finally:
            sys.settrace(None)

    worker = threading.Thread(target=rescan)
    worker.start()
    try:
        assert paused.wait(10), "the swap never reached its return"
        assert inject_async_exception(worker.ident or 0, KeyboardInterrupt) == 1
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    assert outcome == ["KeyboardInterrupt"], "the Stop must propagate out of the rescan"
    assert deliveries == [], "no subscriber ran before the Stop"
    assert name in sea_commands.list_commands()

    # Nothing changed on disk: the swallowed snapshot must still reach the
    # subscriber now, and only once.
    sea_commands.refresh_registry()
    assert len(deliveries) == 1 and name in deliveries[0], deliveries
    sea_commands.refresh_registry()
    assert len(deliveries) == 1, deliveries
