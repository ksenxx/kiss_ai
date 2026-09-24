# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Subscriber notifications from ``refresh_registry`` arrive in publish order.

Regression test for an audit finding: ``refresh_registry()`` used to
compare-and-swap ``_last_broadcast`` under ``_lock`` and then invoke the
subscriber callbacks with no lock held at all.  Two concurrent rescans
that published snapshots A then B could therefore deliver B's callback
before A's, leaving every subscriber (the daemon's ``seaCommands``
broadcast) with the stale list A -- and because ``_last_broadcast``
already equalled B, no later unchanged rescan ever corrected it.

The test forces exactly that interleaving with real threads and a real
``SEAS.md`` folder: a subscriber blocks the first delivery until the
second refresh has had the chance to publish, then asserts the LAST
list handed to the subscriber equals the final registry.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import sea_commands
from kiss.core.config import kiss_home


@pytest.fixture(autouse=True)
def _reset_sea_commands() -> Iterator[None]:
    """Drop the module's in-memory state before and after each test."""
    sea_commands._reset_for_tests()
    yield
    sea_commands._reset_for_tests()


def _touch_sea(folder: Path, name: str) -> Path:
    """Create an empty ``<name>_sea.py`` file in *folder* and return it."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}_sea.py"
    path.write_text("# stub SEA for tests\n", encoding="utf-8")
    return path


def _write_seas_md(lines: list[str]) -> None:
    """Overwrite ``~/.kiss/SEAS.md`` with *lines* (one per row)."""
    home = kiss_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "SEAS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_concurrent_refreshes_notify_subscribers_in_publish_order(
    tmp_path: Path,
) -> None:
    """The last list a subscriber receives is the final registry.

    Interleaving forced by the test:

    1. Thread 1 rescans with only ``alpha_sea.py`` present and publishes
       snapshot A.  The subscriber records A's delivery has started and
       then blocks on ``release_first``.
    2. The test adds ``beta_sea.py`` and starts thread 2, which
       publishes snapshot B (A + beta).  On the unfixed code thread 2
       delivers B immediately and returns; on the fixed code it waits
       for A's delivery to finish first.
    3. The test releases the first delivery and joins both threads.

    With the bug the subscriber sees ``[B, A]`` and ends on the stale
    list A while ``list_commands()`` reports B.  With the fix it sees
    ``[A, B]``.  The subscriber also calls back into
    :func:`list_commands` to prove the fix did not reintroduce the
    re-entrancy deadlock the module guards against.
    """
    folder = tmp_path / "seas"
    _touch_sea(folder, "alpha")
    _write_seas_md([str(folder)])

    delivered: list[list[str]] = []
    first_delivery_started = threading.Event()
    release_first = threading.Event()
    errors: list[BaseException] = []

    def subscriber(commands: list[str]) -> None:
        # Re-entrancy check: a subscriber may read the registry.
        sea_commands.list_commands()
        if not first_delivery_started.is_set():
            first_delivery_started.set()
            assert release_first.wait(timeout=10.0), "first delivery never released"
        delivered.append(commands)

    def run_refresh() -> None:
        try:
            sea_commands.refresh_registry()
        except BaseException as exc:  # pragma: no cover - surfaces thread failures
            errors.append(exc)

    sea_commands.subscribe(subscriber)

    first = threading.Thread(target=run_refresh, name="refresh-A")
    first.start()
    assert first_delivery_started.wait(timeout=10.0), "snapshot A was never delivered"

    _touch_sea(folder, "beta")
    second = threading.Thread(target=run_refresh, name="refresh-B")
    second.start()
    # Give thread 2 the chance to publish AND deliver B before A's
    # delivery is released.  Unfixed code lets it finish here; fixed
    # code parks it until A's delivery completes, so this times out.
    second.join(timeout=1.0)

    release_first.set()
    first.join(timeout=10.0)
    second.join(timeout=10.0)
    assert not first.is_alive(), "refresh-A deadlocked"
    assert not second.is_alive(), "refresh-B deadlocked"
    assert not errors, errors

    final = sea_commands.list_commands()
    assert "alpha" in final and "beta" in final
    assert delivered, "subscriber never notified"
    assert delivered[-1] == final, (
        f"subscriber ended on stale snapshot {delivered[-1]!r}; "
        f"registry is {final!r}; deliveries were {delivered!r}"
    )
    assert delivered == sorted(delivered, key=len), (
        f"snapshots delivered out of publish order: {delivered!r}"
    )
