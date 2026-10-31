# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The SEA command registry publishes scans in order and survives an unparsable script.

``refresh_registry`` read the folders without holding ``_notify_lock``
and took it only to publish, so a rescan that read the folders before a
SEA appeared could publish after a newer rescan that saw it: the
registry and every subscriber fell back to the stale list until the next
watcher tick (and a process without the watcher stayed stale).  Each
rescan is now numbered when it starts, and one that started before the
last published rescan is dropped instead of published.

``parsed_sea`` memoised a revision ``ast.parse`` rejects only for
``SyntaxError`` and ``ValueError``.  A source nested too deeply for the
parser raises ``RecursionError`` (or ``MemoryError``), which escaped every
registry read (``refresh_registry``, ``list_commands``,
``get_command``) for every command, not only for that script.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.sea_settings import parsed_sea
from kiss.core.config import kiss_home
from kiss.core.locked import locked_call

_SEA = """
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    pass
"""

TOO_DEEP = "x = " + " + ".join(['"a"'] * 100_000) + "\n"
"""A valid expression nested too deeply for CPython's parser (``RecursionError``)."""


@pytest.fixture(autouse=True)
def _isolated_registry() -> Iterator[None]:
    """Start and end each test with an empty registry and no ``SEAS.md``."""
    sea_commands._reset_for_tests()
    yield
    sea_commands._reset_for_tests()
    (kiss_home() / "SEAS.md").unlink(missing_ok=True)


def _write_sea(folder: Path, name: str, body: str = _SEA) -> Path:
    """Create the SEA ``<folder>/<name>/<name>_sea.py`` holding *body*; return its path."""
    script = folder / name / f"{name}_sea.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(body, encoding="utf-8")
    return script


def _register_folder(folder: Path) -> None:
    """Point ``$KISS_HOME/SEAS.md`` at *folder*."""
    kiss_home().mkdir(parents=True, exist_ok=True)
    (kiss_home() / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")


def _wait_until(done: Callable[..., bool], *args: Any) -> None:
    """Poll ``done(*args)`` every 10 ms for up to 10 s; fail when it never holds."""
    deadline = time.monotonic() + 10
    while not done(*args):
        assert time.monotonic() < deadline, f"{done.__name__}{args} never held"
        time.sleep(0.01)


def _blocked_on_notify_lock(thread: threading.Thread) -> bool:
    """Whether *thread* is parked taking ``sea_commands._notify_lock``."""
    frame = sys._current_frames().get(thread.ident or 0)
    return (
        frame is not None
        and frame.f_code is locked_call.__code__
        and frame.f_locals.get("lock") is sea_commands._notify_lock
    )


class _HoldingSubscriber:
    """A registry subscriber that holds ``_notify_lock`` through its first call, then rescans.

    The first call (the first publish) sets :attr:`entered` and waits for
    :attr:`release`, then rescans from inside the callback (as a
    subscriber resolving a name through ``get_command`` may), so a newer
    scan is published while another rescan waits for the lock.
    """

    def __init__(self) -> None:
        self.snapshots: list[list[str]] = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, names: list[str]) -> None:
        """Record *names*; on the first call, hold the lock until released, then rescan."""
        self.snapshots.append(names)
        if len(self.snapshots) == 1:
            self.entered.set()
            self.release.wait(10)
            sea_commands.refresh_registry()


def test_an_older_scan_is_never_published_over_a_newer_one(tmp_path: Path) -> None:
    """A rescan that read the folders before a newer rescan published is dropped, not published."""
    folder = tmp_path / "seas"
    _write_sea(folder, "alpha")
    _register_folder(folder)
    subscriber = _HoldingSubscriber()
    sea_commands.subscribe(subscriber)
    first = threading.Thread(target=sea_commands.refresh_registry, daemon=True)
    first.start()
    try:
        assert subscriber.entered.wait(10), "the first rescan never notified"
        queued = threading.Thread(target=sea_commands.refresh_registry, daemon=True)
        queued.start()
        _wait_until(_blocked_on_notify_lock, queued)
        _write_sea(folder, "beta")  # appears after the queued rescan reached the lock
    finally:
        subscriber.release.set()
    first.join(10)
    queued.join(10)
    assert not first.is_alive() and not queued.is_alive()
    assert "beta" in subscriber.snapshots[-1], subscriber.snapshots[-1]
    assert "beta" in sea_commands.list_commands()


def test_a_script_too_deep_to_parse_does_not_break_the_registry(tmp_path: Path) -> None:
    """Every other command still resolves; the unparsable revision is memoised as such."""
    folder = tmp_path / "seas"
    deep = _write_sea(folder, "deep", TOO_DEEP)
    fine = _write_sea(folder, "fine")
    _register_folder(folder)
    assert {"deep", "fine"} <= set(sea_commands.refresh_registry())
    assert sea_commands.get_command("fine") == fine.resolve()
    parsed = parsed_sea(deep)
    assert parsed.stamp is not None
    assert (parsed.bases, parsed.literals) == (set(), {})
    assert parsed_sea(deep) is parsed  # the next scan does not parse the revision again
