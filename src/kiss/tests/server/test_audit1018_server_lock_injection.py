# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The talk player's and the browser tab service's locks survive the runner's Stop.

The runner's Stop delivers ``KeyboardInterrupt`` to a task thread with
``PyThreadState_SetAsyncExc``; CPython raises it at the next bytecode
check point, including the one right after a ``with lock:``'s
``__enter__`` returned (gh-74174), so a plain ``with`` on a task thread
leaks the lock.  Two daemon-wide singletons are reached from task
threads:

* :class:`~kiss.server.talk_player.TalkPlayer` through the ``talk``
  tool (``WebPrinter.broadcast`` -> ``_fanout_talk`` ->
  ``_play_talk_clip_locally`` -> ``shared_player().play``): a leaked
  ``_lock``, ``_shared_lock`` or playback-queue mutex silences every
  later talk of every task;
* :class:`~kiss.server.browser_tab.BrowserTabService` through the web
  tools (``open_for_agent`` -> ``_ensure_loop``, ``tab_for_target``): a
  leaked ``_lock`` wedges every later browser tab and the daemon's
  shutdown.

Each hammer runs in a child process (a lock the worker leaked would
otherwise wedge the rest of the pytest session) and contends the lock
from a second thread, which turns the leak into a hang the harness
reports.  Before the conversion to ``locked_call`` every path here
leaked its lock within the first few hundred injections.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from kiss.server import talk_player
from kiss.server.browser_tab import BrowserTabService
from kiss.server.json_printer import JsonPrinter
from kiss.server.talk_player import TalkPlayer
from kiss.tests.lock_injection import hammer


class _Player:
    """``play`` of a fresh clip on the worker: dedupe, eviction and the queue put.

    The clip is not valid base64 and the event carries no text, so the
    player's worker drops it without starting any process.  The first
    play happens here, so the worker thread exists before the hammer
    starts (an interrupt inside ``Thread.start`` would be the test's
    own leak of ``threading._active_limbo_lock``).
    """

    def __init__(self) -> None:
        self.player = TalkPlayer()
        self.n = 0
        self.player.play({"audioB64": "!!", "talkId": "prime"})

    def __call__(self) -> None:
        self.n += 1
        self.player.play({"audioB64": "!!", "talkId": f"t{self.n}"})


class _Shared:
    """``shared_player()`` on the worker: the singleton's lock."""

    def __init__(self) -> None:
        talk_player.shared_player()

    def __call__(self) -> None:
        talk_player.shared_player()


class _Service:
    """A ``BrowserTabService`` whose loop thread runs but drives no browser."""

    def __init__(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.service = BrowserTabService(JsonPrinter(), Path(self.dir.name))
        self.service._ensure_loop()


class _Loop(_Service):
    """``_ensure_loop`` on the worker: the entry of ``open_for_agent``."""

    def __call__(self) -> None:
        self.service._ensure_loop()


class _Target(_Service):
    """``tab_for_target`` on the worker: the web tools' lookup."""

    def __call__(self) -> None:
        self.service.tab_for_target("no-such-target")


def run_hammer(name: str) -> None:
    """Hammer the path *name* with the Stop injection; the child-process body."""
    if name == "talk_play":
        player = _Player()
        hammer(player, {"_lock": player.player._lock}, contend=player.player._lock)
        # The player's worker still drains: an interrupt inside a
        # ``queue.Queue.put`` (a Python ``with`` over the queue's
        # mutex) used to wedge its ``get`` for good.
        deadline = time.monotonic() + 5
        while not player.player._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert player.player._queue.empty(), "the player's worker no longer drains"
    elif name == "talk_shared":
        hammer(_Shared(), {"_shared_lock": talk_player._shared_lock},
               contend=talk_player._shared_lock)
    else:
        driver = _Loop() if name == "browser_loop" else _Target()
        hammer(driver, {"_lock": driver.service._lock}, contend=driver.service._lock)
        driver.service.shutdown()
        driver.dir.cleanup()


@pytest.mark.parametrize(
    "path", ["talk_play", "talk_shared", "browser_loop", "browser_target"]
)
def test_task_thread_entries_keep_their_locks(path: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {__name__} as m; m.run_hammer({path!r})"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
