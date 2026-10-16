# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Stop landing inside ``PromptCacheKeepAlive.__enter__`` must not leave the ping thread running.

The runner's Stop injects ``KeyboardInterrupt`` into the task thread with
``PyThreadState_SetAsyncExc``; CPython raises it at the next check point,
which can be the one right after ``Thread.start()``'s C-level spawn
returned.  The thread is already running, but ``__enter__`` raises, so
the ``with`` never calls ``__exit__`` and nothing sets ``_stop``: the
thread would ping (and bill) the model every interval for up to
``max_pings`` intervals after the task ended.

The window is reproduced deterministically with a profile hook that
raises ``KeyboardInterrupt`` on the ``c_return`` of
``_thread.start_joinable_thread`` — the same exception at the same
place the async injection lands.
"""

from __future__ import annotations

import _thread
import sys
import threading
import time
from types import FrameType
from typing import Any

import pytest

from kiss.core.prompt_cache_keepalive import PromptCacheKeepAlive


def _interrupt_after_spawn(frame: FrameType, event: str, arg: Any) -> None:
    """Raise ``KeyboardInterrupt`` once the OS thread has been spawned, then unhook."""
    if event == "c_return" and arg is _thread.start_joinable_thread:
        sys.setprofile(None)
        raise KeyboardInterrupt


def test_stop_inside_enter_stops_the_ping_thread() -> None:
    """The thread spawned by an interrupted ``__enter__`` exits instead of pinging for 40 min."""
    keep_alive = PromptCacheKeepAlive(None, {}, None, time.time())
    sys.setprofile(_interrupt_after_spawn)
    try:
        with pytest.raises(KeyboardInterrupt), keep_alive:
            raise AssertionError("the body must not run")
    finally:
        sys.setprofile(None)
    thread: threading.Thread = keep_alive._thread
    # ``join`` refuses a thread whose bootstrap has not set ``_started``
    # yet; the spawn already happened, so the flag arrives at once.
    deadline = time.monotonic() + 5.0
    while thread.ident is None and time.monotonic() < deadline:
        time.sleep(0.001)
    thread.join(5.0)
    assert not thread.is_alive(), "the keep-alive thread outlived its interrupted __enter__"
