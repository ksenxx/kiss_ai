# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The core locks a task thread takes survive the runner's Stop injection.

The runner's Stop delivers ``KeyboardInterrupt`` to a task thread with
``PyThreadState_SetAsyncExc``; CPython raises it at the next check
point of the bytecode, including the one right after a ``with lock:``'s
``__enter__`` returned (gh-74174), so a plain ``with`` on a task thread
leaks the lock.  Every ``kiss.core`` lock a task thread can take and
that outlives the task (process-wide locks, or one re-taken on the
task's own unwind path) therefore goes through
:func:`kiss.core.locked.locked_call` / :func:`kiss.core.locked.locked`.

Each path is hammered from a worker thread with hundreds of injections
in a child process (a leaked process-wide lock would wedge the rest of
the pytest session) and the lock is asserted free after.  Before the
conversion ``Base.__init__`` leaked ``Base._class_lock`` in roughly one
of ten injections, which hangs every later agent construction of the
daemon.

Not hammered: ``ModelEmbedder._client`` (building the embedding model
needs a provider key) and the Windows-only Job Object sections of
:mod:`kiss.core.processes`; both are plain ``locked_call`` wrappers of
the same shape as the paths below.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading

from kiss.core import browser_handoff, config
from kiss.core.base import Base
from kiss.core.memoryfield.index import VectorIndex, hashed_embedding
from kiss.core.memoryfield.pages import MemoryDir
from kiss.core.models import openai_compatible_model as oai
from kiss.core.models.stream_abort import StreamAbortWatchdog
from kiss.core.vscode_config import _config_lock, save_config
from kiss.tests.server.test_audit1013_printer_lock_injection import _hammer, _lock_is_free


def _construct_base() -> None:
    """``Base.__init__`` takes the agent id under ``Base._class_lock``."""
    Base("audit1013-core")


def _artifact_dir() -> None:
    """``get_artifact_dir`` chooses the directory once under ``_artifact_dir_lock``.

    The choice is forgotten first so every call takes the lock (the
    directory is created under the child's temporary working directory).
    """
    config._artifact_dir = None
    config.get_artifact_dir()


def _tool_effort_verdicts() -> None:
    """Record then read an adaptive ``reasoning_effort`` verdict."""
    oai._record_tool_effort_verdict(("http://audit1013", "m"), True)
    oai._record_tool_effort_verdict(("http://audit1013", "m"), False)
    assert oai._tool_effort_verdict(("http://audit1013", "m")) is False


class _BrowserHandoffClaim:
    """Claim, settle and release one hand-off URL under the module condition.

    A fresh URL per call: an interrupt between the claim and its settle
    leaves a pending claim, and a second claim of the same URL would
    wait for it rather than take the lock again.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        url = f"http://audit1013.invalid/{self.calls}"
        assert browser_handoff._claim(url, browser_handoff.BROWSER_TAB) is None
        browser_handoff._settle(url)
        tab = browser_handoff.BROWSER_TAB
        assert browser_handoff._claim(url, tab) == tab
        browser_handoff._release(url)


def _save_last_model() -> None:
    """``save_config`` (the model picker's write) under ``_config_lock`` and its flock."""
    save_config({"last_model": "audit1013"})


class _StreamWatchdog:
    """``beat`` then ``stop`` of a fresh watchdog: the stream loop and its ``finally``."""

    def __init__(self) -> None:
        self.locks: list[threading.RLock] = []

    def __call__(self) -> None:
        watchdog = StreamAbortWatchdog(stream=None, name="audit1013-watchdog")
        self.locks.append(watchdog._lock)
        try:
            watchdog.beat()
        finally:
            watchdog.stop()


class _QueryEmbedding:
    """``embed_query`` (cache miss then hit) under the index's query lock."""

    def __init__(self) -> None:
        root = tempfile.mkdtemp(prefix="audit1013-memory-")
        self.index = VectorIndex(MemoryDir(root), embed=hashed_embedding)
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        self.index.embed_query(f"query {self.calls}")
        self.index.embed_query(f"query {self.calls}")


def run_hammer(name: str) -> None:
    """Child-process body: hammer the path *name* and check its lock stays free."""
    if name == "base":
        _hammer(_construct_base, {"Base._class_lock": Base._class_lock})
    elif name == "artifact_dir":
        os.chdir(tempfile.mkdtemp(prefix="audit1013-artifacts-"))
        _hammer(_artifact_dir, {"_artifact_dir_lock": config._artifact_dir_lock})
    elif name == "verdict":
        lock = oai._ADAPTIVE_TOOL_EFFORT_LOCK
        _hammer(_tool_effort_verdicts, {"_ADAPTIVE_TOOL_EFFORT_LOCK": lock})
    elif name == "handoff":
        _hammer(_BrowserHandoffClaim(), {"browser_handoff._lock": browser_handoff._lock})
    elif name == "config":
        _hammer(_save_last_model, {"_config_lock": _config_lock})
    elif name == "stream":
        call = _StreamWatchdog()
        _hammer(call, {})
        assert all(_lock_is_free(lock) for lock in call.locks), "a watchdog lock leaked"
    else:
        query = _QueryEmbedding()
        _hammer(query, {"_query_lock": query.index._query_lock})


def test_core_lock_paths() -> None:
    """No ``kiss.core`` lock site a task thread runs leaks the lock under injection."""
    for name in ("base", "artifact_dir", "verdict", "handoff", "config", "stream", "query"):
        proc = subprocess.run(
            [sys.executable, "-c", f"import {__name__} as m; m.run_hammer({name!r})"],
            capture_output=True, text=True, timeout=120, check=False,
        )
        assert proc.returncode == 0, f"{name}: {proc.stderr[-3000:]}"
