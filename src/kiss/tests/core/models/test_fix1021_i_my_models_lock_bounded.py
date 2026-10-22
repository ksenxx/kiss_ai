# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A ``MY_MODELS.json`` edit waits a bounded time for the sidecar lock.

``model_info._my_models_flock`` took ``.MY_MODELS.json.kiss.lock`` with no
timeout: a ``flock`` blocking in C that another process (a second daemon
on the same home, a CLI parked in a debugger, an NFS stall) could hold
forever, pinning the asyncio executor thread that runs ``saveMyModel`` /
``deleteMyModel`` and, behind it, every later edit queued on
``_MY_MODELS_EDIT_LOCK``.  The wait is now bounded by
``model_info.MY_MODELS_LOCK_TIMEOUT`` and the ``TimeoutError`` becomes
the error string the settings panel already displays.

The tests hold the lock from a real subprocess, exactly as another kiss
process would, with the bound lowered so the suite stays fast.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.core.models.model_info as mi

_HOLDER = r"""
import sys, time
from pathlib import Path
from kiss.core.file_lock import exclusive_file_lock
with exclusive_file_lock(Path(sys.argv[1])):
    print("held", flush=True)
    time.sleep(60)
"""


@pytest.fixture
def my_models_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "MY_MODELS.json"
    monkeypatch.setattr(mi, "USER_MY_MODELS_PATH", path)
    monkeypatch.setattr(mi, "MY_MODELS_LOCK_TIMEOUT", 1.0)
    return path


@contextlib.contextmanager
def _sidecar_held_by_another_process(my_models_path: Path) -> Iterator[None]:
    """Hold the registry's sidecar lock from a subprocess for the block's duration."""
    lock_path = my_models_path.with_name("." + my_models_path.name + ".kiss.lock")
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(lock_path)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert proc.stdout is not None and proc.stdout.readline().strip() == "held"
        yield
    finally:
        proc.kill()
        proc.wait(timeout=30)


def _call_into(fn, args: tuple, results: list[str | None]) -> None:
    results.append(fn(*args))


def _result_within(fn, *args, seconds: float = 5.0) -> str | None:
    """Run *fn* on a thread and return its result, failing if it is still blocked."""
    results: list[str | None] = []
    worker = threading.Thread(target=_call_into, args=(fn, args, results), daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), "the edit is still waiting for the sidecar lock"
    return results[0]


def _names() -> set[str]:
    return {row["name"] for row in mi.list_custom_models()}


def test_save_reports_a_lock_it_could_not_get(my_models_path: Path) -> None:
    with _sidecar_held_by_another_process(my_models_path):
        error = _result_within(mi.save_custom_model, "m", "http://127.0.0.1:1/v1")
    assert error is not None and "locked" in error
    assert "m" not in _names()
    # The holder is gone: the same edit now goes through.
    assert mi.save_custom_model("m", "http://127.0.0.1:1/v1") is None
    assert "m" in _names()


def test_delete_reports_a_lock_it_could_not_get(my_models_path: Path) -> None:
    assert mi.save_custom_model("m", "http://127.0.0.1:1/v1") is None
    with _sidecar_held_by_another_process(my_models_path):
        error = _result_within(mi.delete_custom_model, "m")
    assert error is not None and "locked" in error
    assert "m" in _names()
    assert mi.delete_custom_model("m") is None
    assert "m" not in _names()
