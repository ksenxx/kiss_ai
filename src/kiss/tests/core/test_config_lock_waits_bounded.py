# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A config or key-store writer waits a bounded time for another process's lock.

``save_config``, ``save_api_key`` and the legacy-key migration took their
cross-process file locks (``.config.lock``, ``.api_keys.env.kiss.lock``,
``<rc>.kiss.lock``) with no timeout: a blocking ``flock`` in C that a
wedged holder (a process paused in a debugger, a five-second RC
sourcing in another daemon) kept forever, and that a task Stop cannot
interrupt.  Because the writer also holds the process-wide
``_config_lock``, every config read in the daemon queued behind it.

The locks are now bounded by ``user_config._LOCK_TIMEOUT_S``; the
``TimeoutError`` is the writer's error, and the legacy-key migration
(best effort at startup) logs it and lets the daemon start.  The tests
hold each lock themselves — a second open file description excludes
the writer exactly as another process would — with the bound lowered
so the suite stays fast.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import pytest

from kiss.core import user_config
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.user_config import load_api_keys, save_api_key, save_config

KEY = "GEMINI_API_KEY"


@pytest.fixture(autouse=True)
def _own_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # what Path.home() reads on Windows
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.setitem(vars(user_config), "CONFIG_DIR", tmp_path / "kiss")
    monkeypatch.setitem(vars(user_config), "CONFIG_PATH", tmp_path / "kiss" / "config.json")
    monkeypatch.setattr(user_config, "_LOCK_TIMEOUT_S", 1.0)
    return home


def _call_recording(fn, args: tuple, outcome: list[BaseException | None]) -> None:
    try:
        fn(*args)
        outcome.append(None)
    except BaseException as exc:
        outcome.append(exc)


def _held_call_times_out(lock_path: Path, fn, *args) -> BaseException | None:
    """Call *fn* on a thread while *lock_path* is held; return what it raised."""
    outcome: list[BaseException | None] = []
    worker = threading.Thread(target=_call_recording, args=(fn, args, outcome), daemon=True)
    with exclusive_file_lock(lock_path):
        worker.start()
        worker.join(5)
        assert not worker.is_alive(), "the writer is still waiting for the lock"
    return outcome[0]


def test_save_config_gives_up_on_a_held_config_lock(tmp_path: Path) -> None:
    error = _held_call_times_out(
        tmp_path / "kiss" / ".config.lock", save_config, {"work_dir": "/elsewhere"},
    )
    assert isinstance(error, TimeoutError) and ".config.lock" in str(error)
    assert not (tmp_path / "kiss" / "config.json").exists()


def test_save_api_key_gives_up_on_a_held_store_lock(tmp_path: Path) -> None:
    error = _held_call_times_out(
        tmp_path / "kiss" / ".api_keys.env.kiss.lock", save_api_key, KEY, "value",
    )
    assert isinstance(error, TimeoutError)
    assert not (tmp_path / "kiss" / "api_keys.env").exists()
    assert KEY not in os.environ


def test_save_api_key_gives_up_on_a_held_rc_lock(tmp_path: Path, _own_home: Path) -> None:
    (_own_home / ".bashrc").write_text("# mine\n")
    error = _held_call_times_out(_own_home / ".bashrc.kiss.lock", save_api_key, KEY, "value")
    assert isinstance(error, TimeoutError) and ".bashrc.kiss.lock" in str(error)
    assert (_own_home / ".bashrc").read_text() == "# mine\n"
    assert KEY not in os.environ


def test_startup_migration_skips_a_held_store_lock(
    tmp_path: Path, _own_home: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    (_own_home / ".bashrc").write_text(f"export {KEY}=legacy\n")
    with caplog.at_level(logging.WARNING, logger="kiss.core.user_config"):
        error = _held_call_times_out(tmp_path / "kiss" / ".api_keys.env.kiss.lock", load_api_keys)
    assert error is None
    assert "Skipping legacy RC key migration" in caplog.text
    assert KEY not in os.environ
    load_api_keys()
    assert os.environ.get(KEY) == "legacy"
