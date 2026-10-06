# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-05 (scope A, ``src/kiss/core`` minus ``models/``).

Two simplifications are pinned here, each by end-to-end tests of the real
functions against real files, threads and subprocesses (nothing mocked):

1. ``config.adopt_legacy_file`` no longer ``mkdir``s the parent before
   taking its rename lock: the legacy file's existence already proves the
   directory exists, and ``exclusive_file_lock`` creates the lock file's
   parent itself.  The tests check the rename, the symlink left behind
   for a still-running old version, and that several *processes* racing
   to adopt the same file (the daemon, the CLI and the extension's helper
   all start by adopting ``history.db``) agree on one result.

2. ``vscode_config.load_api_keys`` and ``load_api_keys_readonly`` used to
   carry two copies of the "parse the store, import into ``os.environ``,
   refresh ``DEFAULT_CONFIG``" loop that differed only in whether an
   unreadable store or a NUL-poisoned line was logged.  Both now share
   ``_import_api_keys_store``; the tests cover each of its branches
   (missing store, unreadable store, skipped line, NUL value, imported
   value) through the two public entry points, and the contract the
   read-only variant exists for: it works where the locked loader cannot
   even create its lock file.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.core.vscode_config as vscode_config
from kiss.core import config as config_module
from kiss.core.config import adopt_legacy_file
from kiss.core.vscode_config import (
    API_KEY_ENV_VARS,
    api_keys_env_path,
    load_api_keys,
    load_api_keys_readonly,
)
from kiss.tests.conftest import IS_WINDOWS, is_root, posix_only

_SYMLINK = posix_only("adopt_legacy_file leaves a symlink only off Windows")
_PERMISSIONS = pytest.mark.skipif(
    IS_WINDOWS or is_root(),
    reason="permission bits are not enforced for root or on Windows",
)


# --------------------------------------------------------------------------
# adopt_legacy_file
# --------------------------------------------------------------------------


@_SYMLINK
def test_adopts_the_legacy_file_and_leaves_a_symlink_for_old_versions(tmp_path: Path) -> None:
    tmp_path = tmp_path / "db"
    tmp_path.mkdir()
    legacy = tmp_path / "sorcar.db"
    legacy.write_text("payload")
    (tmp_path / "sorcar.db-wal").write_text("wal")
    new = tmp_path / "history.db"

    adopt_legacy_file(new, "sorcar.db", ("-wal", "-shm", ""))

    assert new.read_text() == "payload"
    assert (tmp_path / "history.db-wal").read_text() == "wal"
    assert not (tmp_path / "sorcar.db-wal").exists()
    assert legacy.is_symlink() and os.readlink(legacy) == "history.db"
    assert legacy.read_text() == "payload"
    # Adopting again is a no-op (the new path exists), even though the
    # legacy name still resolves through the symlink.
    adopt_legacy_file(new, "sorcar.db", ("-wal", "-shm", ""))
    assert new.read_text() == "payload"


def test_adoption_is_a_no_op_when_nothing_is_there_and_touches_nothing(tmp_path: Path) -> None:
    tmp_path = tmp_path / "db"
    tmp_path.mkdir()
    nested = tmp_path / "missing" / "history.db"
    adopt_legacy_file(nested, "sorcar.db")
    assert not nested.parent.exists()
    assert list(tmp_path.iterdir()) == []


_RACER = textwrap.dedent(
    """
    import os, sys, time
    from pathlib import Path
    from kiss.core.config import adopt_legacy_file
    new = Path(sys.argv[1]); go = Path(sys.argv[2])
    go.with_name(f"ready-{os.getpid()}").touch()
    deadline = time.monotonic() + 60
    while not go.exists():
        if time.monotonic() > deadline:
            sys.exit("racer barrier timed out")
        time.sleep(0.001)
    adopt_legacy_file(new, "sorcar.db", ("-wal", ""))
    print(new.read_text())
    """
)


def _reap(procs: list[subprocess.Popen[str]]) -> None:
    """Terminate, then kill, every racer still alive and close its pipes."""
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                pipe.close()


@_SYMLINK
def test_processes_racing_to_adopt_the_same_file_agree(tmp_path: Path) -> None:
    """Several processes adopting at once: one renames, the rest see the result.

    Every KISS entry point adopts ``history.db`` on startup, so the
    daemon, a CLI run and the extension's config helper can race.  The
    rename lock beside the new name serialises them; each racer must
    read the adopted content and the directory must end up in the
    single-adoption state.
    """
    tmp_path = tmp_path / "db"
    tmp_path.mkdir()
    legacy = tmp_path / "sorcar.db"
    legacy.write_text("payload")
    (tmp_path / "sorcar.db-wal").write_text("wal")
    new = tmp_path / "history.db"
    go = tmp_path / "go"
    env = {**os.environ, "KISS_HOME": str(tmp_path.parent / "home")}
    racers: list[subprocess.Popen[str]] = []
    try:
        for _ in range(4):
            racers.append(
                subprocess.Popen(
                    [sys.executable, "-c", _RACER, str(new), str(go)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
            )
        deadline = time.monotonic() + 60
        while len(list(tmp_path.glob("ready-*"))) < len(racers):
            assert time.monotonic() < deadline, "racers did not start"
            time.sleep(0.01)
        for ready in tmp_path.glob("ready-*"):
            ready.unlink()
        go.touch()
        outputs = [proc.communicate(timeout=60) for proc in racers]
    finally:
        # A failure before ``go.touch()`` must not leave four children
        # spinning on the barrier.
        _reap(racers)

    for proc, (out, err) in zip(racers, outputs, strict=True):
        assert proc.returncode == 0, err
        assert out.strip() == "payload", err
    assert new.read_text() == "payload"
    assert (tmp_path / "history.db-wal").read_text() == "wal"
    assert legacy.is_symlink() and os.readlink(legacy) == "history.db"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "go",
        "history.db",
        "history.db-wal",
        "history.db.rename.lock",
        "sorcar.db",
    ]


# --------------------------------------------------------------------------
# load_api_keys / load_api_keys_readonly
# --------------------------------------------------------------------------

_KEY = "GEMINI_API_KEY"


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Redirect HOME/config to a temp dir and restore the globals the loaders mutate."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))  # what Path.home() reads on Windows
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.setitem(vars(vscode_config), "CONFIG_DIR", fake_home / ".kiss")
    monkeypatch.setitem(vars(vscode_config), "CONFIG_PATH", fake_home / ".kiss" / "config.json")
    # The loaders write os.environ directly, so the keys are restored by an
    # explicit snapshot (monkeypatch.delenv records nothing for an absent key).
    watched = (*API_KEY_ENV_VARS, "KISS_AUDIT_NUL", "KISS_AUDIT_SKIPPED")
    env_snapshot = {key: os.environ.get(key) for key in watched}
    for key in watched:
        os.environ.pop(key, None)
    saved = config_module.DEFAULT_CONFIG
    snapshot = dict(saved.model_copy(deep=True).__dict__)
    yield
    for k, v in snapshot.items():
        setattr(saved, k, v)
    for key, val in env_snapshot.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _write_store(text: str) -> Path:
    env_file = api_keys_env_path()
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(text, encoding="utf-8")
    return env_file


@pytest.mark.parametrize("loader", [load_api_keys, load_api_keys_readonly])
def test_both_loaders_import_the_store_and_skip_junk_lines(
    loader,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write_store(
        "# comment\n"
        f"export {_KEY}=from-store\n"
        "export KISS_AUDIT_SKIPPED=$HOME/needs-expansion\n"
        "export KISS_AUDIT_NUL=abc\x00def\n"
    )
    os.environ[_KEY] = "stale"
    with caplog.at_level(logging.WARNING, logger=vscode_config.__name__):
        loader()
    assert os.environ[_KEY] == "from-store"
    assert config_module.DEFAULT_CONFIG.GEMINI_API_KEY == "from-store"
    assert "KISS_AUDIT_SKIPPED" not in os.environ
    assert "KISS_AUDIT_NUL" not in os.environ
    assert any("KISS_AUDIT_NUL" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("loader", [load_api_keys, load_api_keys_readonly])
def test_both_loaders_treat_a_missing_store_as_empty_and_still_refresh(loader) -> None:
    assert not api_keys_env_path().exists()
    os.environ[_KEY] = "from-environment"
    loader()
    assert os.environ[_KEY] == "from-environment"
    assert config_module.DEFAULT_CONFIG.GEMINI_API_KEY == "from-environment"


@_PERMISSIONS
def test_readonly_loader_logs_an_unreadable_store_and_carries_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    env_file = _write_store(f"export {_KEY}=hidden\n")
    env_file.chmod(0)
    try:
        with caplog.at_level(logging.WARNING, logger=vscode_config.__name__):
            load_api_keys_readonly()
    finally:
        env_file.chmod(0o600)
    assert _KEY not in os.environ
    assert any("Failed to read" in r.getMessage() for r in caplog.records)


@_PERMISSIONS
def test_readonly_loader_works_where_the_locked_loader_cannot_create_its_lock() -> None:
    """The contract behind the read-only variant (used by ``govee`` when
    ``KISS_HOME`` is not writable): ``load_api_keys`` needs to create a
    lock file beside the store and fails on a read-only config dir, while
    ``load_api_keys_readonly`` still imports the keys."""
    env_file = _write_store(f"export {_KEY}=locked-dir\n")
    env_file.parent.chmod(0o500)
    try:
        with pytest.raises(OSError):
            load_api_keys()
        assert _KEY not in os.environ
        load_api_keys_readonly()
        assert os.environ[_KEY] == "locked-dir"
    finally:
        env_file.parent.chmod(0o700)
