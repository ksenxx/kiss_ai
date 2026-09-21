# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Concurrency audit W1 — behaviour locks for two redundancy removals.

1. ``vscode_config._edit_api_keys_env_file`` (the standalone "take both
   locks, then edit" wrapper) had no callers: ``save_api_key`` and
   ``_migrate_legacy_rc_keys`` both call the ``_locked`` variant inside
   their own ``_config_lock`` + store-flock section.  It was removed.
   The test below drives the three real writers/readers of the canonical
   store — ``save_api_key``, ``load_api_keys`` (which runs
   ``_migrate_legacy_rc_keys`` first) — from concurrent threads against
   a temp ``HOME`` and checks that every thread finishes (no deadlock on
   the RLock / sidecar flocks) and that the store ends in the state the
   last completed save left it in.

2. ``utils.atomic_write_text`` chmod-ed the staged temp file, replaced it
   into position, and then chmod-ed the target a second time.  The second
   chmod was a no-op (``os.replace`` keeps the inode and its mode).  The
   test below overwrites a 0644 file with ``mode=0o640`` and checks the
   published file carries 0640 — the property the removed call was
   supposedly guarding.

Everything is real: real threads, real files under a temp HOME, the real
module locks and ``fcntl`` flocks.  Nothing is mocked.
"""

from __future__ import annotations

import os
import stat
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.core.vscode_config as vscode_config
from kiss.core.utils import atomic_write_text
from kiss.core.vscode_config import (
    API_KEY_ENV_VARS,
    api_keys_env_path,
    load_api_keys,
    save_api_key,
)
from kiss.tests.conftest import posix_only

_JOIN_TIMEOUT_S = 30.0
_ROUNDS = 25


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Redirect HOME/SHELL/config to a temp dir and restore mutated globals."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))  # what Path.home() reads on Windows
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.setitem(vars(vscode_config), "CONFIG_DIR", fake_home / ".kiss")
    monkeypatch.setitem(
        vars(vscode_config), "CONFIG_PATH", fake_home / ".kiss" / "config.json",
    )
    for key in API_KEY_ENV_VARS:
        val = os.environ.get(key)
        if val is not None:
            monkeypatch.setenv(key, val)
        else:
            monkeypatch.delenv(key, raising=False)
    from kiss.core import config as config_module

    saved = config_module.DEFAULT_CONFIG
    snapshot = dict(saved.model_copy(deep=True).__dict__)
    yield
    for k, v in snapshot.items():
        setattr(saved, k, v)


def _run_recording_errors(fn, errors: list[BaseException]) -> None:  # type: ignore[no-untyped-def]
    """Run *fn*; append any exception so a crashed thread cannot pass silently."""
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 — recorded and re-raised by the test
        errors.append(exc)


@posix_only("the legacy-key migration sources ~/.bashrc with bash; asserts a 0600 store")
def test_concurrent_save_load_and_migrate_never_deadlock() -> None:
    """Savers, a loader and a legacy-RC migration run together and all finish.

    ``~/.bashrc`` carries a legacy ``export OPENAI_API_KEY=...`` line and
    the store lacks that key, so every ``load_api_keys`` really takes the
    migration's store-flock + RC scan path while ``save_api_key`` threads
    take store flock + RC flock for a different key.  A lock-order or
    reentrancy mistake would leave a thread blocked past the join timeout.

    The workers are daemon threads: if the deadlock this test exists to
    catch ever happens, the timed joins expire and the assertion below
    fails, and the interpreter must not then hang at exit waiting on the
    blocked workers.
    """
    fake_home = Path(os.environ["HOME"])
    (fake_home / ".bashrc").write_text(
        "export OPENAI_API_KEY=legacy-openai\n", encoding="utf-8",
    )
    for round_no in range(_ROUNDS):
        errors: list[BaseException] = []
        threads = [
            threading.Thread(
                target=_run_recording_errors,
                args=(lambda v=f"gem-{round_no}-a": save_api_key("GEMINI_API_KEY", v), errors),
                daemon=True,
            ),
            threading.Thread(
                target=_run_recording_errors,
                args=(lambda v=f"gem-{round_no}-b": save_api_key("GEMINI_API_KEY", v), errors),
                daemon=True,
            ),
            threading.Thread(
                target=_run_recording_errors, args=(load_api_keys, errors), daemon=True,
            ),
            threading.Thread(
                target=_run_recording_errors,
                args=(lambda: save_api_key("ANTHROPIC_API_KEY", ""), errors),
                daemon=True,
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=_JOIN_TIMEOUT_S)
        stuck = [t.name for t in threads if t.is_alive()]
        assert not stuck, f"round {round_no}: threads still blocked: {stuck}"
        assert not errors, f"round {round_no}: {errors!r}"

        store = api_keys_env_path().read_text(encoding="utf-8")
        gemini_lines = [ln for ln in store.splitlines() if ln.startswith("export GEMINI_API_KEY=")]
        assert len(gemini_lines) == 1, store
        assert gemini_lines[0] in (
            f"export GEMINI_API_KEY=gem-{round_no}-a",
            f"export GEMINI_API_KEY=gem-{round_no}-b",
        ), store
        assert os.environ["GEMINI_API_KEY"] == gemini_lines[0].split("=", 1)[1]
        assert "ANTHROPIC_API_KEY" not in os.environ
        assert not any(ln.startswith("export ANTHROPIC_API_KEY=") for ln in store.splitlines())

    # The migration path really ran: the legacy RC key is now in the store
    # and the environment, and the store is 0600.
    store = api_keys_env_path().read_text(encoding="utf-8")
    assert "export OPENAI_API_KEY=legacy-openai\n" in store
    assert os.environ["OPENAI_API_KEY"] == "legacy-openai"
    assert stat.S_IMODE(api_keys_env_path().stat().st_mode) == 0o600


@posix_only("Windows chmod carries no group/other mode bits")
def test_atomic_write_mode_survives_replacing_a_more_permissive_file(tmp_path: Path) -> None:
    """Overwriting a 0644 file with ``mode=0o640`` publishes a 0640 file.

    The mode is applied to the staged temp file before ``os.replace``,
    which moves the inode (and its mode) into position — there is nothing
    left for a post-replace chmod to do.  ``0o640`` is deliberately not
    ``0o600``: ``tempfile.mkstemp`` already creates the staging file as
    0600 whatever the umask, so asking for 0600 would pass even with the
    chmod deleted.  Only the pre-replace ``chmod`` (which the umask does
    not filter) can produce 0640 here.
    """
    target = tmp_path / "files" / "rc"
    target.parent.mkdir()
    target.write_text("old\n", encoding="utf-8")
    os.chmod(target, 0o644)
    assert stat.S_IMODE(target.stat().st_mode) == 0o644

    atomic_write_text(target, "export K=v\n", mode=0o640)

    assert target.read_text(encoding="utf-8") == "export K=v\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert sorted(p.name for p in target.parent.iterdir()) == ["rc"], "no temp file left behind"


def test_atomic_write_without_mode_keeps_content_and_leaves_no_temp(tmp_path: Path) -> None:
    """The ``mode is None`` branch: content published, staging file gone."""
    target = tmp_path / "nested" / "trajectory.yaml"
    atomic_write_text(target, "a: 1\n")
    assert target.read_text(encoding="utf-8") == "a: 1\n"
    assert sorted(p.name for p in target.parent.iterdir()) == ["trajectory.yaml"]
