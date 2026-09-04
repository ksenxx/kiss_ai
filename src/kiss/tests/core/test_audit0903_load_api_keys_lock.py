# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-03 (core-base): ``load_api_keys`` vs ``save_api_key`` race.

The audit originally fixed this lost-update shape in ``source_shell_env``
vs ``save_api_key_to_shell``.  The API-key rewrite (``refactor: unify
local/remote API-key loading``) deleted both functions but reintroduced
the same shape in their replacements: ``save_api_key`` performs its whole
critical section — canonical-store edit, ``os.environ`` update,
``_refresh_config()`` — inside ``_config_lock``, but ``load_api_keys``
snapshotted the store (``read_text``), imported the parsed assignments
into ``os.environ`` and called ``_refresh_config()`` **without** that
lock, so this interleaving was possible:

1. ``load_api_keys`` reads ``api_keys.env`` while it still holds the OLD
   key value;
2. ``save_api_key`` runs to completion: the store, ``os.environ`` and
   ``DEFAULT_CONFIG`` all hold the NEW value;
3. ``load_api_keys`` applies its stale snapshot: ``os.environ`` and
   ``DEFAULT_CONFIG`` silently revert to the OLD value while the store
   keeps the NEW one — the freshly saved API key is lost until the next
   load.  (A save that *deletes* a key is likewise resurrected.)

Reproducing this needs the loader's snapshot→import window to outlast
the saver's whole locked edit+tail (the loader's ``_migrate_legacy_rc_keys``
also takes ``_config_lock`` *before* the snapshot, so the two serialize
up to that point).  The store is therefore padded with long
expansion-skipped assignments (``export PAD_i=aaa…a$KISS_PAD``): the
loader's ``_shell_would_expand`` walks every value character by
character, while the saver's ``_env_line_sets_key`` only scans up to the
``=`` — so the same padding costs the loader ~95 ms of parsing but the
saver only ~7 ms of editing.  Sweeping the save's start across the
loader's run then lands many saves entirely inside the loader's
snapshot→import window (26/40 iterations stale on this machine).  The
fix holds ``_config_lock`` from before the ``read_text`` snapshot until
after ``_refresh_config``, so a saver serialized behind (or ahead of)
the loader can never have its value overwritten by a stale snapshot.

Everything here is real: real threads, the real canonical store in a temp
HOME, the real module lock.  Nothing is mocked.

Branch coverage of the modified code: the fix adds no new branches (it
wraps the existing snapshot/import/refresh body in ``with
_config_lock:``); the pre-existing branches — store missing, unreadable
store, junk lines, NUL values — are exercised by
``test_vscode_config.py``'s ``load_api_keys`` tests, which run against
the same modified function.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.core.vscode_config as vscode_config
from kiss.core.vscode_config import (
    API_KEY_ENV_VARS,
    api_keys_env_path,
    load_api_keys,
    save_api_key,
)

_KEY = "GEMINI_API_KEY"
# Assignment lines prepended to api_keys.env purely to widen the window
# between load_api_keys' read_text snapshot and its import of the target
# key (which save_api_key keeps after them).  Each line's value is long
# and ends in an unquoted `$`, so the loader pays a full
# _shell_would_expand character walk per line (and then skips the import,
# leaving os.environ unpolluted) while the saver's per-line key check
# stops at the `=`.
_PAD_LINES = 2_000
_PAD_VALUE = "a" * 400 + "$KISS_PAD"
# Sweep the saver's start time across the loader's ~100 ms parse in 2 ms
# steps up to 40 ms.  On the unfixed code, saves landing after the
# loader's read_text but before its import of the key (most of the sweep)
# are reverted by the stale snapshot — exactly the lost-update window.
_SWEEP_STEPS = 20
_SWEEP_INCREMENT_S = 0.002
_ROUNDS = 2


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point HOME/SHELL/config at a temp dir and snapshot mutated globals.

    Mirrors ``test_vscode_config.py``'s isolation fixture: the real
    ``api_keys_env_path`` (via the monkeypatched ``CONFIG_DIR``) and the
    real module lock are used; only the locations are redirected so the
    test never touches the developer's actual key store or RC files, and
    ``os.environ`` / ``DEFAULT_CONFIG`` values touched by the functions
    under test are restored afterwards.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
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


def _pad_store() -> None:
    """Prepend expansion-skipped assignment padding to the canonical store.

    The padded lines are never imported into ``os.environ`` (their values
    would need shell expansion), but each one costs the loader a full
    character walk — see the module docstring for why that asymmetry is
    what makes the race window reproducible.
    """
    env_file = api_keys_env_path()
    existing = env_file.read_text() if env_file.exists() else ""
    padding = "".join(
        f"export PAD_{i}={_PAD_VALUE}\n" for i in range(_PAD_LINES)
    )
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(padding + existing)


def _load_recording_errors(errors: list[BaseException]) -> None:
    """Run ``load_api_keys``, appending any exception to *errors*.

    A loader crash would otherwise vanish with the thread and let the
    main-thread save establish the expected values on its own — a race
    iteration that exercised nothing would still count as clean.
    """
    try:
        load_api_keys()
    except BaseException as exc:
        errors.append(exc)


def test_concurrent_save_survives_load_api_keys() -> None:
    """A key saved while the store is being loaded must not be reverted.

    Invariant checked after every save/load pair has fully completed:
    ``os.environ`` and ``DEFAULT_CONFIG`` hold the value of the LAST
    completed ``save_api_key`` — never the stale value from the store
    snapshot ``load_api_keys`` took before the save.  The canonical
    store itself must hold the new value too (the loader never writes
    the store, so a store regression would mean the saver was mangled).
    """
    from kiss.core import config as config_module

    _pad_store()
    stale: list[str] = []
    loader_errors: list[BaseException] = []
    iteration = 0
    for _ in range(_ROUNDS):
        for step in range(_SWEEP_STEPS):
            iteration += 1
            old_value = f"oldval{iteration}"
            new_value = f"newval{iteration}"
            # Seed: store, environment and config all hold the old value.
            save_api_key(_KEY, old_value)
            assert os.environ[_KEY] == old_value

            loader = threading.Thread(
                target=_load_recording_errors, args=(loader_errors,),
            )
            loader.start()
            try:
                time.sleep(step * _SWEEP_INCREMENT_S)
                save_api_key(_KEY, new_value)
            finally:
                # Join even when the save raises: fixture teardown must
                # not run while the loader still mutates process globals.
                loader.join(timeout=30)
            assert not loader.is_alive()
            assert not loader_errors, f"load_api_keys raised: {loader_errors!r}"
            assert f"export {_KEY}={new_value}" in api_keys_env_path().read_text()

            env_value = os.environ.get(_KEY)
            cfg_value = getattr(config_module.DEFAULT_CONFIG, _KEY)
            if env_value != new_value or cfg_value != new_value:
                stale.append(
                    f"iter {iteration} (sleep {step * _SWEEP_INCREMENT_S * 1000:.1f} ms): "
                    f"env={env_value!r} config={cfg_value!r} expected {new_value!r}"
                )

    assert not stale, (
        "load_api_keys applied a stale store snapshot over a completed "
        "save_api_key:\n" + "\n".join(stale)
    )


def test_load_api_keys_still_imports_saved_keys() -> None:
    """Sanity: a sequential load still imports the stored key for real."""
    save_api_key(_KEY, "sequential-value")
    # Drop it from the environment so only the store can restore it.
    os.environ.pop(_KEY, None)
    load_api_keys()
    assert os.environ.get(_KEY) == "sequential-value"
