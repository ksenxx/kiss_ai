# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The API-key store and shell RCs: non-UTF-8 bytes and ``str.splitlines`` separators.

Non-UTF-8 bytes.  The store and the RCs were read with a strict
``read_text(encoding="utf-8")`` guarded by ``except OSError``, but
``UnicodeDecodeError`` is a ``ValueError``: one Latin-1 byte (an old
``café`` comment) in ``~/.bashrc`` or ``api_keys.env`` made
:func:`load_api_keys` raise, so the daemon and the channel CLIs refused
to start, and made :func:`save_api_key` raise half way through (the
store edited, the RC not scrubbed, ``os.environ`` not updated; a delete
left the key in the RC for the next start's migration to bring back).
Both files are now read with ``surrogateescape`` and written back with
the same handler, so the user's bytes survive the edit unchanged.

Separators.  Every reader splits with ``str.splitlines``, which also
breaks at ``\\v \\f \\x1c \\x1d \\x1e \\x85 \\u2028 \\u2029``, while the
save guard refused only ``\\n`` and ``\\r``: ``sk-ab\\u2028cd`` was
stored as ``export K='sk-ab\\u2028cd'`` and reloaded as ``'sk-ab``.  The
guard and the RC-sourcing migration's filter now refuse every character
``splitlines`` breaks at.

Every test runs the real functions on real files under a temporary
``HOME``/``KISS_HOME``; the sourcing test runs the real ``bash``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from kiss.core import config as config_module
from kiss.core.brand import HOME_DIR
from kiss.core.user_config import (
    _RC_HOOK_LINE,
    _RC_SOURCED_MARKER,
    API_KEY_ENV_VARS,
    RC_HOOK_BEGIN,
    RC_HOOK_END,
    _resolve_shell_path,
    api_keys_env_path,
    load_api_keys,
    load_api_keys_readonly,
    save_api_key,
)
from kiss.tests.conftest import posix_only

LATIN1_COMMENT = b"# caf\xe9 (Latin-1, not UTF-8)\n"
HOOK_BLOCK = f"{RC_HOOK_BEGIN}\n{_RC_HOOK_LINE}\n{RC_HOOK_END}\n".encode()
SEPARATORS = ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A fresh ``HOME`` (bash user) and ``KISS_HOME``; key variables and config restored after.

    :func:`load_api_keys` and :func:`save_api_key` write ``os.environ``
    and the ``DEFAULT_CONFIG`` singleton directly, so both are put back
    by hand rather than through ``monkeypatch``.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("KISS_HOME", str(fake_home / HOME_DIR))
    monkeypatch.setenv("SHELL", "/bin/bash")
    saved_env = {key: os.environ.pop(key, None) for key in API_KEY_ENV_VARS}
    config = config_module.DEFAULT_CONFIG
    saved_config = dict(config.model_copy(deep=True).__dict__)
    yield fake_home
    for name, value in saved_config.items():
        setattr(config, name, value)
    for key, value in saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _write_store(content: bytes) -> Path:
    """Write the canonical key store with *content* and return its path."""
    store = api_keys_env_path()
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_bytes(content)
    return store


def test_daemon_start_migrates_keys_from_a_non_utf8_rc(home: Path) -> None:
    rc = home / ".bashrc"
    rc.write_bytes(LATIN1_COMMENT + b"export GEMINI_API_KEY=sk-from-rc\n")
    load_api_keys()
    assert os.environ["GEMINI_API_KEY"] == "sk-from-rc"
    assert api_keys_env_path().read_bytes() == b"export GEMINI_API_KEY=sk-from-rc\n"
    assert rc.read_bytes() == LATIN1_COMMENT + b"export GEMINI_API_KEY=sk-from-rc\n"


@pytest.mark.parametrize("loader", [load_api_keys, load_api_keys_readonly])
def test_daemon_and_cli_start_load_a_non_utf8_store(loader: Callable[[], None]) -> None:
    content = LATIN1_COMMENT + b"export GEMINI_API_KEY=sk-from-store\n"
    store = _write_store(content)
    loader()
    assert os.environ["GEMINI_API_KEY"] == "sk-from-store"
    assert config_module.DEFAULT_CONFIG.GEMINI_API_KEY == "sk-from-store"
    assert store.read_bytes() == content


def test_save_keeps_the_non_utf8_bytes_of_the_store_and_the_rc(home: Path) -> None:
    rc = home / ".bashrc"
    rc.write_bytes(LATIN1_COMMENT + b"export GEMINI_API_KEY=sk-old\n")
    store = _write_store(b"# na\xefve\n")
    save_api_key("GEMINI_API_KEY", "sk-new")
    assert store.read_bytes() == b"# na\xefve\nexport GEMINI_API_KEY=sk-new\n"
    assert rc.read_bytes() == LATIN1_COMMENT + HOOK_BLOCK
    assert os.environ["GEMINI_API_KEY"] == "sk-new"


def test_a_key_deleted_from_a_non_utf8_rc_stays_deleted_after_a_restart(home: Path) -> None:
    rc = home / ".bashrc"
    rc.write_bytes(LATIN1_COMMENT + b"export GEMINI_API_KEY=sk-old\n")
    store = _write_store(b"export GEMINI_API_KEY=sk-old\n")
    os.environ["GEMINI_API_KEY"] = "sk-old"
    save_api_key("GEMINI_API_KEY", "")
    assert rc.read_bytes() == LATIN1_COMMENT
    assert "GEMINI_API_KEY" not in os.environ
    load_api_keys()  # the next daemon start
    assert "GEMINI_API_KEY" not in os.environ
    assert b"GEMINI_API_KEY" not in store.read_bytes()


@pytest.mark.parametrize("sep", SEPARATORS, ids=[f"U+{ord(s):04X}" for s in SEPARATORS])
def test_save_refuses_a_value_splitlines_would_cut(home: Path, sep: str) -> None:
    save_api_key("GEMINI_API_KEY", f"sk-ab{sep}cd")
    assert "GEMINI_API_KEY" not in os.environ
    assert not api_keys_env_path().exists()
    assert not (home / ".bashrc").exists()


def _migrate_indirect_key(home: Path, value: str) -> str:
    """Start the daemon with an RC that assembles ``GEMINI_API_KEY`` from *value*; return the store.

    The textual scan cannot read ``"$V"``, so only sourcing the RC with
    the real ``bash`` finds the key.
    """
    if _resolve_shell_path("bash") is None:
        pytest.skip("bash is not installed")
    rc = home / ".bashrc"
    rc.write_text(f"V='{value}'\nexport GEMINI_API_KEY=\"$V\"\n", encoding="utf-8")
    load_api_keys()
    store = api_keys_env_path().read_text(encoding="utf-8")
    assert _RC_SOURCED_MARKER in store  # the RC really was sourced
    return store


@posix_only("the migration sources the RC with bash")
def test_rc_sourcing_stores_a_single_line_value(home: Path) -> None:
    store = _migrate_indirect_key(home, "sk-indirect")
    assert "export GEMINI_API_KEY=sk-indirect\n" in store
    assert os.environ["GEMINI_API_KEY"] == "sk-indirect"


@posix_only("the migration sources the RC with bash")
def test_rc_sourcing_does_not_store_a_value_splitlines_would_cut(home: Path) -> None:
    store = _migrate_indirect_key(home, "sk-ab\u2028cd")
    assert "GEMINI_API_KEY" not in store
    assert "GEMINI_API_KEY" not in os.environ
