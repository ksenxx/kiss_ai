# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 (core-base): a ``config.json`` that is not valid UTF-8
must be treated like any other unreadable config, not crash the caller.

``load_config`` and ``save_config`` guard the ``json.load`` of the
user-editable ``~/.kiss/config.json`` with ``except (json.JSONDecodeError,
OSError)``.  A hand-edited file holding Latin-1 bytes (or one truncated
inside a multi-byte sequence) raises ``UnicodeDecodeError`` instead —
a sibling ``ValueError`` the guard did not cover — so every daemon
``load_config`` call (and every settings save, which reads the file to
merge) blew up, while the same junk expressed as bad JSON was handled.
The two readers must agree on what "unreadable" means.

Real files in a real temp ``$KISS_HOME``; nothing is mocked.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kiss.core import vscode_config
from kiss.core.vscode_config import DEFAULTS, load_config, save_config

_JUNK = b'{"max_budget": 42, "work_dir": "/tmp/caf\xe9"}'  # Latin-1 é: not UTF-8


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``vscode_config`` at a temporary ``$KISS_HOME`` for the test."""
    home = tmp_path / "kiss_home"
    home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(home))
    for name in ("CONFIG_DIR", "CONFIG_PATH"):
        monkeypatch.delitem(vars(vscode_config), name, raising=False)
    return home / "config.json"


class TestConfigJunkBytes:
    """Invalid UTF-8 in config.json is 'unreadable', same as invalid JSON."""

    def test_load_config_falls_back_to_defaults(self, config_path: Path) -> None:
        config_path.write_bytes(_JUNK)
        assert load_config() == DEFAULTS

    def test_save_config_replaces_unreadable_file(self, config_path: Path) -> None:
        config_path.write_bytes(_JUNK)
        save_config({"max_budget": 7})
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        assert stored == {"max_budget": 7}
        assert load_config()["max_budget"] == 7
        leftovers = [p for p in os.listdir(config_path.parent) if p.startswith(".config.json-")]
        assert leftovers == [], f"staged temp files leaked: {leftovers}"

    def test_missing_file_and_non_object_json_fall_back(self, config_path: Path) -> None:
        assert not config_path.exists()
        assert load_config() == DEFAULTS
        config_path.write_text('["max_budget", 42]', encoding="utf-8")
        assert load_config() == DEFAULTS
        save_config({"max_budget": 3})
        assert json.loads(config_path.read_text(encoding="utf-8")) == {"max_budget": 3}

    def test_bad_json_and_bad_bytes_are_handled_alike(self, config_path: Path) -> None:
        config_path.write_text("{not json", encoding="utf-8")
        from_bad_json = load_config()
        config_path.write_bytes(_JUNK)
        from_bad_bytes = load_config()
        assert from_bad_json == from_bad_bytes == DEFAULTS
