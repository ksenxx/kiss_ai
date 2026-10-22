# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit B fixes: config.json junk integers, unreadable config, adoption lock.

F1: a ``max_budget`` integer too large for a float must not raise
``OverflowError`` out of ``load_config`` / ``apply_config_to_env``.
F2: ``save_config`` must refuse to replace a ``config.json`` it could not
read instead of publishing a file that holds only the new keys.
F3: ``adopt_legacy_file`` must not take its ``.rename.lock`` once the
adoption is done (the legacy name is a link to the new one).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kiss.core import config as config_module
from kiss.core import user_config
from kiss.tests.conftest import is_root, posix_only

HUGE_INT = int("1" * 400)


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".kiss"
    home.mkdir()
    monkeypatch.setitem(vars(user_config), "CONFIG_DIR", home)
    monkeypatch.setitem(vars(user_config), "CONFIG_PATH", home / "config.json")
    return home


def test_load_config_survives_huge_integer_budget(config_home: Path) -> None:
    (config_home / "config.json").write_text(
        '{"max_budget": ' + "1" * 400 + ', "email": "a@b"}',
        encoding="utf-8",
    )
    cfg = user_config.load_config()
    assert cfg["max_budget"] == user_config.DEFAULTS["max_budget"]
    assert cfg["email"] == "a@b"


def test_apply_config_to_env_survives_huge_integer_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module.DEFAULT_CONFIG,
        "max_budget",
        config_module.DEFAULT_CONFIG.max_budget,
    )
    user_config.apply_config_to_env({"max_budget": HUGE_INT})
    assert config_module.DEFAULT_CONFIG.max_budget == user_config.DEFAULTS["max_budget"]
    # An integer that does fit a float is kept exactly.
    user_config.apply_config_to_env({"max_budget": 7})
    assert config_module.DEFAULT_CONFIG.max_budget == 7.0


def test_sanitize_config_huge_integers() -> None:
    assert user_config.sanitize_config({"max_budget": HUGE_INT}) == {
        "max_budget": user_config.DEFAULTS["max_budget"],
    }
    # An int that fits a float stays an int; a numeric string becomes a float.
    assert user_config.sanitize_config({"max_budget": 10**300}) == {"max_budget": 10**300}
    assert user_config.sanitize_config({"max_budget": "1e400"}) == {
        "max_budget": user_config.DEFAULTS["max_budget"],
    }
    assert user_config.sanitize_config({"max_budget": "12.5"}) == {"max_budget": 12.5}


@posix_only("chmod-based permission denial")
@pytest.mark.skipif(is_root(), reason="root can read a 000 file")
def test_save_config_refuses_to_wipe_unreadable_file(config_home: Path) -> None:
    cfg_path = config_home / "config.json"
    cfg_path.write_text(json.dumps({"tunnel_token": "tok", "email": "a@b"}), encoding="utf-8")
    cfg_path.chmod(0)
    try:
        with pytest.raises(OSError):
            user_config.save_config({"work_dir": "/tmp"})
        with pytest.raises(OSError):
            user_config.setdefault_config({"work_dir": "/tmp"})
    finally:
        cfg_path.chmod(0o600)
    assert json.loads(cfg_path.read_text(encoding="utf-8")) == {
        "tunnel_token": "tok",
        "email": "a@b",
    }
    # Readers stay lenient: an unreadable file reads as empty.
    cfg_path.chmod(0)
    try:
        assert user_config.load_config() == user_config.sanitize_config(user_config.DEFAULTS)
    finally:
        cfg_path.chmod(0o600)


@posix_only("chmod-based permission denial")
@pytest.mark.skipif(is_root(), reason="root can traverse a 000 directory")
def test_save_config_refuses_to_replace_link_into_unreadable_dir(config_home: Path) -> None:
    """A ``config.json`` link into a mode-000 directory is unreadable, not missing.

    ``Path.exists()`` answers ``False`` for such a target, so an existence
    precheck would read it as "no file yet" and let the writer replace the
    link with a file holding only its own keys.
    """
    locked = config_home / "locked"
    locked.mkdir()
    target = locked / "config.json"
    target.write_text(json.dumps({"tunnel_token": "tok", "email": "a@b"}), encoding="utf-8")
    cfg_path = config_home / "config.json"
    cfg_path.symlink_to(Path("locked") / "config.json")
    locked.chmod(0)
    try:
        with pytest.raises(OSError):
            user_config.save_config({"work_dir": "/tmp"})
        with pytest.raises(OSError):
            user_config.setdefault_config({"work_dir": "/tmp"})
        assert user_config.load_config() == user_config.sanitize_config(user_config.DEFAULTS)
    finally:
        locked.chmod(0o700)
    assert os.readlink(cfg_path) == os.path.join("locked", "config.json")
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "tunnel_token": "tok",
        "email": "a@b",
    }


def test_save_config_still_replaces_undecodable_file(config_home: Path) -> None:
    cfg_path = config_home / "config.json"
    cfg_path.write_text("{not json", encoding="utf-8")
    user_config.save_config({"work_dir": "/tmp"})
    assert json.loads(cfg_path.read_text(encoding="utf-8")) == {"work_dir": "/tmp"}


@posix_only("the adoption leaves a symlink behind only on POSIX")
def test_adopted_file_does_not_touch_rename_lock(tmp_path: Path) -> None:
    new = tmp_path / "AGENTS.md"
    old = tmp_path / "SORCAR.md"
    lock = tmp_path / "AGENTS.md.rename.lock"
    old.write_text("hello", encoding="utf-8")
    config_module.adopt_legacy_file(new, "SORCAR.md")
    assert new.read_text(encoding="utf-8") == "hello"
    assert os.readlink(old) == "AGENTS.md"
    assert lock.exists()
    lock.unlink()
    for _ in range(3):
        config_module.adopt_legacy_file(new, "SORCAR.md")
    assert not lock.exists()
    assert new.read_text(encoding="utf-8") == "hello"
    assert os.readlink(old) == "AGENTS.md"


@posix_only("the adoption leaves a symlink behind only on POSIX")
def test_legacy_link_to_a_distinct_file_still_goes_through_the_lock(tmp_path: Path) -> None:
    """Only a link to *path* itself counts as done; a link elsewhere is decided under the lock."""
    new = tmp_path / "AGENTS.md"
    old = tmp_path / "SORCAR.md"
    other = tmp_path / "OTHER.md"
    lock = tmp_path / "AGENTS.md.rename.lock"
    other.write_text("other", encoding="utf-8")
    new.write_text("new", encoding="utf-8")
    old.symlink_to("OTHER.md")
    config_module.adopt_legacy_file(new, "SORCAR.md")
    assert lock.exists()
    assert new.read_text(encoding="utf-8") == "new"
    assert os.readlink(old) == "OTHER.md"
    # A dangling link is no legacy file at all: nothing happens, no lock.
    lock.unlink()
    other.unlink()
    config_module.adopt_legacy_file(new, "SORCAR.md")
    assert not lock.exists()
    assert new.read_text(encoding="utf-8") == "new"
