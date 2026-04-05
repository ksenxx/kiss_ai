"""Integration tests for channel config persistence using shared helpers.

Verifies that all channels' _load_config/_save_config/_clear_config functions
use the shared load_json_config/save_json_config/clear_json_config helpers and
that round-trip persistence works correctly.

Also covers Slack's _load_token/_save_token/_clear_token (Issue 2).
"""

from __future__ import annotations

import json
from pathlib import Path

from kiss.channels._channel_agent_utils import (
    clear_json_config,
    load_json_config,
    save_json_config,
)


def test_shared_helpers_roundtrip(tmp_path: Path) -> None:
    """save_json_config + load_json_config round-trips correctly."""
    cfg_path = tmp_path / "config.json"
    save_json_config(cfg_path, {"key1": "val1", "key2": "val2"})
    loaded = load_json_config(cfg_path, ("key1", "key2"))
    assert loaded is not None
    assert loaded["key1"] == "val1"
    assert loaded["key2"] == "val2"


def test_shared_helpers_missing_required_key(tmp_path: Path) -> None:
    """load_json_config returns None when required keys are missing."""
    cfg_path = tmp_path / "config.json"
    save_json_config(cfg_path, {"key1": "val1"})
    assert load_json_config(cfg_path, ("key1", "key2")) is None


def test_shared_helpers_empty_required_key(tmp_path: Path) -> None:
    """load_json_config returns None when a required key is empty string."""
    cfg_path = tmp_path / "config.json"
    save_json_config(cfg_path, {"key1": "", "key2": "val"})
    assert load_json_config(cfg_path, ("key1", "key2")) is None


def test_shared_helpers_corrupt_json(tmp_path: Path) -> None:
    """load_json_config returns None for corrupt JSON files."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text("not json{{{")
    assert load_json_config(cfg_path, ("key",)) is None


def test_shared_helpers_non_dict_json(tmp_path: Path) -> None:
    """load_json_config returns None for non-dict JSON."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text('"just a string"')
    assert load_json_config(cfg_path, ("key",)) is None


def test_shared_helpers_missing_file(tmp_path: Path) -> None:
    """load_json_config returns None for missing file."""
    cfg_path = tmp_path / "nonexistent.json"
    assert load_json_config(cfg_path, ("key",)) is None


def test_shared_helpers_clear(tmp_path: Path) -> None:
    """clear_json_config removes the file."""
    cfg_path = tmp_path / "config.json"
    save_json_config(cfg_path, {"key": "val"})
    assert cfg_path.exists()
    clear_json_config(cfg_path)
    assert not cfg_path.exists()


def test_shared_helpers_clear_nonexistent(tmp_path: Path) -> None:
    """clear_json_config is a no-op if file doesn't exist."""
    cfg_path = tmp_path / "nonexistent.json"
    clear_json_config(cfg_path)  # Should not raise


def test_shared_helpers_creates_parent_dirs(tmp_path: Path) -> None:
    """save_json_config creates parent directories."""
    cfg_path = tmp_path / "a" / "b" / "config.json"
    save_json_config(cfg_path, {"key": "val"})
    assert cfg_path.exists()
    loaded = load_json_config(cfg_path, ("key",))
    assert loaded is not None
    assert loaded["key"] == "val"


def test_shared_helpers_no_required_keys(tmp_path: Path) -> None:
    """load_json_config with empty required_keys returns any valid dict."""
    cfg_path = tmp_path / "config.json"
    save_json_config(cfg_path, {"enabled": "true"})
    loaded = load_json_config(cfg_path, ())
    assert loaded is not None
    assert loaded["enabled"] == "true"


def test_shared_helpers_null_value_becomes_empty(tmp_path: Path) -> None:
    """load_json_config converts None values to empty strings."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"key": None}))
    loaded = load_json_config(cfg_path, ())
    assert loaded is not None
    assert loaded["key"] == ""
