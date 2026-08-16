# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Characterization (lockdown) tests for json_printer and vscode_config.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.agents.vscode.test_simplification_lockdown_printer_config``;
the non-core tests remain there.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path


class _ConfigDirTestCase(unittest.TestCase):
    """Base: redirect CONFIG_DIR/CONFIG_PATH to a fresh temp dir per test."""

    def setUp(self) -> None:
        """Point the config module at an isolated temporary directory."""
        import kiss.core.vscode_config as vc

        self._orig_dir = vc.CONFIG_DIR
        self._orig_path = vc.CONFIG_PATH
        self._tmpdir = tempfile.mkdtemp()
        vc.CONFIG_DIR = Path(self._tmpdir)
        vc.CONFIG_PATH = Path(self._tmpdir) / "config.json"

    def tearDown(self) -> None:
        """Restore the real config locations and remove the temp dir."""
        import kiss.core.vscode_config as vc

        vc.CONFIG_DIR = self._orig_dir
        vc.CONFIG_PATH = self._orig_path
        shutil.rmtree(self._tmpdir, ignore_errors=True)


class TestCustomHeaderParsingParity(_ConfigDirTestCase):
    """Both header consumers parse `custom_headers` identically."""

    def test_get_custom_model_entry_and_build_model_config_agree(self) -> None:
        """Saved headers yield the same extra_headers dict via both APIs."""
        from kiss.core.vscode_config import (
            build_model_config,
            get_custom_model_entry,
            load_config,
            save_config,
        )

        save_config({
            "custom_endpoint": "http://localhost:8080/v1",
            "custom_headers": "A:1\nbad\nB: two : three",
        })
        cfg = load_config()

        expected = {"A": "1", "B": "two : three"}
        entry = get_custom_model_entry(cfg)
        model_config = build_model_config(cfg)
        assert entry is not None
        assert model_config is not None
        assert entry["extra_headers"] == expected, entry
        assert model_config["extra_headers"] == expected, model_config
        assert entry["extra_headers"] == model_config["extra_headers"]

    def test_whitespace_is_stripped_identically(self) -> None:
        """Key/value whitespace stripping is the same on both paths."""
        from kiss.core.vscode_config import (
            build_model_config,
            get_custom_model_entry,
        )

        cfg = {
            "custom_endpoint": "http://localhost:8080/v1",
            "custom_api_key": "",
            "custom_headers": "  X-Pad  :   spaced value  \n\nNoColonLine",
        }
        expected = {"X-Pad": "spaced value"}
        entry = get_custom_model_entry(cfg)
        model_config = build_model_config(cfg)
        assert entry is not None
        assert model_config is not None
        assert entry["extra_headers"] == expected, entry
        assert model_config["extra_headers"] == expected, model_config


class TestConfigRoundTrip(_ConfigDirTestCase):
    """`save_config`/`load_config` semantics that any refactor must keep."""

    def test_save_then_load_overlays_defaults(self) -> None:
        """A partial save round-trips, and every DEFAULTS key is present."""
        from kiss.core.vscode_config import DEFAULTS, load_config, save_config

        save_config({"max_budget": 7})
        cfg = load_config()

        assert cfg["max_budget"] == 7
        for key, default_value in DEFAULTS.items():
            assert key in cfg, f"missing DEFAULTS key {key!r}: {cfg}"
            if key != "max_budget":
                assert cfg[key] == default_value, (key, cfg[key])

    def test_corrupt_file_returns_defaults_without_raising(self) -> None:
        """Invalid JSON on disk falls back to pure DEFAULTS."""
        import kiss.core.vscode_config as vc
        from kiss.core.vscode_config import DEFAULTS, load_config

        vc.CONFIG_PATH.write_text("{this is not json", encoding="utf-8")

        cfg = load_config()
        assert cfg == dict(DEFAULTS), cfg

    def test_save_preserves_disk_keys_and_writes_extension_keys(self) -> None:
        """Untouched keys on disk survive; keys passed in are written.

        This used to assert that non-DEFAULTS *input* keys were dropped.
        That was a defect, not a guarantee: ``tunnel_token``,
        ``skill_permissions``, ``mcp_permissions`` and ``email`` are all
        read at runtime, so accepting them silently and discarding them
        lost the value until the next daemon restart.  Only API keys are
        excluded, and only retired keys are purged.
        """
        import kiss.core.vscode_config as vc
        from kiss.core.vscode_config import load_config, save_config

        vc.CONFIG_PATH.write_text(
            json.dumps({"email": "a@b.c", "max_budget": 3}), encoding="utf-8",
        )

        save_config({
            "max_budget": 7,
            "tunnel_token": "tok-1",
            "ANTHROPIC_API_KEY": "sk-nope",
            "demo_mode": True,
        })

        stored = json.loads(vc.CONFIG_PATH.read_text(encoding="utf-8"))
        assert stored["email"] == "a@b.c", stored
        assert stored["max_budget"] == 7, stored
        assert stored["tunnel_token"] == "tok-1", stored
        assert "ANTHROPIC_API_KEY" not in stored, stored
        assert "demo_mode" not in stored, stored
        cfg = load_config()
        assert cfg["email"] == "a@b.c"
        assert cfg["max_budget"] == 7
        assert cfg["tunnel_token"] == "tok-1"
