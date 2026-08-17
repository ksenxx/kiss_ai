# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit round 6: config persistence (B1) and fast-model selection (B3) fixes.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.server.test_vscode_audit_6``; the non-core tests remain there.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase

from kiss.core.models.model_info import get_fast_model
from kiss.core.vscode_config import CONFIG_PATH, load_config, save_config


class TestFastModelForReturnsActuallyFastModels(TestCase):
    """B3 FIX: ``get_fast_model()`` now returns genuinely cheap/fast
    models for each provider.  Previously, the Gemini branch returned
    ``gemini-2.5-pro`` which is one of the most expensive models.
    """

    def test_gemini_model_is_flash_not_pro(self) -> None:
        """The Gemini fast model should be a flash variant, not pro."""
        from kiss.core import config as config_module

        orig = config_module.DEFAULT_CONFIG
        try:
            config_module.DEFAULT_CONFIG = type(orig)()
            config_module.DEFAULT_CONFIG.ANTHROPIC_API_KEY = ""
            config_module.DEFAULT_CONFIG.OPENROUTER_API_KEY = ""
            config_module.DEFAULT_CONFIG.TOGETHER_API_KEY = ""
            config_module.DEFAULT_CONFIG.GEMINI_API_KEY = "test-key"
            config_module.DEFAULT_CONFIG.OPENAI_API_KEY = ""

            result = get_fast_model()
            assert "flash" in result.lower() or "2.0" in result, (
                f"B3 FIX: Gemini fast model should be a flash variant, "
                f"got '{result}'"
            )
            assert "pro" not in result.lower() or "flash" in result.lower(), (
                f"B3 FIX: Gemini fast model should not be a pro model, "
                f"got '{result}'"
            )
        finally:
            config_module.DEFAULT_CONFIG = orig


class TestSaveConfigPreservesExtraKeys(TestCase):
    """B1 FIX: ``save_config`` now preserves non-DEFAULTS keys that
    were already stored in ``~/.kiss/config.json``.

    Previously, calling ``save_config({"max_budget": 50})`` would
    overwrite config.json with only ``{"max_budget": 50}``, silently
    dropping keys like ``email`` that are documented as living in
    config.json.
    """

    def setUp(self) -> None:
        self._orig_path = str(CONFIG_PATH)
        self._tmpdir = tempfile.mkdtemp()
        self._tmp_config = Path(self._tmpdir) / "config.json"
        import kiss.core.vscode_config as mod

        self._mod = mod
        self._orig_config_path = mod.CONFIG_PATH
        self._orig_config_dir = mod.CONFIG_DIR
        mod.CONFIG_PATH = self._tmp_config
        mod.CONFIG_DIR = Path(self._tmpdir)

    def tearDown(self) -> None:
        self._mod.CONFIG_PATH = self._orig_config_path
        self._mod.CONFIG_DIR = self._orig_config_dir
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_config_preserves_email_key(self) -> None:
        """Saving config should not strip the ``email`` key."""
        self._tmp_config.write_text(
            json.dumps({"email": "user@example.com", "max_budget": 100})
        )
        save_config({"max_budget": 50})
        stored = json.loads(self._tmp_config.read_text())
        assert stored.get("email") == "user@example.com", (
            f"B1 FIX: email should be preserved, got {stored}"
        )
        assert stored.get("max_budget") == 50

    def test_save_config_preserves_arbitrary_extra_keys(self) -> None:
        """Any non-DEFAULTS key already in config.json should survive a save."""
        self._tmp_config.write_text(
            json.dumps({
                "email": "a@b.com",
                "tunnel_token": "tok-123",
                "max_budget": 100,
            })
        )
        save_config({"max_budget": 75, "use_web_browser": False})
        stored = json.loads(self._tmp_config.read_text())
        assert stored["email"] == "a@b.com"
        assert stored["tunnel_token"] == "tok-123"
        assert stored["max_budget"] == 75
        assert stored["use_web_browser"] is False

    def test_save_config_new_file_still_works(self) -> None:
        """When no config.json exists yet, save_config creates one correctly."""
        assert not self._tmp_config.exists()
        save_config({"max_budget": 42})
        stored = json.loads(self._tmp_config.read_text())
        assert stored["max_budget"] == 42

    def test_load_then_save_round_trip(self) -> None:
        """load_config → modify → save_config round-trip preserves extra keys."""
        self._tmp_config.write_text(
            json.dumps({"email": "keep@me.com", "max_budget": 100})
        )
        cfg = load_config()
        cfg["max_budget"] = 200
        save_config(cfg)
        stored = json.loads(self._tmp_config.read_text())
        assert stored["email"] == "keep@me.com"
        assert stored["max_budget"] == 200
