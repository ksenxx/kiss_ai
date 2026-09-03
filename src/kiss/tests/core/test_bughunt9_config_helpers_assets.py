# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt round 9 — e2e repros for three real defects.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.server.test_bughunt9_config_helpers_assets``; the non-core tests remain there.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest

from kiss.core import config as config_module
from kiss.core.vscode_config import (
    DEFAULTS,
    apply_config_to_env,
    save_api_key,
)


class TestApplyConfigBooleanBudget:
    """A boolean ``max_budget`` must fall back to the default budget."""

    @pytest.fixture(autouse=True)
    def _restore_budget(self):  # type: ignore[no-untyped-def]
        saved = config_module.DEFAULT_CONFIG.max_budget
        yield
        config_module.DEFAULT_CONFIG.max_budget = saved

    def test_true_budget_falls_back_to_default(self) -> None:
        """``True`` must not become a $1.00 live budget."""
        config_module.DEFAULT_CONFIG.max_budget = 55.0
        apply_config_to_env({"max_budget": True})
        assert config_module.DEFAULT_CONFIG.max_budget == float(
            DEFAULTS["max_budget"],
        )

    def test_false_budget_falls_back_to_default(self) -> None:
        """``False`` must not become a $0.00 live budget."""
        config_module.DEFAULT_CONFIG.max_budget = 55.0
        apply_config_to_env({"max_budget": False})
        assert config_module.DEFAULT_CONFIG.max_budget == float(
            DEFAULTS["max_budget"],
        )

    def test_genuine_numbers_still_apply(self) -> None:
        apply_config_to_env({"max_budget": 42})
        assert config_module.DEFAULT_CONFIG.max_budget == 42.0
        apply_config_to_env({"max_budget": 7.5})
        assert config_module.DEFAULT_CONFIG.max_budget == 7.5


class TestSaveApiKeyNameValidation:
    """Only valid env-var identifiers may reach the key store."""

    @pytest.fixture(autouse=True)
    def fake_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> Generator[Path]:
        """Point ``HOME``/``SHELL``/key store at a scratch dir."""
        import kiss.core.vscode_config as _vc

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.setitem(vars(_vc), "CONFIG_DIR", tmp_path / ".kiss")
        monkeypatch.setitem(
            vars(_vc), "CONFIG_PATH", tmp_path / ".kiss" / "config.json",
        )
        monkeypatch.setenv("OPENAI_API_KEY", "sentinel-original")
        # ``save_api_key`` updates the DEFAULT_CONFIG singleton IN
        # PLACE, so rebinding the module attribute to the saved reference
        # would restore nothing (the leaked ``sk-valid`` key then fails any
        # later live-API test in the same process).  Snapshot and restore
        # the field values instead.
        saved_config = config_module.DEFAULT_CONFIG
        snapshot = dict(saved_config.model_copy(deep=True).__dict__)
        yield tmp_path
        config_module.DEFAULT_CONFIG = saved_config
        for key, value in snapshot.items():
            setattr(saved_config, key, value)

    def test_newline_name_cannot_inject_rc_commands(
        self, fake_home: Path,
    ) -> None:
        """A name embedding a newline must not write commands to the RC."""
        evil = "OPENAI_API_KEY\ntouch " + str(fake_home / "pwned") + "\n#"
        save_api_key(evil, "sk-x")
        rc = fake_home / ".bashrc"
        assert not rc.exists(), rc.read_text()
        assert evil not in os.environ

    def test_metacharacter_name_rejected(self, fake_home: Path) -> None:
        """Shell metacharacters in the name must never reach the RC."""
        for evil in ("FOO$(touch pwned)", "FOO; rm -rf ~", "FOO BAR", ""):
            save_api_key(evil, "sk-x")
        assert not (fake_home / ".bashrc").exists()

    def test_equals_name_does_not_raise(self, fake_home: Path) -> None:
        """A name containing ``=`` must not raise out of the handler."""
        save_api_key("A=B", "sk-x")
        assert not (fake_home / ".bashrc").exists()

    def test_valid_name_still_written_and_exported(
        self, fake_home: Path,
    ) -> None:
        """The legitimate save path keeps working after the fix."""
        from kiss.core.vscode_config import api_keys_env_path

        save_api_key("OPENAI_API_KEY", "sk-valid")
        store = api_keys_env_path()
        assert store.exists()
        assert "export OPENAI_API_KEY=sk-valid" in store.read_text()
        rc = fake_home / ".bashrc"
        if rc.exists():
            assert "sk-valid" not in rc.read_text()
        assert os.environ["OPENAI_API_KEY"] == "sk-valid"
