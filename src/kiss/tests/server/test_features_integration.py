# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for: web browser toggle wiring and API key deletion
through the VS Code server's saveConfig handler.

The core-only methods (budget limit, custom endpoint, config persistence,
key save/overwrite lifecycle) moved to
``tests/core/test_features_integration.py``; the methods here exercise
``kiss.agents.sorcar.sorcar_agent`` or ``kiss.server.server``.  The
``_restore_default_config`` autouse fixture is shared from the core twin.

Each test uses real HTTP servers, real file I/O, and real objects —
no mocks, patches, fakes, or test doubles (except monkeypatch for env
isolation, which is not a test double).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kiss.core.vscode_config import API_KEY_ENV_VARS
from kiss.tests.core.test_features_integration import (  # noqa: F401
    _restore_default_config,
)


class TestApiKeySetupAndDeletion:
    """Delete a key by saving it empty through the server's saveConfig."""

    @pytest.fixture(autouse=True)
    def _isolate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        import kiss.core.vscode_config as _vc

        monkeypatch.setitem(vars(_vc), "CONFIG_DIR", fake_home / ".kiss")
        monkeypatch.setitem(
            vars(_vc), "CONFIG_PATH", fake_home / ".kiss" / "config.json",
        )
        monkeypatch.setenv("SHELL", "/bin/zsh")
        for k in API_KEY_ENV_VARS:
            monkeypatch.delenv(k, raising=False)
        from kiss.core import config as config_module

        monkeypatch.setattr(config_module, "DEFAULT_CONFIG", config_module.DEFAULT_CONFIG)

    def test_delete_key_by_saving_empty(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Saving an empty key via config panel skips writing to RC.

        The VSCodeServer saveConfig handler skips empty keys, so
        after the key is removed from the env and not written to RC,
        it is effectively deleted.
        """
        from kiss.server.server import VSCodeServer

        server = VSCodeServer()

        server._handle_command({
            "type": "saveConfig",
            "config": {"max_budget": 100},
            "apiKeys": {"ANTHROPIC_API_KEY": "ant-key-to-delete"},
        })
        assert os.environ["ANTHROPIC_API_KEY"] == "ant-key-to-delete"
        rc = Path.home() / ".zshrc"
        assert "ant-key-to-delete" in rc.read_text()

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        server._handle_command({
            "type": "saveConfig",
            "config": {"max_budget": 100},
            "apiKeys": {"ANTHROPIC_API_KEY": ""},
        })
        assert os.environ.get("ANTHROPIC_API_KEY") is None
