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

    def test_delete_key_by_saving_empty(self) -> None:
        """Saving an empty key via the config panel deletes it everywhere.

        The VSCodeServer saveConfig handler forwards empty values to
        ``save_api_key``, which treats them as a delete: the ``export``
        line is removed from the canonical key store and the variable is
        dropped from ``os.environ`` — so the key stays gone across the
        next daemon start too.
        """
        from kiss.core.vscode_config import api_keys_env_path, load_api_keys
        from kiss.server.server import VSCodeServer

        server = VSCodeServer()

        server._handle_command({
            "type": "saveConfig",
            "config": {"max_budget": 100},
            "apiKeys": {"ANTHROPIC_API_KEY": "ant-key-to-delete"},
        })
        assert os.environ["ANTHROPIC_API_KEY"] == "ant-key-to-delete"
        store = api_keys_env_path()
        assert "ant-key-to-delete" in store.read_text()

        server._handle_command({
            "type": "saveConfig",
            "config": {"max_budget": 100},
            "apiKeys": {"ANTHROPIC_API_KEY": ""},
        })
        assert os.environ.get("ANTHROPIC_API_KEY") is None
        content = store.read_text()
        assert "ant-key-to-delete" not in content
        assert "ANTHROPIC_API_KEY" not in content

        # The delete must survive what a daemon restart would do.
        load_api_keys()
        assert os.environ.get("ANTHROPIC_API_KEY") is None
