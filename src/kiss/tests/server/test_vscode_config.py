# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for VS Code configuration panel backend."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any

import pytest

import kiss.core.vscode_config as vscode_config
from kiss.core.vscode_config import (
    load_config,
    save_config,
)
from kiss.tests.core.test_vscode_config import _isolate_config  # noqa: F401


class _Recorder:
    """Lightweight stand-in for ``io.StringIO`` that captures broadcast events.

    Tests historically captured events by redirecting ``sys.stdout``; the
    server now publishes them via ``self.printer.broadcast``.  We patch
    ``printer.broadcast`` to append into ``events`` instead.  ``truncate``
    and ``seek`` are provided so tests can clear the buffer between
    sub-actions in the same way they did with the old ``StringIO``.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def truncate(self, _size: int = 0) -> int:
        self.events.clear()
        return 0

    def seek(self, _pos: int) -> int:  # pragma: no cover - trivial
        return 0


def _make_server_with_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, _Recorder]:
    """Construct a ``VSCodeServer`` and intercept ``printer.broadcast``."""
    from kiss.server.server import VSCodeServer

    server = VSCodeServer()
    recorder = _Recorder()
    monkeypatch.setattr(
        server.printer, "broadcast", lambda event: recorder.events.append(event),
    )
    return server, recorder


class TestCommandHandlerIntegration:
    """Integration tests for getConfig/saveConfig using real VSCodeServer."""

    def _capture_broadcasts(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[Any, _Recorder]:
        """Create a VSCodeServer with ``printer.broadcast`` recorded."""
        return _make_server_with_recorder(monkeypatch)

    @staticmethod
    def _parse_events(captured: _Recorder) -> list[dict]:
        return list(captured.events)

    def test_get_config_broadcasts_defaults(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        server, captured = self._capture_broadcasts(monkeypatch)
        server._handle_command({"type": "getConfig"})
        events = self._parse_events(captured)
        cfg_events = [e for e in events if e["type"] == "configData"]
        assert len(cfg_events) == 1
        assert cfg_events[0]["config"]["max_budget"] == 100
        assert cfg_events[0]["config"]["use_web_browser"] is True

    def test_save_config_persists_and_broadcasts(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        server, captured = self._capture_broadcasts(monkeypatch)
        server._handle_command({
            "type": "saveConfig",
            "config": {"max_budget": 25, "use_web_browser": False},
            "apiKeys": {},
        })
        events = self._parse_events(captured)
        cfg_events = [e for e in events if e["type"] == "configData"]
        assert len(cfg_events) == 1
        assert cfg_events[0]["config"]["max_budget"] == 25
        assert cfg_events[0]["config"]["use_web_browser"] is False
        assert load_config()["max_budget"] == 25

    def test_save_config_with_api_keys(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """saveConfig with API keys writes them to the canonical store and env.

        Keys land in ``$KISS_HOME/api_keys.env`` only; the shell RC gets a
        hook sourcing that file, never a second copy of the key.
        """
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        server, captured = self._capture_broadcasts(monkeypatch)
        server._handle_command({
            "type": "saveConfig",
            "config": {"max_budget": 100},
            "apiKeys": {"OPENROUTER_API_KEY": "or-key-123"},
        })
        assert os.environ["OPENROUTER_API_KEY"] == "or-key-123"
        assert (
            f"export OPENROUTER_API_KEY={shlex.quote('or-key-123')}"
            in vscode_config.api_keys_env_path().read_text()
        )
        rc_content = (Path.home() / ".zshrc").read_text()
        assert "or-key-123" not in rc_content
        assert '. "$HOME/.kiss/api_keys.env"' in rc_content

    def test_save_config_skips_empty_api_keys(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty API key values are not written to shell RC."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        server, captured = self._capture_broadcasts(monkeypatch)
        server._handle_command({
            "type": "saveConfig",
            "config": {},
            "apiKeys": {"GEMINI_API_KEY": ""},
        })
        rc = Path.home() / ".zshrc"
        assert not rc.exists()

    def test_save_config_refreshes_models(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """saveConfig triggers a models broadcast."""
        server, captured = self._capture_broadcasts(monkeypatch)
        server._handle_command({
            "type": "saveConfig",
            "config": {"custom_endpoint": "http://localhost:8080/v1"},
            "apiKeys": {},
        })
        events = self._parse_events(captured)
        model_events = [e for e in events if e["type"] == "models"]
        assert len(model_events) == 1
        names = [m["name"] for m in model_events[0]["models"]]
        assert "custom/v1" in names

    def test_get_config_after_save(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """getConfig returns the config that was previously saved."""
        server, captured = self._capture_broadcasts(monkeypatch)
        save_config({"max_budget": 77, "remote_password": "pw123"})
        server._handle_command({"type": "getConfig"})
        events = self._parse_events(captured)
        cfg_events = [e for e in events if e["type"] == "configData"]
        assert cfg_events[0]["config"]["max_budget"] == 77
        assert cfg_events[0]["config"]["remote_password"] == "pw123"


class TestEndToEndFlows:
    """Full integration flows across multiple functions."""



    def test_custom_model_in_models_list(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Custom endpoint config appears in VSCodeServer._get_models output."""
        save_config({
            "custom_endpoint": "http://localhost:9999/completions",
            "custom_api_key": "ck-test",
        })

        server, recorder = _make_server_with_recorder(monkeypatch)
        server._get_models()

        model_events = [e for e in recorder.events if e["type"] == "models"]
        assert len(model_events) == 1
        custom_models = [
            m for m in model_events[0]["models"] if m["vendor"] == "Custom"
        ]
        assert len(custom_models) == 1
        assert custom_models[0]["name"] == "custom/completions"
        assert custom_models[0]["api_key"] == "ck-test"

    def test_no_custom_model_without_endpoint(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No custom model in list when endpoint is empty."""
        save_config({"custom_endpoint": ""})

        server, recorder = _make_server_with_recorder(monkeypatch)
        server._get_models()

        model_events = [e for e in recorder.events if e["type"] == "models"]
        assert len(model_events) == 1
        custom_models = [
            m for m in model_events[0]["models"] if m.get("vendor") == "Custom"
        ]
        assert len(custom_models) == 0




class TestGetConfigIncludesApiKeys:
    """Test that getConfig command includes current API keys in the response."""

    def test_get_config_includes_api_keys(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """getConfig broadcast includes apiKeys with current env values."""
        monkeypatch.setenv("GEMINI_API_KEY", "gem-test-val")
        monkeypatch.setenv("OPENAI_API_KEY", "oai-test-val")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        server, recorder = _make_server_with_recorder(monkeypatch)
        server._handle_command({"type": "getConfig"})
        cfg_events = [e for e in recorder.events if e["type"] == "configData"]
        assert len(cfg_events) == 1
        assert "apiKeys" in cfg_events[0]
        assert cfg_events[0]["apiKeys"]["GEMINI_API_KEY"] == "gem-test-val"
        assert cfg_events[0]["apiKeys"]["OPENAI_API_KEY"] == "oai-test-val"
        assert cfg_events[0]["apiKeys"]["ANTHROPIC_API_KEY"] == ""

    def test_get_config_api_keys_after_save(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After saving an API key, getConfig returns the updated value."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

        server, recorder = _make_server_with_recorder(monkeypatch)
        server._handle_command({
            "type": "saveConfig",
            "config": {"max_budget": 100},
            "apiKeys": {"TOGETHER_API_KEY": "tog-key-saved"},
        })
        recorder.events.clear()
        server._handle_command({"type": "getConfig"})
        cfg_events = [e for e in recorder.events if e["type"] == "configData"]
        assert len(cfg_events) == 1
        assert cfg_events[0]["apiKeys"]["TOGETHER_API_KEY"] == "tog-key-saved"






class TestRetiredKeys:
    """A setting that no longer exists must be forgotten, not preserved.

    ``config.json`` is written by every previous release, so removing a
    key from :data:`DEFAULTS` cannot be the whole job: ``load_config``
    overlays whatever the file holds, ``sanitize_config`` deliberately
    lets unknown keys through so genuine extension-owned keys survive,
    and ``save_config`` rewrites the file from its own former contents.
    """

    def _write_legacy_config(self) -> Path:
        """Write a config file as an older release would have left it."""
        path = Path(vscode_config.CONFIG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "demo_mode": True,
                "max_budget": 42,
                "tunnel_token": "keep-me",
                "email": "someone@example.com",
            }),
            encoding="utf-8",
        )
        return path





    def test_config_data_reply_omits_a_retired_key(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The config the daemon sends every client must not carry it."""
        self._write_legacy_config()
        server, captured = _make_server_with_recorder(monkeypatch)
        server._handle_command({"type": "getConfig"})
        server._handle_command({
            "type": "saveConfig", "config": {"max_budget": 11},
        })
        replies = [e for e in captured.events if e["type"] == "configData"]
        assert replies, f"no configData event was broadcast: {captured.events}"
        for reply in replies:
            assert "demo_mode" not in reply["config"], (
                f"configData still advertises a retired setting: {reply}"
            )
