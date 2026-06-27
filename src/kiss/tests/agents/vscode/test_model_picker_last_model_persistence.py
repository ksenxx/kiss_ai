# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration regression: model picker refresh re-reads persisted last_model.

A long-running ``kiss-web`` daemon can outlive a VS Code window.  If the
persisted ``last_model`` preference in ``~/.kiss/config.json`` changes while
that daemon is alive, a fresh VS Code activation asks the daemon for the model
list via ``getModels``.  The reply must select the persisted last model rather
than the daemon's stale in-memory default, otherwise reopening VS Code shows an
older default such as ``gpt-5.5`` even though the user previously selected a
Claude model.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from collections.abc import Generator

import pytest

from kiss.agents.sorcar.persistence import _close_db
from kiss.agents.vscode.server import VSCodeServer
from kiss.agents.vscode.vscode_config import save_config
from kiss.core import config as config_module

_API_KEY_NAMES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
    "MINIMAX_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Redirect config/database state and restore API-key environment."""
    import kiss.agents.sorcar.persistence as pm
    import kiss.agents.vscode.vscode_config as vc

    saved_env = {name: os.environ.get(name) for name in _API_KEY_NAMES}
    saved_default_config = config_module.DEFAULT_CONFIG
    _close_db()
    tmpdir = tempfile.mkdtemp()
    monkeypatch.setattr(pm, "_KISS_DIR", type(pm._KISS_DIR)(tmpdir))
    monkeypatch.setattr(pm, "_DB_PATH", type(pm._DB_PATH)(os.path.join(tmpdir, "sorcar.db")))
    monkeypatch.setattr(vc, "CONFIG_DIR", type(vc.CONFIG_DIR)(tmpdir))
    monkeypatch.setattr(
        vc,
        "CONFIG_PATH",
        type(vc.CONFIG_PATH)(os.path.join(tmpdir, "config.json")),
    )
    try:
        yield
    finally:
        _close_db()
        shutil.rmtree(tmpdir, ignore_errors=True)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        config_module.DEFAULT_CONFIG = saved_default_config


class TestModelPickerLastModelPersistence:
    """Fresh ``getModels`` replies must honor the persisted last model."""

    def test_get_models_reloads_last_model_changed_after_daemon_start(self) -> None:
        """A daemon started with ``gpt-5.5`` must not keep selecting it
        after ``config.json`` is updated to ``claude-opus-4-8``."""
        for name in _API_KEY_NAMES:
            os.environ.pop(name, None)
        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
        config_module.DEFAULT_CONFIG = config_module.Config()

        save_config({"last_model": "gpt-5.5"})
        server = VSCodeServer()
        assert server._default_model == "gpt-5.5"

        save_config({"last_model": "claude-opus-4-8"})

        events: list[dict] = []
        lock = threading.Lock()
        orig_broadcast = server.printer.broadcast

        def capture(event: dict) -> None:
            with lock:
                events.append(dict(event))
            orig_broadcast(event)

        server.printer.broadcast = capture  # type: ignore[method-assign]
        server._get_models()

        with lock:
            model_events = [event for event in events if event.get("type") == "models"]
        assert model_events, "Expected a models broadcast"
        event = model_events[-1]
        available = {model["name"] for model in event["models"]}
        assert "claude-opus-4-8" in available
        assert event["selected"] == "claude-opus-4-8", (
            "getModels should re-read config.json last_model on each refresh; "
            f"got selected={event['selected']!r}"
        )
