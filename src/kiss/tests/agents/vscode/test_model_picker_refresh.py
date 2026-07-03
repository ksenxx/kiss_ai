# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: model picker recovers from a stale ``"No model"``.

On a fresh installation the ``VSCodeServer`` is constructed before any
API key is configured, so ``self._default_model`` is initialised to the
``"No model"`` sentinel.  When the user subsequently provides an
``ANTHROPIC_API_KEY`` (either by typing it into the settings panel or by
having it exported in the environment), ``DEFAULT_CONFIG`` is refreshed
but the cached ``self._default_model`` stays ``"No model"`` — so the
model picker keeps showing ``"No model"`` even though Claude models are
now available.

These tests drive ``_get_models`` (the method that builds the ``models``
broadcast consumed by the frontend picker) through that exact sequence
and assert that the broadcast ends up selecting a real, available model.
"""

from __future__ import annotations

import os
import threading
from typing import Any
from unittest import TestCase

from kiss.core import config as config_module


def _make_server() -> Any:
    os.environ.setdefault("KISS_WORKDIR", "/tmp")
    from kiss.agents.vscode.server import VSCodeServer

    return VSCodeServer()


def _clear_all_keys() -> dict[str, str]:
    """Clear every API key from env + DEFAULT_CONFIG; return saved values."""
    names = (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "TOGETHER_API_KEY",
        "OPENROUTER_API_KEY",
        "ZAI_API_KEY",
        "MOONSHOT_API_KEY",
    )
    saved = {n: os.environ.get(n, "") for n in names}
    for n in names:
        os.environ.pop(n, None)
    config_module.DEFAULT_CONFIG = config_module.Config()
    return saved


def _restore_keys(saved: dict[str, str]) -> None:
    for n, v in saved.items():
        if v:
            os.environ[n] = v
        else:
            os.environ.pop(n, None)
    config_module.DEFAULT_CONFIG = config_module.Config()


class TestModelPickerRefresh(TestCase):
    """The picker must not stay stuck on ``"No model"`` after a key
    becomes available."""

    def test_picker_recovers_after_key_provided(self) -> None:
        """Server starts with no keys (``_default_model == "No model"``).
        After ANTHROPIC_API_KEY is set and config refreshed, ``_get_models``
        must broadcast a real, available model as ``selected``."""
        saved = _clear_all_keys()
        saved_path = os.environ.get("PATH", "")
        from kiss.core.models import codex_model as codex_module

        saved_codex_paths = codex_module._UI_CANDIDATE_PATHS
        # A fresh install must also have no persisted ``last_model``
        # preference and no ``KISS_MODEL`` override: the server
        # constructor consults ``_load_last_model()`` (which reads
        # ``vscode_config.CONFIG_PATH``) and ``$KISS_MODEL`` *before*
        # ``get_default_model()``, so a value leaked into the shared
        # per-process config by earlier tests would defeat the
        # ``"No model"`` precondition.  Point the config at an empty
        # temp dir for the duration of the test.
        import shutil
        import tempfile
        from pathlib import Path

        import kiss.agents.vscode.vscode_config as vc

        saved_kiss_model = os.environ.pop("KISS_MODEL", None)
        saved_config_dir, saved_config_path = vc.CONFIG_DIR, vc.CONFIG_PATH
        tmpdir = tempfile.mkdtemp()
        try:
            vc.CONFIG_DIR = Path(tmpdir)
            vc.CONFIG_PATH = Path(tmpdir) / "config.json"
            os.environ["PATH"] = ""
            codex_module._UI_CANDIDATE_PATHS = ()

            server = _make_server()
            # Fresh install: no keys at construction time.
            assert server._default_model == "No model", (
                f"precondition: expected 'No model', got {server._default_model!r}"
            )

            events: list[dict[str, Any]] = []
            lock = threading.Lock()
            orig_broadcast = server.printer.broadcast

            def capture(e: dict[str, Any]) -> None:
                with lock:
                    events.append(dict(e))
                orig_broadcast(e)

            server.printer.broadcast = capture  # type: ignore[assignment]

            # User provides the key; config is refreshed (mirrors
            # save_api_key_to_shell -> _refresh_config).
            os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
            config_module.DEFAULT_CONFIG = config_module.Config()

            server._get_models()

            with lock:
                model_events = [e for e in events if e.get("type") == "models"]
            assert model_events, "Expected a 'models' broadcast"
            ev = model_events[-1]
            names = {m["name"] for m in ev["models"]}
            assert names, "Expected at least one available model after key set"
            assert ev["selected"] != "No model", (
                f"Picker still shows 'No model' after key provided; "
                f"available={sorted(names)[:5]}"
            )
            assert ev["selected"] in names, (
                f"selected {ev['selected']!r} not in available models"
            )
            assert ev["selected"].startswith("claude-"), (
                f"expected a Claude model, got {ev['selected']!r}"
            )
        finally:
            os.environ["PATH"] = saved_path
            codex_module._UI_CANDIDATE_PATHS = saved_codex_paths
            vc.CONFIG_DIR, vc.CONFIG_PATH = saved_config_dir, saved_config_path
            if saved_kiss_model is not None:
                os.environ["KISS_MODEL"] = saved_kiss_model
            shutil.rmtree(tmpdir, ignore_errors=True)
            _restore_keys(saved)

    def test_picker_keeps_valid_user_selection(self) -> None:
        """A model the user explicitly selected must not be overridden by
        ``_get_models`` as long as it is still available."""
        saved = _clear_all_keys()
        try:
            os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
            config_module.DEFAULT_CONFIG = config_module.Config()

            server = _make_server()
            events: list[dict[str, Any]] = []
            lock = threading.Lock()
            orig_broadcast = server.printer.broadcast

            def capture(e: dict[str, Any]) -> None:
                with lock:
                    events.append(dict(e))
                orig_broadcast(e)

            server.printer.broadcast = capture  # type: ignore[assignment]

            server._default_model = "claude-sonnet-4-5"
            server._get_models()

            with lock:
                model_events = [e for e in events if e.get("type") == "models"]
            assert model_events
            ev = model_events[-1]
            assert ev["selected"] == "claude-sonnet-4-5", (
                f"valid user selection overridden: {ev['selected']!r}"
            )
        finally:
            _restore_keys(saved)
