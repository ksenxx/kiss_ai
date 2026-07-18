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
import time
from collections.abc import Generator
from pathlib import Path

import pytest

from kiss.agents.sorcar.persistence import _close_db
from kiss.core import config as config_module
from kiss.core.vscode_config import load_config, save_config
from kiss.server.server import VSCodeServer

_API_KEY_NAMES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
    "ZAI_API_KEY",
    "MOONSHOT_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Redirect config/database state and restore API-key environment."""
    import kiss.agents.sorcar.persistence as pm
    import kiss.core.vscode_config as vc

    saved_env = {name: os.environ.get(name) for name in _API_KEY_NAMES}
    saved_default_config = config_module.DEFAULT_CONFIG
    _close_db()
    tmpdir = tempfile.mkdtemp()
    monkeypatch.setattr(pm, "_KISS_DIR", type(pm._KISS_DIR)(tmpdir))
    monkeypatch.setattr(pm, "_DB_PATH", type(pm._DB_PATH)(os.path.join(tmpdir, "sorcar.db")))
    # ``CONFIG_DIR``/``CONFIG_PATH`` are PEP 562 lazy attributes;
    # ``setattr`` would pin the computed (stale tmp) Path at teardown.
    # ``setitem`` deletes the pin instead, restoring lazy resolution.
    monkeypatch.setitem(vars(vc), "CONFIG_DIR", Path(tmpdir))
    monkeypatch.setitem(
        vars(vc), "CONFIG_PATH", Path(tmpdir) / "config.json",
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

    def test_get_models_selects_custom_model_when_cached_default_is_invalid(
        self,
    ) -> None:
        """When a custom endpoint is the only runnable model, ``getModels``
        must select that custom entry instead of a stale cached default."""
        for name in _API_KEY_NAMES:
            os.environ.pop(name, None)
        saved_path = os.environ.get("PATH", "")
        from kiss.core.models import codex_model as codex_module

        saved_codex_paths = codex_module._UI_CANDIDATE_PATHS
        try:
            os.environ["PATH"] = ""
            codex_module._UI_CANDIDATE_PATHS = ()
            config_module.DEFAULT_CONFIG = config_module.Config()
            save_config({
                "custom_endpoint": "http://localhost:9999/v1",
                "last_model": "stale-unavailable-model",
            })

            server = VSCodeServer()
            assert server._default_model == "stale-unavailable-model"

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
                model_events = [e for e in events if e.get("type") == "models"]
            assert model_events, "Expected a models broadcast"
            event = model_events[-1]
            available = {model["name"] for model in event["models"]}
            assert available == {"custom/v1"}
            assert event["selected"] == "custom/v1", (
                "custom-only model picker should select the custom model; "
                f"got selected={event['selected']!r}"
            )
        finally:
            os.environ["PATH"] = saved_path
            codex_module._UI_CANDIDATE_PATHS = saved_codex_paths

    def test_concurrent_get_models_does_not_revert_just_picked_model(self) -> None:
        """A concurrent ``_get_models`` must not clobber a just-picked
        in-memory selection with the stale on-disk persisted value
        while ``_cmd_select_model``'s ``_save_last_model`` is still in
        flight.

        Reproduces the race between ``_cmd_select_model`` (which updates
        ``self._default_model`` under the lock, then releases the lock
        before persisting to disk via ``_record_model_usage`` →
        ``_save_last_model``) and ``_get_models`` (which reads
        ``_load_last_model()`` outside the lock and assigns
        ``self._default_model = persisted``).  Without the fix, the
        racing ``_get_models`` reads the OLD on-disk value and reverts
        the user's just-picked model.
        """
        import kiss.agents.sorcar.persistence as pm

        for name in _API_KEY_NAMES:
            os.environ.pop(name, None)
        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
        config_module.DEFAULT_CONFIG = config_module.Config()

        save_config({"last_model": "gpt-5.5"})
        server = VSCodeServer()
        assert server._default_model == "gpt-5.5"

        events: list[dict] = []
        elock = threading.Lock()
        orig_broadcast = server.printer.broadcast

        def capture(event: dict) -> None:
            with elock:
                events.append(dict(event))
            orig_broadcast(event)

        server.printer.broadcast = capture  # type: ignore[method-assign]

        # Stall ``_save_last_model`` so we can open the race window
        # deterministically.  Thread A calls ``_cmd_select_model`` which
        # updates ``self._default_model`` to ``claude-opus-4-8`` under
        # the lock, releases the lock, then enters ``_record_model_usage``
        # → ``_save_last_model`` which blocks here.  At that moment the
        # main thread (B) calls ``_get_models``.
        save_started = threading.Event()
        proceed_save = threading.Event()
        real_save = pm._save_last_model

        def slow_save(model: str) -> None:
            save_started.set()
            assert proceed_save.wait(timeout=30.0), (
                "test bug: proceed_save not set within timeout"
            )
            real_save(model)

        pm._save_last_model = slow_save  # type: ignore[assignment]
        try:
            sel_thread = threading.Thread(
                target=server._cmd_select_model,
                args=({"tabId": "tabA", "model": "claude-opus-4-8"},),
                daemon=True,
            )
            sel_thread.start()
            assert save_started.wait(timeout=30.0), (
                "test bug: _save_last_model not entered within timeout"
            )

            # In-memory has been updated by _cmd_select_model, but disk
            # still reflects the OLD value because slow_save hasn't
            # called real_save yet.
            assert server._default_model == "claude-opus-4-8"
            assert load_config()["last_model"] == "gpt-5.5"

            # Concurrent getModels must not revert the just-picked
            # model.  Run it in a thread because, with the fix, the
            # select thread holds ``_state_lock`` until the on-disk
            # write completes, so ``_get_models`` will block waiting
            # on the lock and only proceed AFTER the select finishes.
            get_thread = threading.Thread(
                target=server._get_models, daemon=True,
            )
            get_thread.start()
            # Let the racing _get_models start and reach the lock.
            time.sleep(0.05)

            proceed_save.set()
            sel_thread.join(timeout=30.0)
            get_thread.join(timeout=30.0)
            assert not sel_thread.is_alive(), (
                "test bug: select thread did not finish"
            )
            assert not get_thread.is_alive(), (
                "test bug: get thread did not finish"
            )
        finally:
            pm._save_last_model = real_save  # type: ignore[assignment]

        with elock:
            model_events = [e for e in events if e.get("type") == "models"]
        assert model_events, "Expected a models broadcast"
        event = model_events[-1]
        assert event["selected"] == "claude-opus-4-8", (
            "_get_models reverted the just-picked model to the stale "
            "on-disk value; got selected="
            f"{event['selected']!r}"
        )
        # And the daemon's in-memory state must reflect the user's pick.
        assert server._default_model == "claude-opus-4-8", (
            "_get_models clobbered self._default_model with the stale "
            "on-disk value; got "
            f"{server._default_model!r}"
        )

    def test_concurrent_new_chat_does_not_revert_just_picked_model(self) -> None:
        """A concurrent ``_new_chat`` must not clobber a just-picked
        in-memory selection with the stale on-disk persisted value
        while ``_cmd_select_model``'s ``_save_last_model`` is still in
        flight.

        Reproduces the race between ``_cmd_select_model`` (which now
        persists to disk under ``_state_lock``) and ``_new_chat``
        (which reads ``_load_last_model()`` and assigns
        ``self._default_model = persisted`` outside the lock).
        Without the fix, the racing ``_new_chat`` reads the OLD
        on-disk value and reverts the user's just-picked model on the
        daemon's in-memory state AND on the new tab's
        ``selected_model``.
        """
        import kiss.agents.sorcar.persistence as pm

        for name in _API_KEY_NAMES:
            os.environ.pop(name, None)
        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
        config_module.DEFAULT_CONFIG = config_module.Config()

        save_config({"last_model": "gpt-5.5"})
        server = VSCodeServer()

        save_started = threading.Event()
        proceed_save = threading.Event()
        real_save = pm._save_last_model

        def slow_save(model: str) -> None:
            save_started.set()
            assert proceed_save.wait(timeout=30.0), (
                "test bug: proceed_save not set within timeout"
            )
            real_save(model)

        pm._save_last_model = slow_save  # type: ignore[assignment]
        try:
            sel_thread = threading.Thread(
                target=server._cmd_select_model,
                args=({"tabId": "tabA", "model": "claude-opus-4-8"},),
                daemon=True,
            )
            sel_thread.start()
            assert save_started.wait(timeout=30.0), (
                "test bug: _save_last_model not entered within timeout"
            )

            # In-memory has been updated by the select thread; disk
            # still reflects the OLD value because slow_save is
            # blocked.
            assert server._default_model == "claude-opus-4-8"
            assert load_config()["last_model"] == "gpt-5.5"

            new_thread = threading.Thread(
                target=server._new_chat, args=("tabB",), daemon=True,
            )
            new_thread.start()
            # Give _new_chat time to read disk (outside lock) before
            # we unblock the select thread.
            time.sleep(0.05)

            proceed_save.set()
            sel_thread.join(timeout=30.0)
            new_thread.join(timeout=30.0)
            assert not sel_thread.is_alive(), (
                "test bug: select thread did not finish"
            )
            assert not new_thread.is_alive(), (
                "test bug: new_chat thread did not finish"
            )
        finally:
            pm._save_last_model = real_save  # type: ignore[assignment]

        assert server._default_model == "claude-opus-4-8", (
            "_new_chat clobbered self._default_model with the stale "
            "on-disk value; got "
            f"{server._default_model!r}"
        )
        tab_b = server._get_tab("tabB")
        assert tab_b.selected_model == "claude-opus-4-8", (
            "_new_chat set the new tab's selected_model to the stale "
            "on-disk value; got "
            f"{tab_b.selected_model!r}"
        )
