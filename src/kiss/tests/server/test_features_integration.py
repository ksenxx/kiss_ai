# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for: API key deletion and the Chrome bridge toggle
through the daemon's saveConfig handler, and the toggle's label in the
Settings sheet of a live page.

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

import json
import os
import threading
import time
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from kiss.core.brand import PRODUCT_NAME
from kiss.core.user_config import API_KEY_ENV_VARS
from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.core.test_features_integration import (  # noqa: F401
    _restore_default_config,
)
from kiss.tests.webapp.test_remote_tab_settings_button import _start_live_server


class TestApiKeySetupAndDeletion:
    """Delete a key by saving it empty through the server's saveConfig."""

    @pytest.fixture(autouse=True)
    def _isolate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        import kiss.core.user_config as _vc

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

        The AgentServer saveConfig handler forwards empty values to
        ``save_api_key``, which treats them as a delete: the ``export``
        line is removed from the canonical key store and the variable is
        dropped from ``os.environ`` — so the key stays gone across the
        next daemon start too.
        """
        from kiss.core.user_config import api_keys_env_path, load_api_keys
        from kiss.server.server import AgentServer

        server = AgentServer()

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


class TestChromeBridgeToggle:
    """Ticking "Drive my Chrome" in Settings starts the bridge at once."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import kiss.core.user_config as _vc

        monkeypatch.setitem(vars(_vc), "CONFIG_DIR", tmp_path / ".kiss")
        monkeypatch.setitem(vars(_vc), "CONFIG_PATH", tmp_path / ".kiss" / "config.json")

    def test_save_config_starts_the_bridge_when_enabled(self) -> None:
        from kiss.agents.sorcar.chrome_bridge import CONFIG_FILE_NAME, ChromeBridge, extension_dir
        from kiss.core.user_config import load_config
        from kiss.server.server import AgentServer

        ChromeBridge.shutdown()
        server = AgentServer()
        try:
            # Off (the default): saving leaves the bridge alone.
            server._handle_command({"type": "saveConfig", "config": {"max_budget": 100}})
            assert load_config()["chrome_bridge"] is False
            assert ChromeBridge._instance is None

            server._handle_command({"type": "saveConfig", "config": {"chrome_bridge": True}})
            assert load_config()["chrome_bridge"] is True
            bridge = ChromeBridge._instance
            assert bridge is not None
            # The daemon owns the extension: its port and token are published
            # where the unpacked extension reads them.
            published = json.loads((extension_dir() / CONFIG_FILE_NAME).read_text())
            assert published == {"port": bridge.port, "token": bridge.token}
            # Saving again keeps the running bridge.
            server._handle_command({"type": "saveConfig", "config": {"chrome_bridge": True}})
            assert ChromeBridge._instance is bridge
        finally:
            ChromeBridge.shutdown()
        assert not (extension_dir() / CONFIG_FILE_NAME).exists()

    @pytest.mark.timeout(180)
    def test_the_label_says_how_to_install_the_extension_and_its_steps_work(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The label's own first step, taken in a live page, yields the folder and name it gives."""
        from kiss.agents.sorcar.chrome_bridge import CONFIG_FILE_NAME, ChromeBridge, extension_dir

        # Not the product's usual home, and no extension installed in it yet.
        home = tmp_path / "home"
        home.mkdir()
        # Opted out of the Tips window, which opens over a new home's first page.
        (home / "TIPS_DISABLED").write_text("opt-out\n")
        monkeypatch.setenv("KISS_HOME", str(home))
        ChromeBridge.shutdown()
        ready, done = threading.Event(), threading.Event()
        state: dict[str, object] = {}
        server = threading.Thread(
            target=_start_live_server, args=(tmp_path, ready, done, state), daemon=True
        )
        server.start()
        try:
            assert ready.wait(30) and "error" not in state, state
            with sync_playwright() as p:
                browser = p.chromium.launch(args=["--ignore-certificate-errors"])
                try:
                    page = browser.new_page(ignore_https_errors=True)
                    url = f"https://127.0.0.1:{state['port']}/"
                    goto_retrying_network_change(page, url, wait_until="domcontentloaded")
                    page.wait_for_selector("#more-btn", state="attached")
                    page.wait_for_timeout(200)
                    page.click("#more-btn")
                    page.click("#settings-btn")
                    page.wait_for_selector("#settings-panel.open", state="attached")
                    label = page.locator("label:has(#cfg-chrome-bridge)")
                    label.scroll_into_view_if_needed()
                    # The folder is this daemon's own, wherever its home is.
                    page.wait_for_function(
                        "folder => document.getElementById('cfg-chrome-extension-dir')"
                        ".textContent === folder",
                        arg=str(extension_dir()),
                    )
                    assert " ".join(label.inner_text().split()) == (
                        "Drive my Chrome: tick this box and close Settings. Then, in Chrome on "
                        f"the computer that runs {PRODUCT_NAME}, open chrome://extensions, turn "
                        'on Developer mode, click "Load unpacked" and select the folder '
                        f"{home / 'chrome_extension'}. "
                        f'It is listed there as "{PRODUCT_NAME} Bridge". Leave Developer mode '
                        "on: without it Chrome switches the extension off."
                    )
                    assert label.get_attribute("title") == (
                        f'Ticked: the bridge that the "{PRODUCT_NAME} Bridge" extension connects '
                        "to runs from daemon start. Clear: an agent's first use of your Chrome "
                        "starts it."
                    )
                    assert not extension_dir().exists()
                    page.check("#cfg-chrome-bridge")  # "tick this box"
                    page.click("#settings-panel-close")  # "and close Settings"
                    deadline = time.monotonic() + 30
                    while not (extension_dir() / CONFIG_FILE_NAME).exists():
                        assert time.monotonic() < deadline, "the save started no bridge"
                        time.sleep(0.05)
                finally:
                    browser.close()
            # The folder the label named holds the extension, under the name the label gave.
            manifest = json.loads((home / "chrome_extension" / "manifest.json").read_text())
            assert manifest["name"] == f"{PRODUCT_NAME} Bridge"
            worker = manifest["background"]["service_worker"]
            assert (home / "chrome_extension" / worker).is_file()
        finally:
            done.set()
            server.join(timeout=30)
            ChromeBridge.shutdown()
