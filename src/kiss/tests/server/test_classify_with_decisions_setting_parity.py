# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Parity tests for the settings panel's "Classify with Jev" checkbox.

The checkbox (``#cfg-classify-with-decisions`` in ``media/chat.html``)
is bound by ``media/main.js`` to the ``classify_with_decisions`` config
key, which :func:`kiss.agents.sorcar.task_classifier.decisions_classification_enabled`
reads to decide whether the pre-run classifier asks OpenRouter's
``~typesafe/jev-latest`` decisions model first or goes straight to the
LLM classifier.  These tests pin the whole loop the checkbox rides on,
end to end and without test doubles:

* the exact ``saveConfig`` payload ``main.js`` posts when the box is
  unticked and the panel closes (``{"classify_with_decisions": false}``,
  a merge-style partial payload) travels over a real client connection
  to a real :class:`~kiss.server.web_server.RemoteAccessServer`, through
  the shared command dispatch into the isolated ``config.json``, and
  flips ``decisions_classification_enabled()`` — the daemon-side effect
  the user wants without editing ``config.json`` by hand;
* ticking it back re-enables the decisions route;
* the ``configData`` replies (to ``getConfig``, which initialises the
  checkbox, and to ``saveConfig``, which repaints it) carry the key,
  both at the default and after a save, and reach the requesting
  connection;
* a partial save touches only that key (``classify_tasks`` and the rest
  stay as stored);
* the JSDOM test ``agents/vscode/test/configToggleInit.test.js`` — which
  drives the real ``chat.html`` + ``main.js`` through the checkbox's
  init / poll / save paths — passes under ``node``.

The browser webapp speaks the same newline-delimited JSON over WSS that
the VS Code extension speaks over the UDS used here; both go through
``RemoteAccessServer._dispatch_client_command`` verbatim.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar.task_classifier import (
    clear_classification_cache,
    decisions_classification_enabled,
)
from kiss.core import config as config_module
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import requires_unix_sockets
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome

_KISS_ROOT = Path(__file__).resolve().parents[2]
_VSCODE_DIR = _KISS_ROOT / "agents" / "vscode"
_TOGGLE_TEST_JS = _VSCODE_DIR / "test" / "configToggleInit.test.js"
_JSDOM_PKG = _VSCODE_DIR / "node_modules" / "jsdom" / "package.json"

# What main.js's collectConfigForm posts for the unticked / ticked box.
_UNTICKED_SAVE = {"type": "saveConfig", "config": {"classify_with_decisions": False}}
_TICKED_SAVE = {"type": "saveConfig", "config": {"classify_with_decisions": True}}


class TestClassifyWithDecisionsSettingParity(IsolatedAsyncioTestCase):
    """The checkbox's config key reaches the classifier and back."""

    async def asyncSetUp(self) -> None:
        self.isolated = IsolatedKissHome("kiss-classify-with-decisions-")
        self._saved_key = config_module.DEFAULT_CONFIG.OPENROUTER_API_KEY
        self._saved_budget = config_module.DEFAULT_CONFIG.max_budget
        # The decisions route also needs a key; a placeholder is enough
        # because nothing here makes a request.
        config_module.DEFAULT_CONFIG.OPENROUTER_API_KEY = "test-key"
        clear_classification_cache()

        certfile = self.isolated.tmpdir / "cert.pem"
        keyfile = self.isolated.tmpdir / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.uds_path = self.isolated.tmpdir / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=self.isolated.tmpdir / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        config_module.DEFAULT_CONFIG.OPENROUTER_API_KEY = self._saved_key
        config_module.DEFAULT_CONFIG.max_budget = self._saved_budget
        clear_classification_cache()
        self.isolated.cleanup()

    async def _request(self, cmd: dict[str, Any]) -> dict[str, Any]:
        """Send *cmd* on a fresh client connection; return its ``configData`` reply."""
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path), limit=16 * 1024 * 1024,
        )
        try:
            writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
            await writer.drain()
            for _ in range(50):
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                self.assertTrue(line, "UDS closed before a configData reply")
                event = json.loads(line.decode("utf-8"))
                self.assertIsInstance(event, dict)
                if event.get("type") == "configData":
                    return dict(event)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # pragma: no cover — best-effort close
                pass
        raise AssertionError("no configData reply within 50 events")

    def _stored(self) -> dict[str, Any]:
        stored = json.loads((self.isolated.kiss_home / "config.json").read_text())
        self.assertIsInstance(stored, dict)
        return dict(stored)

    @requires_unix_sockets
    async def test_unticking_the_box_pins_the_llm_classifier(self) -> None:
        """The panel's partial saveConfig turns the decisions route off."""
        self.assertTrue(decisions_classification_enabled())

        reply = await self._request(dict(_UNTICKED_SAVE))

        self.assertIs(self._stored()["classify_with_decisions"], False)
        self.assertFalse(decisions_classification_enabled())
        # The reply echoes the saved config so the checkbox repaints
        # from the persisted state, not from its own click.
        self.assertIs(reply["config"]["classify_with_decisions"], False)

    @requires_unix_sockets
    async def test_ticking_the_box_back_restores_the_decisions_route(self) -> None:
        """A ticked box saved after an unticked one re-enables Jev."""
        await self._request(dict(_UNTICKED_SAVE))
        self.assertFalse(decisions_classification_enabled())

        reply = await self._request(dict(_TICKED_SAVE))

        self.assertIs(self._stored()["classify_with_decisions"], True)
        self.assertTrue(decisions_classification_enabled())
        self.assertIs(reply["config"]["classify_with_decisions"], True)

    @requires_unix_sockets
    async def test_get_config_carries_the_key_for_the_checkbox(self) -> None:
        """``getConfig`` reports the key at its default and after a save."""
        first = await self._request({"type": "getConfig"})
        # Nothing stored yet: load_config seeds the DEFAULTS value, so
        # the box initialises checked without a hardcoded fallback.
        self.assertIs(first["config"]["classify_with_decisions"], True)

        await self._request(dict(_UNTICKED_SAVE))
        second = await self._request({"type": "getConfig"})
        self.assertIs(second["config"]["classify_with_decisions"], False)

    @requires_unix_sockets
    async def test_partial_save_leaves_the_other_settings_alone(self) -> None:
        """Only the edited key changes; the merge keeps the rest."""
        self.isolated.write_config(
            classify_tasks=False, max_budget=7, memory_dir="/tmp/mem", is_worktree=False
        )

        await self._request(dict(_UNTICKED_SAVE))

        stored = self._stored()
        self.assertIs(stored["classify_with_decisions"], False)
        self.assertIs(stored["classify_tasks"], False)
        self.assertIs(stored["is_worktree"], False)
        self.assertEqual(stored["max_budget"], 7)
        self.assertEqual(stored["memory_dir"], "/tmp/mem")

    @requires_unix_sockets
    async def test_junk_value_is_coerced_like_the_other_toggles(self) -> None:
        """A non-boolean payload value is coerced, never stored raw."""
        await self._request({"type": "saveConfig", "config": {"classify_with_decisions": 0}})
        self.assertIs(self._stored()["classify_with_decisions"], False)
        self.assertFalse(decisions_classification_enabled())

    async def test_jsdom_toggle_test_passes(self) -> None:
        """The checkbox's init / poll / save behaviour holds in the real webview."""
        if shutil.which("node") is None:
            self.skipTest("node is not available on PATH")
        if not _JSDOM_PKG.is_file():
            self.skipTest(
                f"jsdom is not installed under {_VSCODE_DIR / 'node_modules'}"
                " — run `npm ci` there"
            )
        proc = await asyncio.to_thread(
            subprocess.run,
            ["node", str(_TOGGLE_TEST_JS)],
            cwd=str(_VSCODE_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"configToggleInit.test.js failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )
        self.assertIn("Classify-with-Jev edit alone is flushed", proc.stdout)
        self.assertIn("all tests passed", proc.stdout)
