# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 7: junk-typed config values must not poison the daemon.

Two untrusted sources feed the configuration dict:

* the ``saveConfig`` command payload of any connected client, and
* the user-editable ``~/.kiss/config.json`` (via ``load_config``).

Iteration 6 fixed only ``max_budget`` (BUG-6G-3).  Every OTHER key was
still consumed without type validation:

* ``saveConfig`` with a non-string ``config.work_dir`` (e.g. a list)
  assigned the raw value to the daemon-global ``self.work_dir`` — the
  exact corruption class fixed for the *top-level* ``workDir`` field in
  BUG-6E-1 — and persisted it to ``config.json``.
* A non-string ``custom_endpoint`` (e.g. ``123`` from a hand-edited
  config.json) made ``get_custom_model_entry`` raise
  ``AttributeError: 'int' object has no attribute 'rstrip'`` out of
  ``_get_models``.  The per-message transport containment keeps the
  connection alive but the ``models`` reply is silently dropped — the
  model picker stays dead in EVERY window, on EVERY reconnect, until
  the user hand-repairs config.json.  ``build_model_config`` in
  ``task_runner`` raises the same way, failing every task run.
* ``custom_headers`` as a JSON list (a natural hand-edit for a
  "one per line" setting) raised ``AttributeError: 'list' object has
  no attribute 'splitlines'`` the same way.

Fix: ``vscode_config.sanitize_config`` coerces every DEFAULTS-keyed
value to its expected type (falling back to the default); applied in
``load_config``, ``save_config`` and ``_cmd_save_config``.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.server import vscode_config
from kiss.server.server import VSCodeServer
from kiss.server.vscode_config import load_config


class _ConfigIsolationMixin(unittest.TestCase):
    """Snapshot/restore the shared test config.json around each test.

    The test process shares one ``KISS_HOME`` (set by conftest), so a
    test that writes junk into ``config.json`` must restore the prior
    contents or it will order-flake unrelated tests (the iter-6
    ``selectModel`` lesson).
    """

    def setUp(self) -> None:
        self._cfg_path = vscode_config.CONFIG_PATH
        self._saved = (
            self._cfg_path.read_bytes() if self._cfg_path.exists() else None
        )

    def tearDown(self) -> None:
        if self._saved is None:
            self._cfg_path.unlink(missing_ok=True)
        else:
            self._cfg_path.write_bytes(self._saved)
        _RunningAgentState.running_agent_states.clear()

    def _write_config(self, data: dict[str, Any]) -> None:
        self._cfg_path.parent.mkdir(parents=True, exist_ok=True)
        self._cfg_path.write_text(json.dumps(data))

    def _server(self) -> tuple[VSCodeServer, list[dict[str, Any]]]:
        server = VSCodeServer()
        events: list[dict[str, Any]] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]
        return server, events


class TestSaveConfigJunkTypes(_ConfigIsolationMixin):
    """``saveConfig`` payloads with junk-typed values must be neutralised."""

    def test_junk_work_dir_does_not_corrupt_daemon_work_dir(self) -> None:
        server, _ = self._server()
        original = server.work_dir
        server._handle_command(
            {"type": "saveConfig", "config": {"work_dir": [1, 2]}},
        )
        self.assertEqual(server.work_dir, original)
        self.assertIsInstance(load_config()["work_dir"], str)

    def test_junk_endpoint_not_persisted_and_models_still_reply(self) -> None:
        server, events = self._server()
        server._handle_command(
            {"type": "saveConfig", "config": {"custom_endpoint": 123}},
        )
        self.assertEqual(load_config()["custom_endpoint"], "")
        self.assertIn("models", [e.get("type") for e in events])

    def test_junk_remote_password_not_persisted(self) -> None:
        # A truthy non-string password (e.g. 123456) must neither be
        # persisted (it would lock the user out of web auth) nor be
        # treated as a genuine password change (which restarts the
        # kiss-web daemon, killing every in-flight task).
        server, _ = self._server()
        server._handle_command(
            {"type": "saveConfig", "config": {"remote_password": 123456}},
        )
        self.assertEqual(load_config()["remote_password"], "")


class TestHandEditedConfigJunkTypes(_ConfigIsolationMixin):
    """Junk types hand-edited into config.json must not break readers."""

    def test_get_models_survives_non_string_custom_endpoint(self) -> None:
        self._write_config({"custom_endpoint": 123})
        server, events = self._server()
        events.clear()
        # Pre-fix: AttributeError ('int' object has no attribute
        # 'rstrip') escaped _handle_command; in production the models
        # reply was silently dropped forever.
        server._handle_command({"type": "getModels"})
        self.assertIn("models", [e.get("type") for e in events])

    def test_get_models_survives_list_custom_headers(self) -> None:
        self._write_config({
            "custom_endpoint": "http://localhost:9/v1",
            "custom_headers": ["Authorization: Bearer x"],
        })
        server, events = self._server()
        events.clear()
        server._handle_command({"type": "getModels"})
        models_events = [e for e in events if e.get("type") == "models"]
        self.assertEqual(len(models_events), 1)
        names = [m["name"] for m in models_events[0]["models"]]
        self.assertIn("custom/v1", names)

    def test_load_config_sanitizes_defaults_keys_only(self) -> None:
        self._write_config({
            "work_dir": 7,
            "last_model": None,
            "use_web_browser": 1,
            "tunnel_token": ["keep", "me"],
        })
        cfg = load_config()
        self.assertEqual(cfg["work_dir"], "")
        self.assertEqual(cfg["last_model"], "")
        self.assertIs(cfg["use_web_browser"], True)
        # Non-DEFAULTS keys pass through untouched.
        self.assertEqual(cfg["tunnel_token"], ["keep", "me"])


if __name__ == "__main__":
    unittest.main()
