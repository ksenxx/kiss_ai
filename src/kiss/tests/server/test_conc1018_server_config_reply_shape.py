# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: ``getConfig`` and ``saveConfig`` answer with the same ``configData``
shape.

The settings sheet repaints from either reply (``populateConfigForm``
writes every API-key field from ``apiKeys`` and the status bar takes
``machine``), so the save echo must carry the same fields as the
initial load; it used to carry neither ``apiKeys`` nor ``machine``
(findings H2-F3).  Real daemon over its local WSS endpoint, one client.
"""

from __future__ import annotations

from typing import Any

from kiss.tests.server.two_client_harness import Client, TwoClientHarness


class ConfigReplyShapeTest(TwoClientHarness):
    """Both config replies carry ``config``, ``apiKeys`` and ``machine``."""

    def _reply(self, client: Client, cmd: dict[str, Any]) -> dict[str, Any]:
        before = len(client.of_type("configData"))
        client.send(cmd)
        replies: list[dict[str, Any]] = self.wait_for(
            lambda: client.of_type("configData")[before:],
            what=f"configData for {cmd['type']}",
        )
        return replies[-1]

    def test_save_reply_matches_get_reply(self) -> None:
        client = self.open_client("a")
        client.ready()
        # ``ready`` fans out its own ``getConfig``: take that reply before
        # sending ours, or a slow daemon's ready reply counts as ours and
        # its get reply (the stock budget) as the save echo.
        self.wait_for(lambda: client.of_type("configData"), what="configData from ready")
        got = self._reply(client, {"type": "getConfig"})
        saved = self._reply(
            client,
            {
                "type": "saveConfig",
                "config": {"max_budget": 3.5},
                "apiKeys": {},
            },
        )
        self.assertEqual(set(saved) - {"ts"}, set(got) - {"ts"})
        self.assertIn("apiKeys", saved)
        self.assertEqual(saved["machine"], got["machine"])
        self.assertEqual(saved["config"]["work_dir"], got["config"]["work_dir"])
        self.assertEqual(saved["config"]["max_budget"], 3.5)
        self.assertIsInstance(saved["config"]["recent_work_dirs"], list)
