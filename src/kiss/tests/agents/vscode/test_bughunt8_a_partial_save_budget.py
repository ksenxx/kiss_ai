# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8 (group A): partial ``saveConfig`` resets the live max_budget.

``_cmd_save_config`` applied the *incoming payload* to the process
environment via ``apply_config_to_env(cfg)``.  When the payload does
not carry ``max_budget`` (any client that saves a single setting —
e.g. the password-only settings flush), ``apply_config_to_env`` falls
back to ``DEFAULTS['max_budget']`` (100) even though ``config.json``
still holds the user's configured budget: ``save_config`` correctly
preserves the on-disk value (it merges only the keys present in the
payload), so disk and process silently disagree, and every
``cost > max_budget`` check in the daemon now enforces the *default*
budget instead of the user's.  The handler must apply the merged
on-disk config, not the raw payload.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode import vscode_config
from kiss.agents.vscode.server import VSCodeServer
from kiss.core import config as core_config


class TestPartialSaveConfigBudget(unittest.TestCase):
    """A payload without ``max_budget`` must not reset the live budget."""

    def setUp(self) -> None:
        self._cfg_path = vscode_config.CONFIG_PATH
        self._saved = (
            self._cfg_path.read_bytes() if self._cfg_path.exists() else None
        )
        self._saved_budget = core_config.DEFAULT_CONFIG.max_budget

    def tearDown(self) -> None:
        if self._saved is None:
            self._cfg_path.unlink(missing_ok=True)
        else:
            self._cfg_path.write_bytes(self._saved)
        core_config.DEFAULT_CONFIG.max_budget = self._saved_budget
        _RunningAgentState.running_agent_states.clear()

    def _write_config(self, data: dict[str, Any]) -> None:
        self._cfg_path.parent.mkdir(parents=True, exist_ok=True)
        self._cfg_path.write_text(json.dumps(data))

    def test_partial_payload_keeps_configured_budget(self) -> None:
        """saveConfig without max_budget keeps the on-disk budget live."""
        self._write_config({"max_budget": 7})
        server = VSCodeServer()
        events: list[dict[str, Any]] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]

        server._handle_command(
            {"type": "saveConfig", "config": {"use_web_browser": True}},
        )

        # Disk still holds the user's budget (save_config merges) ...
        on_disk = json.loads(self._cfg_path.read_text())
        self.assertEqual(on_disk.get("max_budget"), 7)
        # ... and the live process budget must agree with it.
        self.assertEqual(core_config.DEFAULT_CONFIG.max_budget, 7.0)

    def test_payload_with_budget_still_applies(self) -> None:
        """A payload that DOES carry max_budget updates the live value."""
        self._write_config({"max_budget": 7})
        server = VSCodeServer()
        events: list[dict[str, Any]] = []
        server.printer.broadcast = events.append  # type: ignore[assignment]

        server._handle_command(
            {"type": "saveConfig", "config": {"max_budget": 12}},
        )

        self.assertEqual(core_config.DEFAULT_CONFIG.max_budget, 12.0)


if __name__ == "__main__":
    unittest.main()
