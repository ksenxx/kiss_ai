# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 6: a non-numeric ``max_budget`` must not kill the connection.

``vscode_config.apply_config_to_env`` did a bare
``float(cfg["max_budget"])``.  The value comes from two places that
can both hold junk:

* the ``saveConfig`` command payload of any connected client, and
* the user-editable ``~/.kiss/config.json`` (via ``load_config``).

A non-numeric value (``"abc"``, ``None``) raised ``ValueError`` /
``TypeError`` out of ``_cmd_save_config`` → ``_handle_command`` → the
transport receive loop, killing the whole client connection (the same
escape path as the unguarded ``int()`` handler bugs fixed in
iteration 3).  ``SorcarAgent``'s startup caller swallowed the
exception wholesale (``except Exception: pass``), silently skipping
budget application instead of falling back to the default.

Fix: ``apply_config_to_env`` falls back to ``DEFAULTS['max_budget']``
when the stored value is not float-convertible.
"""

from __future__ import annotations

import unittest
from typing import Any

from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer
from kiss.agents.vscode.vscode_config import DEFAULTS, apply_config_to_env
from kiss.core import config as config_module


class TestJunkBudgetDoesNotRaise(unittest.TestCase):
    """Junk ``max_budget`` values must degrade to the default budget."""

    def setUp(self) -> None:
        self._saved_budget = config_module.DEFAULT_CONFIG.max_budget

    def tearDown(self) -> None:
        config_module.DEFAULT_CONFIG.max_budget = self._saved_budget
        _RunningAgentState.running_agent_states.clear()

    def test_apply_config_junk_string_falls_back_to_default(self) -> None:
        apply_config_to_env({"max_budget": "abc"})
        self.assertEqual(
            config_module.DEFAULT_CONFIG.max_budget,
            float(DEFAULTS["max_budget"]),
        )

    def test_apply_config_none_falls_back_to_default(self) -> None:
        apply_config_to_env({"max_budget": None})
        self.assertEqual(
            config_module.DEFAULT_CONFIG.max_budget,
            float(DEFAULTS["max_budget"]),
        )

    def test_apply_config_numeric_string_still_applies(self) -> None:
        apply_config_to_env({"max_budget": "55"})
        self.assertEqual(config_module.DEFAULT_CONFIG.max_budget, 55.0)

    def test_save_config_command_with_junk_budget_does_not_raise(self) -> None:
        """End-to-end: the ``saveConfig`` command path must survive junk."""
        server = VSCodeServer()
        events: list[dict[str, Any]] = []
        server.printer.broadcast = events.append  # type: ignore[method-assign]
        # Pre-fix this raised ValueError out of _handle_command, which
        # in production kills the transport's receive loop (the whole
        # client connection).
        server._handle_command(
            {"type": "saveConfig", "config": {"max_budget": "abc"}},
        )
        self.assertEqual(
            config_module.DEFAULT_CONFIG.max_budget,
            float(DEFAULTS["max_budget"]),
        )


if __name__ == "__main__":
    unittest.main()
