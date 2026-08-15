# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 3: empty-tabId commands must not mint phantom state (BUG-E).

``{"type": "newChat"}`` without a ``tabId`` used to create a permanent
registry entry keyed ``""`` that could never be disposed.  In the
task-keyed architecture the equivalent hazard is a phantom ``""`` key
in the per-tab stickiness dicts (``_tab_models``) or a phantom entry in
``kiss.server.agent_state.agent_states``.  ``selectModel`` without a
``tabId`` must only update the daemon-wide default model.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from typing import Any

import kiss.agents.sorcar.persistence as _pm
import kiss.core.vscode_config as _vc
from kiss.server import agent_state
from kiss.server.server import VSCodeServer


class TestEmptyTabIdPhantom(unittest.TestCase):
    """Empty tabId must not create undisposable per-tab or task state."""

    def setUp(self) -> None:
        _pm._close_db()
        self._tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt3-")
        self._saved_paths = (
            _pm._KISS_DIR, _pm._DB_PATH, _vc.CONFIG_DIR, _vc.CONFIG_PATH,
        )
        _pm._KISS_DIR = type(_pm._KISS_DIR)(self._tmpdir)
        _pm._DB_PATH = type(_pm._DB_PATH)(self._tmpdir) / "sorcar.db"
        _vc.CONFIG_DIR = type(_vc.CONFIG_DIR)(self._tmpdir)
        _vc.CONFIG_PATH = type(_vc.CONFIG_PATH)(self._tmpdir) / "config.json"

        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            self.events.append(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        agent_state.agent_states.clear()
        _pm._close_db()
        (
            _pm._KISS_DIR, _pm._DB_PATH, _vc.CONFIG_DIR, _vc.CONFIG_PATH,
        ) = self._saved_paths
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_new_chat_empty_tab_id_creates_no_phantom(self) -> None:
        self.server._handle_command({"type": "newChat"})
        assert "" not in self.server._tab_models, (
            'BUG: newChat without tabId minted a phantom "" entry in '
            "_tab_models that _cmd_close_tab can never dispose"
        )
        assert "" not in agent_state.agent_states, (
            'BUG: newChat without tabId minted a phantom "" AgentState'
        )

    def test_select_model_empty_tab_id_creates_no_phantom(self) -> None:
        self.server._handle_command({"type": "selectModel", "model": "m-x"})
        assert "" not in self.server._tab_models, (
            'BUG: selectModel without tabId minted a phantom "" entry in '
            "_tab_models that _cmd_close_tab can never dispose"
        )
        assert "" not in agent_state.agent_states, (
            'BUG: selectModel without tabId minted a phantom "" AgentState'
        )
        assert self.server._default_model == "m-x"


if __name__ == "__main__":
    unittest.main()
