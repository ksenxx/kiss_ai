# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 3: empty-tabId commands must not mint phantom tabs (BUG-E).

``{"type": "newChat"}`` without a ``tabId`` created a permanent
registry entry keyed ``""`` (plus a ``WorktreeSorcarAgent`` via
``_get_tab``).  ``_cmd_close_tab`` guards against empty tab ids, so
the phantom could NEVER be disposed — inconsistent with
``_stop_task`` / ``_replay_session``, which both no-op on an empty
tab id.  ``selectModel`` had the same hole via ``_get_tab``.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from typing import Any

import kiss.agents.sorcar.persistence as _pm
import kiss.agents.vscode.vscode_config as _vc
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer


class TestEmptyTabIdPhantom(unittest.TestCase):
    """Empty tabId must not create an undisposable registry entry."""

    def setUp(self) -> None:
        # ``selectModel`` persists ``last_model`` into config.json and
        # increments the bogus test model's ("m-x") count in the SQLite
        # ``model_usage`` table via ``_record_model_usage``.  Redirect
        # BOTH stores into a private temp dir so neither the developer's
        # real ``~/.kiss`` data nor later tests in this process (e.g.
        # ``_load_model_usage() == {}`` assertions) get polluted.
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
        _RunningAgentState.running_agent_states.clear()
        _pm._close_db()
        (
            _pm._KISS_DIR, _pm._DB_PATH, _vc.CONFIG_DIR, _vc.CONFIG_PATH,
        ) = self._saved_paths
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_new_chat_empty_tab_id_creates_no_phantom(self) -> None:
        self.server._handle_command({"type": "newChat"})
        assert "" not in _RunningAgentState.running_agent_states, (
            "BUG: newChat without tabId minted a phantom registry entry "
            'keyed "" that _cmd_close_tab can never dispose'
        )

    def test_select_model_empty_tab_id_creates_no_phantom(self) -> None:
        self.server._handle_command({"type": "selectModel", "model": "m-x"})
        assert "" not in _RunningAgentState.running_agent_states, (
            "BUG: selectModel without tabId minted a phantom registry "
            'entry keyed "" that _cmd_close_tab can never dispose'
        )
        # The daemon-wide default is still updated.
        assert self.server._default_model == "m-x"


if __name__ == "__main__":
    unittest.main()
