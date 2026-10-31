# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a settings or work-dir change whose write fails still answers the page.

* ``setWorkDir`` switched the live directory and then persisted it; a
  failed write (``config.json`` unreadable, its lock held too long by
  another process) escaped after the switch: tasks ran in the new
  directory, no client was told, nothing was persisted, and the sender
  got neither its notice nor its ``reply`` ``configData``.
* ``saveConfig`` whose ``save_config`` write failed answered nothing:
  no ``error`` and no ``configData``.
* ``saveConfig`` started the Chrome bridge before its replies, and a
  start failure other than ``ChromeBridgeError`` (its serving thread
  refused under thread exhaustion) cost the save its ``models`` and
  ``configData``.

Real daemon over its local WSS endpoint, a real client connection.  The
failures are real: ``config.json`` made unreadable (mode 000), and
``RLIMIT_NPROC`` lowered to 1 for exactly the bridge thread's start by
a :class:`Pause` hook (instrumentation, nothing replaced).
"""

from __future__ import annotations

import contextlib
import sys
import threading
import unittest
from pathlib import Path
from typing import Any

from websockets.sync.server import Server

from kiss.agents.sorcar import chrome_bridge
from kiss.core import user_config
from kiss.tests.conftest import is_root, nproc_limit_lowered_to_one, thread_start_can_be_starved
from kiss.tests.server.pause import Pause
from kiss.tests.server.two_client_harness import Client, TwoClientHarness


@unittest.skipIf(
    sys.platform == "win32" or is_root(),
    "needs POSIX permission bits that deny the owner (not root, not Windows)",
)
class SettingsWriteFailureTest(TwoClientHarness):
    """Each failed write is answered; nothing half-applied is left behind."""

    def tearDown(self) -> None:
        # Readable again before the daemon stops (it may read or write it).
        if user_config.CONFIG_PATH.exists():
            user_config.CONFIG_PATH.chmod(0o600)
        super().tearDown()

    def _client(self) -> Client:
        client = self.open_client("a")
        client.ready()
        self.wait_event(client, "models")
        return client

    def _make_config_unreadable(self) -> None:
        """Leave ``config.json`` in place but unreadable (until :meth:`tearDown`)."""
        path = user_config.CONFIG_PATH
        if not path.exists():
            user_config.save_config({})
        path.chmod(0)

    def _wait_reply(self, client: Client, before: int) -> dict:
        """Wait for a ``configData`` past the first *before* ones; return it."""
        self.wait_for(
            lambda: len(client.of_type("configData")) > before, timeout=20, what="configData",
        )
        return client.of_type("configData")[-1]

    def test_a_work_dir_the_config_cannot_record_is_not_adopted(self) -> None:
        """Refused with a notice and answered; adopted once the file is readable again."""
        agent_server = self.server._agent_server
        a = self._client()
        old = agent_server.work_dir
        new_dir = Path(self.tmpdir) / "elsewhere"
        new_dir.mkdir()
        self._make_config_unreadable()

        before = len(a.of_type("configData"))
        a.send({"type": "setWorkDir", "workDir": str(new_dir), "reply": True})
        reply = self._wait_reply(a, before)
        self.assertEqual(reply["config"]["work_dir"], old)
        self.assertEqual(agent_server.work_dir, old)
        self.assertIn(f"keeping {old}", self.wait_event(a, "notice")["text"])
        self.assertEqual(a.of_type("workDirChanged"), [])

        user_config.CONFIG_PATH.chmod(0o600)
        a.send({"type": "setWorkDir", "workDir": str(new_dir)})
        self.wait_event(a, "workDirChanged", workDir=str(new_dir))
        self.assertEqual(agent_server.work_dir, str(new_dir))
        self.assertEqual(user_config.load_config()["work_dir"], str(new_dir))

    def test_a_settings_save_the_config_cannot_take_is_answered(self) -> None:
        """The sender gets an ``error`` and the settings as they stand."""
        a = self._client()
        self._make_config_unreadable()
        before = len(a.of_type("configData"))
        a.send({"type": "saveConfig", "config": {"max_budget": 7.0}, "apiKeys": {}})
        reply = self._wait_reply(a, before)
        self.assertNotEqual(reply["config"]["max_budget"], 7.0)
        self.assertIn("Could not save the settings", self.wait_event(a, "error")["text"])

    @unittest.skipUnless(
        thread_start_can_be_starved(),
        "needs a host where RLIMIT_NPROC makes Thread.start fail (non-root Linux)",
    )
    def test_a_chrome_bridge_that_cannot_start_leaves_the_save_its_reply(self) -> None:
        """The bridge's serving thread cannot start; the save is still answered."""
        a = self._client()
        bridge_config = chrome_bridge.extension_dir() / chrome_bridge.CONFIG_FILE_NAME
        if not bridge_config.exists():
            self.addCleanup(bridge_config.unlink, missing_ok=True)
        starved = contextlib.ExitStack()
        self.addCleanup(starved.close)
        lowered = threading.Event()

        def starve_the_bridge_thread(frame_locals: dict[str, Any]) -> bool:
            serving = getattr(frame_locals.get("target"), "__self__", None)
            if isinstance(serving, Server):
                starved.enter_context(nproc_limit_lowered_to_one())
                lowered.set()
            return False

        before = len(a.of_type("configData"))
        with Pause(threading.Thread.__init__.__code__, starve_the_bridge_thread):
            a.send({"type": "saveConfig", "config": {"chrome_bridge": True}, "apiKeys": {}})
            self.wait_for(
                lambda: lowered.is_set() and not chrome_bridge.ChromeBridge._instance_lock.locked(),
                timeout=20, what="the bridge's start to fail",
            )
        starved.close()
        reply = self._wait_reply(a, before)
        self.assertIs(reply["config"]["chrome_bridge"], True)
        self.assertIsNone(chrome_bridge.ChromeBridge._instance)
