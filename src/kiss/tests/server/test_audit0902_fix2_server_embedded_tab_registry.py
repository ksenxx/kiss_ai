# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the channel launcher's embedded daemon never touches ``tabs.json``.

``TabRegistry`` loads its file once and publishes its complete
in-memory list on every mutation, so two live registries on ONE file
overwrite each other's tabs.  The design assumes a single owner per
``KISS_HOME`` — and the canonical ``kiss-web`` daemon is that owner —
but ``_kiss_web_launcher._ensure_api_server`` builds a second,
private-UDS ``RemoteAccessServer`` in the channel-agent process whose
``VSCodeServer`` used to bind the SAME ``KISS_HOME/tabs.json``.  Every
channel run registers a tab, so the canonical daemon's tabs were
erased by the embedded server's stale snapshot (and vice versa), and
the embedded server's transient tabs leaked into every VS Code window.

The fix gives the embedded server a private registry of its own
(``VSCodeServer.use_private_tab_registry``, called by the launcher).

The test runs the real canonical server and the real launcher
entry point (``_ensure_api_server``) in one process against one KISS
home, has both open tabs concurrently (real ``openTab`` command
handlers, released by a barrier) and checks the canonical file.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any
from unittest import TestCase

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.third_party_agents import _kiss_web_launcher as launcher
from kiss.server.web_server import RemoteAccessServer

_TABS_PER_SIDE = 12


def _tab_ids(path: Path) -> set[str]:
    """Return the tab ids persisted in the registry file at *path*."""
    return {
        entry["tabId"]
        for entry in json.loads(path.read_text(encoding="utf-8"))["tabs"]
    }


class TestEmbeddedLauncherServerOwnsPrivateTabRegistry(TestCase):
    """Canonical tabs survive an embedded server; embedded tabs stay out."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-embedded-tabs-")).resolve()
        kiss_dir = self.tmp / ".kiss"
        kiss_dir.mkdir()
        self._saved_persistence = (
            _persistence._DB_PATH, _persistence._db_conn, _persistence._KISS_DIR,
        )
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None
        self.canonical_tabs = kiss_dir / "tabs.json"
        # A tab bound to a chat, persisted by an earlier daemon session:
        # it must survive, and its chat binding must not be inherited by
        # the embedded server.
        self.canonical_tabs.write_text(json.dumps({"tabs": [{
            "tabId": "vscode-earlier", "chatId": "chat-earlier",
            "title": "earlier chat", "workDir": str(self.tmp),
        }]}), encoding="utf-8")
        self._saved_launcher = (launcher._API_SERVER, launcher._API_SERVER_SOCK)
        launcher._API_SERVER = None
        launcher._API_SERVER_SOCK = ""

    def tearDown(self) -> None:
        embedded = launcher._API_SERVER
        if embedded is not None and embedded._loop is not None:
            embedded._loop.call_soon_threadsafe(embedded._loop.stop)
        launcher._API_SERVER, launcher._API_SERVER_SOCK = self._saved_launcher
        (
            _persistence._DB_PATH, _persistence._db_conn, _persistence._KISS_DIR,
        ) = self._saved_persistence
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _open_tabs(
        server: Any, prefix: str, barrier: threading.Barrier,
    ) -> None:
        barrier.wait(timeout=30)
        for index in range(_TABS_PER_SIDE):
            server._vscode_server._cmd_open_tab({
                "type": "openTab",
                "tabId": f"{prefix}-{index}",
                "title": f"{prefix} {index}",
                "workDir": "",
            })

    def test_canonical_registry_keeps_every_tab_and_none_of_the_embedded(
        self,
    ) -> None:
        canonical = RemoteAccessServer(
            uds_path=str(self.tmp / "canonical.sock"), work_dir=str(self.tmp),
        )
        # The launcher's real entry point: a private-UDS daemon in this
        # process sharing the KISS home (database, chats) with the
        # canonical one.
        sock_path = launcher._ensure_api_server()
        embedded = launcher._API_SERVER
        assert embedded is not None
        self.assertTrue(Path(sock_path).exists())

        barrier = threading.Barrier(2)
        threads = [
            threading.Thread(
                target=self._open_tabs, args=(canonical, "vscode", barrier),
            ),
            threading.Thread(
                target=self._open_tabs, args=(embedded, "channel", barrier),
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
            self.assertFalse(t.is_alive())

        expected_canonical = {"vscode-earlier"} | {
            f"vscode-{i}" for i in range(_TABS_PER_SIDE)
        }
        expected_embedded = {f"channel-{i}" for i in range(_TABS_PER_SIDE)}
        self.assertEqual(
            {e["tabId"] for e in canonical._vscode_server.tab_registry.snapshot()},
            expected_canonical,
        )
        self.assertEqual(
            {e["tabId"] for e in embedded._vscode_server.tab_registry.snapshot()},
            expected_embedded,
        )
        on_disk = _tab_ids(self.canonical_tabs)
        self.assertEqual(
            on_disk & expected_embedded, set(),
            "embedded server's tabs leaked into the canonical tabs.json",
        )
        self.assertEqual(
            on_disk, expected_canonical,
            "canonical tabs.json lost tabs to the embedded server",
        )
        # A fresh canonical daemon (restart) sees exactly its own tabs.
        restarted = RemoteAccessServer(
            uds_path=str(self.tmp / "restarted.sock"), work_dir=str(self.tmp),
        )
        self.assertEqual(
            {e["tabId"] for e in restarted._vscode_server.tab_registry.snapshot()},
            expected_canonical,
        )
        self.assertNotIn(
            "vscode-earlier", embedded._vscode_server._tab_chat_views,
            "the embedded server inherited a canonical tab's chat binding",
        )

    def test_embedded_server_tab_mutations_never_write_the_canonical_file(
        self,
    ) -> None:
        before = self.canonical_tabs.read_bytes()
        launcher._ensure_api_server()
        embedded = launcher._API_SERVER
        assert embedded is not None
        vscode = embedded._vscode_server
        vscode._cmd_open_tab({"type": "openTab", "tabId": "channel-x", "title": "x"})
        vscode.tab_registry.update_tab("channel-x", chat_id="chat-x", title="renamed")
        vscode.tab_registry.close_tab("channel-x")
        self.assertEqual(self.canonical_tabs.read_bytes(), before)
        self.assertEqual(vscode.tab_registry.snapshot(), [])

    def test_embedded_server_is_created_once_per_process(self) -> None:
        first = launcher._ensure_api_server()
        server = launcher._API_SERVER
        self.assertEqual(launcher._ensure_api_server(), first)
        self.assertIs(launcher._API_SERVER, server)
        loop = server._loop if server is not None else None
        self.assertIsInstance(loop, asyncio.AbstractEventLoop)
