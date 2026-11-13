# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Daemon start-up and shutdown obligations.

Both tests drive a **real** :class:`~kiss.server.web_server.RemoteAccessServer`
bound to a temporary Unix-domain socket and an ephemeral TCP port, with
an isolated ``KISS_HOME`` and a scratch history database.  Nothing is
mocked, patched or doubled.

* **K2-5** — ``apply_config_to_env`` had exactly one caller repo-wide,
  the ``saveConfig`` command handler.  A freshly started ``kiss-web``
  daemon therefore never applied the *persisted* ``max_budget`` to the
  process config: it kept the declared default until the user happened
  to open and close the settings panel.

* **K2-6** — no module under ``kiss/server/`` referenced MCP at all, so
  the stdio **child processes** ``MCPManager`` spawns were reaped only
  by its ``atexit`` hook — which does not run when the daemon is
  killed.  Every shutdown that was not a clean interpreter exit
  orphaned them.

* **L2-2** — only the synchronous ``start()`` finalizer stopped the
  ``@``-picker's file-index worker thread; ``stop_async`` ran
  ``_shutdown_children`` without it, so an embedder that stops the
  daemon through the async API kept a worker blocked on its queue for
  the life of the process, with the server graph it retains.

* **L2-1** — the ghost-text autocomplete worker had no stop at all:
  neither shutdown API joined it, so every daemon that served one
  ``complete`` command left a thread blocked on its queue, retaining the
  server graph and still answering requests queued behind it.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.mcp_servers import MCPManager, MCPServerConfig
from kiss.core import config as config_module
from kiss.core import user_config
from kiss.core.processes import pid_alive
from kiss.server import agent_state, autocomplete
from kiss.server.server import AgentServer
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import _generate_self_signed_cert, open_local_connection
from kiss.tests.server._memory_printer import MemoryPrinter
from kiss.tests.server.conftest import EntryGate, find_free_port

#: A real MCP server: it records its pid, then serves the stdio
#: transport for real.  Used to prove the daemon reaps its children.
_MCP_SERVER_SOURCE = """
import os
import sys

from mcp.server.fastmcp import FastMCP

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    handle.write(str(os.getpid()))

server = FastMCP("k2-probe")


@server.tool()
def ping() -> str:
    \"\"\"Return a constant so the handshake has something to advertise.\"\"\"
    return "pong"


server.run()
"""


def _wait_pid_dead(pid: int, timeout: float) -> bool:
    """Poll until *pid* is gone or *timeout* seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.1)
    return not pid_alive(pid)


class _DaemonHarness(IsolatedAsyncioTestCase):
    """A real daemon with a temp local endpoint, fully isolated from ``~/.kiss``."""

    #: Config written before the server is constructed.
    config: dict[str, Any] = {}

    async def asyncSetUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-k2-daemon-"))
        self.kiss_home = self.tmpdir / ".kiss"
        self.kiss_home.mkdir(parents=True, exist_ok=True)
        self.work_dir = self.tmpdir / "repo"
        self.work_dir.mkdir()

        self._saved_env = os.environ.get("KISS_HOME")
        os.environ["KISS_HOME"] = str(self.kiss_home)
        self._saved_cfg = (
            user_config.CONFIG_DIR,
            user_config.CONFIG_PATH,
        )
        user_config.CONFIG_DIR = self.kiss_home
        user_config.CONFIG_PATH = self.kiss_home / "config.json"
        (self.kiss_home / "config.json").write_text(
            json.dumps({"work_dir": str(self.work_dir), **self.config}),
            encoding="utf-8",
        )
        self._saved_budget = config_module.DEFAULT_CONFIG.max_budget

        self._saved_db = (th._DB_PATH, th._KISS_DIR)
        th._KISS_DIR = self.kiss_home
        th._DB_PATH = self.kiss_home / "history.db"

        certfile = self.tmpdir / "cert.pem"
        keyfile = self.tmpdir / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=find_free_port(),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=self.tmpdir / "remote-url.json",
            local_endpoint_file=self.tmpdir / "sorcar-local.json",
            work_dir=str(self.work_dir),
        )
        self._stopped = False

    async def asyncTearDown(self) -> None:
        if not self._stopped:
            await self.server.stop_async()
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
        th._close_thread_db()
        th._DB_PATH, th._KISS_DIR = self._saved_db
        config_module.DEFAULT_CONFIG.max_budget = self._saved_budget
        user_config.CONFIG_DIR, user_config.CONFIG_PATH = self._saved_cfg
        if self._saved_env is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self._saved_env
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _stop(self) -> None:
        """Stop the daemon exactly once."""
        await self.server.stop_async()
        self._stopped = True


class TestStartupAppliesPersistedConfig(_DaemonHarness):
    """K2-5: a fresh daemon must honour the saved budget immediately."""

    config = {"max_budget": 42.5}

    async def test_persisted_budget_reaches_the_process_config(self) -> None:
        """Starting the daemon applies the saved ``max_budget``."""
        self.assertAlmostEqual(
            config_module.DEFAULT_CONFIG.max_budget,
            42.5,
            msg="a freshly started daemon kept the declared default "
            "budget: the persisted config was never applied to the "
            "process, so it took effect only once the user opened and "
            "closed the settings panel",
        )

    async def test_startup_and_save_config_agree(self) -> None:
        """The start-up path and the ``saveConfig`` path apply the same value."""
        await self.server.start_async()
        reader, writer = await open_local_connection(self.server, limit=1 << 20)
        try:
            writer.write(
                json.dumps(
                    {
                        "type": "saveConfig",
                        "config": {"max_budget": 55.25},
                    },
                ).encode("utf-8")
                + b"\n",
            )
            await writer.drain()
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if config_module.DEFAULT_CONFIG.max_budget == 55.25:
                    break
                await asyncio.sleep(0.05)
        finally:
            writer.close()
        self.assertAlmostEqual(config_module.DEFAULT_CONFIG.max_budget, 55.25)
        del reader

        # Restarting the daemon in this process must land on the value
        # saveConfig just persisted, with no settings panel involved.
        config_module.DEFAULT_CONFIG.max_budget = 1.0
        await self._stop()
        restarted = RemoteAccessServer(
            host="127.0.0.1",
            port=find_free_port(),
            certfile=str(self.tmpdir / "cert.pem"),
            keyfile=str(self.tmpdir / "key.pem"),
            url_file=self.tmpdir / "remote-url-2.json",
            local_endpoint_file=self.tmpdir / "sorcar-local-2.json",
            work_dir=str(self.work_dir),
        )
        self.assertAlmostEqual(
            config_module.DEFAULT_CONFIG.max_budget,
            55.25,
            msg="a restarted daemon ignored the budget saveConfig persisted",
        )
        del restarted


class TestShutdownReapsMcpChildren(_DaemonHarness):
    """K2-6: the daemon must not orphan MCP stdio children."""

    async def asyncTearDown(self) -> None:
        MCPManager.instance().shutdown()
        if self._mcp_pid and pid_alive(self._mcp_pid):  # pragma: no cover
            os.kill(self._mcp_pid, 9)
        await super().asyncTearDown()

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self._mcp_pid = 0

    async def test_stop_async_kills_the_stdio_child(self) -> None:
        """Stopping the daemon terminates the MCP server it hosted."""
        script = self.tmpdir / "k2_mcp_server.py"
        script.write_text(_MCP_SERVER_SOURCE, encoding="utf-8")
        pidfile = self.tmpdir / "mcp.pid"

        await self.server.start_async()
        conn = await asyncio.to_thread(
            MCPManager.instance().connect,
            MCPServerConfig(
                name="k2-probe",
                transport="stdio",
                command=sys.executable,
                args=(str(script), str(pidfile)),
            ),
        )
        self.assertFalse(conn.error, f"MCP handshake failed: {conn.error}")
        self._mcp_pid = int(pidfile.read_text(encoding="utf-8"))
        self.assertTrue(
            pid_alive(self._mcp_pid), "the MCP child died before shutdown",
        )

        await self._stop()

        self.assertTrue(
            _wait_pid_dead(self._mcp_pid, 30.0),
            "the MCP stdio child outlived the daemon: no server module "
            "ever disconnected the MCP manager, so its children were "
            "left to an atexit hook that a killed daemon never runs",
        )


class TestShutdownStopsFileIndexWorker(_DaemonHarness):
    """L2-2: ``stop_async`` stops the file-index worker like ``start()`` does."""

    config: dict[str, Any] = {}

    async def test_stop_async_stops_the_file_index_worker(self) -> None:
        """The worker thread a picker request started is joined by stop_async."""
        await self.server.start_async()
        registry = self.server._agent_server._file_index
        ready = threading.Event()
        self.assertTrue(registry.ensure(str(self.work_dir), ready.set))
        self.assertTrue(ready.wait(30.0), "the file-index build never called back")
        worker = registry._worker
        assert worker is not None
        self.assertTrue(worker.is_alive())

        await self._stop()

        worker.join(timeout=5.0)
        self.assertFalse(
            worker.is_alive(),
            "stop_async left the file-index worker blocked on its queue: only "
            "start()'s finalizer stopped it, _shutdown_children never did",
        )
        self.assertFalse(
            registry.ensure(str(self.work_dir), None),
            "a stopped daemon's index registry must refuse new builds",
        )


class TestShutdownStopsAutocompleteWorker(_DaemonHarness):
    """L2-1: ``stop_async`` stops the ghost-text autocomplete worker."""

    config: dict[str, Any] = {}

    async def test_stop_async_stops_the_autocomplete_worker(self) -> None:
        """The worker a ``complete`` command started is joined; none starts after."""
        await self.server.start_async()
        agent = self.server._agent_server
        agent._cmd_complete(
            {"type": "complete", "query": "he", "connId": "c1", "tabId": "t1"},
        )
        worker = agent._complete_worker
        assert worker is not None
        self.assertTrue(worker.is_alive())

        await self._stop()

        worker.join(timeout=5.0)
        self.assertFalse(
            worker.is_alive(),
            "stop_async left the autocomplete worker blocked on its queue: "
            "_shutdown_children never stopped it",
        )
        agent._cmd_complete(
            {"type": "complete", "query": "hel", "connId": "c1", "tabId": "t1"},
        )
        self.assertIs(
            agent._complete_worker, worker,
            "a stopped daemon must not start a new autocomplete worker",
        )
        assert agent._complete_queue is not None
        self.assertTrue(agent._complete_queue.empty(), "a stopped daemon queues no request")


class TestAutocompleteWorkerStop(unittest.TestCase):
    """L2-1: the stop drops queued requests and silences a late completion."""

    def tearDown(self) -> None:
        threading.setprofile(None)

    def test_stop_drops_the_queue_and_a_late_completion_publishes_nothing(self) -> None:
        """A completion computing at the stop publishes nothing; the queue is dropped."""
        printer = MemoryPrinter()
        server = AgentServer(printer=printer)
        # Hold the worker between computing the completions of the first
        # request and publishing them (``_ghost_suffix`` is the first
        # call after ``_complete_many``).
        gate = EntryGate(autocomplete._ghost_suffix.__code__)
        gate.start(
            server._cmd_complete,
            {"type": "complete", "query": "hel", "connId": "c1", "tabId": "t1"},
        ).join(10)
        self.assertTrue(gate.reached.wait(10), "the worker never computed a completion")
        worker = server._complete_worker
        assert worker is not None
        # Queued behind the held request.
        server._cmd_complete(
            {"type": "complete", "query": "hello", "connId": "c2", "tabId": "t2"},
        )

        stopper = threading.Thread(target=server._stop_complete_worker, daemon=True)
        stopper.start()
        deadline = time.monotonic() + 10
        while not server._complete_stopped and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(server._complete_stopped, "the stop never marked the worker")
        gate.release.set()
        stopper.join(10)
        self.assertFalse(stopper.is_alive(), "the stop did not return")
        worker.join(5)
        self.assertFalse(worker.is_alive(), "the worker outlived the stop")

        published = [
            e for e in printer.emitted if e.get("type") in ("ghost", "completions")
        ]
        self.assertEqual(published, [], "a completion finishing after the stop published")
        # After the stop: no new worker, nothing queued, a second stop is a no-op.
        server._cmd_complete(
            {"type": "complete", "query": "help", "connId": "c3", "tabId": "t3"},
        )
        self.assertIs(server._complete_worker, worker)
        assert server._complete_queue is not None
        self.assertTrue(server._complete_queue.empty())
        server._stop_complete_worker()


class TestBrowserTabHandoffRegistration(_DaemonHarness):
    """Connector sign-in pages reach the daemon's Browser tab only while it serves."""

    config: dict[str, Any] = {}

    async def test_opener_is_registered_while_serving_and_cleared_on_stop(self) -> None:
        """start_async registers the Browser-tab opener; stop_async clears it."""
        from kiss.core import browser_handoff

        # Process-global: a daemon an earlier test of this process left
        # serving (or still stopping) may have registered its opener.
        # What is asserted is this daemon's own registration and clearing.
        browser_handoff._browser_tab_opener = None
        await self.server.start_async()
        opener = browser_handoff._browser_tab_opener
        self.assertIsNotNone(opener, "a served daemon must hand sign-in pages to its tab")
        assert opener is not None
        self.assertEqual(
            opener.__func__,  # type: ignore[attr-defined]
            type(self.server._agent_server.browser_tabs).open_for_user,
        )
        self.assertIs(
            opener.__self__,  # type: ignore[attr-defined]
            self.server._agent_server.browser_tabs,
        )
        await self._stop()
        self.assertIsNone(
            browser_handoff._browser_tab_opener,
            "a stopped daemon's tab has no surfaces: hand-offs must fall back",
        )

    async def test_failed_setup_leaves_no_opener_behind(self) -> None:
        """A setup that fails after registering rolls the registration back too."""
        from kiss.core import browser_handoff

        # Occupy the daemon's port so binding the WSS listener fails at once.
        blocker = socket.socket()
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", self.server.port))
        blocker.listen(1)
        try:
            with self.assertRaises(SystemExit):  # "cannot bind" is fatal for the daemon
                await self.server.start_async()
        finally:
            blocker.close()
        self._stopped = True
        self.assertIsNone(self.server._ws_server)
        self.assertIsNone(browser_handoff._browser_tab_opener)


if __name__ == "__main__":  # pragma: no cover — manual runs
    unittest.main()
