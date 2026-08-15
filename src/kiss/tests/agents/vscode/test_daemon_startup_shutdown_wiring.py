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
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.mcp_servers import MCPManager, MCPServerConfig
from kiss.core import config as config_module
from kiss.core import vscode_config
from kiss.server import agent_state
from kiss.server.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
)

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


def _find_free_port() -> int:
    """Return an available TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        port: int = sock.getsockname()[1]
        return port


def _pid_alive(pid: int) -> bool:
    """Return True while *pid* still names a live process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover — foreign-owned pid
        return True
    return True


def _wait_pid_dead(pid: int, timeout: float) -> bool:
    """Poll until *pid* is gone or *timeout* seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    return not _pid_alive(pid)


class _DaemonHarness(IsolatedAsyncioTestCase):
    """A real daemon over a temp UDS socket, fully isolated from ``~/.kiss``."""

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
            vscode_config.CONFIG_DIR,
            vscode_config.CONFIG_PATH,
        )
        vscode_config.CONFIG_DIR = self.kiss_home
        vscode_config.CONFIG_PATH = self.kiss_home / "config.json"
        (self.kiss_home / "config.json").write_text(
            json.dumps({"work_dir": str(self.work_dir), **self.config}),
            encoding="utf-8",
        )
        self._saved_budget = config_module.DEFAULT_CONFIG.max_budget

        self._saved_db = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = self.kiss_home
        th._DB_PATH = self.kiss_home / "sorcar.db"
        th._db_conn = None

        certfile = self.tmpdir / "cert.pem"
        keyfile = self.tmpdir / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=_find_free_port(),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=self.tmpdir / "remote-url.json",
            uds_path=self.tmpdir / "sorcar.sock",
            work_dir=str(self.work_dir),
        )
        self._stopped = False

    async def asyncTearDown(self) -> None:
        if not self._stopped:
            await self.server.stop_async()
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
        if th._db_conn is not None:
            th._db_conn.close()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self._saved_db
        config_module.DEFAULT_CONFIG.max_budget = self._saved_budget
        vscode_config.CONFIG_DIR, vscode_config.CONFIG_PATH = self._saved_cfg
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
        reader, writer = await asyncio.open_unix_connection(
            str(self.tmpdir / "sorcar.sock"), limit=1 << 20,
        )
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
            port=_find_free_port(),
            certfile=str(self.tmpdir / "cert.pem"),
            keyfile=str(self.tmpdir / "key.pem"),
            url_file=self.tmpdir / "remote-url-2.json",
            uds_path=self.tmpdir / "sorcar-2.sock",
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
        if self._mcp_pid and _pid_alive(self._mcp_pid):  # pragma: no cover
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
            _pid_alive(self._mcp_pid), "the MCP child died before shutdown",
        )

        await self._stop()

        self.assertTrue(
            _wait_pid_dead(self._mcp_pid, 30.0),
            "the MCP stdio child outlived the daemon: no server module "
            "ever disconnected the MCP manager, so its children were "
            "left to an atexit hook that a killed daemon never runs",
        )


if __name__ == "__main__":  # pragma: no cover — manual runs
    unittest.main()
