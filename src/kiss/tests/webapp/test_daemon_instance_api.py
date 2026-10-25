# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: ``GET /api/daemon`` names the daemon process.

The webapp asks this endpoint on every ``auth_ok`` (``media/transport.js``)
and reloads itself when a reconnect reaches a daemon process other than
the one it first authenticated with: ``install.sh`` replaced the package
and restarted the daemon, so the page's JavaScript is older than the
daemon it is talking to.  The token must therefore be stable for the
life of one process and differ between processes, and the answer must
never be cached.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace
from typing import cast

from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.http11 import Request

from kiss.server import web_server


def _get(path: str) -> tuple[int, Headers, bytes]:
    """GET *path* from a fresh server as a loopback client; return status, headers, body."""
    server = web_server.RemoteAccessServer(host="127.0.0.1", port=0)
    conn = SimpleNamespace(remote_address=("127.0.0.1", 0))
    response = asyncio.run(
        server._process_request(
            cast(ServerConnection, conn), Request(path=path, headers=Headers())
        ),
    )
    assert response is not None
    return response.status_code, response.headers, bytes(response.body or b"")


def test_api_daemon_names_this_process_and_its_version() -> None:
    """The endpoint answers JSON with the process token and the version, uncached."""
    status, headers, body = _get(web_server.DAEMON_API)
    assert web_server.DAEMON_API == "/api/daemon"
    assert status == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert headers["Cache-Control"] == "no-cache, no-store, must-revalidate"
    info = json.loads(body)
    assert set(info) == {"instance", "version"}
    assert info["instance"] == web_server.DAEMON_INSTANCE
    assert len(info["instance"]) == 16 and int(info["instance"], 16) >= 0
    assert info["version"] == web_server._read_version() != ""


def test_api_daemon_token_is_stable_per_process_and_differs_between_processes() -> None:
    """Two servers of one process agree; a daemon in another process answers another token."""
    first = json.loads(_get(web_server.DAEMON_API)[2])["instance"]
    second = json.loads(_get(web_server.DAEMON_API)[2])["instance"]
    assert first == second == web_server.DAEMON_INSTANCE
    script = (
        "from kiss.server import web_server\n"
        "print(web_server.DAEMON_INSTANCE)\n"
    )
    other = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True,
        timeout=120,
    ).stdout.strip()
    assert len(other) == 16
    assert other != first
