# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: every remote-webapp instance shares ONE work_dir.

Each browser tab running the standalone web client is one webapp
instance.  The working directory is a single daemon-wide value: a
``setWorkDir`` from any instance (the "Working directory" panel, the
Explorer's check mark) adopts it for every task of every instance and
every VS Code window, persists it as ``config.json`` ``work_dir`` and
is broadcast back as ``workDirChanged``.  Nothing is pinned per tab:
the WS shim keeps no ``sessionStorage`` copy and replays nothing on
reconnect, so a reloaded or reconnected instance simply sees the
global value.

* The shim tests replay the REAL ``_WS_SHIM_JS`` source in Node with a
  fake WebSocket, asserting that ``setWorkDir`` is an ordinary queued
  command (sent after ``auth``, in order) and that no pin is kept or
  replayed.
* The WSS tests open real ``wss://`` connections (the webapp's actual
  transport) against a real :class:`RemoteAccessServer`.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import ssl
import subprocess
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import ClientConnection, connect

import kiss.agents.sorcar.persistence as th
import kiss.core.vscode_config as vc
from kiss.server import web_server
from kiss.server.web_server import RemoteAccessServer

_SHIM_PRELUDE = """
'use strict';
const timers = [];
globalThis.setTimeout = (fn, ms) => { timers.push(fn); return 0; };
const _ss = {};
globalThis.sessionStorage = {
  getItem: k => (k in _ss ? _ss[k] : null),
  setItem: (k, v) => { _ss[k] = String(v); },
  removeItem: k => { delete _ss[k]; },
};
const _ls = {};
globalThis.localStorage = {
  getItem: k => (k in _ls ? _ls[k] : null),
  setItem: (k, v) => { _ls[k] = String(v); },
  removeItem: k => { delete _ls[k]; },
};
globalThis.location = {host: 'example.test'};
// readyState 'complete' makes the shim's _dispatchToApp deliver
// events immediately instead of queueing them for the (never firing)
// DOMContentLoaded of this DOM-less harness.
globalThis.document = {
  getElementById: () => null,
  querySelector: () => null,
  readyState: 'complete',
  addEventListener: () => {},
};
globalThis.window = globalThis;
globalThis.dispatchEvent = () => {};
globalThis.MessageEvent = class {
  constructor(type, init) { this.data = init && init.data; }
};
class FakeWS {
  constructor(url) {
    this.url = url;
    this.readyState = FakeWS.OPEN;
    this.sent = [];
    FakeWS.instances.push(this);
  }
  send(d) { this.sent.push(d); }
}
FakeWS.OPEN = 1;
FakeWS.instances = [];
globalThis.WebSocket = FakeWS;
globalThis.FakeWS = FakeWS;
const out = {};
"""

_SHIM_EPILOGUE = """
out.sessionWorkDir = _ss['sorcar-work-dir'] || '';
console.log(JSON.stringify(out));
"""


def _run_shim_harness(scenario_js: str) -> dict[str, Any]:
    """Run the REAL ``_WS_SHIM_JS`` in Node followed by *scenario_js*.

    The prelude installs fake ``WebSocket`` / ``sessionStorage`` /
    ``localStorage`` / ``setTimeout`` globals; the scenario drives the
    shim through its handlers (``onopen`` / ``onmessage`` / ``onclose``)
    and records observations into ``out``, which is printed as JSON.
    """
    script = (
        _SHIM_PRELUDE + web_server._WS_SHIM_JS + scenario_js + _SHIM_EPILOGUE
    )
    result = subprocess.run(  # the script exceeds Windows' command-line limit: use stdin
        ["node", "-"],
        input=script,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"node error: {result.stderr}\nstdout: {result.stdout}"
    )
    parsed = json.loads(result.stdout.strip())
    assert isinstance(parsed, dict)
    return parsed


class TestWsShimNoWorkDirPin(unittest.TestCase):
    """The browser WS shim neither pins nor replays a work_dir."""

    def test_set_work_dir_is_an_ordinary_queued_command(self) -> None:
        """``postMessage({type:'setWorkDir'})`` writes no per-tab pin;
        it goes out after ``auth`` in the order it was posted, like any
        other command."""
        out = _run_shim_harness("""
const api = window.acquireVsCodeApi();
const ws0 = FakeWS.instances[0];
ws0.onopen();
api.postMessage({type: 'setWorkDir', workDir: '/inst/a'});
api.postMessage({type: 'getFiles', prefix: ''});
ws0.onmessage({data: JSON.stringify({type: 'auth_ok'})});
out.sent = ws0.sent.map(s => JSON.parse(s));
""")
        self.assertEqual(out["sessionWorkDir"], "")
        self.assertEqual(
            [m["type"] for m in out["sent"]][:3],
            ["auth", "setWorkDir", "getFiles"],
        )
        self.assertEqual(out["sent"][1]["workDir"], "/inst/a")

    def test_reconnect_replays_nothing(self) -> None:
        """After a dropped WebSocket post-auth the shim re-authenticates
        the new connection and flushes what the page posted during the
        outage -- and nothing else: there is no pin to replay, so the
        reconnected instance sees the daemon's global work_dir."""
        out = _run_shim_harness("""
globalThis.location.reload = () => {
  out.reloaded = (out.reloaded || 0) + 1;
};
const api = window.acquireVsCodeApi();
const ws0 = FakeWS.instances[0];
ws0.onopen();
api.postMessage({type: 'setWorkDir', workDir: '/inst/a'});
ws0.onmessage({data: JSON.stringify({type: 'auth_ok'})});
ws0.onmessage({data: JSON.stringify({type: 'pong'})});
ws0.onclose();
// The last timer is the reconnect backoff (auth_ok armed the
// stale-socket check before it; a closed socket makes that a no-op).
timers.pop()();
const ws1 = FakeWS.instances[1];
ws1.onopen();
api.postMessage({type: 'saveConfig', config: {edited: 'during the outage'}});
ws1.onmessage({data: JSON.stringify({type: 'auth_ok'})});
out.sent1 = ws1.sent.map(s => JSON.parse(s));
""")
        self.assertEqual(out.get("reloaded"), None, out)
        self.assertEqual(
            [m["type"] for m in out["sent1"]],
            ["auth", "saveConfig", "ping"],
        )
        self.assertEqual(out["sessionWorkDir"], "")

    def test_fresh_instance_sends_only_auth(self) -> None:
        """A fresh instance sends nothing but the auth frame on connect,
        even when an older page left a ``sorcar-work-dir`` key behind."""
        out = _run_shim_harness("""
_ss['sorcar-work-dir'] = '/inst/stale';  // left by an older build
const ws0 = FakeWS.instances[0];
ws0.onopen();
ws0.onmessage({data: JSON.stringify({type: 'auth_ok'})});
out.sent = ws0.sent.map(s => JSON.parse(s));
""")
        self.assertEqual([m["type"] for m in out["sent"]], ["auth"])


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "history.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _find_free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _no_verify_ssl() -> ssl.SSLContext:
    """Return an SSL client context that skips certificate verification."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _file_names(event: dict[str, Any]) -> list[str]:
    """Extract the file-name strings from a ``files`` event."""
    names: list[str] = []
    for entry in event.get("files", []):
        if isinstance(entry, dict):
            names.append(str(entry.get("text", "")))
        else:
            names.append(str(entry))
    return names


class TestWebappInstanceWorkDirOverWss(IsolatedAsyncioTestCase):
    """Two real WSS connections (= two webapp instances), one daemon."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)

        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        self.dir_a = Path(self.tmpdir) / "inst_a"
        self.dir_b = Path(self.tmpdir) / "inst_b"
        self.dir_a.mkdir()
        self.dir_b.mkdir()
        (self.dir_a / "alpha.txt").write_text("alpha")
        (self.dir_b / "beta.txt").write_text("beta")

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.server.web_server import _generate_self_signed_cert
        _generate_self_signed_cert(certfile, keyfile)

        self.port = _find_free_port()
        self.url = f"wss://127.0.0.1:{self.port}/ws"
        self.ctx = _no_verify_ssl()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()
        self._sockets: list[ClientConnection] = []

    async def asyncTearDown(self) -> None:
        for ws in self._sockets:
            try:
                await ws.close()
            except Exception:
                pass
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        vc.CONFIG_DIR = self._orig_cfg_dir
        vc.CONFIG_PATH = self._orig_cfg_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect_instance(self) -> ClientConnection:
        """Open + authenticate one WSS connection (one webapp instance)."""
        ws = await connect(self.url, ssl=self.ctx)
        self._sockets.append(ws)
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        self.assertEqual(resp["type"], "auth_ok")
        return ws

    async def _send(self, ws: ClientConnection, cmd: dict[str, Any]) -> None:
        await ws.send(json.dumps(cmd))

    async def _drain_until(
        self,
        ws: ClientConnection,
        predicate: Callable[[dict[str, Any]], bool],
        max_events: int = 100,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Read events until *predicate* matches or the budget expires."""
        for _ in range(max_events):
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            msg = json.loads(raw)
            assert isinstance(msg, dict)
            if predicate(msg):
                return msg
        raise AssertionError(
            f"predicate never matched within {max_events} events",
        )

    @staticmethod
    def _files_event_with(name: str) -> Callable[[dict[str, Any]], bool]:
        """Predicate: a populated ``files`` event containing *name*."""
        def _pred(msg: dict[str, Any]) -> bool:
            return (
                msg.get("type") == "files"
                and not msg.get("loading")
                and name in _file_names(msg)
            )
        return _pred

    async def test_two_instances_share_one_work_dir(self) -> None:
        """Instance B picking folder B moves instance A too: A's
        work_dir-dependent commands (sent WITHOUT explicit workDir) run
        in folder B, and both instances receive ``workDirChanged``."""
        ws_a = await self._connect_instance()
        ws_b = await self._connect_instance()
        await self._send(
            ws_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(
            ws_b, {"type": "setWorkDir", "workDir": str(self.dir_b)},
        )
        for ws in (ws_a, ws_b):
            await self._drain_until(
                ws,
                lambda m: (
                    m.get("type") == "workDirChanged"
                    and m.get("workDir") == str(self.dir_b)
                ),
            )

        await self._send(ws_a, {"type": "getFiles", "prefix": ""})
        ev_a = await self._drain_until(
            ws_a, self._files_event_with("./beta.txt"),
        )
        self.assertNotIn("./alpha.txt", _file_names(ev_a))

        await self._send(ws_b, {"type": "getFiles", "prefix": ""})
        ev_b = await self._drain_until(
            ws_b, self._files_event_with("./beta.txt"),
        )
        self.assertNotIn("./alpha.txt", _file_names(ev_b))

    async def test_reconnected_instance_sees_global_work_dir(self) -> None:
        """A reconnecting instance replays nothing (there is no pin) and
        simply works in the global work_dir another instance picked."""
        ws_a = await self._connect_instance()
        ws_b = await self._connect_instance()
        await self._send(
            ws_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(
            ws_b, {"type": "setWorkDir", "workDir": str(self.dir_b)},
        )
        await self._drain_until(
            ws_b,
            lambda m: (
                m.get("type") == "workDirChanged"
                and m.get("workDir") == str(self.dir_b)
            ),
        )

        await ws_a.close()
        ws_a2 = await self._connect_instance()
        await self._send(ws_a2, {"type": "getFiles", "prefix": ""})
        ev = await self._drain_until(
            ws_a2, self._files_event_with("./beta.txt"),
        )
        self.assertNotIn("./alpha.txt", _file_names(ev))

    async def test_set_work_dir_replaces_persisted_value(self) -> None:
        """A pick (``setWorkDir``) replaces a previously persisted
        work_dir: ``getConfig`` reports the picked folder and
        ``config.json`` now holds it, so a daemon restart keeps it."""
        vc.save_config({"work_dir": str(self.dir_b)})
        ws_a = await self._connect_instance()
        await self._send(
            ws_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(ws_a, {"type": "getConfig"})
        await self._drain_until(
            ws_a,
            lambda m: (
                m.get("type") == "configData"
                and m.get("config", {}).get("work_dir") == str(self.dir_a)
            ),
        )
        self.assertEqual(vc.load_config().get("work_dir"), str(self.dir_a))
