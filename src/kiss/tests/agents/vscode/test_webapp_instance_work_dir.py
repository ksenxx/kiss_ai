# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests: each remote-webapp instance keeps its own work_dir.

Each browser tab running the standalone web client is one webapp
instance.  The instance pins its work_dir in ``sessionStorage`` (key
``sorcar-work-dir``, scoped per tab) via the WS shim's ``postMessage``
hook, and the shim replays ``setWorkDir`` to the server right after
every successful (re)authentication — mirroring how each VS Code
window re-announces its workspace folder on every UDS (re)connect.
Server-side, ``RemoteAccessServer._dispatch_client_command`` records
the folder per connection and stamps it onto every later command from
the same connection that lacks an explicit ``workDir``.

Invariant under test: two webapp instances sharing one daemon can
never observe each other's folder, across reconnects and reloads.

* The shim tests replay the REAL ``_WS_SHIM_JS`` source in Node with a
  fake WebSocket, asserting the sessionStorage pin and the
  auth_ok-time ``setWorkDir`` replay (ordered BEFORE any queued
  commands, so the server stamps them with the right folder).
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
import kiss.server.vscode_config as vc
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
globalThis.document = {getElementById: () => null};
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
    result = subprocess.run(
        ["node", "-e", script],
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


class TestWsShimWorkDirPin(unittest.TestCase):
    """The browser WS shim pins and replays the instance work_dir."""

    def test_post_message_set_work_dir_pins_session_storage(self) -> None:
        """``postMessage({type:'setWorkDir'})`` writes the per-tab pin
        and the subsequent ``auth_ok`` replays it BEFORE flushing the
        queued commands, so the server stamps them correctly."""
        out = _run_shim_harness("""
const api = window.acquireVsCodeApi();
const ws0 = FakeWS.instances[0];
ws0.onopen();
api.postMessage({type: 'setWorkDir', workDir: '/inst/a'});
api.postMessage({type: 'getFiles', prefix: ''});
ws0.onmessage({data: JSON.stringify({type: 'auth_ok'})});
out.sent = ws0.sent.map(s => JSON.parse(s));
""")
        self.assertEqual(out["sessionWorkDir"], "/inst/a")
        sent = out["sent"]
        self.assertEqual(sent[0]["type"], "auth")
        # The replay must come first; the queued getFiles arrives only
        # after the connection's work_dir is established.
        self.assertEqual(sent[1]["type"], "setWorkDir")
        self.assertEqual(sent[1]["workDir"], "/inst/a")
        types_after = [m["type"] for m in sent[2:]]
        self.assertIn("getFiles", types_after)
        self.assertLess(
            [m["type"] for m in sent].index("setWorkDir"),
            [m["type"] for m in sent].index("getFiles"),
        )

    def test_reconnect_after_prior_auth_reloads_page(self) -> None:
        """After a dropped WebSocket post-auth the shim reloads the page
        on the next ``auth_ok`` (recovering from a server restart).
        The reload preserves ``sessionStorage`` so the post-reload
        shim's first ``auth_ok`` replays ``setWorkDir`` from the pin
        (covered by
        ``test_fresh_shim_with_pinned_work_dir_replays_on_auth_ok``).

        Together these two tests preserve the per-tab work_dir
        invariant across reconnects under the auto-reload design
        introduced to recover from server restarts: the in-place
        replay is replaced by a reload-then-replay round-trip whose
        net observable effect on the server is the same
        ``setWorkDir`` frame on the new connection."""
        out = _run_shim_harness("""
globalThis.location.reload = () => {
  out.reloaded = (out.reloaded || 0) + 1;
};
const api = window.acquireVsCodeApi();
const ws0 = FakeWS.instances[0];
ws0.onopen();
api.postMessage({type: 'setWorkDir', workDir: '/inst/a'});
ws0.onmessage({data: JSON.stringify({type: 'auth_ok'})});
ws0.onclose();
timers.shift()();  // fire the reconnect timer
const ws1 = FakeWS.instances[1];
ws1.onopen();
ws1.onmessage({data: JSON.stringify({type: 'auth_ok'})});
out.sent1 = ws1.sent.map(s => JSON.parse(s));
""")
        # Reload must have been triggered exactly once: this is what
        # refreshes the page so the new server's restored state is
        # visible to the user.
        self.assertEqual(out.get("reloaded"), 1, out)
        # The second connection sent ONLY ``auth`` — no in-place
        # setWorkDir replay, because the shim short-circuits to a
        # reload on the post-disconnect ``auth_ok``.  The replay
        # happens on the fresh post-reload shim from sessionStorage.
        sent1 = out["sent1"]
        self.assertEqual([m["type"] for m in sent1], ["auth"])
        # And the sessionStorage pin survives the simulated reload
        # (real ``location.reload()`` preserves sessionStorage).
        self.assertEqual(out["sessionWorkDir"], "/inst/a")

    def test_fresh_shim_with_pinned_work_dir_replays_on_auth_ok(self) -> None:
        """A fresh shim whose sessionStorage already holds a pinned
        work_dir (e.g. because the prior shim instance reloaded the
        page on reconnect) MUST replay ``setWorkDir`` on its first
        ``auth_ok``.  This is the post-reload half of the
        reconnect-replay round-trip."""
        out = _run_shim_harness("""
_ss['sorcar-work-dir'] = '/inst/a';  // pin from a prior page instance
const ws0 = FakeWS.instances[0];
ws0.onopen();
ws0.onmessage({data: JSON.stringify({type: 'auth_ok'})});
out.sent = ws0.sent.map(s => JSON.parse(s));
""")
        sent = out["sent"]
        self.assertEqual(sent[0]["type"], "auth")
        self.assertEqual(sent[1]["type"], "setWorkDir")
        self.assertEqual(sent[1]["workDir"], "/inst/a")

    def test_no_pin_means_no_replay(self) -> None:
        """A fresh instance with no pinned work_dir sends nothing but
        the auth frame on connect — the server's fallback applies."""
        out = _run_shim_harness("""
const ws0 = FakeWS.instances[0];
ws0.onopen();
ws0.onmessage({data: JSON.stringify({type: 'auth_ok'})});
out.sent = ws0.sent.map(s => JSON.parse(s));
""")
        self.assertEqual([m["type"] for m in out["sent"]], ["auth"])
        self.assertEqual(out["sessionWorkDir"], "")


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
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

        # Isolate config.json so auth uses an empty remote_password and
        # this test never sees (or pollutes) the user's real config.
        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        # Two folders, one per simulated webapp instance.
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
            uds_path=Path(self.tmpdir) / "sorcar.sock",
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

    async def test_two_instances_keep_independent_work_dirs(self) -> None:
        """Instance B pinning folder B must never redirect instance A's
        work_dir-dependent commands (sent WITHOUT explicit workDir) to
        folder B, even though B synced last (daemon fallback = B)."""
        ws_a = await self._connect_instance()
        ws_b = await self._connect_instance()
        await self._send(
            ws_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(
            ws_b, {"type": "setWorkDir", "workDir": str(self.dir_b)},
        )

        await self._send(ws_a, {"type": "getFiles", "prefix": ""})
        ev_a = await self._drain_until(
            ws_a, self._files_event_with("alpha.txt"),
        )
        self.assertNotIn("beta.txt", _file_names(ev_a))

        await self._send(ws_b, {"type": "getFiles", "prefix": ""})
        ev_b = await self._drain_until(
            ws_b, self._files_event_with("beta.txt"),
        )
        self.assertNotIn("alpha.txt", _file_names(ev_b))

    async def test_reconnect_replay_restores_instance_work_dir(self) -> None:
        """A reconnecting instance that replays ``setWorkDir`` (exactly
        what the WS shim does after ``auth_ok``) gets its folder back,
        even though another instance moved the daemon fallback."""
        ws_a = await self._connect_instance()
        ws_b = await self._connect_instance()
        await self._send(
            ws_a, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(
            ws_b, {"type": "setWorkDir", "workDir": str(self.dir_b)},
        )

        # Instance A's WebSocket drops (page reload / network blip).
        await ws_a.close()
        ws_a2 = await self._connect_instance()
        await self._send(
            ws_a2, {"type": "setWorkDir", "workDir": str(self.dir_a)},
        )
        await self._send(ws_a2, {"type": "getFiles", "prefix": ""})
        ev = await self._drain_until(
            ws_a2, self._files_event_with("alpha.txt"),
        )
        self.assertNotIn("beta.txt", _file_names(ev))

    async def test_get_config_reports_instance_pin_over_persisted(
        self,
    ) -> None:
        """The settings panel of a pinned instance must show ITS folder
        even when another instance persisted a different work_dir
        globally via saveConfig."""
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
