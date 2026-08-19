# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Tests that stay behind from ``kiss.tests.agents.vscode.test_web_server``
(now ``kiss.tests.server.test_web_server``): they assert on assets outside
kiss.core/kiss.agents.sorcar/kiss.server (e.g. the real
chat.html/main.js media content), so they keep their original
location while the server-only majority of the file moved to
tests/server.
"""


from __future__ import annotations

import asyncio
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import RemoteAccessServer, _build_html
from kiss.tests.server.test_web_server import _find_free_port, _no_verify_ssl


class TestBuildHtml(unittest.TestCase):
    """Test HTML template generation."""

    def test_html_contains_key_elements(self) -> None:
        """The generated HTML includes all essential chat UI components."""
        html = _build_html()
        self.assertIn("<title>KISS Sorcar</title>", html)
        self.assertIn('id="tab-bar"', html)
        self.assertIn('id="output"', html)
        self.assertIn('id="task-input"', html)
        self.assertIn('id="input-area"', html)
        self.assertIn('id="model-picker"', html)
        self.assertIn('id="sidebar"', html)
        self.assertIn('id="settings-panel"', html)
        self.assertIn('id="frequent-panel"', html)
        self.assertIn('id="ask-user-modal"', html)
        self.assertIn('id="send-btn"', html)
        self.assertIn('id="stop-btn"', html)

class TestWebappServerLoadingOverlay(unittest.TestCase):
    """End-to-end regression for the "KISS Sorcar Server is starting ..."
    overlay in the remote webapp.

    Bug: the WebSocket shim served by :func:`_build_html` never
    dispatched a ``daemonStatus`` message, so ``media/main.js``'s
    overlay toggle (driven by that exact message) kept the
    ``#kiss-server-loading`` overlay covering ``#app`` forever — the
    remote webapp showed "KISS Sorcar Server is starting ..." on
    every load and never recovered.

    The fix synthesises ``daemonStatus`` events from the shim's
    WebSocket lifecycle:
      * ``auth_ok``   → ``daemonStatus connected:true``  (reveal #app)
      * ``onclose``   → ``daemonStatus connected:false`` (re-show overlay)

    These assertions pin the source-level contract.  The companion
    end-to-end test ``test/webappServerLoadingOverlay.test.js`` runs
    the shim inside jsdom against a fake WebSocket and verifies the
    actual DOM toggles.
    """

    def test_overlay_visible_by_default_in_rendered_html(self) -> None:
        """The rendered remote-webapp HTML must paint the overlay on top.

        ``#kiss-server-loading`` is present, contains the user-visible
        "KISS Sorcar Server is starting ..." string, and ``#app``
        starts with ``display:none`` so the overlay is the only
        visible UI until the daemonStatus toggle hides it.
        """
        html = _build_html()
        self.assertIn('id="kiss-server-loading"', html)
        self.assertIn("KISS Sorcar Server is starting ...", html)
        self.assertIn('id="app" style="display:none;"', html)

    def test_webapp_e2e_with_jsdom(self) -> None:
        """Drive the real shim through node + jsdom against a fake WSS.

        Reproduces the full bug scenario: load the rendered remote
        webapp HTML, stub ``WebSocket``, run the shim, simulate the
        server replying with ``auth_ok``, and assert the overlay node
        is hidden and ``#app`` is revealed.

        Skipped automatically when node or jsdom are unavailable —
        the source-level assertions above still pin the contract in
        that case.
        """
        import shutil
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not available")
        vscode_dir = (
            Path(__file__).resolve().parents[3]
            / "agents"
            / "vscode"
        )
        jsdom_dir = vscode_dir / "node_modules" / "jsdom"
        if not jsdom_dir.exists():
            self.skipTest("jsdom not installed in vscode/node_modules")

        html = _build_html()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            html_file = tmp_path / "index.html"
            html_file.write_text(html, encoding="utf-8")
            script = tmp_path / "drive.js"
            script.write_text(
                """
'use strict';
const fs = require('fs');
const path = require('path');
const {JSDOM} = require(path.join(process.argv[3], 'node_modules', 'jsdom'));
const html = fs.readFileSync(process.argv[2], 'utf-8');
const dom = new JSDOM(html, {
  url: 'https://example.test/',
  runScripts: 'outside-only',
});
const {window} = dom;

// Replace the inline shim's <script> tag impact: rebuild the auth
// modal nodes the shim looks up (they are part of chat.html template).
// jsdom did not run inline <script>s (runScripts:outside-only), so we
// must locate and eval the shim ourselves.
const shimTag = Array.from(window.document.querySelectorAll('script'))
  .find((s) => s.textContent.indexOf('acquireVsCodeApi') !== -1);
if (!shimTag) {
  console.error('FAIL: shim <script> not found in HTML');
  process.exit(2);
}
const shimSrc = shimTag.textContent;

// Install a fake WebSocket.
const sockets = [];
function FakeWS(url) {
  this.url = url; this.readyState = 0; this.sent = [];
  this.onopen = null; this.onmessage = null; this.onclose = null; this.onerror = null;
  sockets.push(this);
}
FakeWS.CONNECTING = 0; FakeWS.OPEN = 1; FakeWS.CLOSING = 2; FakeWS.CLOSED = 3;
FakeWS.prototype.send = function(d) { this.sent.push(d); };
FakeWS.prototype.close = function() {
  this.readyState = FakeWS.CLOSED;
  if (this.onclose) this.onclose();
};
window.WebSocket = FakeWS;

// Mirror media/main.js's overlay toggle.
window.addEventListener('message', (ev) => {
  const d = ev.data;
  if (d && d.type === 'daemonStatus') {
    const ov = window.document.getElementById('kiss-server-loading');
    const ap = window.document.getElementById('app');
    if (ov) ov.style.display = d.connected ? 'none' : '';
    if (ap) ap.style.display = d.connected ? '' : 'none';
  }
});

window.eval(shimSrc);
if (sockets.length !== 1) {
  console.error('FAIL: expected 1 socket, got', sockets.length); process.exit(3);
}
const s = sockets[0];
s.readyState = FakeWS.OPEN;
s.onopen && s.onopen();
s.onmessage && s.onmessage({data: JSON.stringify({type: 'auth_ok'})});

const ov = window.document.getElementById('kiss-server-loading');
const ap = window.document.getElementById('app');
if (ov.style.display !== 'none') {
  console.error('FAIL: overlay still visible after auth_ok:', JSON.stringify(ov.style.display));
  process.exit(4);
}
if (ap.style.display === 'none') {
  console.error('FAIL: #app still hidden after auth_ok'); process.exit(5);
}

s.readyState = FakeWS.CLOSED;
s.onclose && s.onclose();
if (ov.style.display === 'none') {
  console.error('FAIL: overlay still hidden after socket close'); process.exit(6);
}
if (ap.style.display !== 'none') {
  console.error('FAIL: #app still visible after socket close'); process.exit(7);
}
console.log('OK');
                """,
                encoding="utf-8",
            )
            result = subprocess.run(
                [node, str(script), str(html_file), str(vscode_dir)],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"node drive failed:\nstdout={result.stdout}\nstderr={result.stderr}",
            )
            self.assertIn("OK", result.stdout)


class TestRemoteAccessServerHTTP(IsolatedAsyncioTestCase):
    """Test HTTPS serving of HTML and static assets."""

    async def asyncSetUp(self) -> None:
        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def _http_get(self, path: str) -> tuple[int, str]:
        """Make an HTTPS GET request in a thread to avoid blocking the loop."""
        import urllib.error
        import urllib.request

        url = f"https://127.0.0.1:{self.port}{path}"
        ctx = _no_verify_ssl()

        def _fetch() -> tuple[int, str]:
            try:
                resp = urllib.request.urlopen(url, timeout=5, context=ctx)
                return resp.status, resp.read().decode()
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode() if e.fp else ""

        return await asyncio.get_event_loop().run_in_executor(None, _fetch)

    async def test_serve_css(self) -> None:
        """GET /media/main.css returns the CSS file."""
        status, body = await self._http_get("/media/main.css")
        self.assertEqual(status, 200)
        self.assertIn("#app", body)

    async def test_serve_js(self) -> None:
        """GET /media/main.js returns the JS file."""
        status, body = await self._http_get("/media/main.js")
        self.assertEqual(status, 200)
        self.assertIn("acquireVsCodeApi", body)
