# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a machine's browser page streamed here over a slow link keeps its tab.

A real local daemon with two pages (:class:`TwoClientHarness`) shows a
page of a real remote daemon's browser (the machine ``box``, see
:mod:`kiss.server.remote_views`).  The machine link runs through a TCP
proxy whose machine-to-here direction carries 150 KB/s, a tenth of what
the page's screencast produces (it repaints every animation frame), and
the link's keepalive is tightened to 3 s.

Without flow control the frames piled up in the socket buffers, the
keepalive pong travelled behind them, the link timed out and
``RemoteViews.dropped`` closed the tab on every page here within one
keepalive round (39 s at the production 20 s settings, the "stream
browser crash").  Checked on the wire:

1. The tab stays open and frames keep arriving for the whole watch, at
   the link's pace: the stream acknowledges each frame to the machine,
   which keeps at most two in flight on the link.
2. A page here that acknowledges its frames gets them steadily; one
   that never does gets two per :data:`~kiss.server.frame_flow.ACK_GRACE`.
3. An acknowledgement for a page the machine never announced is ignored.
4. On a link so slow that one frame takes longer than the grace (5 KB/s,
   a small viewport), the tab still survives several keepalive rounds:
   the machine holds the acknowledging stream to its window and never
   counts a frame as delivered by age (that would rebuild the backlog).
5. A storm of resizes from the page here (a viewport announced every
   250 ms) grants no extra frames: the window stays the viewer's, and
   the tab survives on the slow link.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import machines as mach
from kiss.agents.sorcar import web_stealth
from kiss.server.frame_flow import ACK_GRACE, MAX_IN_FLIGHT
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server.two_client_harness import Client, TwoClientHarness

_PLAYWRIGHT_CACHE = Path.home() / ".cache" / "ms-playwright"
_LINK_BYTES_PER_SECOND = 150_000
_KEEPALIVE = 3.0
_WATCH = 20.0
_SLOW_WATCH = 40.0
_STORM_WATCH = 12.0

_PAGE = b"""<!doctype html><title>Remote Animated</title>
<canvas id=c width=1100 height=700></canvas>
<script>
const c = document.getElementById('c').getContext('2d'); let t = 0;
function f() {
  t++;
  for (let i = 0; i < 40; i++) {
    c.fillStyle = 'hsl(' + ((t * 7 + i * 37) % 360) + ',80%,50%)';
    c.fillRect((i * 53 + t * 5) % 1100, (i * 97 + t * 3) % 700, 120, 80);
  }
  requestAnimationFrame(f);
}
f();
</script>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(_PAGE)))
        self.end_headers()
        self.wfile.write(_PAGE)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return


class ThrottledProxy:
    """A TCP proxy to ``target`` whose target-to-client direction is rate limited."""

    def __init__(self, target: tuple[str, int], bytes_per_second: int) -> None:
        self.target = target
        self.rate = bytes_per_second
        self.bytes_down = 0
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(16)
        self.port = sock.getsockname()[1]
        self._sock = sock
        self._loop = asyncio.new_event_loop()
        self._stopped = threading.Event()
        threading.Thread(target=self._run, name="throttled-proxy", daemon=True).start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        server = await asyncio.start_server(self._handle, sock=self._sock)
        async with server:
            await self._loop.run_in_executor(None, self._stopped.wait)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            up_reader, up_writer = await asyncio.open_connection(*self.target)
        except OSError:
            writer.close()
            return
        await asyncio.gather(
            self._pump(reader, up_writer, limited=False),
            self._pump(up_reader, writer, limited=True),
        )

    async def _pump(
        self, src: asyncio.StreamReader, dst: asyncio.StreamWriter, limited: bool,
    ) -> None:
        try:
            while True:
                data = await src.read(4096)
                if not data:
                    break
                if limited:
                    self.bytes_down += len(data)
                    await asyncio.sleep(len(data) / self.rate)
                dst.write(data)
                await dst.drain()
        except (OSError, asyncio.CancelledError):
            pass
        finally:
            with contextlib.suppress(Exception):
                dst.close()

    def stop(self) -> None:
        self._stopped.set()


class RemoteViewsFrameFlowTest(TwoClientHarness):
    """The relayed page of a slow machine link survives, at the link's pace."""

    def setUp(self) -> None:
        if not _PLAYWRIGHT_CACHE.is_dir():
            self.skipTest("Playwright browsers are not installed")
        super().setUp()
        mach.save_machines([])
        self._keepalive = (mach.LINK_PING_INTERVAL, mach.LINK_PING_TIMEOUT)
        self.remote_dir = tempfile.mkdtemp(prefix="remote_views_flow_")
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}/"
        self.remote = start_remote_daemon(
            Path(self.remote_dir) / "home", password="pw-remote",
            env={"KISS_HEADLESS": "1", "KISS_BROWSER_HOME": self.base},
        )
        host, port = self.remote.url.removeprefix("https://").split(":")
        self.target = (host, int(port))
        self.proxy: ThrottledProxy | None = None
        self.a = self.open_client("a")
        self.b = self.open_client("b")

    def _link(
        self, bytes_per_second: int, ping_interval: float, ping_timeout: float,
    ) -> ThrottledProxy:
        """Register the machine ``box`` behind a proxy carrying *bytes_per_second* to here."""
        mach.LINK_PING_INTERVAL, mach.LINK_PING_TIMEOUT = ping_interval, ping_timeout
        self.proxy = proxy = ThrottledProxy(self.target, bytes_per_second)
        self.a.send({
            "type": "machineAction", "action": "add", "name": "box",
            "address": f"https://127.0.0.1:{proxy.port}", "machinePassword": "pw-remote",
        })
        self.assertEqual(self.wait_event(self.a, "machines")["error"], "")
        return proxy

    def _open(self, width: int, height: int, *clients: Client) -> str:
        """Open the animated page on ``box`` from ``a``; *clients* show it at *width* x *height*."""
        self.a.send({"type": "browserOpen", "url": self.base, "machine": "box"})
        tab_id = str(self.wait_event(self.a, "openBrowserTab", timeout=120, focus=True)["tab_id"])
        self.assertTrue(tab_id.endswith("@box"))
        for client in clients:
            client.send({
                "type": "browserViewport", "tab_id": tab_id,
                "width": width, "height": height, "visible": True,
            })
        return tab_id

    def _ack_all(self, client: Client, tab_id: str, acked: int) -> int:
        """Acknowledge the frames *client* got since the *acked* first; the frames it has."""
        got = self._frames(client, tab_id)
        for _ in range(got - acked):
            client.send({"type": "browserFrameAck", "tab_id": tab_id})
        return got

    def tearDown(self) -> None:
        super().tearDown()
        self.remote.stop()
        if self.proxy is not None:
            self.proxy.stop()
        self.httpd.shutdown()
        self.httpd.server_close()
        web_stealth.stop_virtual_display()
        mach.LINK_PING_INTERVAL, mach.LINK_PING_TIMEOUT = self._keepalive
        mach.save_machines([])
        shutil.rmtree(self.remote_dir, ignore_errors=True)

    @staticmethod
    def _frames(client: Client, tab_id: str) -> int:
        return len(client.of_type("browserFrame", tab_id=tab_id))

    def test_the_page_of_a_slow_link_keeps_streaming(self) -> None:
        a, b = self.a, self.b
        proxy = self._link(_LINK_BYTES_PER_SECOND, _KEEPALIVE, _KEEPALIVE)
        tab_id = self._open(1200, 760, a, b)
        a.send({"type": "browserFrameAck", "tab_id": "browser__77@box"})  # unknown: ignored
        self.wait_for(lambda: self._frames(a, tab_id) > 0, timeout=60, what="a: first frame")

        acked = 0
        counts: list[int] = []
        deadline = time.monotonic() + _WATCH
        while time.monotonic() < deadline:
            acked = self._ack_all(a, tab_id, acked)
            if len(counts) < int(time.monotonic() - deadline + _WATCH):
                counts.append(acked)
            self.assertEqual(a.of_type("closeBrowserTab"), [], "the tab stayed open")
            time.sleep(0.05)

        # 1. The tab survived several keepalive rounds; frames kept coming
        #    throughout, far more than the link could carry unthrottled
        #    (the proxy saw the frames the stream let through, no backlog).
        self.assertEqual(a.of_type("closeBrowserTab"), [])
        self.assertEqual(a.of_type("browserError"), [])
        self.assertGreater(counts[-1], counts[len(counts) // 2], "frames still arriving")
        self.assertGreaterEqual(acked, 10, "the stream acknowledges, the machine keeps sending")
        self.assertLess(
            proxy.bytes_down, _LINK_BYTES_PER_SECOND * (_WATCH + 10),
            "no more crossed the link than it carries",
        )
        # 2. b never acknowledged: MAX_IN_FLIGHT per grace period at most.
        self.assertLessEqual(
            self._frames(b, tab_id), MAX_IN_FLIGHT * (int(_WATCH / ACK_GRACE) + 2),
        )
        self.assertGreaterEqual(self._frames(b, tab_id), MAX_IN_FLIGHT)
        self.assertGreater(acked, self._frames(b, tab_id) * 2, "a streamed, b was held")
        # b's acknowledgement releases the frame held back for it.
        before = self._frames(b, tab_id)
        b.send({"type": "browserFrameAck", "tab_id": tab_id})
        self.wait_for(lambda: self._frames(b, tab_id) > before, what="b: the held frame")
        self.assertIn("box", self.server._server_api.remote_views._streams)

    def test_a_link_slower_than_the_grace_keeps_the_page(self) -> None:
        # One frame of the 600x400 page (under 20 KB) takes some 4 s on
        # a 5 KB/s link, close to ACK_GRACE: were an unacknowledged frame
        # counted as delivered by age, two more would be pushed onto the
        # link every grace period, the backlog (and the pong behind it)
        # would outgrow the 12 s ping timeout, and the tab would close
        # during the watch (it did, after 25 s).  The stream acknowledges,
        # so the machine holds it to two frames in flight (8 s of link),
        # under the timeout, however slow the link.
        a = self.a
        self._link(5_000, 3.0, 12.0)
        tab_id = self._open(600, 400, a)
        self.wait_for(lambda: self._frames(a, tab_id) > 0, timeout=90, what="a: first frame")

        acked = 0
        deadline = time.monotonic() + _SLOW_WATCH
        while time.monotonic() < deadline:
            acked = self._ack_all(a, tab_id, acked)
            self.assertEqual(a.of_type("closeBrowserTab"), [], "the tab stayed open")
            time.sleep(0.05)
        self.assertEqual(a.of_type("browserError"), [])
        self.assertGreaterEqual(acked, 4, "frames kept coming at the link's pace")
        self.assertIn("box", self.server._server_api.remote_views._streams)

    def test_a_storm_of_resizes_grants_no_extra_frames(self) -> None:
        # The page here announces its viewport every 250 ms (a window
        # being dragged): each announcement reaches the machine.  Were
        # the viewer's window reset by it, the frames in flight would
        # be forgotten and two more sent each time, the link would fill
        # and the keepalive fail within seconds.
        a = self.a
        self._link(_LINK_BYTES_PER_SECOND, _KEEPALIVE, _KEEPALIVE)
        tab_id = self._open(1200, 760, a)
        self.wait_for(lambda: self._frames(a, tab_id) > 0, timeout=60, what="a: first frame")

        acked = 0
        resizes = 0
        deadline = time.monotonic() + _STORM_WATCH
        next_resize = time.monotonic()
        while time.monotonic() < deadline:
            acked = self._ack_all(a, tab_id, acked)
            if time.monotonic() >= next_resize:
                a.send({
                    "type": "browserViewport", "tab_id": tab_id,
                    "width": 1200 + resizes % 2, "height": 760, "visible": True,
                })
                resizes += 1
                next_resize += 0.25
            self.assertEqual(a.of_type("closeBrowserTab"), [], "the tab stayed open")
            time.sleep(0.05)
        self.assertGreaterEqual(resizes, _STORM_WATCH * 3)
        self.assertEqual(a.of_type("browserError"), [])
        self.assertGreaterEqual(acked, 10, "frames kept coming through the storm")
        self.assertIn("box", self.server._server_api.remote_views._streams)
