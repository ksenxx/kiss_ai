# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a browser tab's frames are flow-controlled per client (``browserFrameAck``).

The real daemon on its local WSS endpoint with two real client
connections (``two_client_harness``) and a real headless Chromium on a
page that repaints every animation frame, so the screencast produces
frames continuously.  Checked on the wire:

1. A client that acknowledges every frame keeps receiving them; one
   that acknowledges none gets :data:`~kiss.server.frame_flow.MAX_IN_FLIGHT`
   frames and then waits: the daemon holds the newest frame back for it.
2. A client that never acknowledges is not starved: after
   :data:`~kiss.server.frame_flow.ACK_GRACE` seconds its frames count
   as delivered and the frame held back for it goes out by itself.
3. Once a client has acknowledged, an acknowledgement releases exactly
   one frame (the held one), and the client is held to its window for
   good: silent again, it gets nothing more, however long (no grace).
4. A resize (the viewport announced again, visible) grants nothing: the
   frames in flight stay on the client's account; acknowledging them
   streams on.
5. An acknowledgement for a tab that does not exist is ignored, and one
   from a client that stopped showing the tab sends nothing.
6. A client that never acknowledged and stops showing the tab while a
   grace release is scheduled for it leaves nothing behind: once the
   grace has run out the page accounts for its current viewers only.
"""

from __future__ import annotations

import http.server
import os
import threading
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import web_stealth
from kiss.server.frame_flow import ACK_GRACE, MAX_IN_FLIGHT
from kiss.tests.server.two_client_harness import Client, TwoClientHarness

_PLAYWRIGHT_CACHE = Path.home() / ".cache" / "ms-playwright"

_PAGE = b"""<!doctype html><title>Animated</title>
<canvas id=c width=380 height=280></canvas>
<script>
const c = document.getElementById('c').getContext('2d'); let t = 0;
function f() {
  t++;
  for (let i = 0; i < 30; i++) {
    c.fillStyle = 'hsl(' + ((t * 7 + i * 37) % 360) + ',80%,50%)';
    c.fillRect((i * 53 + t * 5) % 380, (i * 97 + t * 3) % 280, 90, 60);
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


class BrowserFrameFlowTest(TwoClientHarness):
    """Frames flow at the pace each client acknowledges them."""

    def setUp(self) -> None:
        if not _PLAYWRIGHT_CACHE.is_dir():
            self.skipTest("Playwright browsers are not installed")
        self._saved_env = {
            k: os.environ.get(k) for k in ("KISS_HEADLESS", "KISS_BROWSER_HOME")
        }
        os.environ["KISS_HEADLESS"] = "1"
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        os.environ["KISS_BROWSER_HOME"] = f"http://127.0.0.1:{self.httpd.server_address[1]}/"
        super().setUp()
        self.service = self.server._agent_server.browser_tabs

    def tearDown(self) -> None:
        self.service.shutdown()
        web_stealth.stop_virtual_display()
        self.httpd.shutdown()
        self.httpd.server_close()
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        super().tearDown()

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _frames(client: Client, tab_id: str) -> int:
        return len(client.of_type("browserFrame", tab_id=tab_id))

    @staticmethod
    def _show(client: Client, tab_id: str, visible: bool = True, width: int = 400) -> None:
        client.send({
            "type": "browserViewport", "tab_id": tab_id,
            "width": width, "height": 300, "visible": visible,
        })

    def _ack_all(self, client: Client, tab_id: str, seconds: float) -> int:
        """Acknowledge every frame *client* gets for *seconds*; the frames received."""
        acked = 0
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            got = self._frames(client, tab_id)
            for _ in range(got - acked):
                client.send({"type": "browserFrameAck", "tab_id": tab_id})
            acked = got
            time.sleep(0.02)
        return acked

    def _settled(self, client: Client, tab_id: str, seconds: float) -> int:
        """The frames *client* has after *seconds* more without acknowledging any."""
        time.sleep(seconds)
        return self._frames(client, tab_id)

    # -- the test ------------------------------------------------------------

    def test_frames_wait_for_the_client_s_acknowledgements(self) -> None:
        a = self.open_client("a")
        b = self.open_client("b")
        a.send({"type": "browserOpen", "url": ""})
        tab_id = str(self.wait_event(a, "openBrowserTab", timeout=120, focus=True)["tab_id"])
        self.wait_event(a, "browserState", timeout=60, tab_id=tab_id, title="Animated")
        self._show(a, tab_id)
        self._show(b, tab_id)

        # 1. a acknowledges and streams on; b is silent and gets MAX_IN_FLIGHT.
        a_frames = self._ack_all(a, tab_id, 3.0)
        self.assertGreaterEqual(a_frames, 10, "an acknowledging client streams")
        self.assertEqual(self._frames(b, tab_id), MAX_IN_FLIGHT, "a silent client is held")

        # 2. b never acknowledged: after ACK_GRACE its frames count as
        #    delivered and the frame held back for it goes out by itself.
        self.wait_for(
            lambda: self._frames(b, tab_id) > MAX_IN_FLIGHT,
            timeout=ACK_GRACE + 5, what="b: the held frame after the grace",
        )

        # 3. One acknowledgement releases exactly the frame held back, and
        #    b, having acknowledged, is held to its window from now on:
        #    silent again, nothing more comes, grace or no grace.
        before = self._settled(b, tab_id, 1.0)
        b.send({"type": "browserFrameAck", "tab_id": tab_id})
        self.wait_for(lambda: self._frames(b, tab_id) == before + 1, what="b: the held frame")
        self.assertEqual(self._settled(b, tab_id, ACK_GRACE + 1.0), before + 1)

        # 4. A resize grants nothing: the frames in flight stay on b's
        #    account; acknowledging them streams on.
        self._show(b, tab_id, width=401)
        self.assertEqual(self._settled(b, tab_id, 1.0), before + 1)
        self.assertGreater(self._ack_all(b, tab_id, 1.0), before + 3, "b streams once it acks")

        # 5. Unknown tab: ignored.  A hidden tab: the ack sends nothing.
        a.send({"type": "browserFrameAck", "tab_id": "browser__999"})
        self._show(b, tab_id, visible=False)
        time.sleep(0.5)
        hidden_at = self._frames(b, tab_id)
        b.send({"type": "browserFrameAck", "tab_id": tab_id})
        self.assertEqual(self._settled(b, tab_id, 1.0), hidden_at)
        self.assertEqual(a.of_type("browserError"), [])
        self.assertEqual(b.of_type("browserError"), [])
        self.assertGreater(self._ack_all(a, tab_id, 1.0), a_frames, "a still streams")

        # 6. c never acknowledges: MAX_IN_FLIGHT frames, then one is held and
        #    its grace release scheduled.  Hidden before the grace runs out,
        #    c is forgotten, and the release finds nothing to account for.
        c = self.open_client("c")
        self._show(c, tab_id)
        self.wait_for(lambda: self._frames(c, tab_id) == MAX_IN_FLIGHT, what="c: its window")
        time.sleep(0.5)  # a frame is held back for c now, its release scheduled
        self._show(c, tab_id, visible=False)
        time.sleep(ACK_GRACE + 1.0)
        rec = self.server._agent_server.browser_tabs._pages[tab_id]
        self.assertEqual(len(rec.viewers), 1, "a is the one viewer left")
        self.assertEqual(set(rec.flow._in_flight), set(rec.viewers))
        self.assertEqual(set(rec.flow._held) - set(rec.viewers), set())
        self.assertEqual(self._frames(c, tab_id), MAX_IN_FLIGHT)
