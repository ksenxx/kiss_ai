# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 V8 F2: a ``browserTabs`` snapshot never contradicts a close it follows.

The snapshot a ``browserList`` (or a ``ready``) answers was captured on
the command thread and sent later, while the page's close ran on the
browser loop in between: the asking surface got ``closeBrowserTab``
and then a snapshot still listing the page, took the snapshot as
authoritative and reopened the tab, while every other surface had
closed it.  The snapshot is now captured and sent in one step of the
browser loop, the thread every open and close is announced from, so
it either lists a page and precedes its close, or omits it and follows.

A real daemon, two real local clients and a real headless Chromium.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import web_stealth
from kiss.server.web_server import WebPrinter
from kiss.tests.server.pause import Pause
from kiss.tests.server.two_client_harness import Client, TwoClientHarness

_PLAYWRIGHT_CACHE = Path.home() / ".cache" / "ms-playwright"


def _folded_tabs(client: Client) -> list[str]:
    """The browser tabs *client* holds after its wire events, as the webapp reconciles them."""
    tabs: set[str] = set()
    for ev in list(client.events):
        if ev.get("type") == "openBrowserTab":
            tabs.add(ev["tab_id"])
        elif ev.get("type") == "closeBrowserTab":
            tabs.discard(ev["tab_id"])
        elif ev.get("type") == "browserTabs" and not ev.get("remote"):
            tabs = {e["tab_id"] for e in ev["tabs"]}
    return sorted(tabs)


class BrowserSnapshotAgreesWithCloseTest(TwoClientHarness):
    """Both surfaces end with the daemon's page set, whichever order the race lands in."""

    def setUp(self) -> None:
        if not _PLAYWRIGHT_CACHE.is_dir():
            self.skipTest("Playwright browsers are not installed")
        self._saved_headless = os.environ.get("KISS_HEADLESS")
        os.environ["KISS_HEADLESS"] = "1"
        super().setUp()
        self.service = self.server._agent_server.browser_tabs

    def tearDown(self) -> None:
        self.service.shutdown()
        web_stealth.stop_virtual_display()
        if self._saved_headless is None:
            os.environ.pop("KISS_HEADLESS", None)
        else:
            os.environ["KISS_HEADLESS"] = self._saved_headless
        super().tearDown()

    def test_close_landing_inside_a_snapshot_leaves_no_ghost_tab(self) -> None:
        a = self.open_client("a")
        b = self.open_client("b")
        a.send({"type": "browserOpen", "url": "about:blank"})
        tab = self.wait_event(a, "openBrowserTab")["tab_id"]
        self.wait_event(b, "openBrowserTab", tab_id=tab)

        def is_snapshot(frame_locals: dict[str, Any]) -> bool:
            event = frame_locals.get("event")
            return isinstance(event, dict) and event.get("type") == "browserTabs"

        # Held where the captured snapshot is handed to the printer.
        with Pause(WebPrinter.broadcast.__code__, is_snapshot) as hold:
            a.send({"type": "browserList"})
            self.assertTrue(hold.paused.wait(30), "the snapshot was never captured")
            b.send({"type": "browserClose", "tab_id": tab})
            # Lands inside the window when the snapshot was captured off
            # the browser loop; a snapshot captured on the loop holds
            # the close back until the release, and this wait runs out.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and tab in self.service._pages:
                time.sleep(0.05)
            hold.release()
            self.wait_event(a, "browserTabs")
            self.wait_event(a, "closeBrowserTab", tab_id=tab)
            self.wait_event(b, "closeBrowserTab", tab_id=tab)
        self.wait_for(lambda: not self.service.open_events(), what="daemon forgets the tab")
        self.assertEqual(_folded_tabs(b), [], "the closing surface dropped the tab")
        lifecycle = ("openBrowserTab", "closeBrowserTab", "browserTabs")
        wire_order = [e["type"] for e in list(a.events) if e.get("type") in lifecycle]
        self.assertEqual(
            _folded_tabs(a), [], f"the asking surface kept a ghost tab; wire order: {wire_order}",
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
