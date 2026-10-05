# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: a terminal opened from the ... menu is a tab on EVERY surface.

A real daemon (``DaemonLocalHarness``) and real chat webviews
(``media/main.js`` + ``terminalTab.js`` + the vendored xterm.js under
jsdom, each on its own daemon connection through
``multiSurfaceBridge.js``) exercise the whole path: the "Terminal" item
of the composer's ... menu, ``kiss/server/terminal_tab.py`` running the
user's shell in a pseudo-terminal, and the tab plumbing in ``main.js``.

Checked, in order:

1. Clicking "Terminal" on one surface opens the tab on every connected
   surface; only the surface that clicked switches to it.
2. Output reaches the surfaces showing the tab; a surface that never
   showed it has an empty terminal until the user switches to it there,
   at which point it receives the scrollback kept by the daemon.
3. A surface connecting MID-SESSION gets the tab from its ``ready``
   snapshot, unfocused, and the scrollback once it shows the tab.
4. A resize from one surface changes the pty's size (``stty size``).
5. A surface whose connection drops and comes back catches up with
   what the shell printed while it was away.
6. ``exit`` in the shell closes the tab on ALL surfaces.
7. Closing the tab by hand on ONE surface closes it on ALL and ends the
   shell; a surface connecting afterwards sees no terminal tab.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from functools import partial
from typing import Any, cast

from kiss.tests.server.test_run_agent_subagent_tab import DaemonLocalHarness
from kiss.tests.server.test_subagent_tabs_all_surfaces import (
    _JSDOM_PKG,
    SurfaceBridge,
)


def _wait(pred: Callable[[], Any], what: str, timeout: float = 30) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = pred()
        if last:
            return last
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what} (last={last!r})")


class TerminalTabAllSurfacesTest(DaemonLocalHarness):
    """Open everywhere while the shell lives, closed everywhere when it ends."""

    def setUp(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not available on PATH")
        if not _JSDOM_PKG.is_file():
            self.skipTest("jsdom is not installed under agents/vscode")
        if os.name == "nt":
            self.skipTest("terminal tabs need a Unix pseudo-terminal")
        super().setUp()
        self.bridge = SurfaceBridge(str(self.endpoint_file))

    def tearDown(self) -> None:
        self.bridge.quit()
        self.server._vscode_server.terminal_tabs.shutdown()
        super().tearDown()

    # -- helpers -----------------------------------------------------------

    def _announcements(self, surface: str) -> list[dict[str, Any]]:
        """Every ``openTerminalTab`` *surface* received, live or via a ``terminalTabs`` snapshot."""
        found: list[dict[str, Any]] = []
        for e in self.bridge.events(surface):
            if not isinstance(e, dict):
                continue
            if e["type"] == "openTerminalTab":
                found.append(e)
            elif e["type"] == "terminalTabs":
                found.extend(e["tabs"])
        return found

    def _terminal_tab_ids(self, surface: str) -> list[str]:
        ids: list[str] = []
        for e in self._announcements(surface):
            if e["tab_id"] not in ids:
                ids.append(e["tab_id"])
        return ids

    def _info(self, surface: str, tab_id: str) -> dict[str, Any]:
        reply = self.bridge.call("terminal", name=surface, tabId=tab_id)
        assert reply["op"] == "terminal", reply
        return dict(reply["info"])

    def _has_tab(self, surface: str, tab_id: str) -> bool:
        return tab_id in self._terminal_tab_ids(surface)

    def _in_tab_bar(self, surface: str, tab_id: str) -> bool:
        return bool(self._info(surface, tab_id)["inTabBar"])

    def _gone(self, surface: str, tab_id: str) -> bool:
        return not self._in_tab_bar(surface, tab_id)

    def _wait_text(self, surface: str, tab_id: str, fragment: str) -> dict[str, Any]:
        """Poll the xterm buffer on *surface* until it shows *fragment*."""

        def _probe() -> dict[str, Any] | None:
            info = self._info(surface, tab_id)
            return info if fragment in info["text"] else None

        return cast("dict[str, Any]", _wait(_probe, f"{surface}: terminal shows {fragment!r}"))

    def _type(self, surface: str, tab_id: str, text: str) -> None:
        reply = self.bridge.call("terminalType", name=surface, tabId=tab_id, text=text)
        self.assertTrue(reply["found"], f"{surface} has no terminal view for {tab_id}")

    def _show(self, surface: str, tab_id: str) -> None:
        reply = self.bridge.call("click", name=surface, selector=f'#tab-list [data-tab-id="{tab_id}"]')
        self.assertTrue(reply["found"], f"{surface}: no tab strip entry for {tab_id}")

    def _assert_no_webview_errors(self, *surfaces: str) -> None:
        for name in surfaces:
            self.bridge.tabs(name)  # raises on webview errors

    # -- the test ------------------------------------------------------------

    def test_terminal_tab_open_everywhere_closed_everywhere(self) -> None:
        bridge = self.bridge
        service = self.server._vscode_server.terminal_tabs
        bridge.open("remote1", ' class="remote-chat"')
        bridge.open("sidebar")

        # 1. "Terminal" in the ... menu of ONE surface.
        reply = bridge.call("click", name="remote1", selector="#terminal-btn")
        self.assertTrue(reply["found"], "chat.html has no #terminal-btn in the ... menu")
        tab_id = str(
            _wait(
                lambda: next(iter(self._terminal_tab_ids("remote1")), None),
                "openTerminalTab on remote1",
            )
        )
        _wait(partial(self._has_tab, "sidebar", tab_id), "openTerminalTab on the sidebar")
        for name in ("remote1", "sidebar"):
            info = self._info(name, tab_id)
            self.assertTrue(info["inTabBar"] and info["isTerminalTab"], info)
            self.assertTrue(info["title"].startswith("Terminal"), info["title"])
        self.assertTrue(self._info("remote1", tab_id)["visible"], "the clicking surface switches")
        self.assertTrue(self._info("remote1", tab_id)["hasXterm"], "xterm opened on remote1")
        sidebar = self._info("sidebar", tab_id)
        self.assertFalse(sidebar["visible"], "other surfaces stay on their chat")
        self.assertFalse(sidebar["hasXterm"], "a hidden tab does not attach")
        self.assertFalse(
            any(e["focus"] for e in self._announcements("sidebar")),
            "only the surface that clicked is asked to switch",
        )
        [term] = service._terms.values()
        self.assertEqual(term.cwd, str(self.repo), "the shell starts in the chat's folder")

        # 2. Output reaches remote1 (attached); the sidebar's terminal
        #    stays empty until the user switches to the tab there.
        self._type("remote1", tab_id, "echo marker-$((40+2))\r")
        self._wait_text("remote1", tab_id, "marker-42")
        time.sleep(0.5)
        self.assertEqual(self._info("sidebar", tab_id)["text"], "")
        self._show("sidebar", tab_id)
        self._wait_text("sidebar", tab_id, "marker-42")
        self.assertTrue(self._info("sidebar", tab_id)["hasXterm"])

        # 3. A surface connecting mid-session gets the tab from `ready`,
        #    unfocused, and the scrollback once it shows the tab.
        bridge.open("remote2", ' class="remote-chat"')
        _wait(partial(self._has_tab, "remote2", tab_id), "replay on remote2")
        late = self._info("remote2", tab_id)
        self.assertTrue(late["inTabBar"] and late["isTerminalTab"], late)
        self.assertFalse(late["visible"], "a replayed terminal tab must not steal focus")
        self.assertFalse(any(e["focus"] for e in self._announcements("remote2")))
        self._show("remote2", tab_id)
        self._wait_text("remote2", tab_id, "marker-42")

        # 4. A resize from one surface sizes the pty; the shell sees it.
        bridge.call(
            "post",
            name="remote2",
            msg={"type": "terminalResize", "tab_id": tab_id, "cols": 100, "rows": 30},
        )
        _wait(lambda: (term.cols, term.rows) == (100, 30), "pty resized")
        self._type("remote2", tab_id, "stty size\r")
        for name in ("remote1", "sidebar", "remote2"):
            self._wait_text(name, tab_id, "30 100")

        # 5. remote1 drops its connection; the shell prints while it is
        #    away; on reconnect remote1 re-attaches and catches up.
        reply = bridge.call("disconnect", name="remote1")
        self.assertEqual(reply["op"], "disconnected")
        self._type("sidebar", tab_id, "echo while-$((100+1))-away\r")
        self._wait_text("sidebar", tab_id, "while-101-away")
        time.sleep(0.3)
        self.assertNotIn("while-101-away", self._info("remote1", tab_id)["text"])
        reply = bridge.call("reconnect", name="remote1")
        self.assertEqual(reply["op"], "reconnected")
        caught_up = self._wait_text("remote1", tab_id, "while-101-away")
        self.assertIn("marker-42", caught_up["text"], "the replay carries the earlier scrollback")
        self.assertEqual(caught_up["text"].count("marker-42"), 1, "nothing is painted twice")

        # 6. `exit` closes the tab on every surface.
        self._type("remote2", tab_id, "exit\r")
        for name in ("remote1", "sidebar", "remote2"):
            _wait(partial(self._gone, name, tab_id), f"terminal tab closed on {name}")
        _wait(lambda: not service.open_events(), "daemon forgets the terminal")

        # 7. A second terminal, closed by hand on ONE surface, closes on
        #    ALL and its shell is gone.
        reply = bridge.call("click", name="remote2", selector="#terminal-btn")
        self.assertTrue(reply["found"])
        second = str(
            _wait(
                lambda: next((t for t in self._terminal_tab_ids("remote2") if t != tab_id), None),
                "second terminal on remote2",
            )
        )
        for name in ("remote1", "sidebar"):
            _wait(partial(self._has_tab, name, second), f"second terminal on {name}")
        self.assertTrue(self._info("remote2", second)["visible"])
        self.assertTrue(self._info("remote2", second)["title"].startswith("Terminal 2"))
        proc = service._terms[second].proc
        reply = bridge.call("closeTab", name="sidebar", tabId=second)
        self.assertTrue(reply["found"])
        for name in ("remote1", "sidebar", "remote2"):
            _wait(partial(self._gone, name, second), f"second terminal closed on {name}")
        _wait(lambda: proc.poll() is not None, "the shell ended", timeout=10)
        _wait(lambda: not service.open_events(), "daemon forgets the second terminal")
        bridge.open("remote3", ' class="remote-chat"')
        time.sleep(0.5)
        self.assertEqual(self._terminal_tab_ids("remote3"), [])
        self._assert_no_webview_errors("remote1", "sidebar", "remote2", "remote3")
