# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: losing a remote machine mid-task marks the tab's terminal ``status`` with ``lost``.

When the stream of a tab whose task runs on a remote machine is lost,
the daemon ends the tab with ``status running:false`` followed by a
``prompt_refused`` error (``RemoteTabs.dropped`` -> ``_refuse_run``),
the order the frontends need to re-enable the composer.  An API client
waiting on the run (``daemon_client.run``) took that terminal status for
the task's own end, although the task may still be running on the
machine.  The status now names the lost machine (``lost: "box"``), so a
client can tell the loss from a finished task; a refusal of a run that
never started (an unknown machine) carries no marker.

A surface that was away at the loss heard none of it, and a page keeps
its tabs through an outage: it still shows the tab spinning and bound.
Its ``ready`` is answered with every binding there is (``tabMachines``:
the page drops the ones it holds that ended meanwhile) and with the run
state of each tab on no chat and no machine
(``RemoteTabs.replay_bindings``); while the tab runs on the machine its
run state stays the stream's to tell.  The same answer brings a tab
home on a surface that was away while another one ran a task of this
machine in it.

Real local daemon, real remote daemon registered as ``box`` (killed, or
its stream cut, mid-task), one client on the local connection, and the
real chat page (``media/main.js`` under jsdom) for what a surface shows.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import unittest
from pathlib import Path
from typing import Any

from kiss.tests.local_ws import open_local_connection
from kiss.tests.server.test_remote_tabs import RemoteTabsHarness

_WEBAPP_TEST_DIR = Path(__file__).resolve().parents[2] / "webapp" / "test"

# Plays the frames on stdin into a page showing the tab argv[1] and prints
# what the page shows for it: spinner, Stop button, the header's machine.
_PAGE_SCRIPT = r"""
const {makePreloadedPage, send} = require(process.argv[2] + '/simplify2_harness');
const {tabEl} = require(process.argv[2] + '/subagent_tab_helpers');
const page = makePreloadedPage(process.argv[1]);
for (const ev of JSON.parse(require('fs').readFileSync(0, 'utf8'))) send(page.win, ev);
const doc = page.win.document;
process.stdout.write(JSON.stringify({
  spinning: !!tabEl(page.win, process.argv[1]).querySelector('.chat-tab-spinner'),
  stopShown: doc.getElementById('stop-btn').style.display !== 'none',
  machine: doc.getElementById('chat-machine').value,
}));
page.close();
"""


def _page_tab(tab_id: str, frames: list[dict[str, Any]]) -> dict[str, Any]:
    """What the real chat page shows for its tab *tab_id* after the daemon's *frames*."""
    if shutil.which("node") is None:
        raise unittest.SkipTest("node is not available on PATH")
    page = subprocess.run(
        ["node", "--no-maglev", "--no-concurrent-sparkplug", "-e", _PAGE_SCRIPT,
         tab_id, str(_WEBAPP_TEST_DIR)],
        input=json.dumps(frames), capture_output=True, text=True, timeout=120,
        cwd=str(_WEBAPP_TEST_DIR.parent),
    )
    assert page.returncode == 0, page.stderr
    shown: dict[str, Any] = json.loads(page.stdout)
    return shown


# What its transport tells a page whose daemon is there, and one whose daemon went away.
_UP = {"type": "daemonStatus", "connected": True}
_DOWN = {"type": "daemonStatus", "connected": False, "reconnecting": True}


def _told(frames: list[dict[str, Any]], tab_id: str) -> list[tuple[str, Any]]:
    """What *frames* tell of *tab_id*: its run state and its machine (``""`` for none).

    The machine is told by a ``tabMachine`` of the tab, or by a
    ``tabMachines`` (the answer to a ``ready``), which names every bound tab.
    """
    told: list[tuple[str, Any]] = []
    for f in frames:
        if f.get("type") == "tabMachines":
            told.append(("tabMachines", f["machines"].get(tab_id, "")))
        elif f.get("tabId") == tab_id and f.get("type") == "status":
            told.append(("status", f.get("running")))
        elif f.get("tabId") == tab_id and f.get("type") == "tabMachine":
            told.append(("tabMachine", f.get("machine")))
    return told


def _status_before_error(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """The last ``status`` frame among *frames* before the first ``error``."""
    status: dict[str, Any] = {}
    for frame in frames:
        if frame.get("type") == "error":
            return status
        if frame.get("type") == "status":
            status = frame
    raise AssertionError(f"no error frame in {frames}")


class TestLostMachineStatusMarker(RemoteTabsHarness):
    """The terminal status of a lost stream says which machine was lost."""

    async def asyncSetUp(self) -> None:
        self.frames: list[dict[str, Any]] = []  # every frame the surface online got
        await super().asyncSetUp()

    async def _next(self, timeout: float = 90.0) -> dict[str, Any]:
        msg = await super()._next(timeout)
        self.frames.append(msg)
        return msg

    async def _frames_on_ready(self) -> list[dict[str, Any]]:
        """Every frame a surface connecting now gets on its ``ready``.

        Its ``ready`` answers one ``inputHistory``; the second, to its
        own request, comes once the ``ready`` is answered in full.
        """
        reader, writer = await open_local_connection(self.server)
        try:
            for name in ("ready", "getInputHistory"):
                writer.write(json.dumps({"type": name}).encode() + b"\n")
            await writer.drain()
            frames: list[dict[str, Any]] = []
            while [f["type"] for f in frames].count("inputHistory") < 2:
                frames.append(json.loads(await asyncio.wait_for(reader.readline(), 90)))
            return frames
        finally:
            writer.close()

    async def test_a_surface_away_at_the_loss_is_told_the_tab_stopped_on_no_machine(self) -> None:
        tab = "tab-lost-away"
        await self._fan_out(tab)  # the task is still running on box
        # Running there, the tab is its stream's to tell: bound, running.
        self.assertEqual(
            _told(await self._frames_on_ready(), tab), [("tabMachines", "box"), ("status", True)],
        )
        before = list(self.frames)  # what a surface going away now has seen
        stream = self.server._server_api.remote_tabs._tabs[tab]
        await asyncio.to_thread(stream._ws.close)  # lost here; the machine runs on
        while (await self._until("tabMachine", tab))["machine"]:
            pass  # the surface online has been told all of the loss
        away = await self._frames_on_ready()
        self.assertEqual(_told(away, tab), [("tabMachines", ""), ("status", False)], away)
        # To that connection alone: the surface online is told nothing twice.
        mark = len(self.frames)
        await self._ask({"type": "getInputHistory"}, "inputHistory")
        self.assertEqual(_told(self.frames[mark:], tab), [])
        online = await asyncio.to_thread(_page_tab, tab, self.frames)
        self.assertEqual(online, {"spinning": False, "stopShown": False, "machine": ""})
        returned = await asyncio.to_thread(_page_tab, tab, [_UP, *before, _DOWN, _UP, *away])
        self.assertEqual(returned, online, "the surface that was away shows another state")

    async def test_a_surface_away_when_the_tab_came_home_shows_it_on_this_machine(self) -> None:
        tab = "tab-home-away"
        await self._submit(tab, "Say hello", "box")
        await self._run_until_idle(tab)  # ran on box: the tab stays bound to it
        before = list(self.frames)  # what a surface going away now has seen
        await self._submit(tab, "Say hello here", "")  # this machine: the tab comes home
        await self._run_until_idle(tab)
        away = await self._frames_on_ready()
        # Its machine first: the chat's replay is rendered on the machine the tab is on.
        self.assertEqual(_told(away, tab), [("tabMachines", ""), ("status", False)], away)
        online = await asyncio.to_thread(_page_tab, tab, self.frames)
        self.assertEqual(online, {"spinning": False, "stopShown": False, "machine": ""})
        returned = await asyncio.to_thread(_page_tab, tab, [_UP, *before, _DOWN, _UP, *away])
        self.assertEqual(returned, online, "the surface that was away shows another state")

    async def test_a_lost_stream_marks_the_terminal_status(self) -> None:
        tab = "tab-lost-marker"
        await self._fan_out(tab)  # the task is still running on box
        self.remote.stop()
        frames: list[dict[str, Any]] = []
        for _ in range(400):
            msg = await self._next()
            if msg.get("tabId") != tab:
                continue
            frames.append(msg)
            if msg.get("type") == "tabMachine" and msg.get("machine") == "":
                break
        else:
            raise AssertionError("the tab was never unbound")
        errors = [f for f in frames if f.get("type") == "error"]
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("lost the connection to box", errors[0]["text"])
        status = _status_before_error(frames)
        self.assertIs(status.get("running"), False, frames)
        self.assertEqual(status.get("lost"), "box", frames)

    async def test_a_run_that_never_started_is_refused_without_the_marker(self) -> None:
        tab = "tab-unknown-machine"
        await self._submit(tab, "Say hello", "nowhere")
        frames: list[dict[str, Any]] = []
        for _ in range(100):
            msg = await self._next()
            if msg.get("tabId") != tab:
                continue
            frames.append(msg)
            if msg.get("type") == "error":
                break
        self.assertEqual(frames[-1].get("code"), "prompt_refused", frames)
        status = _status_before_error(frames)
        self.assertIs(status.get("running"), False, frames)
        self.assertNotIn("lost", status)
