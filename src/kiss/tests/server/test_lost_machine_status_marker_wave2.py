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

Real local daemon, real remote daemon registered as ``box`` (killed
mid-task), one client on the local connection.
"""

from __future__ import annotations

from typing import Any

from kiss.tests.server.test_remote_tabs import RemoteTabsHarness


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
