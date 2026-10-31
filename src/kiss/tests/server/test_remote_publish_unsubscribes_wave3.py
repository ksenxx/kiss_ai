# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a tab bound to a machine stops streaming the local task it watched.

A tab watching a chat whose task runs here (opened from the CHATS
panel while another tab runs it) is subscribed to that task's stream.
Submitting in that tab with a machine picked binds it to the machine:
the tab shows the machine's chat from then on, so the local task's
later events must not reach it any more, interleaved with the
machine's.  The bind's publication dropped the tab's chat view and
task pin but left its printer subscriptions, and the local run's
result streamed into the bound tab.

Real daemons (this one, its model a local scripted endpoint that holds
the local run until the test releases it, and a remote one registered
as machine "box"), real client connection; no mocks, patches or test
doubles.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from kiss.core import user_config as vc
from kiss.tests.server.parallel_agent_harness import StandInModelServer, finish_response
from kiss.tests.server.test_remote_tab_binding_sweep1030 import _BindHarness, _restore_config

LOCAL_SUMMARY = "<p>local run result 7c1e</p>"


class BoundTabLeavesTheLocalStreamTest(_BindHarness):
    """A viewer of a running local task, then bound to a machine."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.standin = StandInModelServer(self._hold_then_finish)
        self.addCleanup(self.standin.stop)
        config_path = vc._config_path()
        self.addCleanup(
            _restore_config, config_path.read_text() if config_path.exists() else None,
        )
        vc.save_config({
            "custom_endpoint": self.standin.url,
            "custom_api_key": "k",
            "max_budget": 100,
        })

    async def asyncTearDown(self) -> None:
        self.release.set()  # a run still held must not hold up the daemon's stop
        await super().asyncTearDown()

    def _hold_then_finish(self, request: dict[str, Any]) -> dict[str, Any]:
        """Hold the local run's agent call until released, then finish it."""
        if not request.get("tools"):  # a task classifier's call, not the agent loop's
            return finish_response("classified")
        self.entered.set()
        self.release.wait(60)
        return finish_response(LOCAL_SUMMARY)

    async def test_the_local_run_does_not_stream_into_the_bound_tab(self) -> None:
        runner, viewer = "tab-runs-locally", "tab-watches-then-binds"
        printer = self.server._printer
        await self._submit(runner, "Work here", "")
        self.assertTrue(await asyncio.to_thread(self.entered.wait, 60), "the local run started")
        await self._poll("the runner's chat", lambda: (self._row(runner) or {}).get("chatId"))
        chat_id = self._row(runner)["chatId"]  # type: ignore[index]

        # The viewer opens the running chat: it streams the local task.
        await self._send({"type": "resumeSession", "chatId": chat_id, "tabId": viewer})
        await self._poll("the viewer's subscription", lambda: printer.tasks_for_tab(viewer))

        await self._submit(viewer, "Say hello", "box")
        await self._poll(
            "the bind", lambda: self.server._server_api.remote_tabs.machine_of(viewer) == "box",
        )
        await self._bind_settled(viewer)

        # The local run goes on: none of its frames reach the bound tab.
        self.release.set()
        leaked: list[dict[str, Any]] = []
        for _ in range(2000):
            msg = await self._next()
            if msg.get("tabId") == viewer and LOCAL_SUMMARY in json.dumps(msg):
                leaked.append(msg)
            if (
                msg.get("tabId") == runner
                and msg.get("type") == "status"
                and not msg.get("running")
                and not msg.get("replayId")
            ):
                break
        else:
            raise AssertionError("the local run never ended")
        self.assertEqual(leaked, [], "the local run's result reached the bound tab")
        self.assertEqual(printer.tasks_for_tab(viewer), [], "the bound tab watches no local task")
