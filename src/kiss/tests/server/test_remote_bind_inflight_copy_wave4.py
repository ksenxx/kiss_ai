# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a local task's copy in flight never reaches a tab after the tab's bind to a machine.

A tab watching a task that runs here (opened from the CHATS panel while
another tab runs it) is bound to a machine by a submit naming it; the
bind's publication drops the tab's printer subscriptions
(``RemoteTabs._publish`` → ``cleanup_tab``).  A transient broadcast of
the local task (``worktree_created``, ``launch_phase``, ``askUserDone``)
and a ``talk`` fan-out resolved their watching tabs first and sent one
copy per tab afterwards, outside the lock that orders deliveries.  A
copy resolved for the tab before the retirement but sent after it
reached the bound tab after its ``tabMachine`` binding, interleaved
with the machine's chat, and a leaked ``worktree_created`` put the
local worktree back as the bound tab's file-resolution fallback.

Real daemons (this one, its model a local scripted endpoint that holds
the local run, and a remote one registered as machine "box"), real
client connection.  The copy is held at the entry of the printer's
delivery by a ``sys.settrace`` hook (nothing is replaced) while the
bind runs; no mocks, patches or test doubles.
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from collections.abc import Callable
from typing import Any

from kiss.core import user_config as vc
from kiss.server.web_server import WebPrinter
from kiss.tests.server.parallel_agent_harness import StandInModelServer, finish_response
from kiss.tests.server.pause import Pause, Pauses
from kiss.tests.server.test_remote_tab_binding_sweep1030 import _BindHarness, _restore_config

RUNNER, VIEWER = "tab-runs-locally", "tab-watches-then-binds"
LOCAL_TREE = "/w4-local-worktree-5d1a"
LOCAL_SPEECH = "spoken by the local task 3b9e"


def _say_from_task(printer: WebPrinter, task_key: str) -> None:
    """Broadcast a ``talk`` event from a thread running the task *task_key*."""
    printer.adopt_thread_binding(task_key)
    printer.broadcast({"type": "talk", "text": LOCAL_SPEECH, "talkId": "w4-talk"})


def _viewer_worktree_copy(args: dict[str, Any]) -> bool:
    """The viewer's copy of the local ``worktree_created`` entering ``broadcast``."""
    event = args.get("event") or {}
    return event.get("type") == "worktree_created" and event.get("tabId") == VIEWER


def _viewer_talk_copy(args: dict[str, Any]) -> bool:
    """The viewer's copy of the local ``talk`` entering ``_send_to_ws_clients``."""
    data = args.get("data")
    return args.get("tab_id") == VIEWER and isinstance(data, str) and LOCAL_SPEECH in data


class BindRacesALocalCopyTest(_BindHarness):
    """A viewer of a running local task bound to a machine while a copy for it is in flight."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cleanup_entered = threading.Event()
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
        self.release.set()  # the held local run must not hold up the daemon's stop
        await super().asyncTearDown()

    def _hold_then_finish(self, request: dict[str, Any]) -> dict[str, Any]:
        """Hold the local run's agent call until released, then finish it."""
        if not request.get("tools"):  # a task classifier's call, not the agent loop's
            return finish_response("classified")
        self.entered.set()
        self.release.wait(60)
        return finish_response("<p>local run result</p>")

    def _note_cleanup(self, args: dict[str, Any]) -> bool:
        """Note the bind retiring the viewer's subscriptions; never hold the thread."""
        if args.get("tab_id") == VIEWER:
            self.cleanup_entered.set()
        return False

    def _release_after_the_bind(self, hold: Pause) -> None:
        """Let the held copy go once the bind settled, or after it waited 2 s on the copy.

        Runs on a worker thread: the daemon shares the test's event
        loop, and a bind waiting on the held copy holds locks the loop
        may need.
        """
        self.cleanup_entered.wait(60)
        remote_tabs = self.server._server_api.remote_tabs
        deadline = time.monotonic() + 2.0
        while VIEWER in remote_tabs._connecting and time.monotonic() < deadline:
            time.sleep(0.02)
        hold.release()

    async def _watch_the_local_run(self) -> str:
        """Run a task here in RUNNER, open its chat in VIEWER; return the task's key."""
        printer = self.server._printer
        await self._submit(RUNNER, "Work here", "")
        self.assertTrue(await asyncio.to_thread(self.entered.wait, 60), "the local run started")
        await self._poll("the runner's chat", lambda: (self._row(RUNNER) or {}).get("chatId"))
        chat_id = self._row(RUNNER)["chatId"]  # type: ignore[index]
        await self._send({"type": "resumeSession", "chatId": chat_id, "tabId": VIEWER})
        await self._poll("the viewer's subscription", lambda: printer.tasks_for_tab(VIEWER))
        return printer.tasks_for_tab(VIEWER)[0]

    async def _bind_with_a_copy_in_flight(
        self, hold: Pause, send: Callable[[], None],
    ) -> list[dict[str, Any]]:
        """Bind VIEWER to "box" while *hold* parks the viewer's copy of what *send* broadcasts.

        Returns:
            The viewer's frames up to the end of the machine's run.
        """
        cleanup = Pause(WebPrinter.cleanup_tab.__code__, self._note_cleanup)
        with Pauses(hold, cleanup):
            sender = threading.Thread(target=send, daemon=True)
            sender.start()
            self.assertTrue(
                await asyncio.to_thread(hold.paused.wait, 60), "the viewer's copy is in flight",
            )
            await self._submit(VIEWER, "Say hello", "box")
            await asyncio.to_thread(self._release_after_the_bind, hold)
            await asyncio.to_thread(sender.join, 60)
        self.assertTrue(self.cleanup_entered.is_set(), "the bind retired the viewer's watch")
        await self._bind_settled(VIEWER)
        return await self._run_until_idle(VIEWER)

    def _assert_before_the_binding(
        self, frames: list[dict[str, Any]], local: list[int],
    ) -> None:
        """The local copies (indexes *local* of *frames*) came before the ``tabMachine`` frame."""
        binding = next(
            i for i, f in enumerate(frames)
            if f.get("type") == "tabMachine" and f.get("machine") == "box"
        )
        self.assertTrue(local, "the copy resolved before the bind still reached the tab")
        self.assertLess(max(local), binding, "a local copy reached the tab after its binding")
        self.assertEqual(self.server._printer.tasks_for_tab(VIEWER), [])

    async def test_a_transient_copy_lands_before_the_binding(self) -> None:
        printer = self.server._printer
        task_key = await self._watch_the_local_run()
        event = {"type": "worktree_created", "worktreeDir": LOCAL_TREE, "branch": "kiss/w4"}
        frames = await self._bind_with_a_copy_in_flight(
            Pause(WebPrinter.broadcast.__code__, _viewer_worktree_copy),
            functools.partial(printer.broadcast_transient, event, task_id=task_key),
        )
        local = [i for i, f in enumerate(frames) if f.get("worktreeDir") == LOCAL_TREE]
        self._assert_before_the_binding(frames, local)
        self.assertEqual(
            printer.worktree_dir_for_tab(VIEWER), "", "the bound tab resolves no local worktree",
        )

    async def test_a_talk_copy_lands_before_the_binding(self) -> None:
        printer = self.server._printer
        task_key = await self._watch_the_local_run()
        frames = await self._bind_with_a_copy_in_flight(
            Pause(WebPrinter._send_to_ws_clients.__code__, _viewer_talk_copy),
            functools.partial(_say_from_task, printer, task_key),
        )
        local = [i for i, f in enumerate(frames) if f.get("text") == LOCAL_SPEECH]
        self._assert_before_the_binding(frames, local)
