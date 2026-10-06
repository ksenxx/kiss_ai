# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a third-party agent launched via ``kiss.server.sorcar.run`` is
visible and interactable from a remote webview.

Wires up a real :class:`RemoteAccessServer` with a temporary local
endpoint (the same WSS transport the production ``kiss-web`` daemon
serves to browser/VS Code webviews), launches a
``third_party_agents.slack_sea.SlackAgent`` through
``run_agent_via_kiss_web`` (i.e. through the ``kiss.server.sorcar.run``
API against that daemon), and asserts:

1. a remote webview connection to the SAME daemon receives the task's
   live events (``clear`` / ``status running=True`` / ``prompt``)
   stamped with the API launch's tab id — i.e. the agent task can be
   *opened* remotely; and
2. an ``appendUserMessage`` command sent from the webview is echoed
   back as a ``prompt`` event and is drained by the real agent's
   pre-step hook into the model conversation as a ``user`` message —
   i.e. the agent can be *interacted with* remotely.

The run is a real ``SorcarAgent`` run against the streaming stand-in
model server (:class:`kiss.tests.server.parallel_agent_harness.StandInModelServer`);
the model answers ``finish`` once the test releases it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import time
import unittest
from collections.abc import Callable
from typing import Any

import yaml

from kiss.agents.sorcar import channel_workspace, local_endpoint
from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.third_party_agents import _kiss_web_launcher as launcher
from kiss.agents.third_party_agents._kiss_web_launcher import (
    run_agent_via_kiss_web,
)
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import LocalReader, LocalWriter, open_local_connection
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
)

STUB_SUMMARY = "<p>remote webview run done</p>"
FOLLOW_UP = "follow-up from the webview"


class TestRemoteWebviewInteraction(unittest.TestCase):
    """Third-party agent tasks are open/interactable via remote webview.

    The daemon under test is reached over its local WSS channel (the
    launcher's private endpoint file).
    """

    def setUp(self) -> None:
        # Every global mutation registers its restoration with
        # ``addCleanup`` immediately: cleanups run (in LIFO order) even
        # when ``setUp`` itself fails partway, unlike ``tearDown``.
        self.home = IsolatedKissHome(prefix="kiss-tp-webview-")
        self.addCleanup(self.home.cleanup)
        # Runs before ``home.cleanup``: stops the event writer and
        # invalidates every thread's cached connection, so the harness
        # finds no live connection to close raw under another thread.
        self.addCleanup(_persistence._close_db)
        self.endpoint_file = str(self.home.tmpdir / "sorcar-local.json")
        self.repo = str(self.home.repo)

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever,
            daemon=True,
        )
        self.loop_thread.start()
        self.addCleanup(self._stop_loop)

        self.server = RemoteAccessServer(
            local_endpoint_file=self.endpoint_file,
            work_dir=self.repo,
        )

        self._viewer_writer: LocalWriter | None = None
        self._reader_task: concurrent.futures.Future[None] | None = None
        self._received_cv = threading.Condition()
        asyncio.run_coroutine_threadsafe(
            self.server.start_private_async(),
            self.loop,
        ).result(timeout=30)
        self.addCleanup(self._shutdown_server)
        # Launches go to this test's daemon, not the process-global one.
        self._saved_endpoint_override = launcher._ENDPOINT_FILE_OVERRIDE
        launcher._ENDPOINT_FILE_OVERRIDE = self.endpoint_file
        self.addCleanup(self._restore_endpoint_override)

        self.model_requests: list[dict[str, Any]] = []
        self.model_seen = threading.Event()
        self.model_gate = threading.Event()
        self.model_server = StandInModelServer(self._answer_model_request)
        self.addCleanup(self.model_server.stop)
        # Registered last so they run first: release any in-flight model
        # call, then join the task workers while the daemon, the model
        # server and the history DB are all still up.
        self.addCleanup(self._join_tasks)
        self.addCleanup(self.model_gate.set)

    def _restore_endpoint_override(self) -> None:
        launcher._ENDPOINT_FILE_OVERRIDE = self._saved_endpoint_override

    def _answer_model_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Record *request*, hold it until the test opens ``model_gate``, then ``finish``.

        ``model_seen`` tells the test the run is inside a model call.
        """
        self.model_requests.append(request)
        self.model_seen.set()
        self.model_gate.wait(timeout=60)
        return finish_response(STUB_SUMMARY)

    def _shutdown_server(self) -> None:
        async def _shutdown() -> None:
            try:
                if self._viewer_writer is not None:
                    self._viewer_writer.close()
                    await self._viewer_writer.wait_closed()
            except Exception:
                pass
            ws_server = self.server._ws_server
            if ws_server is not None:
                ws_server.close()
                await ws_server.wait_closed()
            local_endpoint.remove_endpoint_if_owned(
                self.server._local_endpoint_file,
                self.server._local_token,
            )
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(
                _shutdown(),
                self.loop,
            ).result(timeout=5)
        except Exception:
            pass

    def _stop_loop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()

    def _join_tasks(self) -> None:
        # The daemon answers the launcher before its task thread has
        # finished its bookkeeping on the test's history.db; join those
        # threads before the DB is closed and the tmpdir removed.  The
        # worker clears ``state.task_thread`` itself as it finishes, so
        # read the attribute once and keep the reference.
        for state in agent_state.snapshot():
            thread = state.task_thread
            if thread is None:
                continue
            thread.join(timeout=30)
            if thread.is_alive():
                raise AssertionError(
                    f"task thread of tab {state.tab_id} still alive; not closing its DB"
                )

    def _open_viewer(self) -> list[dict[str, Any]]:
        """Open a remote-webview local connection and drain its inbox.

        Every received event is appended under ``_received_cv`` and the
        condition is notified, so waiters block on it instead of polling.
        """

        async def _open() -> tuple[LocalReader, LocalWriter]:
            return await open_local_connection(self.server)

        reader, writer = asyncio.run_coroutine_threadsafe(
            _open(),
            self.loop,
        ).result(timeout=5)
        self._viewer_writer = writer

        received: list[dict[str, Any]] = []
        cv = self._received_cv

        async def _drain() -> None:
            while True:
                line = await reader.readline()
                if not line:
                    return
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                with cv:
                    received.append(event)
                    cv.notify_all()

        self._reader_task = asyncio.run_coroutine_threadsafe(
            _drain(),
            self.loop,
        )
        return received

    def _wait_until(self, predicate: Callable[[], bool], timeout: float) -> bool:
        """Block on ``_received_cv`` until *predicate* holds or *timeout* passes."""
        deadline = time.monotonic() + timeout
        with self._received_cv:
            while not predicate():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._received_cv.wait(remaining)
            return True

    def _sync_viewer(self, received: list[dict[str, Any]], timeout: float = 10.0) -> None:
        """Round-trip a ``ping`` so every earlier broadcast has reached the viewer.

        The daemon answers ``pong`` from the connection's command loop,
        which only starts once the peer is registered as a local client,
        and per-endpoint sends are FIFO (``WebPrinter._locked_send``): a
        reply cannot overtake a broadcast already scheduled for the same
        peer.  So the ``pong`` proves both that the viewer is registered
        and that every event broadcast before the ping was delivered.
        """
        before = sum(1 for e in list(received) if e.get("type") == "pong")
        self._send_from_viewer({"type": "ping"})
        assert self._wait_until(
            lambda: sum(1 for e in received if e.get("type") == "pong") > before, timeout
        ), "the daemon never answered the viewer's ping"

    def _send_from_viewer(self, cmd: dict[str, Any]) -> None:
        """Send a JSON command over the viewer's local connection."""
        writer = self._viewer_writer
        assert writer is not None

        async def _send() -> None:
            writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
            await writer.drain()

        asyncio.run_coroutine_threadsafe(_send(), self.loop).result(
            timeout=5,
        )

    @staticmethod
    def _events_for_tab(
        received: list[dict[str, Any]],
        tab_id: str,
        ev_type: str,
    ) -> list[dict[str, Any]]:
        return [e for e in list(received) if e.get("type") == ev_type and e.get("tabId") == tab_id]

    def _wait_for_events(
        self,
        received: list[dict[str, Any]],
        tab_id: str,
        ev_type: str,
        text: str = "",
        timeout: float = 10.0,
        running: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Return the tab's matching *ev_type* events once any arrive.

        An event matches when its ``text`` contains *text* and, when
        *running* is given, its ``running`` flag equals it.
        """

        def matching() -> list[dict[str, Any]]:
            return [
                e
                for e in self._events_for_tab(received, tab_id, ev_type)
                if text in str(e.get("text", ""))
                and (running is None or e.get("running") is running)
            ]

        self._wait_until(lambda: bool(matching()), timeout)
        return matching()

    def test_launched_agent_open_and_interact_via_remote_webview(
        self,
    ) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        received = self._open_viewer()
        self._sync_viewer(received)

        agent = SlackAgent()
        out: dict[str, Any] = {}

        def launch() -> None:
            out["result"] = run_agent_via_kiss_web(
                agent,
                "remote webview task",
                model_name=STANDIN_MODEL,
                work_dir=self.repo,
                model_config=self.model_server.model_config,
            )

        t = threading.Thread(target=launch, daemon=True)
        t.start()
        try:
            assert self.model_seen.wait(timeout=60), "the run never called the model"

            # ``_cmd_run`` installs the worker thread in the registry and
            # broadcasts ``clear`` before it starts the thread, so the
            # first ``clear`` on an ``api-`` tab names a registered task.
            def api_clears() -> list[dict[str, Any]]:
                return [
                    e
                    for e in list(received)
                    if e.get("type") == "clear" and str(e.get("tabId", "")).startswith("api-")
                ]

            assert self._wait_until(lambda: bool(api_clears()), 10), (
                "remote webview never saw the task's clear event"
            )
            tab_id = str(api_clears()[0]["tabId"])
            states = [st for st in agent_state.snapshot() if st.tab_id == tab_id]
            assert states, "API launch never appeared in the registry"
            worker = states[0].task_thread
            assert worker is not None, "the API launch has no task thread"

            assert self._wait_for_events(received, tab_id, "status", running=True), (
                "remote webview never saw status running=True for the task"
            )
            assert self._wait_for_events(
                received,
                tab_id,
                "prompt",
                "remote webview task",
            ), "remote webview never saw the task's prompt event"

            self._send_from_viewer(
                {
                    "type": "appendUserMessage",
                    "tabId": tab_id,
                    "prompt": FOLLOW_UP,
                }
            )
            assert self._wait_for_events(received, tab_id, "prompt", FOLLOW_UP), (
                "the webview never received the prompt echo for its appendUserMessage"
            )
        finally:
            # Release the model: it answers ``finish``; with the follow-up
            # still queued the agent refuses that finish, and the next
            # step's pre-step hook drains the message into the
            # conversation before the model is asked again.
            self.model_gate.set()
            t.join(timeout=60)

        parsed = yaml.safe_load(out.get("result") or "")
        assert parsed and parsed.get("success") is True, out.get("result")
        assert parsed.get("summary") == STUB_SUMMARY
        agentic = [r for r in self.model_requests if r.get("tools")]
        assert agentic, "the run sent no agentic model request"
        user_turns = [
            str(m.get("content", "")) for m in agentic[-1]["messages"] if m.get("role") == "user"
        ]
        assert any(f"User says: {FOLLOW_UP}" in c for c in user_turns), (
            "the appendUserMessage from the remote webview was never drained "
            f"into the running agent's conversation: {user_turns}"
        )

        assert self._wait_for_events(received, tab_id, "status", running=False), (
            "remote webview never saw status running=False"
        )
        # The drained follow-up must not come back as a second run on
        # the tab.  The task runner re-submits leftovers as the LAST
        # step of the worker thread's cleanup (after ``running=False``),
        # and a re-submitted run registers its thread and broadcasts its
        # ``clear`` synchronously inside that step; so once the worker
        # has been joined, every re-dispatch artefact already exists: a
        # second ``clear`` event (delivered before the ping barrier's
        # ``pong``), a fresh task thread in the registry and the slack
        # SEA's ``default`` channel workspace held again.
        worker.join(timeout=30)
        assert not worker.is_alive(), "the launched run's task thread did not finish"
        self._sync_viewer(received)
        assert len(self._events_for_tab(received, tab_id, "clear")) == 1, (
            "an undrained follow-up was re-submitted as a new run"
        )
        assert dict(channel_workspace._ACTIVE_WORKSPACES) == {}
        assert not any(st.task_thread is not None for st in agent_state.snapshot()), (
            "a task thread outlived the launched run"
        )


if __name__ == "__main__":
    unittest.main()
