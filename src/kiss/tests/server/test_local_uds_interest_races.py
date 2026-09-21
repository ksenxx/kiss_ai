# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: local-UDS talk bookkeeping vs. tab-registry races.

The printer's local-UDS bookkeeping used to MIRROR "which tabs a local
webview shows": every registry removal pruned the tab from every UDS
connection, every ``ready`` re-copied a registry snapshot into the
connection's set, and the talk fan-out trusted that copy.  Registry
and printer keep separate locks, so the mirror could interleave with
the registry mutation it copied (gpt-5.6-sol review, three latent
races):

1. A still-stale local client re-recorded a tab right after a close
   pruned it; nothing ever undid the add, so the closed tab's
   still-running task kept playing talk natively.
2. ``ready`` copied a registry snapshot, a concurrent close removed
   and pruned a tab from it, and the sync then re-added the closed
   tab from the stale copy.
3. A ``resumeSession`` reopen republished a closed tab before the
   close's prune ran, so the prune dropped the reopened tab's
   bookkeeping while registry and every client showed it.

The rule now (``VSCodeServer._local_tab_shown``, consulted by
``WebPrinter.shown_local_uds_tabs`` at talk time), evaluated in order:
(1) a tab listed in the canonical registry is shown by every attached
chat webview, so it counts when some UDS connection has announced
``ready``; otherwise (2) a target some UDS peer addressed counts while
its own agent state is alive and not ``frontend_closed`` (a
``run_agent`` dispatch's ``api-…`` tab, a running sub-agent's tab, a
headless client's own registry tab with no webview attached);
otherwise (3) an addressed target with no state of its own is a
still-subscribed viewer of a live task (a ``run_parallel`` child's
visible sub-tab, a tab resuming a running chat before its publication
lands) and counts unless it is a registry tab — the one arm that
requires absence from the registry.  The
bookkeeping is no longer a copy of registry state, so no interleaving
of a close with a record, a ``ready`` sync or a reopen can leave a
stale decision behind; what remains is the read itself — the printer
facts and the canonical facts are read one after the other, so an
event landing between the two reads can misjudge the utterance being
fanned out, and the next decision is correct again.

The race tests drive a REAL ``RemoteAccessServer`` over its UDS
listener with a real audio-player child process
(``KISS_SORCAR_PLAY_CMD``); the racing interleavings are forced
through seams on the server object itself (a wrapped method that
parks or runs the racing close), never through mocks of the code
under test.  The four preserved-behaviour controls at the end feed
the rule the inputs production produces (agent states as
``agent_task_allocated`` / ``_cmd_run`` register them, subscriptions
as ``_attach_viewer_to_running_chat`` adds them) rather than driving
a whole run or a persisted-history resume; the run/resume wiring
itself is covered by the run-agent and sub-agent multi-view suites.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state, talk_player
from kiss.server.agent_state import AgentState
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import requires_unix_sockets
from kiss.tests.server.test_talk_endpoint_muting import (
    _find_free_port,
    _redirect_persistence,
    _restore_persistence,
    _talk_event,
    _write_player,
)


def _encode(cmd: dict[str, Any]) -> bytes:
    """Serialize one wire command as a newline-terminated JSON line."""
    return (json.dumps(cmd) + "\n").encode("utf-8")


class _CloseDuringSync:
    """Seam for race 2: run the racing close INSIDE the ``ready`` sync.

    Wraps ``WebPrinter.sync_local_uds_tabs``, the bookkeeping step of
    ``ServerApi.ready``: closing the tab here reproduces "ready is
    mid-flight, the close completes, ready finishes" deterministically
    and on the event-loop thread (no parked loop, no second thread).
    """

    def __init__(self, real_sync: Any, close_tab: Any, tab_id: str) -> None:
        self._real_sync = real_sync
        self._close_tab = close_tab
        self._tab_id = tab_id
        self.fired = False

    def __call__(
        self, conn_id: str, tab_ids: set[str], local_tabs: set[str],
    ) -> None:
        if not self.fired:
            self.fired = True
            self._close_tab(self._tab_id)
        self._real_sync(conn_id, tab_ids, local_tabs)


class _ParkingCall:
    """Seam that parks a server method until the test releases it.

    Used for race 3 (park the close's prune step) and for the sibling
    ordering (park the reopen's publication step) so the test can run
    the other half of the race in between.  Parking is best-effort: a
    server that never reaches the wrapped step simply never parks,
    and the test asserts only on the final outcome.
    """

    def __init__(self, real: Any) -> None:
        self._real = real
        self.parked = threading.Event()
        self.release = threading.Event()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.parked.set()
        if not self.release.wait(timeout=30):
            raise AssertionError("parked server call was never released")
        return self._real(*args, **kwargs)


@requires_unix_sockets
class TestLocalUdsInterestRaces(IsolatedAsyncioTestCase):
    """The three documented races plus the retained sub-agent behaviour."""

    async def asyncSetUp(self) -> None:
        agent_state.agent_states.clear()
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        self.marker_dir = Path(self.tmpdir) / "markers"
        player = _write_player(Path(self.tmpdir), self.marker_dir)
        self.saved_play_cmd = os.environ.get("KISS_SORCAR_PLAY_CMD")
        os.environ["KISS_SORCAR_PLAY_CMD"] = (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(player))}"
        )
        talk_player.reset_shared_player_for_tests()
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        from kiss.server.web_server import _generate_self_signed_cert

        _generate_self_signed_cert(certfile, keyfile)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=_find_free_port(),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()
        self.backend = self.server._vscode_server
        self.registry = self.backend.tab_registry
        self.printer = self.server._printer
        self.task_id = uuid.uuid4().hex
        self._writers: list[asyncio.StreamWriter] = []

    async def asyncTearDown(self) -> None:
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        await self.server.stop_async()
        agent_state.agent_states.clear()
        if self.saved_play_cmd is None:
            os.environ.pop("KISS_SORCAR_PLAY_CMD", None)
        else:
            os.environ["KISS_SORCAR_PLAY_CMD"] = self.saved_play_cmd
        talk_player.reset_shared_player_for_tests()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ----------------------------------------------------------- helpers

    async def _open_uds(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open one UDS client and pin its work dir (no ``ready`` yet)."""
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path), limit=16 * 1024 * 1024
        )
        self._writers.append(writer)
        writer.write(_encode({"type": "setWorkDir", "workDir": self.tmpdir}))
        await writer.drain()
        return reader, writer

    async def _send(self, writer: asyncio.StreamWriter, cmd: dict[str, Any]) -> None:
        writer.write(_encode(cmd))
        await writer.drain()

    async def _ready(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
        tab_id: str,
    ) -> None:
        """Announce ``ready`` and wait for the snapshot it triggers."""
        await self._send(
            writer, {"type": "ready", "tabId": tab_id, "workDir": self.tmpdir},
        )
        await self._wait_tabs_state(reader, lambda ids: True)

    async def _wait_tabs_state(
        self,
        reader: asyncio.StreamReader,
        accept: Any,
        timeout: float = 10.0,
    ) -> list[str]:
        """Read UDS events until a ``tabs_state`` satisfying *accept*."""
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            self.assertGreater(remaining, 0, "expected tabs_state never arrived")
            line = await asyncio.wait_for(reader.readline(), timeout=remaining)
            self.assertTrue(line, "UDS connection closed unexpectedly")
            msg = json.loads(line.decode("utf-8"))
            if isinstance(msg, dict) and msg.get("type") == "tabs_state":
                ids = [t.get("tabId") for t in msg.get("tabs", [])]
                if accept(ids):
                    return ids

    async def _collect_talk(
        self, reader: asyncio.StreamReader, timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Read UDS events until one ``talk`` copy arrives."""
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            self.assertGreater(remaining, 0, "talk copy never arrived")
            line = await asyncio.wait_for(reader.readline(), timeout=remaining)
            self.assertTrue(line, "UDS connection closed unexpectedly")
            msg = json.loads(line.decode("utf-8"))
            if isinstance(msg, dict) and msg.get("type") == "talk":
                return msg

    async def _collect_talks(
        self, reader: asyncio.StreamReader, count: int, timeout: float = 5.0,
    ) -> dict[str, dict[str, Any]]:
        """Read UDS events until *count* ``talk`` copies arrive; by tab."""
        talks: dict[str, dict[str, Any]] = {}
        deadline = asyncio.get_event_loop().time() + timeout
        while len(talks) < count:
            remaining = deadline - asyncio.get_event_loop().time()
            self.assertGreater(remaining, 0, "talk copies never arrived")
            line = await asyncio.wait_for(reader.readline(), timeout=remaining)
            self.assertTrue(line, "UDS connection closed unexpectedly")
            msg = json.loads(line.decode("utf-8"))
            if isinstance(msg, dict) and msg.get("type") == "talk":
                talks[msg["tabId"]] = msg
        return talks

    def _wait_registry(self, present: bool, tab_id: str) -> None:
        """Block until the registry does / does not list *tab_id*."""
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.registry.has_tab(tab_id) is present:
                return
            time.sleep(0.01)
        self.fail(f"registry never reached has_tab({tab_id!r}) == {present}")

    def _markers(self) -> int:
        return len(list(self.marker_dir.glob("*.json")))

    def _await_markers(self, count: int, timeout: float = 5.0) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self._markers() < count:
            time.sleep(0.05)
        return self._markers()

    async def _assert_talk_native(
        self, reader: asyncio.StreamReader, tab_id: str, talk_id: str,
    ) -> None:
        """The daemon plays the clip and the UDS copy is muted."""
        self.printer.subscribe_tab(self.task_id, tab_id)
        self.printer.broadcast(_talk_event(self.task_id, talk_id))
        talk = await self._collect_talk(reader)
        self.assertEqual(talk["tabId"], tab_id)
        self.assertTrue(
            talk.get("muted"),
            "a local webview shows the tab, so its UDS copy must be muted "
            "in favour of daemon-native playback",
        )
        self.assertEqual(
            await asyncio.to_thread(self._await_markers, 1), 1,
            "the daemon must play the clip natively for a tab a local "
            "webview shows",
        )

    async def _assert_talk_not_native(
        self, reader: asyncio.StreamReader, tab_id: str, talk_id: str,
    ) -> None:
        """The UDS copy stays playable and the daemon stays silent."""
        self.printer.subscribe_tab(self.task_id, tab_id)
        self.printer.broadcast(_talk_event(self.task_id, talk_id))
        talk = await self._collect_talk(reader)
        self.assertEqual(talk["tabId"], tab_id)
        self.assertFalse(
            talk.get("muted"),
            "no local webview shows the tab, so its copy must stay playable",
        )
        await asyncio.sleep(0.5)
        self.assertEqual(
            self._markers(), 0,
            "the daemon must not play a talk for a tab no local webview shows",
        )

    # ------------------------------------------------------------- tests

    async def test_stale_command_after_close_cannot_revive_native_playback(
        self,
    ) -> None:
        """Race 1: a stale tab-bearing command after the close is inert.

        Client A still shows the tab when another client's ``closeTab``
        lands; A's next command names the closed tab and dispatch
        records A's interest in it.  That interest must not make the
        closed tab's still-running task play natively: the tab no
        longer exists in the registry.
        """
        tab = "closed-" + uuid.uuid4().hex[:8]
        self.registry.update_tab(tab, title="busy chat", create=True)
        # The tab's task is still running: the close defers teardown
        # and the task keeps its subscription to the closed tab.
        self._register_live_state(tab, task_id=self.task_id, running=True)
        reader_a, writer_a = await self._open_uds()
        await self._ready(reader_a, writer_a, "placeholder-a")
        reader_b, writer_b = await self._open_uds()
        await self._ready(reader_b, writer_b, "placeholder-b")

        await self._send(writer_b, {"type": "closeTab", "tabId": tab})
        await self._wait_tabs_state(reader_a, lambda ids: tab not in ids)

        # A is stale: its command still names the closed tab.  The
        # ``getTabsState`` reply doubles as the "dispatch finished"
        # signal (dispatch records the tab BEFORE the handler runs).
        await self._send(writer_a, {"type": "getTabsState", "tabId": tab})
        await self._wait_tabs_state(reader_a, lambda ids: True)
        self.assertTrue(agent_state.find_by_tab(tab).frontend_closed)  # type: ignore[union-attr]

        await self._assert_talk_not_native(reader_a, tab, "race1-talk")

    async def test_ready_racing_a_close_cannot_resurrect_the_closed_tab(
        self,
    ) -> None:
        """Race 2: a close landing inside ``ready`` leaves nothing stale.

        ``ready`` used to copy the registry snapshot into the
        connection's bookkeeping; a close completing between the copy
        and the sync re-added the closed tab.  ``ready`` no longer
        copies registry state at all — the fan-out reads the registry
        at talk time — so however the close interleaves with the
        ready, the closed tab's talk must not play natively.
        """
        tab = "snap-" + uuid.uuid4().hex[:8]
        self.registry.update_tab(tab, title="snapshot chat", create=True)
        reader, writer = await self._open_uds()
        seam = _CloseDuringSync(
            self.printer.sync_local_uds_tabs, self.backend._close_tab, tab,
        )
        self.printer.sync_local_uds_tabs = seam  # type: ignore[method-assign]
        try:
            await self._ready(reader, writer, "placeholder")
        finally:
            del self.printer.sync_local_uds_tabs
        self.assertTrue(seam.fired, "ready never reached the sync")
        self.assertFalse(self.registry.has_tab(tab))

        await self._assert_talk_not_native(reader, tab, "race2-talk")

    async def test_resume_reopen_survives_a_close_prune_landing_late(
        self,
    ) -> None:
        """Race 3 (the reviewer's order): reopen, then the close's tail.

        The close removes the tab and parks before its cleanup tail; a
        ``resumeSession`` from a local webview republishes the tab; the
        tail resumes.  Registry and every client show the reopened
        tab, so its talk must play natively.
        """
        tab = "reopen-" + uuid.uuid4().hex[:8]
        chat = "chat-" + uuid.uuid4().hex[:8]
        self.registry.update_tab(tab, chat_id=chat, create=True)
        reader, writer = await self._open_uds()
        await self._ready(reader, writer, "placeholder")

        seam = _ParkingCall(self.backend._prune_local_uds_tab)
        self.backend._prune_local_uds_tab = seam  # type: ignore[method-assign]
        closer = threading.Thread(
            target=self.backend._close_tab, args=(tab,), daemon=True,
        )
        try:
            closer.start()
            await asyncio.to_thread(self._wait_registry, False, tab)
            await self._send(
                writer, {"type": "resumeSession", "id": chat, "tabId": tab},
            )
            await asyncio.to_thread(self._wait_registry, True, tab)
        finally:
            seam.release.set()
            closer.join(timeout=10)
            del self.backend._prune_local_uds_tab
        self.assertFalse(closer.is_alive(), "close never finished")
        self.assertTrue(self.registry.has_tab(tab))

        await self._assert_talk_native(reader, tab, "race3-talk")

    async def test_resume_reopen_survives_a_close_between_record_and_publish(
        self,
    ) -> None:
        """Race 3 (sibling order): the close lands inside the reopen.

        Dispatch records the resuming webview's interest, the replay
        parks before its publication, another client's close removes
        the tab, and the publication then reopens it.  Interest
        recorded before the close must still count once the tab
        exists again.
        """
        tab = "mid-" + uuid.uuid4().hex[:8]
        chat = "chat-" + uuid.uuid4().hex[:8]
        self.registry.update_tab(tab, chat_id=chat, create=True)
        reader, writer = await self._open_uds()
        await self._ready(reader, writer, "placeholder")

        seam = _ParkingCall(self.backend._registry_update_tab)
        self.backend._registry_update_tab = seam  # type: ignore[method-assign]
        try:
            await self._send(
                writer, {"type": "resumeSession", "id": chat, "tabId": tab},
            )
            self.assertTrue(
                await asyncio.to_thread(seam.parked.wait, 10),
                "the replay never reached its publication",
            )
            await asyncio.to_thread(self.backend._close_tab, tab)
            self.assertFalse(self.registry.has_tab(tab))
        finally:
            seam.release.set()
            del self.backend._registry_update_tab
        await asyncio.to_thread(self._wait_registry, True, tab)

        await self._assert_talk_native(reader, tab, "race3b-talk")

    def _register_live_state(
        self,
        tab_id: str,
        task_id: str = "",
        server_owned: bool = False,
        running: bool = False,
    ) -> AgentState:
        """Register a live agent state for *tab_id*.

        *running* marks the task active, so a close of the tab DEFERS
        the state's teardown (``frontend_closed`` is raised, the task
        subscription is retained) exactly like a close during a real
        run; otherwise the close retires the state at once.
        """
        state = AgentState(
            task_id or "task-" + uuid.uuid4().hex[:8],
            tab_id=tab_id, chat_id="chat-x", server_owned=server_owned,
        )
        state.is_task_active = running
        with agent_state.STATE_LOCK:
            agent_state.register(state)
        return state

    async def _wait_subagent_close(
        self, reader: asyncio.StreamReader, tab_id: str,
    ) -> None:
        deadline = asyncio.get_event_loop().time() + 10
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            self.assertGreater(remaining, 0, "closeSubagentTab never arrived")
            line = await asyncio.wait_for(reader.readline(), timeout=remaining)
            msg = json.loads(line.decode("utf-8"))
            if msg.get("type") == "closeSubagentTab" and msg.get("tab_id") == tab_id:
                return

    async def test_subagent_tab_counts_while_live_and_stops_when_closed(
        self,
    ) -> None:
        """Control: a sub-agent tab running its own task never enters
        the registry: it counts while a UDS peer addressed it and its
        agent state is not closed.  Closing it mid-run raises
        ``frontend_closed`` (the busy state and its subscription are
        retained until the task ends), so the talk stops playing
        natively — the pre-existing behaviour, now derived from the
        state instead of from a pruned bookkeeping copy.
        """
        sub_tab = "parent-" + uuid.uuid4().hex[:8] + "__sub_task1"
        self._register_live_state(sub_tab, running=True)
        reader, writer = await self._open_uds()
        await self._ready(reader, writer, "placeholder")
        await self._send(writer, {"type": "getTabsState", "tabId": sub_tab})
        await self._wait_tabs_state(reader, lambda ids: True)

        await self._assert_talk_native(reader, sub_tab, "sub-open-talk")
        for marker in self.marker_dir.glob("*.json"):
            marker.unlink()

        await self._send(writer, {"type": "closeTab", "tabId": sub_tab})
        await self._wait_subagent_close(reader, sub_tab)

        await self._assert_talk_not_native(reader, sub_tab, "sub-closed-talk")

    async def test_headless_peer_live_tab_counts_until_its_close(self) -> None:
        """Control: a ``run_agent`` dispatch — the daemon client is a
        UDS peer that never announces ``ready`` and runs its sub-agent
        in an ``api-…`` tab outside the registry.  The state and the
        peer's interest are reproduced directly (the state carries no
        parent metadata, so its close takes the plain, not the
        ``closeSubagentTab``, branch).  Its talk plays natively while
        the run's state is alive and not closed (the pre-existing
        behaviour) and stops once the tab is closed.
        """
        api_tab = "api-" + uuid.uuid4().hex
        self._register_live_state(api_tab, server_owned=True, running=True)
        reader, writer = await self._open_uds()
        await self._send(writer, {"type": "getTabsState", "tabId": api_tab})
        await self._wait_tabs_state(reader, lambda ids: True)

        await self._assert_talk_native(reader, api_tab, "api-live-talk")
        for marker in self.marker_dir.glob("*.json"):
            marker.unlink()

        # A webview user closes the (still running) dispatch's tab:
        # teardown is deferred, the state is marked closed.
        await self._send(writer, {"type": "closeTab", "tabId": api_tab})
        await self._send(writer, {"type": "getTabsState"})
        await self._wait_tabs_state(reader, lambda ids: True)
        self.assertTrue(agent_state.find_by_tab(api_tab).frontend_closed)  # type: ignore[union-attr]

        await self._assert_talk_not_native(reader, api_tab, "api-closed-talk")

    async def test_registry_tab_needs_an_attached_webview(self) -> None:
        """A registry tab addressed only by a headless UDS peer is shown
        nowhere locally; it counts as soon as a chat webview attaches
        (``ready``), even though that webview never addressed it.
        """
        tab = "bg-" + uuid.uuid4().hex[:8]
        self.registry.update_tab(tab, title="background chat", create=True)
        reader, writer = await self._open_uds()
        await self._send(writer, {"type": "getTabsState", "tabId": tab})
        await self._wait_tabs_state(reader, lambda ids: True)

        await self._assert_talk_not_native(reader, tab, "headless-talk")

        await self._ready(reader, writer, "placeholder")
        await self._assert_talk_native(reader, tab, "webview-talk")

    async def test_tab_published_after_ready_counts_without_being_touched(
        self,
    ) -> None:
        """A tab another client publishes AFTER this webview's ``ready``
        is adopted from ``tabs_state`` without any command naming it;
        its talk must still play natively here.  (The old per-tab copy
        only knew tabs present at ``ready`` time or named later.)
        """
        reader, writer = await self._open_uds()
        await self._ready(reader, writer, "placeholder")
        tab = "late-" + uuid.uuid4().hex[:8]
        self.registry.update_tab(tab, title="late chat", create=True)

        await self._assert_talk_native(reader, tab, "late-talk")

    async def test_run_parallel_viewer_counts_while_subscribed(self) -> None:
        """Control: a ``run_parallel`` child runs under a synthetic
        server tab (``task-<key>__sub_<i>``, the id
        ``agent_task_allocated`` stores on its state) while every
        webview shows it as a VIEWER tab of its own id
        (``<parent>__sub_<taskId>``) that the resume triggered by the
        ``new_tab`` broadcast subscribes to the child's task.  This
        test reproduces those inputs directly (state + the two
        subscriptions + the resume's dispatch record) rather than
        driving the persisted-history resume.  The viewer has no agent
        state of its own: it counts while it is subscribed and a UDS
        peer addressed it, and its real ``closeTab`` (which
        unsubscribes it) ends native playback.
        """
        child_task = "child-" + uuid.uuid4().hex[:8]
        synthetic = f"task-{uuid.uuid4().hex[:8]}__sub_0"
        viewer = f"parent-{uuid.uuid4().hex[:8]}__sub_{child_task}"
        self._register_live_state(synthetic, task_id=child_task, running=True)
        self.printer.subscribe_tab(child_task, synthetic)
        reader, writer = await self._open_uds()
        await self._ready(reader, writer, "placeholder")
        # What ``_attach_viewer_to_running_chat`` does for the resume,
        # plus the resume's own dispatch record of the viewer id.
        self.printer.subscribe_tab(child_task, viewer)
        await self._send(writer, {"type": "getTabsState", "tabId": viewer})
        await self._wait_tabs_state(reader, lambda ids: True)

        self.printer.broadcast(_talk_event(child_task, "viewer-open-talk"))
        talks = await self._collect_talks(reader, 2)
        self.assertEqual(set(talks), {synthetic, viewer})
        self.assertTrue(
            talks[viewer].get("muted"),
            "the visible sub-agent viewer is shown by the local webview: "
            "its copy must be muted while the daemon plays natively",
        )
        self.assertFalse(
            talks[synthetic].get("muted"),
            "the synthetic server-side id is shown by no webview",
        )
        self.assertEqual(await asyncio.to_thread(self._await_markers, 1), 1)
        for marker in self.marker_dir.glob("*.json"):
            marker.unlink()

        await self._send(writer, {"type": "closeTab", "tabId": viewer})
        await self._wait_subagent_close(reader, viewer)

        self.printer.broadcast(_talk_event(child_task, "viewer-closed-talk"))
        talks = await self._collect_talks(reader, 1)
        self.assertEqual(set(talks), {synthetic}, "the closed viewer was unsubscribed")
        self.assertFalse(talks[synthetic].get("muted"))
        await asyncio.sleep(0.5)
        self.assertEqual(self._markers(), 0, "no local webview shows the child")

    async def test_headless_registry_run_counts_without_a_webview(self) -> None:
        """Control: a standalone ``daemon_client.run`` (no
        ``parentTaskId``) publishes its ``api-…`` tab into the registry
        (``_cmd_run``), yet no chat webview may be attached at all.
        The registry row, the server-owned running state and the
        headless peer's interest are reproduced directly here.  Its
        talk still plays natively while the headless owner is connected
        and the run's state is alive (the pre-existing behaviour), and
        stops once the tab is closed while the run continues (state
        marked closed).
        """
        api_tab = "api-" + uuid.uuid4().hex
        self.registry.update_tab(api_tab, chat_id="chat-x", create=True)
        self._register_live_state(api_tab, server_owned=True, running=True)
        reader, writer = await self._open_uds()
        await self._send(writer, {"type": "getTabsState", "tabId": api_tab})
        await self._wait_tabs_state(reader, lambda ids: True)

        await self._assert_talk_native(reader, api_tab, "headless-run-talk")
        for marker in self.marker_dir.glob("*.json"):
            marker.unlink()

        await self._send(writer, {"type": "closeTab", "tabId": api_tab})
        await self._wait_tabs_state(reader, lambda ids: api_tab not in ids)
        self.assertTrue(agent_state.find_by_tab(api_tab).frontend_closed)  # type: ignore[union-attr]

        await self._assert_talk_not_native(reader, api_tab, "headless-closed-talk")
