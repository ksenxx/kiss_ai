# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of chat tabs that run their tasks on a remote machine.

A real local daemon, a real remote daemon (:mod:`kiss.tests.remote_daemon`,
its model a fake OpenAI endpoint) registered as machine ``box``, and a
client on the local connection:

* a ``submit`` with ``machine: "box"`` runs on the remote daemon: the
  client sees ``tabMachine`` then the run's ``status``/``result``
  frames stamped with its tab id, the local history has no row for the
  task while the remote history does, and the tab is published in the
  local registry (and the remote's);
* surfaces connecting later (``ready``) each get the binding and one
  ``task_events`` replay of the remote chat, nobody else does;
* a ``stop`` of a bound tab is forwarded; ``closeTab`` unbinds it here
  and on the remote; a later ``submit`` with an empty ``machine`` runs
  here again and announces the unbinding;
* a run that fans out: the sub-agent's ``new_tab`` is relayed, a
  client's ``resumeSession`` of the sub-task is forwarded and answered
  with the sub-agent tab's binding (``tabMachine``: the chat header
  shows the machine for it as for its parent), the sub-agent's
  transcript and ``subagentDone``, a surface
  joining meanwhile gets the running sub-agent announced
  (``openSubagentTab``) after the parent's replay, and closing the
  sub-agent tab leaves the parent bound;
* a ``submit`` naming an unknown machine or an unreachable machine is
  refused with ``status running:false`` and a ``prompt_refused`` error;
* losing the remote daemon mid-binding ends the tab with an error, and
  the sub-agent tabs its stream relayed with ``subagentDone``.

A ``closeTab`` landing while the first connect is still in flight is
covered by ``test_remote_bind_does_not_block_connection.py`` (a
listener that never answers the TLS handshake pauses the connect); a
second ``submit`` racing the first from another page (``RemoteTabs._bind``
keeps the winner) by ``test_remote_bound_tab_browser_pages.py``.  From
the same page the two cannot be in flight at once: the per-connection,
per-tab ordering of ``ServerApi._dispatch_detached`` serialises them.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
from collections.abc import AsyncIterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import machines as mach
from kiss.agents.sorcar import persistence
from kiss.server.remote_tabs import RemoteTab
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import (
    LocalReader,
    LocalWriter,
    _generate_self_signed_cert,
    open_local_connection,
)
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server.pause import Pause
from kiss.tests.server.remote_tabs_probe import machine_of
from kiss.tests.server.two_client_harness import page_after

SUMMARY = "<p>Hello from box</p>"

# A SEA that needs no model: its run is held at the first model call until a
# ``release`` file appears beside the script, and ends there (as failed).
HELD_SEA = '''
import pathlib
import time

from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def llm_call_hook(self, new_messages):
        release, deadline = pathlib.Path(__file__).with_name("release"), time.monotonic() + 60
        while not release.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        raise RuntimeError("released")
'''


class _FinishingModel(BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint whose every answer is a ``finish`` call.

    With ``parallel_on_first_call`` the first answer is a
    ``run_parallel`` of one sub-task instead (the sub-agent's own call,
    the second one, is answered after ``hold_sub_agent`` seconds, or
    once ``released`` is set, so a viewer can attach while it runs);
    the counter is per test.
    """

    parallel_on_first_call = False
    hold_sub_agent = 2.0
    released = threading.Event()
    calls = 0

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        _FinishingModel.calls += 1
        if _FinishingModel.parallel_on_first_call and _FinishingModel.calls == 1:
            function = {
                "name": "run_parallel",
                "arguments": json.dumps({"tasks": json.dumps(["Say hi from the sub-agent"])}),
            }
        else:
            if _FinishingModel.parallel_on_first_call and _FinishingModel.calls == 2:
                _FinishingModel.released.wait(_FinishingModel.hold_sub_agent)
            function = {
                "name": "finish",
                "arguments": json.dumps({"success": True, "summary_in_html": SUMMARY}),
            }
        call = {"index": 0, "id": f"call-{_FinishingModel.calls}", "type": "function",
                "function": function}
        base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": "gpt-4o-mini"}
        chunks = [
            {**base, "choices": [{
                "index": 0, "delta": {"role": "assistant", "tool_calls": [call]},
                "finish_reason": None,
            }]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {**base, "choices": [], "usage": {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            }},
        ]
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            pass  # the daemon was killed while its call was held


async def _frames_to(reader: LocalReader, kind: str, tab_id: str = "") -> list[dict[str, Any]]:
    """The frames *reader* yields up to the first *kind* one (of *tab_id* when given)."""
    frames = [json.loads(await asyncio.wait_for(reader.readline(), 90))]
    while frames[-1]["type"] != kind or tab_id not in ("", frames[-1].get("tabId")):
        frames.append(json.loads(await asyncio.wait_for(reader.readline(), 90)))
    return frames


async def _late_frames(reader: LocalReader) -> AsyncIterator[dict[str, Any]]:
    """The frames a connecting surface is sent, as a freshly loaded page takes them.

    The ``tabMachines`` frame of its ``ready`` names every binding at
    once: it is the ``tabMachine`` of each tab it names.
    """
    for _ in range(400):
        msg = json.loads(await asyncio.wait_for(reader.readline(), 90))
        if msg.get("type") != "tabMachines":
            yield msg
            continue
        for tab, machine in msg["machines"].items():
            yield {"type": "tabMachine", "tabId": tab, "machine": machine}


def _ended_tabs(frames: list[dict[str, Any]]) -> list[str]:
    """The sub-agent tabs *frames* tell a surface are over (``subagentDone``)."""
    return [str(f.get("tab_id")) for f in frames if f.get("type") == "subagentDone"]


def _release_model() -> None:
    """Let a held model call answer: it blocks the single-threaded model server."""
    _FinishingModel.released.set()


class RemoteTabsHarness(IsolatedAsyncioTestCase):
    """One local daemon, one remote daemon ("box") with a fake model, one client."""

    async def asyncSetUp(self) -> None:
        # Cleanups, each registered as its resource starts, not asyncTearDown:
        # unittest skips the tear-down when the set-up fails, and the model
        # server such a set-up left serving kept pytest from exiting.
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        mach.save_machines([])
        self.addCleanup(mach.save_machines, [])
        _FinishingModel.parallel_on_first_call = False
        _FinishingModel.hold_sub_agent = 2.0
        _FinishingModel.released = threading.Event()
        _FinishingModel.calls = 0
        self.model_port = find_free_port()
        self.model_server = HTTPServer(("127.0.0.1", self.model_port), _FinishingModel)
        self.addCleanup(self.model_server.server_close)
        threading.Thread(target=self.model_server.serve_forever, daemon=True).start()
        self.addCleanup(self.model_server.shutdown)
        self.addCleanup(_release_model)
        endpoint = f"http://127.0.0.1:{self.model_port}/v1"
        self.model = "gpt-4o-mini"
        self.remote = await asyncio.to_thread(
            start_remote_daemon, Path(self.tmpdir) / "remote-home", password="pw-remote",
            config={"custom_endpoint": endpoint, "custom_api_key": "k", "max_budget": 100},
        )
        self.addCleanup(self.remote.stop)
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), certfile=str(certfile),
            keyfile=str(keyfile), url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        # Registered before the start: one that fails part-way is stopped too.
        self.addAsyncCleanup(self.server.stop_async)
        await self.server.start_async()
        self.reader, self.writer = await open_local_connection(self.server)
        self.addCleanup(self.writer.close)
        added = await self._ask({
            "type": "machineAction", "action": "add", "name": "box",
            "address": self.remote.url, "machinePassword": "pw-remote",
        }, "machines")
        self.assertEqual(added["error"], "")

    async def _send(self, cmd: dict[str, Any]) -> None:
        self.writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await self.writer.drain()

    async def _next(self, timeout: float = 90.0) -> dict[str, Any]:
        line = await asyncio.wait_for(self.reader.readline(), timeout=timeout)
        assert line, "local connection closed"
        msg: dict[str, Any] = json.loads(line.decode("utf-8"))
        return msg

    async def _ask(self, cmd: dict[str, Any], reply_type: str) -> dict[str, Any]:
        """Send *cmd* and return the first *reply_type* message."""
        await self._send(cmd)
        return await self._until(reply_type)

    async def _until(
        self, reply_type: str, tab_id: str = "", timeout: float = 90.0,
    ) -> dict[str, Any]:
        """Return the first *reply_type* message (of *tab_id* when given)."""
        for _ in range(300):
            msg = await self._next(timeout)
            if msg.get("type") == reply_type and (not tab_id or msg.get("tabId") == tab_id):
                return msg
        raise AssertionError(f"no {reply_type} reply")

    async def _submit(self, tab_id: str, prompt: str, machine: str, prompt_id: str = "") -> None:
        await self._send({
            "type": "submit", "prompt": prompt, "tabId": tab_id,
            "machine": machine, "model": self.model, "workDir": self.tmpdir,
            **({"promptId": prompt_id} if prompt_id else {}),
        })

    async def _run_until_idle(self, tab_id: str) -> list[dict[str, Any]]:
        """Collect the tab's frames up to the run's own ``status running:false``.

        A ``status`` stamped with a ``replayId`` answers a replay (a
        ``ready`` replays every bound tab to its connection, a history
        click resumes one) and says nothing about the run just
        submitted: the answer to a replay asked for before the submit
        can land after it, so such a frame is kept but not taken as
        the end of the run.
        """
        frames: list[dict[str, Any]] = []
        for _ in range(500):
            msg = await self._next()
            if msg.get("tabId") != tab_id:
                continue
            frames.append(msg)
            if (
                msg.get("type") == "status"
                and not msg.get("running")
                and not msg.get("replayId")
            ):
                return frames
        raise AssertionError("the tab never went idle")

    async def _surface_frames(
        self, reader: LocalReader, writer: LocalWriter, tab_id: str, awaited: str,
    ) -> tuple[dict[str, dict[str, Any]], int]:
        """Read a surface's frames of *tab_id* until an *awaited* one, then fence the stream.

        Returns the first frame of each type seen for the tab and how
        many ``task_events`` arrived.  Once the awaited frame is in, a
        ``getInputHistory`` is sent: its reply (sent to this connection
        alone, in order) fences the stream, so a second replay would
        have shown up before it.
        """
        seen: dict[str, dict[str, Any]] = {}
        replays = 0
        fenced = False
        async for msg in _late_frames(reader):
            if msg.get("tabId") == tab_id:
                seen.setdefault(str(msg["type"]), msg)
                replays += msg["type"] == "task_events"
            if awaited in seen and not fenced:
                fenced = True
                writer.write(json.dumps({"type": "getInputHistory"}).encode() + b"\n")
                await writer.drain()
            elif fenced and msg.get("type") == "inputHistory":
                return seen, replays
        raise AssertionError(f"no {awaited} for {tab_id}")

    async def _titles(self, machine: str) -> list[str]:
        """The chat titles of *machine*'s history page."""
        reply = await self._ask({"type": "getHistory", "machine": machine}, "history")
        return [str(s["title"]) for s in reply["sessions"]]

    async def _remote_tabs_state(self) -> list[str]:
        """The tab ids the remote daemon's registry holds."""
        box = next(m for m in mach.load_machines() if m.name == "box")
        reply = await asyncio.to_thread(
            mach.link_for(box).request, {"type": "ready"}, "tabs_state",
        )
        return [str(t["tabId"]) for t in reply["tabs"]]

    async def _fan_out(self, tab: str, hold: float = 60.0) -> dict[str, Any]:
        """Submit a fanning-out run on *tab* to "box" and return the relayed ``new_tab``.

        The sub-agent's model call is held for *hold* seconds (or until
        ``_FinishingModel.released`` is set), so it is still running
        when the caller acts on the announcement.
        """
        _FinishingModel.parallel_on_first_call = True
        _FinishingModel.hold_sub_agent = hold
        _FinishingModel.calls = 0
        await self._submit(tab, "Fan out to a sub-agent", "box")
        for _ in range(400):
            msg = await self._next()
            if msg.get("type") == "new_tab" and msg.get("parent_tab_id") == tab:
                return msg
        raise AssertionError("the sub-agent's new_tab never arrived")


class TestRemoteTabs(RemoteTabsHarness):
    """A bound tab's run, replay, sub-agent tabs and loss."""

    async def test_deleting_a_running_task_of_the_machine_is_refused_here(self) -> None:
        """The CHATS panel's delete of a task running on "box" shows box's refusal."""
        sub = await self._fan_out("tab-del")
        await self._send({
            "type": "deleteTask", "taskId": sub["task_id"], "tabId": "tab-del",
            "machine": "box",
        })
        notice = await self._until("notice", "tab-del")
        self.assertEqual(
            notice["text"],
            "box: Cannot delete the task: its task is still running. Stop it first.",
        )
        _FinishingModel.released.set()

    async def test_submit_runs_on_the_selected_machine(self) -> None:
        tab = "tab-remote"
        await self._submit(tab, "Say hello", "box", "p-box")
        frames = await self._run_until_idle(tab)
        types = [f["type"] for f in frames]
        self.assertIn("status", types)
        # The machine answers the prompt under the id the page sent it by.
        self.assertEqual(
            [f.get("promptIds") for f in frames if f["type"] == "clear"], [["p-box"]],
        )
        # The binding is announced to every surface before the run's frames.
        self.assertEqual(
            (types[0], frames[0]["machine"]), ("tabMachine", "box"),
        )
        result = next(f for f in frames if f["type"] == "result")
        self.assertIn(SUMMARY, json.dumps(result))
        # The remote run's own terminal event reaches the surfaces under
        # the tab's id: the page's finish handling (the victory sound
        # among it) keys on exactly that frame.
        done = next(f for f in frames if f["type"] == "task_done")
        self.assertEqual(done["tabId"], tab)
        self.assertIn(tab, await self._remote_tabs_state(), "the remote has the tab")

        # Two surfaces connecting now (a reload, another window) each
        # learn the binding, even with their replays in flight together.
        # The one showing the tab gets the transcript once, as for a
        # local chat; the other gets the tab's state and an offer, and
        # the transcript when it takes the offer (the user opened the
        # tab), as a local tab is deferred.
        surfaces = [await open_local_connection(self.server) for _ in range(2)]
        try:
            (shown_reader, shown_writer), (other_reader, other_writer) = surfaces
            shown_writer.write(
                json.dumps({"type": "ready", "tabId": tab, "restoredTabs": []}).encode() + b"\n",
            )
            await shown_writer.drain()
            other_writer.write(json.dumps({"type": "ready", "restoredTabs": []}).encode() + b"\n")
            await other_writer.drain()
            seen, replays = await self._surface_frames(
                shown_reader, shown_writer, tab, "task_events",
            )
            self.assertEqual(seen["tabMachine"]["machine"], "box")
            self.assertEqual(seen["task_events"]["task"], "Say hello")
            self.assertIn(SUMMARY, json.dumps(seen["task_events"]["events"]))
            self.assertEqual(replays, 1, "one replay for the surface showing the tab")
            self.assertNotIn("replay_deferred", seen)
            chat_id = str(seen["task_events"]["chat_id"])
            seen, replays = await self._surface_frames(
                other_reader, other_writer, tab, "replay_deferred",
            )
            self.assertEqual(seen["tabMachine"]["machine"], "box")
            self.assertEqual(replays, 0, "no transcript for a tab the surface does not show")
            self.assertEqual(
                (seen["replay_deferred"]["chatId"], seen["replay_deferred"]["taskId"]),
                (chat_id, ""),
            )
            self.assertFalse(seen["status"]["running"], "the tab's state comes with the offer")
            self.assertNotIn("replayId", seen["status"])
            # Taking the offer (the page's scoped resumeSession, without
            # a machine) fetches the machine's transcript into the tab,
            # which stays bound, for this surface alone.
            other_writer.write(json.dumps({
                "type": "resumeSession", "id": chat_id, "tabId": tab, "scoped": True,
            }).encode() + b"\n")
            await other_writer.drain()
            seen, replays = await self._surface_frames(
                other_reader, other_writer, tab, "task_events",
            )
            self.assertEqual(seen["task_events"]["task"], "Say hello")
            self.assertIn(SUMMARY, json.dumps(seen["task_events"]["events"]))
            self.assertEqual(replays, 1)
            self.assertEqual(machine_of(self.server._server_api.remote_tabs, tab), "box")
            shown_writer.write(json.dumps({"type": "getInputHistory"}).encode() + b"\n")
            await shown_writer.drain()
            for _ in range(300):
                line = await asyncio.wait_for(shown_reader.readline(), timeout=90.0)
                msg = json.loads(line.decode("utf-8"))
                self.assertNotEqual(msg.get("type"), "task_events", "the fetch is the asker's")
                if msg.get("type") == "inputHistory":
                    break
        finally:
            for _, writer in surfaces:
                writer.close()
        # The replays went to the connecting surfaces alone: nothing of
        # them reached the first one, which was up to date.
        await self._send({"type": "getInputHistory"})
        for _ in range(300):
            msg = await self._next()
            self.assertNotEqual(msg.get("type"), "task_events")
            if msg.get("type") == "inputHistory":
                break
        # The task is the remote daemon's: its history has it, ours does not.
        self.assertNotIn("Say hello", await self._titles(mach.LOCAL_MACHINE))
        self.assertIn("Say hello", await self._titles("box"))
        # The tab is published in the local registry for every surface.
        tabs = await self._ask({"type": "ready", "tabId": tab}, "tabs_state")
        self.assertIn(tab, [t["tabId"] for t in tabs["tabs"]])
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, tab), "box")

        # A stop is forwarded to the machine (the idle tab answers quietly)
        # and the tab stays bound; closing the tab unbinds it.
        await self._send({"type": "stop", "tabId": tab})
        await asyncio.sleep(0.5)
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, tab), "box")
        await self._send({"type": "closeTab", "tabId": tab})
        await self._until("tabs_state")
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, tab), "")
        for _ in range(20):  # the forwarded closeTab reaches the remote asynchronously
            if tab not in await self._remote_tabs_state():
                break
            await asyncio.sleep(0.25)
        self.assertNotIn(tab, await self._remote_tabs_state(), "the remote closed it too")

    async def test_an_empty_machine_runs_here_and_unbinds(self) -> None:
        tab = "tab-home"
        await self._submit(tab, "Say hello", "box")
        await self._run_until_idle(tab)
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, tab), "box")
        # Back home: the local daemon has no model endpoint, so the run
        # is refused or fails HERE, but the tab is unbound either way.
        await self._submit(tab, "Say hello again", "")
        frames = await self._run_until_idle(tab)
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, tab), "")
        self.assertEqual(
            [f["machine"] for f in frames if f["type"] == "tabMachine"], [""],
            "the unbinding is announced",
        )

    async def test_a_fetch_into_a_tab_whose_first_task_is_starting_leaves_it_bound(self) -> None:
        """A page's fetch of the chat the tab showed before a submit bound it reads the tab.

        The tab shows no chat yet (the run's first frame is still to
        be relayed) when the fetch arrives.  Taken for a resume of a
        local chat, it unbound the tab: the run went on, on the
        machine, shown on no surface.
        """
        tab = "tab-first-task"
        remote_tabs = self.server._server_api.remote_tabs
        first_frame = Pause(
            RemoteTab._relay_frame.__code__, lambda local: local["msg"].get("tabId") == tab,
        )
        with first_frame:
            await self._submit(tab, "Say hello", "box")
            self.assertTrue(await asyncio.to_thread(first_frame.paused.wait, 30))
            stream = remote_tabs._tabs[tab]
            self.assertEqual(stream.chat_id, "")
            (page,) = self.server._printer._conn_endpoints
            fetch = {
                "type": "resumeSession", "chatId": "a-local-chat", "tabId": tab, "scoped": True,
                "connId": page,
            }
            self.assertTrue(await asyncio.to_thread(remote_tabs.route, fetch))
            self.assertIs(remote_tabs._tabs.get(tab), stream)
            # Nor is a surface connecting now replayed or offered a chat the tab has not.
            await asyncio.to_thread(remote_tabs.replay_to, page, tab)
            await asyncio.to_thread(remote_tabs.replay_to, page, "another-tab")
            self.assertEqual(stream.replay_sinks, {})
            first_frame.release()
        types = [f["type"] for f in await self._run_until_idle(tab)]
        self.assertIn("result", types)
        self.assertNotIn("replay_deferred", types)

    async def _machine_pin(self, stream: RemoteTab) -> str:
        """The task the machine's own registry pins *stream*'s tab to ("" for none)."""
        reply = await asyncio.to_thread(
            mach.link_for(stream.machine).request, {"type": "ready"}, "tabs_state",
        )
        return next(str(t["taskId"]) for t in reply["tabs"] if t["tabId"] == stream.tab_id)

    async def _pin_read_before_a_submit(self, fetch: bool) -> None:
        """A resume of the pinned task, read before a submit, reaches the machine before it.

        The tab is pinned to its finished task when a surface's
        ``ready`` replay of it (*fetch*: a page's own fetch of it)
        reads the pin and is held right before its send, and a task is
        submitted, which drops the pin.  The machine takes the resume
        as navigation: sent after the task had run, it pinned the
        machine's tab to the old task again, which this daemon showed
        at the new one.
        """
        tab = "tab-pinned"
        await self._submit(tab, "Say hello", "box")
        task = next(f["taskId"] for f in await self._run_until_idle(tab) if f["type"] == "result")
        remote_tabs = self.server._server_api.remote_tabs
        stream = remote_tabs._tabs[tab]
        view = {"type": "resumeSession", "tabId": tab, "chatId": stream.chat_id, "taskId": task}
        await self._send({**view, "machine": "box"})  # a CHATS-panel click on the task
        await self._until("task_events", tab)
        self.assertEqual(await self._machine_pin(stream), task)
        (page,) = self.server._printer._conn_endpoints
        stale = Pause(
            RemoteTab.send.__code__,
            lambda local: local["self"] is stream and bool(local["cmd"].get("replayId")),
        )
        with stale:
            resume = asyncio.create_task(asyncio.to_thread(
                remote_tabs.route, {**view, "scoped": True, "connId": page},
            ) if fetch else asyncio.to_thread(remote_tabs.replay_to, page, tab))
            self.assertTrue(await asyncio.to_thread(stale.paused.wait, 30))
            await self._submit(tab, "Second task", "box")
            idle = asyncio.create_task(self._run_until_idle(tab))
            # The task runs in the window if anything lets it past the held resume.
            await asyncio.wait([idle], timeout=3.0)
            stale.release()
            await resume
        await idle
        for _ in range(900):  # until the machine answered every forwarded resume
            if not stream.replay_sinks:
                break
            await asyncio.sleep(0.05)
        self.assertEqual(stream.replay_sinks, {})
        self.assertEqual(stream.shown_task_id, "")
        self.assertEqual(await self._machine_pin(stream), "")

    async def test_a_replay_of_the_pinned_task_read_before_a_submit_is_sent_before_it(self) -> None:
        """A surface's ``ready`` replay of the tab it shows."""
        await self._pin_read_before_a_submit(fetch=False)

    async def test_a_fetch_of_the_pinned_task_read_before_a_submit_is_sent_before_it(self) -> None:
        """A page's own fetch of the tab (its answer to a ``replay_deferred`` offer)."""
        await self._pin_read_before_a_submit(fetch=True)

    async def test_a_sub_agent_tab_brought_home_is_unbound_on_every_surface(self) -> None:
        """A prompt for this machine sent from a sub-agent tab on its parent's stream unbinds it.

        The tab left the stream without a word: the surfaces online
        kept it on the machine (its file links checked there, the
        machine's files in the sidebar) while its task ran here, where
        a freshly loaded page showed it.
        """
        root = "parent-sub-home"
        spawned = await self._fan_out(root)
        child = f"{root}__sub_{spawned['task_id']}"
        tabs = self.server._server_api.remote_tabs
        await self._send({"type": "resumeSession", "taskId": spawned["task_id"], "tabId": child})
        await self._until("task_events", child)
        self.assertEqual(machine_of(tabs, child), "box")
        await self._submit(child, "Say hello here", "")
        unbound = await asyncio.wait_for(self._until("tabMachine", child), 30)
        self.assertEqual(unbound["machine"], "")
        self.assertEqual((machine_of(tabs, child), machine_of(tabs, root)), ("", "box"))
        _FinishingModel.released.set()
        await self._run_until_idle(root)

    async def test_unknown_and_unreachable_machines_are_refused(self) -> None:
        """A submit no stream can take is refused by name (``promptIds``).

        ``RemoteTabs`` refuses a command with the same ``cmd=cmd`` at
        two more places no test reaches: a stream that dies between its
        connect and its registration (``_bind``), and a send that fails
        for a sub-agent tab before the relay thread has dropped the
        stream (``_forward``).  Each is a window of a few statements
        that only a test double could hold open.
        """
        await self._submit("tab-x", "Say hello", "nowhere", "p-nowhere")
        error = await self._until("error", "tab-x")
        self.assertEqual(error["code"], "prompt_refused")
        self.assertEqual(error["promptIds"], ["p-nowhere"], "the refusal names the prompt")
        self.assertIn("nowhere is not a registered machine", error["text"])

        dead = mach.Machine(name="dead", url=f"https://127.0.0.1:{find_free_port()}",
                            password="x")
        mach.save_machines([*mach.load_machines(), dead])
        await self._submit("tab-y", "Say hello", "dead", "p-dead")
        error = await self._until("error", "tab-y")
        self.assertEqual(error["code"], "prompt_refused")
        self.assertEqual(error["promptIds"], ["p-dead"])
        self.assertIn("dead (https://127.0.0.1:", error["text"])
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, "tab-y"), "")

    async def test_a_bind_still_connecting_at_shutdown_stands_down(self) -> None:
        """A shutdown during a bind's connect leaves no stream behind and forwards nothing.

        The bind connects outside the lock; ``close_all`` finds nothing
        to retire in that window, and the bind used to register its
        stream afterwards and forward the run to the machine.
        """
        tabs = self.server._server_api.remote_tabs
        tab = "tab-late"
        held = Pause(RemoteTab.__init__.__code__, lambda local: local["tab_id"] == tab)
        with held:
            await self._submit(tab, "Run after shutdown", "box")
            self.assertTrue(
                await asyncio.to_thread(held.paused.wait, 30), "the bind never connected",
            )
            await asyncio.to_thread(tabs.close_all)
            held.release()
        for _ in range(600):
            if not tabs._connecting:
                break
            await asyncio.sleep(0.05)
        self.assertEqual(tabs._connecting, {}, "the bind never ended")
        self.assertEqual(machine_of(tabs, tab), "")
        self.assertNotIn(tab, await self._remote_tabs_state())
        self.assertNotIn("Run after shutdown", await self._titles("box"))

    async def test_losing_the_machine_ends_the_tab_with_an_error(self) -> None:
        tab = "tab-lost"
        await self._submit(tab, "Say hello", "box")
        await self._run_until_idle(tab)
        self.remote.stop()
        error = await self._until("error", tab)
        self.assertIn("lost the connection to box", error["text"])
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, tab), "")

    async def test_losing_the_machine_ends_its_sub_agent_tabs(self) -> None:
        """A lost stream ends the sub-agent tabs it relayed, as a local run's end would.

        The remote is killed while its sub-agent still runs (its model
        call is held): nobody can send the sub-agent's ``subagentDone``
        any more, so the daemon does, after the root tab's error, or
        every surface would keep the sub-agent tab spinning for good.
        """
        _FinishingModel.parallel_on_first_call = True
        _FinishingModel.hold_sub_agent = 60.0
        tab = "tab-lost-parent"
        child = f"{tab}__sub"
        await self._submit(tab, "Fan out to a sub-agent", "box")
        for _ in range(400):
            msg = await self._next()
            if msg.get("type") == "new_tab" and msg.get("parent_tab_id") == tab:
                break
        else:
            raise AssertionError("the sub-agent's new_tab never arrived")
        await self._send({"type": "resumeSession", "taskId": msg["task_id"], "tabId": child})
        await self._until("task_events", child)
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, child), "box")
        self.remote.stop()
        # The unbinding (``tabMachine`` "") is the last thing the loss tells.
        seen: list[dict[str, Any]] = []
        for _ in range(400):
            msg = await self._next()
            seen.append(msg)
            if msg.get("type") == "tabMachine" and msg.get("tabId") == tab:
                self.assertEqual(msg["machine"], "")
                break
        else:
            raise AssertionError("the tab was never unbound")
        types = [(m["type"], m.get("tabId") or m.get("tab_id")) for m in seen]
        self.assertIn(("subagentDone", child), types, "the sub-agent tab was ended")
        errors = [m for m in seen if m.get("type") == "error" and m.get("tabId") == tab]
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("lost the connection to box", errors[0]["text"])
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, tab), "")
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, child), "")

    async def test_losing_the_machine_ends_a_sub_agent_tab_nobody_resumed_yet(self) -> None:
        """A sub-agent tab only announced so far (``new_tab``) ends with a lost stream too.

        The surfaces open the ``{parent}__sub_{task_id}`` tab on the
        relayed ``new_tab`` and only then ask for its transcript; the
        remote is killed in that window, before any ``resumeSession``
        joined the tab to the stream, so the stream never mapped the
        tab: the daemon must still end it, once, from the announcement
        alone.
        """
        _FinishingModel.parallel_on_first_call = True
        _FinishingModel.hold_sub_agent = 60.0
        tab = "tab-lost-unresumed-parent"
        await self._submit(tab, "Fan out to a sub-agent", "box")
        for _ in range(400):
            msg = await self._next()
            if msg.get("type") == "new_tab" and msg.get("parent_tab_id") == tab:
                break
        else:
            raise AssertionError("the sub-agent's new_tab never arrived")
        child = f"{tab}__sub_{msg['task_id']}"
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, child), "")
        self.remote.stop()
        seen: list[dict[str, Any]] = []
        for _ in range(400):
            msg = await self._next()
            seen.append(msg)
            if msg.get("type") == "tabMachine" and msg.get("tabId") == tab:
                self.assertEqual(msg["machine"], "")
                break
        else:
            raise AssertionError("the tab was never unbound")
        types = [(m["type"], m.get("tabId") or m.get("tab_id")) for m in seen]
        self.assertEqual(types.count(("subagentDone", child)), 1, types)
        errors = [m for m in seen if m.get("type") == "error" and m.get("tabId") == tab]
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("lost the connection to box", errors[0]["text"])
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, tab), "")

    async def _late_surface(self, tab: str) -> dict[str, dict[str, Any]]:
        """Connect a surface showing *tab*, send ``ready``, and collect its replay of *tab*.

        Returns the first ``tabMachine``, ``task_events`` of *tab*,
        ``openSubagentTab`` under it, the ``tabMachine`` of that
        sub-agent tab (``sub_tabMachine``, announced before it) and the
        offer of its transcript (``sub_replay_deferred``), by key.  The
        surface names *tab* as the one it shows: any other bound tab,
        and every sub-agent tab, is only offered (``replay_deferred``),
        not replayed.
        """
        reader, writer = await open_local_connection(self.server)
        try:
            writer.write(
                json.dumps({"type": "ready", "tabId": tab, "restoredTabs": []}).encode() + b"\n",
            )
            await writer.drain()
            seen: dict[str, dict[str, Any]] = {}
            bindings: dict[str, dict[str, Any]] = {}  # every tabMachine, by tab
            async for msg in _late_frames(reader):
                kind = str(msg.get("type"))
                if kind == "tabMachine":
                    bindings.setdefault(str(msg["tabId"]), msg)
                if msg.get("tabId") == tab and kind in ("tabMachine", "task_events"):
                    seen.setdefault(kind, msg)
                elif kind == "openSubagentTab" and msg.get("parent_tab_id") == tab:
                    seen.setdefault(kind, msg)
                    self.assertIn(
                        msg["tab_id"], bindings,
                        "the sub-agent tab's binding precedes its announcement",
                    )
                    seen["sub_tabMachine"] = bindings[msg["tab_id"]]
                elif (
                    kind == "replay_deferred" and "openSubagentTab" in seen
                    and msg.get("tabId") == seen["openSubagentTab"]["tab_id"]
                ):
                    seen.setdefault("sub_replay_deferred", msg)
                if len(seen) == 5:
                    return seen
            raise AssertionError(f"incomplete replay: {sorted(seen)}")
        finally:
            writer.close()

    async def test_sub_agent_tabs_follow_the_stream(self) -> None:
        """A remote run's sub-agent tab is announced, resumable, and ends here too."""
        _FinishingModel.parallel_on_first_call = True
        tab = "tab-parent"
        child = f"{tab}__sub"
        await self._submit(tab, "Fan out to a sub-agent", "box")
        seen: list[dict[str, Any]] = []
        resumed = False
        sub_tab = ""
        for _ in range(400):
            msg = await self._next()
            if msg.get("tabId") not in (tab, child, "", None):
                continue
            seen.append(msg)
            if msg.get("type") == "new_tab" and not resumed:
                # What every chat page does on new_tab: open a sub-agent tab
                # and ask for its transcript.
                self.assertEqual(msg["parent_tab_id"], tab)
                resumed = True
                await self._send({
                    "type": "resumeSession", "taskId": msg["task_id"], "tabId": child,
                })
                # A surface joining while the sub-agent runs: its ready
                # replays the parent AND announces the running sub-agent
                # (openSubagentTab, the remote's own sub-tab id), then
                # offers the sub-agent's transcript, all to that surface
                # alone.
                late = await self._late_surface(tab)
                self.assertEqual(late["tabMachine"]["machine"], "box")
                self.assertEqual(late["task_events"]["task"], "Fan out to a sub-agent")
                self.assertEqual(late["openSubagentTab"]["parent_tab_id"], tab)
                sub_tab = late["openSubagentTab"]["tab_id"]
                self.assertEqual(late["sub_replay_deferred"]["tabId"], sub_tab)
                self.assertEqual(
                    machine_of(self.server._server_api.remote_tabs, sub_tab), "box",
                    "the announced sub-agent tab joined the stream",
                )
                self.assertEqual(
                    late["sub_tabMachine"]["machine"], "box",
                    "the surface shows box for the sub-agent tab too",
                )
                # A second surface: the stream knows the sub-agent tab
                # now (nothing to adopt), so its binding is replayed
                # with the ready, still ahead of its offer.
                second = await self._late_surface(tab)
                self.assertEqual(second["openSubagentTab"]["tab_id"], sub_tab)
                self.assertEqual(second["sub_tabMachine"]["machine"], "box")
            if msg.get("type") == "status" and msg.get("tabId") == tab and not msg["running"]:
                break
        # This surface's own resume of the child answered with the child's
        # announcement and transcript; the late surface's did not reach it.
        self.assertEqual(
            [m["tab_id"] for m in seen if m["type"] == "openSubagentTab"], [child],
        )
        self.assertEqual(
            [m["tabId"] for m in seen if m["type"] == "task_events"], [child],
        )
        types = [(m["type"], m.get("tabId") or m.get("tab_id")) for m in seen]
        self.assertTrue(resumed, f"no new_tab among {types}")
        self.assertIn(("task_events", child), types, "the sub-agent's transcript replayed")
        # The sub-agent tab's binding is announced when it joins the
        # stream, before its transcript: switching to the tab shows box
        # in the chat header, not this machine.
        self.assertIn(("tabMachine", child), types, "the sub-agent tab's binding announced")
        self.assertLess(types.index(("tabMachine", child)), types.index(("task_events", child)))
        self.assertIn(("subagentDone", child), types, "the sub-agent's end reached the tab")
        # The end takes the sub-agent tab off the stream, with no page
        # closing it: its binding is announced gone, the parent's stays.
        self.assertEqual(
            [m["machine"] for m in seen if m["type"] == "tabMachine" and m["tabId"] == child],
            ["box", ""],
        )
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, child), "")
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, tab), "box")
        # Both tabs that showed the sub-agent are told their end, whichever
        # the machine named first, and the stream holds neither any more.
        self.assertLessEqual({child, sub_tab}, set(_ended_tabs(seen)))
        stream = self.server._server_api.remote_tabs._tabs[tab]
        with stream.lock:
            held = [
                tid for tid in (child, sub_tab)
                for kept in (stream.tab_ids, stream.children, stream.last_status)
                if tid in kept
            ]
        self.assertEqual(held, [], "the stream still holds an ended sub-agent's tab")

    async def _reuse_child_locally(self, child: str) -> None:
        task, chat = persistence._add_task("Local replacement")
        await self._send({
            "type": "resumeSession", "taskId": task, "chatId": chat, "tabId": child,
        })
        replay = await self._until("task_events", child)
        self.assertEqual(replay["chat_id"], chat)

    async def _late_child_announcement(self, joined: bool) -> None:
        root = "parent-late-child"
        spawned = await self._fan_out(root)
        child = f"{root}__sub_{spawned['task_id']}"
        tabs = self.server._server_api.remote_tabs
        if joined:
            await self._send({
                "type": "resumeSession", "taskId": spawned["task_id"], "tabId": child,
            })
            await self._until("task_events", child)
            held = Pause(RemoteTab._relay_frame.__code__, lambda v: (
                v["msg"].get("type") == "openSubagentTab" and v["msg"].get("tab_id") == child
            ))
        else:
            held = Pause(type(tabs).adopt.__code__, lambda v: v["sub_tab_id"] == child)
        with held:
            await self._send({"type": "ready", "tabId": root})
            self.assertTrue(await asyncio.to_thread(held.paused.wait, 30))
            await self._reuse_child_locally(child)
            held.release()
            _FinishingModel.released.set()
            leaked = []
            for _ in range(400):
                msg = await self._next()
                if msg.get("tabId") == child or msg.get("tab_id") == child:
                    leaked.append(msg)
                if (msg.get("type") == "status" and msg.get("tabId") == root
                        and not msg.get("running") and not msg.get("replayId")):
                    break
            else:
                self.fail("parent did not finish")
        self.assertEqual(machine_of(tabs, child), "", leaked)
        self.assertEqual(leaked, [], "old child frames reached the local replacement")

    async def test_late_child_announcement_cannot_reclaim_detached_tab(self) -> None:
        await self._late_child_announcement(joined=True)

    async def test_split_child_adoption_cannot_reclaim_local_tab(self) -> None:
        await self._late_child_announcement(joined=False)

    async def test_orphan_cleanup_cannot_close_local_replacement(self) -> None:
        root = "parent-orphan"
        spawned = await self._fan_out(root)
        child = f"{root}__sub_{spawned['task_id']}"
        await self._send({
            "type": "resumeSession", "taskId": spawned["task_id"], "tabId": child,
        })
        await self._until("task_events", child)
        tabs = self.server._server_api.remote_tabs
        stream = tabs._tabs[root]
        held = Pause(RemoteTab.close.__code__, lambda v: v["self"] is stream)
        with held:
            closing = asyncio.create_task(asyncio.to_thread(
                tabs.route, {"type": "newChat", "tabId": root},
            ))
            self.assertTrue(await asyncio.to_thread(held.paused.wait, 30))
            await self._reuse_child_locally(child)
            held.release()
            await closing
            seen = []
            for _ in range(300):
                msg = await self._next()
                seen.append(msg)
                if msg.get("type") == "tabMachine" and msg.get("tabId") == root:
                    break
            else:
                self.fail("parent did not unbind")
        self.assertFalse(any(
            m.get("type") == "subagentDone" and m.get("tab_id") == child for m in seen
        ), seen)
        _FinishingModel.released.set()

    async def test_lost_stream_cannot_stop_replacement_spinner(self) -> None:
        root = "root-replaced-loss"
        await self._submit(root, "First run", "box")
        await self._run_until_idle(root)
        tabs = self.server._server_api.remote_tabs
        stream = tabs._tabs[root]
        server = tabs._backend._agent_server
        held = Pause(type(server)._refuse_run.__code__, lambda v: v.get("lost") == "box")
        with held:
            await asyncio.to_thread(stream._ws.close)
            self.assertTrue(await asyncio.to_thread(held.paused.wait, 30))
            # If retirement still admits replacement while its refusal is held,
            # let the new task start first. Otherwise release the locked refusal.
            admits_replacement = tabs._lock.acquire(blocking=False)
            if admits_replacement:
                tabs._lock.release()
            _FinishingModel.parallel_on_first_call = True
            _FinishingModel.calls = 0
            _FinishingModel.hold_sub_agent = 60
            await self._submit(root, "Replacement run", "box")
            if not admits_replacement:
                held.release()
            while True:
                msg = await self._until("status", root)
                if msg.get("running") and not msg.get("replayId"):
                    break
            held.release()
            await asyncio.to_thread(stream._thread.join, 30)
            self.assertFalse(stream._thread.is_alive())
        await self._send({"type": "getInputHistory"})
        stale = []
        for _ in range(300):
            msg = await self._next()
            if msg.get("tabId") == root and (msg.get("lost") or msg.get("type") == "error"):
                stale.append(msg)
            if msg.get("type") == "inputHistory":
                break
        _FinishingModel.released.set()
        self.assertEqual(stale, [], "old loss cleared the new task's spinner")
        await self._run_until_idle(root)

    async def test_failed_bind_cannot_refuse_local_replacement(self) -> None:
        tab = "failed-bind-reused"
        dead = mach.Machine(name="dead", url=f"https://127.0.0.1:{find_free_port()}",
                            password="x")
        mach.save_machines([*mach.load_machines(), dead])
        tabs = self.server._server_api.remote_tabs
        held = Pause(RemoteTab.__init__.__code__, lambda v: v["tab_id"] == tab)
        with held:
            pending = asyncio.create_task(asyncio.to_thread(
                tabs.route, {"type": "submit", "tabId": tab, "machine": "dead"},
            ))
            self.assertTrue(await asyncio.to_thread(held.paused.wait, 30))
            await self._reuse_child_locally(tab)
            held.release()
            await pending
        await self._send({"type": "getInputHistory"})
        for _ in range(300):
            msg = await self._next()
            self.assertFalse(msg.get("tabId") == tab and msg.get("type") in ("status", "error"),
                             msg)
            if msg.get("type") == "inputHistory":
                break

    async def _frames_until_idle(self, root: str) -> list[dict[str, Any]]:
        """Every frame up to the end of *root*'s run (its own ``status running:false``)."""
        seen: list[dict[str, Any]] = []
        for _ in range(600):
            msg = await self._next()
            seen.append(msg)
            if (msg.get("type") == "status" and msg.get("tabId") == root
                    and not msg.get("running") and not msg.get("replayId")):
                return seen
        raise AssertionError(f"the run of {root} never ended here")

    async def test_a_bind_of_a_joined_sub_agent_tab_leaves_its_parents_stream_running(
        self,
    ) -> None:
        """A bind takes a sub-agent tab off its parent's stream; the stream itself stays.

        The tab joins its parent's stream (a surface's resume of the
        sub-agent) while a bind of it to another machine connects.  The
        bind, registering, found the tab on a stream and retired that
        stream as it retires a bind that won the race: the running
        parent's, with every other tab on it.
        """
        root = "parent-child-rebound"
        spawned = await self._fan_out(root)
        child = f"{root}__sub_{spawned['task_id']}"
        other = await asyncio.to_thread(
            start_remote_daemon, Path(self.tmpdir) / "other-home", password="pw-other",
            seed=[{"chat_id": "chat-other", "task": "Other's chat"}],
        )
        self.addCleanup(other.stop)
        await asyncio.to_thread(mach.add_machine, "other", other.url, "pw-other")
        tabs = self.server._server_api.remote_tabs
        held = Pause(RemoteTab.__init__.__code__, lambda v: v["tab_id"] == child)
        with held:
            binding = asyncio.create_task(asyncio.to_thread(tabs.route, {
                "type": "resumeSession", "id": "chat-other",
                "taskId": other.task_ids["chat-other/Other's chat"],
                "tabId": child, "machine": "other", "title": "Other's chat",
            }))
            self.assertTrue(await asyncio.to_thread(held.paused.wait, 30))
            await self._send({
                "type": "resumeSession", "taskId": spawned["task_id"], "tabId": child,
            })
            await self._until("task_events", child)
            self.assertEqual(machine_of(tabs, child), "box")
            held.release()
            await binding
        self.assertEqual(machine_of(tabs, child), "other")
        self.assertEqual(
            machine_of(tabs, root), "box", "the bind of the sub-agent tab retired its parent",
        )
        _FinishingModel.released.set()
        seen = await self._frames_until_idle(root)
        self.assertFalse(any(
            m.get("type") == "subagentDone" and m.get("tab_id") == child for m in seen
        ), "the sub-agent's end closed the tab that shows another machine's chat")

    async def test_a_bind_of_an_announced_sub_agent_tab_is_not_ended_by_the_sub_agent(
        self,
    ) -> None:
        """A sub-agent tab bound before any surface resumed it is no longer its announcer's.

        The parent's stream had only announced the tab (``new_tab``)
        when a machine's chat was opened in it: nothing mapped the tab,
        so nothing took it off that stream, whose relay of the
        sub-agent's ``subagentDone`` then closed the tab on every
        surface while it showed the chat.
        """
        await self._submit("tab-a", "Say hello", "box")
        await self._run_until_idle("tab-a")
        await self._send({"type": "closeTab", "tabId": "tab-a"})
        await self._until("tabs_state")
        hello = await self._box_row("Say hello")
        root = "parent-announced-rebound"
        spawned = await self._fan_out(root)
        child = f"{root}__sub_{spawned['task_id']}"
        tabs = self.server._server_api.remote_tabs
        self.assertEqual(machine_of(tabs, child), "", "no surface resumed the sub-agent tab")
        await self._click_history_row(child, hello)
        replay = await self._replay_of(child)
        self.assertEqual(replay["task_events"]["chat_id"], hello["id"])
        _FinishingModel.released.set()
        seen = await self._frames_until_idle(root)
        self.assertFalse(any(
            m.get("type") == "subagentDone" and m.get("tab_id") == child for m in seen
        ), "the sub-agent's end closed the tab that shows the machine's chat")
        self.assertEqual(machine_of(tabs, child), "box")
        late = await self._background_inventory(root)
        self.assertFalse(any(
            m["type"] == "subagentDone" and m.get("tab_id") == child for m in late
        ), late)

    async def _chat_opened_in_a_sub_agent_tab(self, joined: bool, resumed_meanwhile: bool) -> None:
        """A chat of the machine opened in a sub-agent tab on its parent's stream is that tab's.

        The tab is on its parent's stream (a surface's resume of the
        sub-agent: before the bind when *joined*, during its connect
        when *resumed_meanwhile*, as a page resumes on the ``new_tab``
        it is told) when another surface opens a chat of the same
        machine in it.  The bind kept the parent's stream for the tab,
        so the chat and its pinned task were written there: a surface
        connecting later was shown that chat in the parent's tab, and
        the sub-agent's end closed the tab that showed it.  A tab taken
        off that stream before the connect was no better: the resume
        landing meanwhile stood the bind down, and the tab showed
        neither the chat nor its sub-agent any more.
        """
        await self._submit("tab-a", "Say hello", "box")
        await self._run_until_idle("tab-a")
        await self._send({"type": "closeTab", "tabId": "tab-a"})
        await self._until("tabs_state")
        hello = await self._box_row("Say hello")
        root = "parent-joined-rebound"
        spawned = await self._fan_out(root)
        child = f"{root}__sub_{spawned['task_id']}"
        tabs = self.server._server_api.remote_tabs
        join = {"type": "resumeSession", "taskId": spawned["task_id"], "tabId": child}
        _, peer = await open_local_connection(self.server)
        self.addCleanup(peer.close)
        if joined:
            await self._send(join)
            await self._until("task_events", child)
            # A prompt typed there for a machine that is down is refused
            # and leaves the tab on its parent's stream: taken off it
            # before the connect, it had left for good.
            dead = mach.Machine(
                name="dead", url=f"https://127.0.0.1:{find_free_port()}", password="x",
            )
            mach.save_machines([*mach.load_machines(), dead])
            await self._submit(child, "Typed for a machine that is down", "dead")
            error = await asyncio.wait_for(self._until("error", child), 30)
            self.assertEqual(error["code"], "prompt_refused")
            self.assertEqual(machine_of(tabs, child), "box", "the failed bind unjoined the tab")
        if resumed_meanwhile:
            with Pause(RemoteTab.__init__.__code__, lambda v: v["tab_id"] == child) as held:
                await self._click_history_row(child, hello, writer=peer)
                self.assertTrue(await asyncio.to_thread(held.paused.wait, 30))
                await self._send(join)
                await asyncio.wait_for(self._until("task_events", child), 30)
        else:
            await self._click_history_row(child, hello, writer=peer)
        replay = await self._replay_of(child)
        self.assertEqual(replay["task_events"]["chat_id"], hello["id"])
        self.assertEqual(
            (await self._late_replay(root))["task_events"]["task"], "Fan out to a sub-agent",
            "the chat opened in the sub-agent tab is shown in its parent's tab",
        )
        self.assertEqual((await self._late_replay(child))["task_events"]["chat_id"], hello["id"])
        _FinishingModel.released.set()
        seen = await self._frames_until_idle(root)
        self.assertFalse(any(
            m.get("type") == "subagentDone" and m.get("tab_id") == child for m in seen
        ), "the sub-agent's end closed the tab that shows the machine's chat")
        self.assertEqual((machine_of(tabs, root), machine_of(tabs, child)), ("box", "box"))
        # Nor did the end, which no page answers, take anything of the
        # chat from the tab on either daemon.
        self.assertEqual((await self._late_replay(child))["task_events"]["chat_id"], hello["id"])
        # Closed on a surface of the machine: the tab bound here keeps
        # its stream, as any other bound tab closed there does.
        box_reader, box = await open_local_connection(self.remote.home)
        self.addCleanup(box.close)
        for cmd in ({"type": "closeTab", "tabId": child}, {"type": "getInputHistory"}):
            box.write(json.dumps(cmd).encode() + b"\n")
        await box.drain()
        await _frames_to(box_reader, "inputHistory")
        self.assertEqual((await self._late_replay(child))["task_events"]["chat_id"], hello["id"])

    async def test_a_chat_opened_in_a_joined_sub_agent_tab_is_not_its_parents(self) -> None:
        """The tab was on its parent's stream when the chat was opened in it."""
        await self._chat_opened_in_a_sub_agent_tab(joined=True, resumed_meanwhile=False)

    async def test_a_chat_opened_in_a_sub_agent_tab_joining_meanwhile_is_not_its_parents(
        self,
    ) -> None:
        """The tab joined its parent's stream while the chat's bind was connecting."""
        await self._chat_opened_in_a_sub_agent_tab(joined=False, resumed_meanwhile=True)

    async def test_a_chat_opened_in_a_sub_agent_tab_resumed_meanwhile_still_shows(self) -> None:
        """The tab was on its parent's stream, and another page resumed its sub-agent meanwhile."""
        await self._chat_opened_in_a_sub_agent_tab(joined=True, resumed_meanwhile=True)

    async def test_a_sub_agents_end_closes_no_chat_opened_in_its_tab_on_the_machine(self) -> None:
        """A sub-agent's end takes its tabs off the stream here and no chat's tab on the machine.

        No page tells a daemon that it closed a sub-agent's tab: the
        machine ends the views it holds, and this daemon takes the tabs
        off the stream as it relays the end.  A surface of the machine
        opened a chat in the tab named after the sub-agent meanwhile:
        the machine does not end that tab, which ends here with the
        view the machine's end names (a page closes the tab showing the
        task an end names), and the chat keeps its tab there.
        """
        await self._submit("tab-a", "Say hello", "box")
        await self._run_until_idle("tab-a")
        await self._send({"type": "closeTab", "tabId": "tab-a"})
        await self._until("tabs_state")
        hello = await self._box_row("Say hello")
        root = "parent-own-close"
        spawned = await self._fan_out(root)
        child, view = f"{root}__sub_{spawned['task_id']}", "view-own-close"
        tabs = self.server._server_api.remote_tabs
        for tab in (child, view):
            await self._send({"type": "resumeSession", "taskId": spawned["task_id"], "tabId": tab})
            await self._until("task_events", tab)
        box_reader, box = await open_local_connection(self.remote.home)  # a surface of the machine
        self.addCleanup(box.close)
        box.write(json.dumps({
            "type": "resumeSession", "id": hello["id"], "taskId": hello["task_id"], "tabId": child,
        }).encode() + b"\n")
        await box.drain()
        await _frames_to(box_reader, "task_events", child)
        _FinishingModel.released.set()
        seen = await self._frames_until_idle(root)
        self.assertEqual(_ended_tabs(seen), [view, child], "one end per tab that showed the task")
        self.assertEqual((machine_of(tabs, child), machine_of(tabs, view)), ("", ""))
        self.assertEqual(await self._announced_on_connect(), [], "a finished sub-agent's tab")
        self.assertIn(child, await self._remote_tabs_state(), "the machine's chat lost its tab")

    async def _background_inventory(self, root: str) -> list[dict[str, Any]]:
        reader, writer = await open_local_connection(self.server)
        try:
            writer.write(json.dumps({"type": "ready", "restoredTabs": []}).encode() + b"\n")
            await writer.drain()
            seen = []
            fenced = False
            async for msg in _late_frames(reader):
                if msg.get("type") in (
                    "tabMachine", "new_tab", "openSubagentTab", "status", "task_events",
                    "subagentDone",
                ):
                    seen.append(msg)
                if msg.get("type") == "replay_deferred" and msg.get("tabId") == root:
                    fenced = True
                    writer.write(json.dumps({"type": "getInputHistory"}).encode() + b"\n")
                    await writer.drain()
                if fenced and msg.get("type") == "inputHistory":
                    return seen
            self.fail("ready did not finish")
        finally:
            writer.close()

    async def test_background_ready_includes_live_children_but_not_finished(self) -> None:
        root = "parent-background"
        spawned = await self._fan_out(root)
        child = f"{root}__sub_{spawned['task_id']}"
        await self._send({
            "type": "resumeSession", "taskId": spawned["task_id"], "tabId": child,
        })
        await self._until("task_events", child)
        seen = await self._background_inventory(root)
        self.assertTrue(any(
            m.get("type") == "openSubagentTab" and m.get("tab_id") == child for m in seen
        ), seen)
        self.assertTrue(any(
            m.get("type") == "status" and m.get("tabId") == child and m.get("running")
            for m in seen
        ), seen)
        self.assertFalse(any(m.get("type") == "task_events" for m in seen), seen)
        self.assertFalse(any(m.get("replayId") for m in seen), seen)
        # Every replay of the parent announces the running child again,
        # to the page that launched it too.
        chat = self.server._server_api.remote_tabs._tabs[root].chat_id
        await self._send({"type": "resumeSession", "id": chat, "tabId": root, "scoped": True})
        again = await _frames_to(self.reader, "replay_deferred", child)
        self.assertIn(child, [m.get("tab_id") for m in again if m["type"] == "openSubagentTab"])
        _FinishingModel.released.set()
        await self._run_until_idle(root)
        seen = await self._background_inventory(root)
        self.assertFalse(any(m.get("type") in ("new_tab", "openSubagentTab") for m in seen))

    async def _frames_on_connect(self) -> list[dict[str, Any]]:
        """Every frame a surface connecting now gets on its ``ready``.

        Its ``ready`` answers one ``inputHistory`` before the bound tabs are
        replayed; the second, to its own request, comes after the replay.
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

    async def _announced_on_connect(self) -> list[dict[str, Any]]:
        """The sub-agent tabs a surface connecting now is told to open."""
        frames = await self._frames_on_connect()
        return [f for f in frames if f["type"] in ("new_tab", "openSubagentTab")]

    async def _closed_api_run_tab(self, resumed: bool) -> None:
        """A local task's run on "box": its tab, once closed, is not announced on connect.

        The run's stream (an ``api-…`` tab no surface shows) replays the
        run's ``new_tab`` to a connecting surface.  A close of the tab it
        opens, *resumed* by a page or only announced so far, ends that
        while the run goes on, and a page's own resume of the ``new_tab``
        it got before the close, landing after it, brings nothing back.
        """
        script = Path(self.tmpdir) / "held_sea.py"
        script.write_text(HELD_SEA, encoding="utf-8")
        await self._send({
            "type": "run", "prompt": "Run on box", "tabId": "api-run", "taskId": "token",
            "machine": "box", "model": self.model, "workDir": self.tmpdir,
            "parentTaskId": "c" * 32, "parentTabId": "caller", "seaPath": str(script),
        })
        spawn = await self._until("new_tab")
        # The held run's last event: its transcript is settled from here on.
        self.assertEqual((await self._until("prompt", "api-run"))["taskId"], spawn["task_id"])
        child = f"caller__sub_{spawn['task_id']}"
        resume = {"type": "resumeSession", "taskId": spawn["task_id"], "tabId": child}
        if resumed:
            await self._send(resume)
            await self._until("task_events", child)
        self.assertEqual(await self._announced_on_connect(), [spawn])
        await self._send({"type": "closeTab", "tabId": child})
        await self._ask({"type": "getInputHistory", "tabId": child}, "inputHistory")
        self.assertEqual(await self._announced_on_connect(), [])
        await self._send(resume)  # a page's own, of the ``new_tab`` it got before the close
        await self._send({"type": "getInputHistory", "tabId": child})
        late = [await self._next()]
        while late[-1]["type"] != "inputHistory":
            late.append(await self._next())
        self.assertFalse([f for f in late[:-1] if f.get("tabId") == child], "a closed tab answered")
        self.assertEqual(await self._announced_on_connect(), [])
        script.with_name("release").touch()
        await self._run_until_idle("api-run")

    async def test_an_api_run_tab_closed_after_its_resume_is_not_announced_on_connect(self) -> None:
        """The closed tab had joined the run's stream (a page resumed it)."""
        await self._closed_api_run_tab(resumed=True)

    async def test_an_api_run_tab_closed_before_a_resume_is_not_announced_on_connect(self) -> None:
        """The closed tab was only announced: no page had resumed it yet."""
        await self._closed_api_run_tab(resumed=False)

    async def test_background_ready_ends_a_child_that_finished_while_the_surface_was_away(
        self,
    ) -> None:
        """A surface that missed a child's ``subagentDone`` gets it on its ``ready``.

        A page keeps the sub-agent tabs the registry snapshot does not
        list: without the end replayed, the child would spin for good
        on a page that was away.  The child left the stream at its end,
        with no page closing it.  A page's resume of the finished child
        is answered with the end too, by this daemon (the machine's
        ``subagentDone``, which names the sub-agent's task; the machine
        is not asked about a tab that left its stream), and the end
        stays kept, past the parent's stream as well.
        """
        root = "parent-away"
        spawned = await self._fan_out(root)
        child = f"{root}__sub_{spawned['task_id']}"
        resume = {"type": "resumeSession", "taskId": spawned["task_id"], "tabId": child}
        await self._send(resume)
        await self._until("task_events", child)
        _FinishingModel.released.set()
        # The child's subagentDone precedes the root's terminal status on
        # the stream.
        await self._run_until_idle(root)
        seen = await self._background_inventory(root)
        types = [(m["type"], m.get("tabId") or m.get("tab_id")) for m in seen]
        self.assertEqual(_ended_tabs(seen), [child], types)
        self.assertEqual(
            [m["task_id"] for m in seen if m["type"] == "subagentDone"], [spawned["task_id"]],
        )
        self.assertNotIn(("tabMachine", child), types, "the ended child is still on the stream")
        self.assertFalse(any(m["type"] in ("new_tab", "openSubagentTab") for m in seen), types)
        self.assertFalse(any(m["type"] == "status" and m.get("tabId") == child for m in seen))
        # A page that asks for the finished child (its resume of a
        # ``new_tab`` it took late) is told the end kept here.
        await self._send(resume)
        done = await self._until("subagentDone")
        self.assertEqual((done["tab_id"], done["task_id"]), (child, spawned["task_id"]))
        self.assertNotIn("replayId", done, "the machine was asked about a tab that left its stream")
        self.assertEqual(_ended_tabs(await self._background_inventory(root)), [child])
        # The parent's stream retires (a new chat in its tab): the surface
        # that was away shows the child as before, so the end outlives it.
        await self._send({"type": "newChat", "tabId": root})
        await self._until("tabMachine", root)
        self.assertEqual(_ended_tabs(await self._frames_on_connect()), [child])

    async def test_an_online_surfaces_auto_close_keeps_the_end_for_the_surfaces_away(
        self,
    ) -> None:
        """A ``closeTab`` of a finished sub-agent's tab keeps the end for the pages away.

        A page closes a finished sub-agent's tab and tells the daemon
        nothing; an older page, or a script, still sends a ``closeTab``
        for it.  A page that was away still has the tab, which the
        registry snapshot never listed: only the ``subagentDone``
        replayed on its ``ready`` closes it, so that close must not
        erase it.  The child is off the stream (no binding, no state is
        replayed for it).
        """
        root = "parent-auto-closed"
        spawned = await self._fan_out(root)
        child = f"{root}__sub_{spawned['task_id']}"
        await self._send({
            "type": "resumeSession", "taskId": spawned["task_id"], "tabId": child,
        })
        await self._until("task_events", child)
        _FinishingModel.released.set()
        await self._run_until_idle(root)
        await self._send({"type": "closeTab", "tabId": child})
        await self._send({"type": "getInputHistory", "tabId": child})
        await self._until("inputHistory")
        seen = await self._background_inventory(root)
        types = [(m["type"], m.get("tabId") or m.get("tab_id")) for m in seen]
        self.assertEqual(_ended_tabs(seen), [child], types)
        self.assertNotIn(("tabMachine", child), types)
        self.assertFalse(any(
            m["type"] in ("new_tab", "openSubagentTab", "status")
            and (m.get("tabId") == child or m.get("tab_id") == child) for m in seen
        ), types)

    async def test_a_closed_childs_tab_reused_for_a_local_chat_is_not_ended_by_the_child(
        self,
    ) -> None:
        """A local chat resumed into a closed child's tab is not closed by the child's kept end.

        A child resumed into a viewer tab of any id (not derived from
        the parent's) finishes, an online page closes the tab and the
        ``subagentDone`` is kept for the pages away.  A local chat then
        resumed into the same tab id makes the tab that chat's: the kept
        end must go with the reuse (the tab is no longer on the stream
        and its id does not name the stream, so only the detach itself
        can drop it), or a surface connecting later would close the
        local chat's view.  The parent's stream retires in between (a
        new chat in its tab): the pages away show the child as before,
        so the end outlives the stream, and the reuse still drops it.
        """
        root = "parent-alias-reused"
        spawned = await self._fan_out(root)
        child = "alias-child-view"
        await self._send({
            "type": "resumeSession", "taskId": spawned["task_id"], "tabId": child,
        })
        await self._until("task_events", child)
        _FinishingModel.released.set()
        await self._run_until_idle(root)
        await self._send({"type": "closeTab", "tabId": child})
        await self._send({"type": "getInputHistory", "tabId": child})
        await self._until("inputHistory")
        seen = await self._background_inventory(root)
        # Kept for the pages away (the machine ends the derived
        # ``{root}__sub_{task}`` tab of the pages that only saw the
        # child's ``new_tab`` as well; that end is not this tab's).
        self.assertIn(child, _ended_tabs(seen), seen)
        await self._send({"type": "newChat", "tabId": root})
        await self._until("tabMachine", root)
        self.assertIn(child, _ended_tabs(await self._frames_on_connect()))
        await self._reuse_child_locally(child)
        self.assertNotIn(child, _ended_tabs(await self._frames_on_connect()))

    async def test_a_tab_ended_with_its_stream_is_ended_for_the_surfaces_away_too(self) -> None:
        """What a stream's end tells the surfaces online is kept for the ones away.

        The stream is lost while its sub-agent runs: the sub-agent's tab
        is ended on every surface online.  A page that was away shows it
        as before, and its parent's stream, retired, replays nothing: the
        end is kept past the stream.  Not past the tab's return: the chat
        opened in the parent's tab again brings a stream on which the
        machine announces the sub-agent, still running there, and the
        end kept would close that tab on the next surface to connect.
        Nor does anything a page did about the tab the lost stream ended
        reach the tab announced again: a page tells the daemon nothing
        of a sub-agent's tab, so the sub-agent's own end closes it.
        """
        root = "parent-lost-then-back"
        spawned = await self._fan_out(root)
        child = f"{root}__sub_{spawned['task_id']}"
        await self._send({
            "type": "resumeSession", "taskId": spawned["task_id"], "tabId": child,
        })
        opened = await self._until("openSubagentTab")
        await self._until("task_events", child)
        stream = self.server._server_api.remote_tabs._tabs[root]
        await asyncio.to_thread(stream._ws.close)  # lost here; the machine runs on
        # The stream's end, not the sub-agent's: the frame names no task.
        ended = await self._until("subagentDone")
        self.assertEqual((ended["tab_id"], ended["task_id"]), (child, ""))
        await self._until("tabMachine", root)
        away = await self._frames_on_connect()
        self.assertEqual(_ended_tabs(away), [child], away)
        page = await asyncio.to_thread(page_after, root, [spawned, opened, *away])
        self.assertEqual(page["shown"], [], "the page that was away keeps the ended tab")
        stale = page["closed"]  # what that page sent the daemon for the ended tab
        # A page asking for the ended tab by name (its question after a
        # reconnect, its resume of the ``new_tab`` in after the loss) is
        # told that end, which stays kept for the surfaces still away.
        # (Every connection's command carries its ``connId``, so the
        # relay's "no connection to answer" branch cannot be driven here.)
        await self._send({
            "type": "resumeSession", "taskId": spawned["task_id"], "tabId": child, "scoped": True,
        })
        await self._send({"type": "getInputHistory"})
        answer = [await self._next()]
        while answer[-1]["type"] != "inputHistory":
            answer.append(await self._next())
        self.assertEqual(_ended_tabs(answer), [child], answer)
        self.assertEqual(_ended_tabs(await self._frames_on_connect()), [child])
        await self._click_history_row(root, await self._box_row("Fan out to a sub-agent"))
        self.assertEqual((await self._until("openSubagentTab"))["tab_id"], child)
        offer = await self._until("replay_deferred", child)
        seen = await self._background_inventory(root)
        self.assertEqual(_ended_tabs(seen), [], seen)
        page = await asyncio.to_thread(page_after, root, seen)
        self.assertEqual(page["shown"], [child], "the tab its machine announced again is closed")
        # A page fetches the sub-agent announced again, and what the
        # first page sent lands only now: the sub-agent's own end still
        # closes the tab on every page.
        await self._send({
            "type": "resumeSession", "chatId": offer["chatId"], "taskId": offer["taskId"],
            "tabId": child, "scoped": True,
        })
        await self._until("task_events", child)
        for cmd in stale:
            await self._send(cmd)
        await self._ask({"type": "getInputHistory"}, "inputHistory")
        _release_model()
        run = await self._frames_until_idle(root)
        self.assertIn(
            (child, spawned["task_id"]),
            [(m.get("tab_id"), m.get("task_id")) for m in run if m["type"] == "subagentDone"],
            "a close of the tab the lost stream ended took the running sub-agent's tab away",
        )
        page = await asyncio.to_thread(page_after, root, [*seen, *run])
        self.assertEqual((page["shown"], page["closed"]), ([], []))

    async def test_a_chat_opened_in_another_tab_takes_its_sub_agent_tab_along_on_every_page(
        self,
    ) -> None:
        """A running chat opened in another tab shows its sub-agent there, on every page.

        One chat, one tab: the daemon closes the tab the chat was in and
        ends the sub-agent's tab under it, then the machine announces the
        sub-agent, still running, under the new tab.  A page that had the
        closed tab on screen falls back to the first chat left, another
        one, so the sub-agent's tab opens in a group that is not on
        screen.  The page that was online throughout and one that
        connects afterwards both end with that one sub-agent tab.
        """
        first, other, second = "parent-first-tab", "another-chat", "parent-second-tab"
        spawned = await self._fan_out(first)
        old = f"{first}__sub_{spawned['task_id']}"
        await self._send({"type": "openTab", "tabId": other, "title": "new chat", "ack": "other"})
        while "other" not in (await self._until("tabs_state")).get("acks", []):
            pass
        await self._send({"type": "resumeSession", "taskId": spawned["task_id"], "tabId": old})
        online = [spawned, await self._until("openSubagentTab")]
        await self._until("task_events", old)
        await self._click_history_row(second, await self._box_row("Fan out to a sub-agent"))
        child = f"{second}__sub_{spawned['task_id']}"
        while (online[-1]["type"], online[-1].get("tab_id")) != ("openSubagentTab", child):
            online.append(await self._next())
        self.assertEqual(_ended_tabs(online), [old], "the tab under the closed one is not ended")
        for name, frames in (
            ("online throughout", online),
            ("connecting afterwards", await self._frames_on_connect()),
        ):
            page = await asyncio.to_thread(page_after, first, frames)
            self.assertEqual(page["shown"], [child], name)

    async def test_the_ends_kept_past_a_stream_are_a_thousand_at_most(self) -> None:
        """A retired stream's ends are kept for the surfaces away, a thousand tabs at most.

        Each one is told to every surface that connects for as long as
        the daemon runs, and only a reuse of its tab id forgets it:
        unbounded, every remote sub-agent of a long session would be
        told to every page.  The oldest end goes first.
        """
        root = "parent-many-ended"
        spawned = await self._fan_out(root, hold=0)
        child = f"{root}__sub_{spawned['task_id']}"
        await self._run_until_idle(root)  # the sub-agent's end, told for its own tab: the oldest
        # A thousand more views of the finished sub-agent: the machine
        # answers each resume with the end, which takes the view off
        # the stream.
        views = {f"view-{n}" for n in range(1000)}
        for tab in views:
            await self._send({"type": "resumeSession", "taskId": spawned["task_id"], "tabId": tab})
        ended: set[str] = set()
        left: set[str] = set()
        while ended != views or left != views:
            msg = await self._next()
            ended.update(views.intersection(_ended_tabs([msg])))
            if msg["type"] == "tabMachine" and not msg["machine"]:
                left.update(views.intersection([msg["tabId"]]))
        self.assertEqual(set(_ended_tabs(await self._frames_on_connect())), views | {child})
        await self._send({"type": "newChat", "tabId": root})
        await self._until("tabMachine", root)
        kept = _ended_tabs(await self._frames_on_connect())
        self.assertEqual(len(kept), 1000)
        self.assertEqual(set(kept), views, "the oldest end is not the one forgotten")

    async def _box_row(self, title: str) -> dict[str, Any]:
        """The history row of box's task *title*, as the CHATS panel gets it."""
        reply = await self._ask({"type": "getHistory", "machine": "box"}, "history")
        row: dict[str, Any] = next(s for s in reply["sessions"] if s["title"] == title)
        self.assertEqual(row["machine"], "box")
        return row

    async def _click_history_row(
        self, tab: str, row: dict[str, Any], machine: str = "box", scoped: bool = False,
        writer: LocalWriter | None = None,
    ) -> None:
        """Send what a click on a CHATS-panel row of a remote machine sends."""
        cmd: dict[str, Any] = {
            "type": "resumeSession", "id": row["id"], "taskId": row["task_id"],
            "tabId": tab, "machine": machine, "title": row["title"],
        }
        if scoped:
            cmd["scoped"] = True
        if writer is None:
            await self._send(cmd)
        else:
            writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
            await writer.drain()

    async def _replay_of(self, tab: str) -> dict[str, Any]:
        """The binding and transcript a resume of *tab* answers with, by frame type."""
        seen: dict[str, dict[str, Any]] = {}
        for _ in range(300):
            msg = await self._next()
            if msg.get("tabId") == tab and msg["type"] in ("tabMachine", "task_events"):
                seen.setdefault(str(msg["type"]), msg)
            if "task_events" in seen:
                return seen
        raise AssertionError(f"no replay of {tab}: {sorted(seen)}")

    async def test_a_history_click_shows_the_machines_chat_here(self) -> None:
        """A CHATS-panel row of a remote machine opens its chat in a local tab.

        The click's ``resumeSession`` names the machine: the daemon binds
        the tab to it and forwards the resume, the machine replays the
        chat into the tab, the tab's next prompt continues that chat on
        the machine, and every surface shows it; the machine's own web
        UI is never involved.
        """
        remote_tabs = self.server._server_api.remote_tabs
        # Two chats on box, their tabs closed: nothing of them is shown here.
        await self._submit("tab-a", "Say hello", "box")
        await self._run_until_idle("tab-a")
        await self._submit("tab-b", "Say goodbye", "box")
        await self._run_until_idle("tab-b")
        for tab in ("tab-a", "tab-b"):
            await self._send({"type": "closeTab", "tabId": tab})
            await self._until("tabs_state")
        hello = await self._box_row("Say hello")
        goodbye = await self._box_row("Say goodbye")
        self.assertNotEqual(hello["id"], goodbye["id"], "two chats")

        # The click: a fresh tab, bound to box on the resume.
        tab = "tab-click"
        await self._send({"type": "openTab", "tabId": tab, "title": "Say hello"})
        await self._click_history_row(tab, hello)
        seen = await self._replay_of(tab)
        self.assertEqual(seen["tabMachine"]["machine"], "box", "the binding is announced")
        self.assertEqual(seen["task_events"]["task"], "Say hello")
        self.assertEqual(seen["task_events"]["chat_id"], hello["id"])
        self.assertIn(SUMMARY, json.dumps(seen["task_events"]["events"]))
        self.assertEqual(machine_of(remote_tabs, tab), "box")
        tabs = await self._ask({"type": "ready", "tabId": tab}, "tabs_state")
        self.assertEqual(
            next(t["title"] for t in tabs["tabs"] if t["tabId"] == tab), "Say hello",
            "the registry names the tab after the task",
        )
        self.assertNotIn("Say hello", await self._titles(mach.LOCAL_MACHINE),
                         "nothing of the chat was copied here")

        # The next prompt continues the chat on box (the chat header's
        # drop-down shows the binding, so the submit names the machine).
        await self._submit(tab, "Say hello again", "box")
        frames = await self._run_until_idle(tab)
        self.assertIn(SUMMARY, json.dumps(next(f for f in frames if f["type"] == "result")))
        again = await self._box_row("Say hello again")
        self.assertEqual(again["id"], hello["id"], "the same chat, continued on box")

        # A surface connecting now gets the binding and the chat at its
        # latest task (the submit's).
        late = await self._late_replay(tab)
        self.assertEqual(late["tabMachine"]["machine"], "box")
        self.assertEqual(late["task_events"]["chat_id"], hello["id"])
        self.assertEqual(late["task_events"]["task"], "Say hello again")

        # Clicking the chat's FIRST task pins the tab to it: a surface
        # connecting later sees that task, as it would for a local chat.
        await self._click_history_row(tab, hello)
        seen = await self._replay_of(tab)
        self.assertEqual(seen["task_events"]["task"], "Say hello")
        late = await self._late_replay(tab)
        self.assertEqual(late["task_events"]["task"], "Say hello")

        # Scrolling past the end of that task asks for the chat's next
        # one: box holds the chat, so box names it.
        after = await self._ask(
            {"type": "getAdjacentTask", "tabId": tab, "taskId": hello["task_id"],
             "direction": "next"},
            "adjacent_task_events",
        )
        self.assertEqual(
            (after["task"], after["adjacent_to"]), ("Say hello again", hello["task_id"]),
        )

        # Another of box's chats clicked into the same tab: the tab stays
        # bound (one stream), shows that chat, and a surface connecting
        # later gets that chat, not the first.
        await self._click_history_row(tab, goodbye)
        seen = await self._replay_of(tab)
        self.assertEqual(seen["task_events"]["chat_id"], goodbye["id"])
        self.assertEqual(machine_of(remote_tabs, tab), "box")
        late = await self._late_replay(tab)
        self.assertEqual(late["task_events"]["chat_id"], goodbye["id"])

        # A scoped resume (the fetch of a deferred replay) answers the
        # asking connection alone: this one sees no transcript of it.
        reader, writer = await open_local_connection(self.server)
        try:
            await self._click_history_row(tab, hello, scoped=True, writer=writer)
            for _ in range(300):
                line = await asyncio.wait_for(reader.readline(), timeout=90.0)
                msg = json.loads(line.decode("utf-8"))
                if msg.get("type") == "task_events" and msg.get("tabId") == tab:
                    self.assertEqual(msg["chat_id"], hello["id"])
                    break
            else:
                raise AssertionError("the scoped resume was not answered")
        finally:
            writer.close()
        await self._send({"type": "getInputHistory"})
        for _ in range(300):
            msg = await self._next()
            self.assertNotEqual((msg.get("type"), msg.get("tabId")), ("task_events", tab))
            if msg.get("type") == "inputHistory":
                break

        # A local chat clicked into the tab brings it home (no machine).
        await self._send({"type": "resumeSession", "id": "no-such-chat", "tabId": tab})
        for _ in range(40):
            if not machine_of(remote_tabs, tab):
                break
            await asyncio.sleep(0.25)
        self.assertEqual(machine_of(remote_tabs, tab), "")

        # A row of a machine that is not registered (removed meanwhile).
        await self._click_history_row("tab-z", hello, machine="nowhere")
        error = await self._until("error", "tab-z")
        self.assertEqual(error["code"], "prompt_refused")
        self.assertIn("nowhere is not a registered machine", error["text"])
        self.assertEqual(machine_of(remote_tabs, "tab-z"), "")

    async def _late_replay(self, tab: str) -> dict[str, dict[str, Any]]:
        """Connect a surface showing *tab*, send ``ready``; return its frames of *tab* by type."""
        reader, writer = await open_local_connection(self.server)
        try:
            writer.write(
                json.dumps({"type": "ready", "tabId": tab, "restoredTabs": []}).encode() + b"\n",
            )
            await writer.drain()
            seen: dict[str, dict[str, Any]] = {}
            async for msg in _late_frames(reader):
                if msg.get("tabId") == tab and msg["type"] in ("tabMachine", "task_events"):
                    seen.setdefault(str(msg["type"]), msg)
                if len(seen) == 2:
                    return seen
            raise AssertionError(f"incomplete replay: {sorted(seen)}")
        finally:
            writer.close()

    async def test_a_running_sub_agent_of_a_chat_shown_from_a_machine_is_fetched(self) -> None:
        """The transcript a machine offers for a sub-agent tab is fetched over its stream.

        A chat opened from a machine's row while its sub-agent runs:
        the stream never saw the sub-agent's ``new_tab``.  The
        machine's replay announces the tab and offers its transcript
        (``replay_deferred``), and the page answers the offer when the
        user opens the tab, with a ``scoped`` resume that names the
        sub-agent's chat.  Taken for a local chat opened in the tab,
        that fetch took the tab off the stream: it showed the
        sub-agent's task and none of its events, for good.
        """
        remote_tabs = self.server._server_api.remote_tabs
        spawned = await self._fan_out("tab-p")
        await self._send({"type": "closeTab", "tabId": "tab-p"})
        await self._until("tabs_state")
        tab = "tab-click"
        sub = f"{tab}__sub_{spawned['task_id']}"
        await self._send({"type": "openTab", "tabId": tab, "title": "Fan out to a sub-agent"})
        await self._click_history_row(tab, await self._box_row("Fan out to a sub-agent"))
        await self._replay_of(tab)
        # A page showing the chat is offered the running sub-agent's
        # transcript and fetches it as requestDeferredReplay (main.js) does.
        reader, writer = await open_local_connection(self.server)
        self.addCleanup(writer.close)
        writer.write(
            json.dumps({"type": "ready", "tabId": tab, "restoredTabs": []}).encode() + b"\n",
        )
        await writer.drain()
        offer = (await _frames_to(reader, "replay_deferred", sub))[-1]
        writer.write(json.dumps({
            "type": "resumeSession", "chatId": offer["chatId"], "tabId": sub,
            "scoped": True, "taskId": offer["taskId"],
        }).encode() + b"\n")
        await writer.drain()
        events: dict[str, Any] = {}
        async for msg in _late_frames(reader):
            self.assertNotEqual(
                (msg.get("type"), msg.get("tabId"), msg.get("machine")), ("tabMachine", sub, ""),
                "the fetch took the sub-agent tab off its machine's stream",
            )
            if (msg.get("type"), msg.get("tabId")) == ("task_events", sub):
                events = msg
                break
        self.assertEqual(str(events.get("task_id")), spawned["task_id"])
        self.assertEqual(machine_of(remote_tabs, sub), "box")
        # Still the stream's tab: the sub-agent's end reaches it.
        _release_model()
        ended: list[str] = []
        async for msg in _late_frames(reader):
            ended = _ended_tabs([msg])
            if ended == [sub]:
                break
        self.assertEqual(ended, [sub])

    async def test_a_page_that_loads_in_pieces_gets_the_machines_transcript_in_pieces(self) -> None:
        """A bound tab's transcript reaches a page that loads in pieces as the machine's pieces.

        Such a page's ``ready`` says so (``lazy``): the replay asked of
        the machine is then the finished task's Trajectory stub and its
        result, and the page's ``getTaskEvents`` for the stub, stamped
        with the tab's machine, is answered by the machine's daemon,
        which alone holds the events.  A page whose ``ready`` does not
        say so gets the whole transcript.
        """
        tab = "tab-lazy"
        await self._submit(tab, "Say hello", "box")
        await self._run_until_idle(tab)
        ready = {"type": "ready", "tabId": tab, "restoredTabs": []}
        # The events of this run that open a panel: all are the Trajectory panel's.
        folded = ("system_prompt", "prompt", "tool_call")

        reader, writer = await open_local_connection(self.server)
        self.addCleanup(writer.close)
        writer.write(json.dumps({**ready, "lazy": True}).encode() + b"\n")
        await writer.drain()
        seen, replays = await self._surface_frames(reader, writer, tab, "task_events")
        self.assertEqual(seen["tabMachine"]["machine"], "box")
        self.assertEqual(replays, 1, "one replay for the surface showing the tab")
        task_id = str(seen["task_events"]["task_id"])
        first = seen["task_events"]["events"]
        stubs = [ev["lazy"] for ev in first if ev["type"] == "trajectory"]
        self.assertEqual([stub["taskId"] for stub in stubs], [task_id])
        self.assertEqual([ev["type"] for ev in first if ev["type"] in folded], [])
        self.assertIn(SUMMARY, json.dumps([ev for ev in first if ev["type"] == "result"]))
        # The Trajectory panel is opened: the fetch carries the stub's
        # handle and names the tab's machine, box answers.
        fetch = {"type": "getTaskEvents", **stubs[0], "tabId": tab, "token": "lazy-1"}
        writer.write(json.dumps({**fetch, "machine": "box"}).encode() + b"\n")
        await writer.drain()
        piece = (await _frames_to(reader, "taskEventsPiece"))[-1]
        self.assertEqual(
            {k: piece.get(k) for k in ("taskId", "seg", "token", "machine")},
            {"taskId": task_id, "seg": -1, "token": "lazy-1", "machine": "box"},
        )
        panels = [ev for ev in piece["events"] if ev["type"] in folded]
        self.assertIn("finish", [ev.get("name") for ev in panels])
        # Asked of this daemon (no machine named), the task has no events: they are box's alone.
        writer.write(json.dumps({**fetch, "token": "lazy-2"}).encode() + b"\n")
        await writer.drain()
        here = (await _frames_to(reader, "taskEventsPiece"))[-1]
        self.assertEqual((here["token"], here["events"], here.get("machine")), ("lazy-2", [], None))

        # A page that does not load in pieces is replayed the whole
        # transcript: the panels the other page fetched, and no stub.
        reader, writer = await open_local_connection(self.server)
        self.addCleanup(writer.close)
        writer.write(json.dumps(ready).encode() + b"\n")
        await writer.drain()
        seen, replays = await self._surface_frames(reader, writer, tab, "task_events")
        self.assertEqual(replays, 1)
        whole = seen["task_events"]["events"]
        self.assertEqual([ev for ev in whole if ev["type"] in folded], panels)
        self.assertEqual(
            [ev["type"] for ev in whole if ev["type"] in ("trajectory", "lazy_steps", "lazy_gap")],
            [],
        )
