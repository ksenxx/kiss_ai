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
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import machines as mach
from kiss.agents.sorcar import persistence
from kiss.server.remote_tabs import RemoteTab
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import LocalReader, LocalWriter, open_local_connection
from kiss.tests.remote_daemon import start_remote_daemon
from kiss.tests.server.pause import Pause
from kiss.tests.server.remote_tabs_probe import machine_of

SUMMARY = "<p>Hello from box</p>"


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


class RemoteTabsHarness(IsolatedAsyncioTestCase):
    """One local daemon, one remote daemon ("box") with a fake model, one client."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        mach.save_machines([])
        _FinishingModel.parallel_on_first_call = False
        _FinishingModel.hold_sub_agent = 2.0
        _FinishingModel.released = threading.Event()
        _FinishingModel.calls = 0
        self.model_port = find_free_port()
        self.model_server = HTTPServer(("127.0.0.1", self.model_port), _FinishingModel)
        self.model_thread = asyncio.get_running_loop().run_in_executor(
            None, self.model_server.serve_forever,
        )
        endpoint = f"http://127.0.0.1:{self.model_port}/v1"
        self.model = "gpt-4o-mini"
        self.remote = await asyncio.to_thread(
            start_remote_daemon, Path(self.tmpdir) / "remote-home", password="pw-remote",
            config={"custom_endpoint": endpoint, "custom_api_key": "k", "max_budget": 100},
        )
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), certfile=str(certfile),
            keyfile=str(keyfile), url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=Path(self.tmpdir) / "sorcar-local.json",
        )
        await self.server.start_async()
        self.reader, self.writer = await open_local_connection(self.server)
        added = await self._ask({
            "type": "machineAction", "action": "add", "name": "box",
            "address": self.remote.url, "machinePassword": "pw-remote",
        }, "machines")
        self.assertEqual(added["error"], "")

    async def asyncTearDown(self) -> None:
        self.writer.close()
        await self.server.stop_async()
        self.remote.stop()
        _FinishingModel.released.set()  # a held call blocks the single-threaded server
        self.model_server.shutdown()
        await self.model_thread
        mach.save_machines([])
        shutil.rmtree(self.tmpdir, ignore_errors=True)

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

    async def _submit(self, tab_id: str, prompt: str, machine: str) -> None:
        await self._send({
            "type": "submit", "prompt": prompt, "tabId": tab_id,
            "machine": machine, "model": self.model, "workDir": self.tmpdir,
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
        for _ in range(300):
            line = await asyncio.wait_for(reader.readline(), timeout=90.0)
            msg = json.loads(line.decode("utf-8"))
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
        await self._submit(tab, "Say hello", "box")
        frames = await self._run_until_idle(tab)
        types = [f["type"] for f in frames]
        self.assertIn("status", types)
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

    async def test_unknown_and_unreachable_machines_are_refused(self) -> None:
        await self._submit("tab-x", "Say hello", "nowhere")
        error = await self._until("error", "tab-x")
        self.assertEqual(error["code"], "prompt_refused")
        self.assertIn("nowhere is not a registered machine", error["text"])

        dead = mach.Machine(name="dead", url=f"https://127.0.0.1:{find_free_port()}",
                            password="x")
        mach.save_machines([*mach.load_machines(), dead])
        await self._submit("tab-y", "Say hello", "dead")
        error = await self._until("error", "tab-y")
        self.assertEqual(error["code"], "prompt_refused")
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
        sub-agent tab (``sub_tabMachine``, announced before it) and its
        ``task_events`` (``sub_task_events``), by key.  The surface
        names *tab* as the one it shows: any other bound tab is only
        offered (``replay_deferred``), not replayed.
        """
        reader, writer = await open_local_connection(self.server)
        try:
            writer.write(
                json.dumps({"type": "ready", "tabId": tab, "restoredTabs": []}).encode() + b"\n",
            )
            await writer.drain()
            seen: dict[str, dict[str, Any]] = {}
            bindings: dict[str, dict[str, Any]] = {}  # every tabMachine, by tab
            for _ in range(400):
                line = await asyncio.wait_for(reader.readline(), timeout=90.0)
                msg = json.loads(line.decode("utf-8"))
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
                    kind == "task_events" and "openSubagentTab" in seen
                    and msg.get("tabId") == seen["openSubagentTab"]["tab_id"]
                ):
                    seen.setdefault("sub_task_events", msg)
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
                # the sub-agent's transcript, all to that surface alone.
                late = await self._late_surface(tab)
                self.assertEqual(late["tabMachine"]["machine"], "box")
                self.assertEqual(late["task_events"]["task"], "Fan out to a sub-agent")
                self.assertEqual(late["openSubagentTab"]["parent_tab_id"], tab)
                sub_tab = late["openSubagentTab"]["tab_id"]
                self.assertEqual(late["sub_task_events"]["tabId"], sub_tab)
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
                # with the ready, still ahead of its transcript.
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
        self.assertEqual(
            [m["machine"] for m in seen if m["type"] == "tabMachine" and m["tabId"] == child],
            ["box"],
        )
        self.assertIn(("subagentDone", child), types, "the sub-agent's end reached the tab")
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, child), "box")
        # Closing the sub-agent tab (never in the local registry, so no
        # tabs_state follows) leaves the parent bound.
        await self._send({"type": "closeTab", "tabId": child})
        for _ in range(40):
            if not machine_of(self.server._server_api.remote_tabs, child):
                break
            await asyncio.sleep(0.25)
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, child), "")
        self.assertEqual(machine_of(self.server._server_api.remote_tabs, tab), "box")

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

    async def _background_inventory(self, root: str) -> list[dict[str, Any]]:
        reader, writer = await open_local_connection(self.server)
        try:
            writer.write(json.dumps({"type": "ready", "restoredTabs": []}).encode() + b"\n")
            await writer.drain()
            seen = []
            fenced = False
            for _ in range(400):
                msg = json.loads(await asyncio.wait_for(reader.readline(), 90))
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

    async def test_background_ready_includes_live_children_but_not_closed_or_finished(self) -> None:
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
        await self._send({"type": "closeTab", "tabId": child})
        await self._send({"type": "getInputHistory", "tabId": child})
        await self._until("inputHistory")
        seen = await self._background_inventory(root)
        self.assertFalse(any(m.get("type") in ("new_tab", "openSubagentTab") for m in seen))
        _FinishingModel.released.set()
        await self._run_until_idle(root)
        seen = await self._background_inventory(root)
        self.assertFalse(any(m.get("type") in ("new_tab", "openSubagentTab") for m in seen))

    async def test_background_ready_ends_a_child_that_finished_while_the_surface_was_away(
        self,
    ) -> None:
        """A surface that missed a child's ``subagentDone`` gets it on its ``ready``.

        A page keeps the sub-agent tabs the registry snapshot does not
        list, and the ``closeTab`` a page sends when it closes a finished
        child never came from one that was away: without the end
        replayed, the child would spin for good on it.  A child reopened
        since (an expanded fan-out panel's resume) stays open, and one
        closed since is nobody's business any more.
        """
        root = "parent-away"
        spawned = await self._fan_out(root)
        child = f"{root}__sub_{spawned['task_id']}"
        await self._send({
            "type": "resumeSession", "taskId": spawned["task_id"], "tabId": child,
        })
        await self._until("task_events", child)
        _FinishingModel.released.set()
        # The child's subagentDone precedes the root's terminal status on
        # the stream; this client never answers it with a closeTab.
        await self._run_until_idle(root)
        seen = await self._background_inventory(root)
        types = [(m["type"], m.get("tabId") or m.get("tab_id")) for m in seen]
        self.assertEqual(
            [m["tab_id"] for m in seen if m["type"] == "subagentDone"], [child], types,
        )
        self.assertLess(types.index(("tabMachine", child)), types.index(("subagentDone", child)))
        self.assertFalse(any(m["type"] in ("new_tab", "openSubagentTab") for m in seen), types)
        self.assertFalse(any(m["type"] == "status" and m.get("tabId") == child for m in seen))
        # Reopened from the fan-out panel: its transcript shows again, and
        # the end that would close it is not replayed any more.
        await self._send({
            "type": "resumeSession", "taskId": spawned["task_id"], "tabId": child,
            "reopen": True,
        })
        await self._until("task_events", child)
        seen = await self._background_inventory(root)
        self.assertFalse(any(m["type"] == "subagentDone" for m in seen), seen)
        # Closed by a page: off the stream, so nothing of it is replayed.
        await self._send({"type": "closeTab", "tabId": child})
        await self._send({"type": "getInputHistory", "tabId": child})
        await self._until("inputHistory")
        seen = await self._background_inventory(root)
        self.assertFalse(any(m.get("tab_id") == child or m.get("tabId") == child for m in seen))

    async def test_an_online_surfaces_auto_close_keeps_the_end_for_the_surfaces_away(
        self,
    ) -> None:
        """The ``closeTab`` a page answers a ``subagentDone`` with keeps the end for the pages away.

        A page that is online when a sub-agent finishes closes its tab
        and tells the daemon so (``closeTab``, what the webapp does on
        ``subagentDone``).  A page that was away still has the tab, which
        the registry snapshot never listed: only the ``subagentDone``
        replayed on its ``ready`` closes it, so that close must not
        erase it.  The child is off the stream (no binding, no state is
        replayed for it).  Reopened from the fan-out panel since, the
        child stays open on every surface: nothing ends it any more.
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
        # This client stands in for the online page: it closes the
        # finished child as the webapp does on its subagentDone.
        await self._send({"type": "closeTab", "tabId": child})
        await self._send({"type": "getInputHistory", "tabId": child})
        await self._until("inputHistory")
        seen = await self._background_inventory(root)
        types = [(m["type"], m.get("tabId") or m.get("tab_id")) for m in seen]
        self.assertEqual(
            [m["tab_id"] for m in seen if m["type"] == "subagentDone"], [child], types,
        )
        self.assertNotIn(("tabMachine", child), types)
        self.assertFalse(any(
            m["type"] in ("new_tab", "openSubagentTab", "status")
            and (m.get("tabId") == child or m.get("tab_id") == child) for m in seen
        ), types)
        # Reopened from the fan-out panel (the resume names the parent
        # tab, as the panel's does): open again everywhere, so the end
        # is not replayed any more.
        await self._send({
            "type": "resumeSession", "taskId": spawned["task_id"], "tabId": child,
            "parentTabId": root, "reopen": True,
        })
        await self._until("task_events", child)
        seen = await self._background_inventory(root)
        self.assertFalse(any(m["type"] == "subagentDone" for m in seen), seen)
        self.assertIn(("tabMachine", child), [
            (m["type"], m.get("tabId")) for m in seen
        ])

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
        local chat's view.
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
        self.assertIn(child, [m["tab_id"] for m in seen if m["type"] == "subagentDone"], seen)
        await self._reuse_child_locally(child)
        seen = await self._background_inventory(root)
        self.assertFalse(any(
            m["type"] == "subagentDone" and m.get("tab_id") == child for m in seen
        ), seen)

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
            for _ in range(400):
                line = await asyncio.wait_for(reader.readline(), timeout=90.0)
                msg = json.loads(line.decode("utf-8"))
                if msg.get("tabId") == tab and msg["type"] in ("tabMachine", "task_events"):
                    seen.setdefault(str(msg["type"]), msg)
                if len(seen) == 2:
                    return seen
            raise AssertionError(f"incomplete replay: {sorted(seen)}")
        finally:
            writer.close()

    async def test_sub_agents_of_a_chat_shown_from_a_machine_are_fetched(self) -> None:
        """A finished sub-agent of a chat opened from a machine's row is fetched over its stream.

        The stream never saw the sub-agent's ``new_tab`` (the run ended
        before the chat was opened here), so the fan-out panel's resume
        names the parent tab, whose stream the sub-agent tab joins.
        """
        _FinishingModel.parallel_on_first_call = True
        remote_tabs = self.server._server_api.remote_tabs
        await self._submit("tab-p", "Fan out to a sub-agent", "box")
        child_task = ""
        for _ in range(500):
            msg = await self._next()
            if msg.get("type") == "new_tab" and msg.get("parent_tab_id") == "tab-p":
                child_task = str(msg["task_id"])
            if msg.get("type") == "status" and msg.get("tabId") == "tab-p" and not msg["running"]:
                break
        self.assertTrue(child_task, "the run fanned out")
        await self._send({"type": "closeTab", "tabId": "tab-p"})
        await self._until("tabs_state")

        tab = "tab-click"
        await self._send({"type": "openTab", "tabId": tab, "title": "Fan out to a sub-agent"})
        await self._click_history_row(tab, await self._box_row("Fan out to a sub-agent"))
        seen = await self._replay_of(tab)
        self.assertEqual(seen["task_events"]["task"], "Fan out to a sub-agent")
        # What unfolding the fan-out panel sends: a sub-agent tab under
        # the parent, resumed by the child's task id.
        sub = f"{tab}__sub_{child_task}"
        await self._send({
            "type": "resumeSession", "taskId": child_task, "tabId": sub, "parentTabId": tab,
        })
        # The sub-agent tab's binding is announced before its transcript,
        # as the parent's was: the chat header shows box for it too.
        seen = await self._replay_of(sub)
        self.assertEqual(seen["tabMachine"]["machine"], "box")
        events = seen["task_events"]
        self.assertEqual(str(events["task_id"]), child_task)
        self.assertIn(SUMMARY, json.dumps(events["events"]))
        self.assertEqual(machine_of(remote_tabs, sub), "box", "the sub-agent tab joined the stream")
        self.assertEqual(machine_of(remote_tabs, tab), "box")
        # A surface connecting now (a reload restoring both tabs) learns
        # the sub-agent tab's binding along with the parent's.
        bindings = await self._late_bindings(tab, sub)
        self.assertEqual(bindings, {tab: "box", sub: "box"})

    async def _late_bindings(self, *tabs: str) -> dict[str, str]:
        """Connect a surface, send ``ready``, and return the machine its replay binds *tabs* to."""
        reader, writer = await open_local_connection(self.server)
        try:
            writer.write(json.dumps({"type": "ready", "restoredTabs": []}).encode() + b"\n")
            await writer.drain()
            bindings: dict[str, str] = {}
            for _ in range(400):
                line = await asyncio.wait_for(reader.readline(), timeout=90.0)
                msg = json.loads(line.decode("utf-8"))
                if msg.get("type") == "tabMachine" and msg.get("tabId") in tabs:
                    bindings.setdefault(str(msg["tabId"]), str(msg["machine"]))
                if len(bindings) == len(tabs):
                    return bindings
            raise AssertionError(f"incomplete replay: {sorted(bindings)}")
        finally:
            writer.close()
