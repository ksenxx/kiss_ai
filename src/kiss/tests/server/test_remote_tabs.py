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
  with the sub-agent's transcript and ``subagentDone``, a surface
  joining meanwhile gets the running sub-agent announced
  (``openSubagentTab``) after the parent's replay, and closing the
  sub-agent tab leaves the parent bound;
* a ``submit`` naming an unknown machine or an unreachable machine is
  refused with ``status running:false`` and a ``prompt_refused`` error;
* losing the remote daemon mid-binding ends the tab with an error.

Not covered here: a ``closeTab`` or a second ``submit`` landing while
the first connect is still in flight (``RemoteTabs._bind`` stands down
or keeps the winner); that needs the connect paused, which no real
daemon offers.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import machines as mach
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import LocalWriter, open_local_connection
from kiss.tests.remote_daemon import start_remote_daemon

SUMMARY = "<p>Hello from box</p>"


class _FinishingModel(BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint whose every answer is a ``finish`` call.

    With ``parallel_on_first_call`` the first answer is a
    ``run_parallel`` of one sub-task instead (the sub-agent's own call,
    the second one, is answered after a pause so a viewer can attach
    while it runs); the counter is per test.
    """

    parallel_on_first_call = False
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
                time.sleep(2.0)
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
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class TestRemoteTabs(IsolatedAsyncioTestCase):
    """One local daemon, one remote daemon with a fake model, one client."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        mach.save_machines([])
        _FinishingModel.parallel_on_first_call = False
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
            "address": self.remote.url, "password": "pw-remote",
        }, "machines")
        self.assertEqual(added["error"], "")

    async def asyncTearDown(self) -> None:
        self.writer.close()
        await self.server.stop_async()
        self.remote.stop()
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
        """Collect the tab's frames up to the first ``status running:false``."""
        frames: list[dict[str, Any]] = []
        for _ in range(500):
            msg = await self._next()
            if msg.get("tabId") != tab_id:
                continue
            frames.append(msg)
            if msg.get("type") == "status" and not msg.get("running"):
                return frames
        raise AssertionError("the tab never went idle")

    async def _titles(self, machine: str) -> list[str]:
        """The chat titles of *machine*'s history page."""
        reply = await self._ask({"type": "getHistory", "machine": machine}, "history")
        return [str(s["title"]) for s in reply["sessions"]]

    async def _remote_tabs_state(self) -> list[str]:
        """The tab ids the remote daemon's registry holds."""
        box = next(m for m in mach.load_machines() if m.name == "box")
        reply = await asyncio.to_thread(
            mach.link_for(box).request, {"type": "getTabsState"}, "tabs_state",
        )
        return [str(t["tabId"]) for t in reply["tabs"]]

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
        self.assertIn(tab, await self._remote_tabs_state(), "the remote has the tab")

        # Two surfaces connecting now (a reload, another window) each
        # learn the binding and get the transcript once, as for a local
        # chat, even with their replays in flight together.
        surfaces = [await open_local_connection(self.server) for _ in range(2)]
        try:
            for _, writer in surfaces:
                writer.write(json.dumps({"type": "ready", "restoredTabs": []}).encode() + b"\n")
                await writer.drain()
            for reader, writer in surfaces:
                seen: dict[str, dict[str, Any]] = {}
                replays = 0
                fenced = False  # a request sent once the replay arrived
                for _ in range(300):
                    line = await asyncio.wait_for(reader.readline(), timeout=90.0)
                    msg = json.loads(line.decode("utf-8"))
                    if msg.get("tabId") == tab:
                        seen.setdefault(str(msg["type"]), msg)
                        replays += msg["type"] == "task_events"
                    if "task_events" in seen and not fenced:
                        # Its reply (sent to this connection alone, in
                        # order) fences the stream: a second replay
                        # would have shown up before it.
                        fenced = True
                        writer.write(json.dumps({"type": "getInputHistory"}).encode() + b"\n")
                        await writer.drain()
                    elif fenced and msg.get("type") == "inputHistory":
                        break
                self.assertEqual(seen["tabMachine"]["machine"], "box")
                self.assertEqual(seen["task_events"]["task"], "Say hello")
                self.assertIn(SUMMARY, json.dumps(seen["task_events"]["events"]))
                self.assertEqual(replays, 1, "one replay per surface")
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
        tabs = await self._ask({"type": "getTabsState"}, "tabs_state")
        self.assertIn(tab, [t["tabId"] for t in tabs["tabs"]])
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(tab), "box")

        # A stop is forwarded to the machine (the idle tab answers quietly)
        # and the tab stays bound; closing the tab unbinds it.
        await self._send({"type": "stop", "tabId": tab})
        await asyncio.sleep(0.5)
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(tab), "box")
        await self._send({"type": "closeTab", "tabId": tab})
        await self._until("tabs_state")
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(tab), "")
        for _ in range(20):  # the forwarded closeTab reaches the remote asynchronously
            if tab not in await self._remote_tabs_state():
                break
            await asyncio.sleep(0.25)
        self.assertNotIn(tab, await self._remote_tabs_state(), "the remote closed it too")

    async def test_an_empty_machine_runs_here_and_unbinds(self) -> None:
        tab = "tab-home"
        await self._submit(tab, "Say hello", "box")
        await self._run_until_idle(tab)
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(tab), "box")
        # Back home: the local daemon has no model endpoint, so the run
        # is refused or fails HERE, but the tab is unbound either way.
        await self._submit(tab, "Say hello again", "")
        frames = await self._run_until_idle(tab)
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(tab), "")
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
        self.assertEqual(self.server._server_api.remote_tabs.machine_of("tab-y"), "")

    async def test_losing_the_machine_ends_the_tab_with_an_error(self) -> None:
        tab = "tab-lost"
        await self._submit(tab, "Say hello", "box")
        await self._run_until_idle(tab)
        self.remote.stop()
        error = await self._until("error", tab)
        self.assertIn("lost the connection to box", error["text"])
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(tab), "")

    async def _late_surface(self, tab: str) -> dict[str, dict[str, Any]]:
        """Connect a surface, send ``ready``, and collect its replay of *tab*.

        Returns the first ``tabMachine``, ``task_events`` of *tab*,
        ``openSubagentTab`` under it and ``task_events`` of that
        sub-agent tab (``sub_task_events``), by key.
        """
        reader, writer = await open_local_connection(self.server)
        try:
            writer.write(json.dumps({"type": "ready", "restoredTabs": []}).encode() + b"\n")
            await writer.drain()
            seen: dict[str, dict[str, Any]] = {}
            for _ in range(400):
                line = await asyncio.wait_for(reader.readline(), timeout=90.0)
                msg = json.loads(line.decode("utf-8"))
                kind = str(msg.get("type"))
                if msg.get("tabId") == tab and kind in ("tabMachine", "task_events"):
                    seen.setdefault(kind, msg)
                elif kind == "openSubagentTab" and msg.get("parent_tab_id") == tab:
                    seen.setdefault(kind, msg)
                elif (
                    kind == "task_events" and "openSubagentTab" in seen
                    and msg.get("tabId") == seen["openSubagentTab"]["tab_id"]
                ):
                    seen.setdefault("sub_task_events", msg)
                if len(seen) == 4:
                    return seen
            raise AssertionError(f"incomplete replay: {sorted(seen)}")
        finally:
            writer.close()

    async def test_sub_agent_tabs_follow_the_stream(self) -> None:
        """A remote run's sub-agent tab is announced, resumable, and ends here too."""
        _FinishingModel.parallel_on_first_call = True
        tab = "tab-parent"
        child = f"{tab}__sub"
        await self._submit(tab, "Fan out", "box")
        seen: list[dict[str, Any]] = []
        resumed = False
        for _ in range(400):
            msg = await self._next()
            if msg.get("tabId") not in (tab, child, "", None):
                continue
            seen.append(msg)
            if msg.get("type") == "new_tab" and not resumed:
                # What every webview does on new_tab: open a sub-agent tab
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
                self.assertEqual(late["task_events"]["task"], "Fan out")
                self.assertEqual(late["openSubagentTab"]["parent_tab_id"], tab)
                sub_tab = late["openSubagentTab"]["tab_id"]
                self.assertEqual(late["sub_task_events"]["tabId"], sub_tab)
                self.assertEqual(
                    self.server._server_api.remote_tabs.machine_of(sub_tab), "box",
                    "the announced sub-agent tab joined the stream",
                )
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
        self.assertIn(("subagentDone", child), types, "the sub-agent's end reached the tab")
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(child), "box")
        # Closing the sub-agent tab (never in the local registry, so no
        # tabs_state follows) leaves the parent bound.
        await self._send({"type": "closeTab", "tabId": child})
        for _ in range(40):
            if not self.server._server_api.remote_tabs.machine_of(child):
                break
            await asyncio.sleep(0.25)
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(child), "")
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(tab), "box")

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
        self.assertEqual(remote_tabs.machine_of(tab), "box")
        tabs = await self._ask({"type": "getTabsState"}, "tabs_state")
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
        self.assertEqual(remote_tabs.machine_of(tab), "box")
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
            if not remote_tabs.machine_of(tab):
                break
            await asyncio.sleep(0.25)
        self.assertEqual(remote_tabs.machine_of(tab), "")

        # A row of a machine that is not registered (removed meanwhile).
        await self._click_history_row("tab-z", hello, machine="nowhere")
        error = await self._until("error", "tab-z")
        self.assertEqual(error["code"], "prompt_refused")
        self.assertIn("nowhere is not a registered machine", error["text"])
        self.assertEqual(remote_tabs.machine_of("tab-z"), "")

    async def _late_replay(self, tab: str) -> dict[str, dict[str, Any]]:
        """Connect a surface, send ``ready``, and collect its binding and transcript of *tab*."""
        reader, writer = await open_local_connection(self.server)
        try:
            writer.write(json.dumps({"type": "ready", "restoredTabs": []}).encode() + b"\n")
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
        await self._submit("tab-p", "Fan out", "box")
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
        await self._send({"type": "openTab", "tabId": tab, "title": "Fan out"})
        await self._click_history_row(tab, await self._box_row("Fan out"))
        seen = await self._replay_of(tab)
        self.assertEqual(seen["task_events"]["task"], "Fan out")
        # What unfolding the fan-out panel sends: a sub-agent tab under
        # the parent, resumed by the child's task id.
        sub = f"{tab}__sub_{child_task}"
        await self._send({
            "type": "resumeSession", "taskId": child_task, "tabId": sub, "parentTabId": tab,
        })
        events = await self._until("task_events", sub)
        self.assertEqual(str(events["task_id"]), child_task)
        self.assertIn(SUMMARY, json.dumps(events["events"]))
        self.assertEqual(remote_tabs.machine_of(sub), "box", "the sub-agent tab joined the stream")
        self.assertEqual(remote_tabs.machine_of(tab), "box")
