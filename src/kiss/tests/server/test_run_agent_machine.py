# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of ``run_agent(..., machine=...)``: a sub-agent on a remote machine.

A real local daemon whose model (a fake OpenAI endpoint) answers its
task with one ``run_agent`` call naming the machine, a real remote
daemon (:mod:`kiss.tests.remote_daemon`) registered as ``box`` on the
same fake model, and a client on the local connection:

* the sub-task runs on the remote daemon (its history has the row, the
  local history does not) and its result comes back to the calling
  task with ``machine=box`` on the ``ran:`` line;
* the local surfaces see it as a nested sub-agent of the caller: the
  relayed ``new_tab`` names the caller's tab, a ``resumeSession`` of the
  sub-task is answered with its transcript, ``subagentDone`` ends its
  tab, no ``api-…`` tab is published in the local registry, and the
  dispatcher's tab is unbound once the call returns;
* two sub-agents sent to the machine at once each keep their own tab;
* a command runs as the machine's own copy (``seaCommand``), and one
  the machine lacks fails the sub-task with a clear message;
* this machine's own name or address (``localhost``, its IP) runs the
  agent here, an ordinary call;
* an unregistered machine is refused before anything runs.

The dispatcher's own choices (a machine named by its host/IP, no SEA
for the default agent, a command by name, the ``work_dir`` option as
a path on the machine, no chat or container inheritance) are checked
on the planned command without a daemon.
"""

from __future__ import annotations

import asyncio
import functools
import ipaddress
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import machines as mach
from kiss.agents.sorcar.local_endpoint import LOCAL_ENDPOINT_ENV
from kiss.core import vscode_config as vc
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import open_local_connection
from kiss.tests.remote_daemon import start_remote_daemon

CHILD_SUMMARY = "<p>Hello from the box sub-agent</p>"
PARENT_SUMMARY = "<p>The box answered</p>"
CHILD_TASK = "Say hi from box"


def _finish(summary: str, pause: float = 0.0) -> dict[str, Any]:
    """A scripted answer: ``finish(summary)`` after *pause* seconds."""
    return {"pause": pause, "calls": [
        ("finish", {"success": True, "summary_in_html": summary}),
    ]}


def _run_agent(**arguments: Any) -> dict[str, Any]:
    """A scripted answer: one ``run_agent(**arguments)`` call."""
    return {"pause": 0.0, "calls": [("run_agent", arguments)]}


def _default_script(machine: str) -> list[dict[str, Any]]:
    """The one round trip: delegate, the child finishes, the caller finishes."""
    return [
        _run_agent(task=CHILD_TASK, machine=machine),
        _finish(CHILD_SUMMARY, pause=2.0),  # the sub-agent: a viewer can attach meanwhile
        _finish(PARENT_SUMMARY),
    ]


class _DelegatingModel(BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint answering from a script, one entry per call.

    Both daemons share it, so the script interleaves the local task's
    answers and its remote sub-agents' (a sub-agent's call comes while
    its caller waits).  An ``agent_job`` call is completed with the job
    id of the last ``run_agent`` notice the model was shown.  The tool
    results the model was given are kept, so a test can read what the
    calling model saw.  The counter is per test.
    """

    script: list[dict[str, Any]] = []
    calls = 0
    tool_results: list[str] = []

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)))
        _DelegatingModel.calls += 1
        for message in body.get("messages", []):
            if message.get("role") == "tool":
                _DelegatingModel.tool_results.append(str(message.get("content", "")))
        script = _DelegatingModel.script
        answer = script[min(_DelegatingModel.calls, len(script)) - 1]
        time.sleep(answer["pause"])
        tool_calls = []
        for index, (name, arguments) in enumerate(answer["calls"]):
            if name == "agent_job":
                jobs = re.findall(r"agent-[0-9a-f]{8}", " ".join(_DelegatingModel.tool_results))
                arguments = {**arguments, "job_id": jobs[-1] if jobs else ""}
            tool_calls.append({
                "index": index, "id": f"call-{_DelegatingModel.calls}-{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            })
        base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": "gpt-4o-mini"}
        chunks = [
            {**base, "choices": [{
                "index": 0, "delta": {"role": "assistant", "tool_calls": tool_calls},
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


def _remote_rows(home: Path) -> list[tuple[str, str]]:
    """The ``(task, parent_task_id)`` rows of the remote daemon's history."""
    with sqlite3.connect(home / "history.db") as db:
        return [
            (str(task), str(parent or ""))
            for task, parent in db.execute("SELECT task, parent_task_id FROM task_history")
        ]


class TestRunAgentOnMachine(IsolatedAsyncioTestCase):
    """One local daemon and one remote daemon (``box``), both on the fake model."""

    async def asyncSetUp(self) -> None:
        # The real path: the daemon canonicalises the work dir, and on
        # macOS mkdtemp's /var/... is a symlink to /private/var/...
        self.tmpdir = os.path.realpath(tempfile.mkdtemp())
        mach.save_machines([])
        _DelegatingModel.script = _default_script("box")
        _DelegatingModel.calls = 0
        _DelegatingModel.tool_results = []
        self.model_port = find_free_port()
        self.model_server = HTTPServer(("127.0.0.1", self.model_port), _DelegatingModel)
        self.model_thread = asyncio.get_running_loop().run_in_executor(
            None, self.model_server.serve_forever,
        )
        endpoint = f"http://127.0.0.1:{self.model_port}/v1"
        self.model = "gpt-4o-mini"
        config = {"custom_endpoint": endpoint, "custom_api_key": "k", "max_budget": 100}
        self.saved_config = vc._config_path().read_text() if vc._config_path().exists() else None
        vc.save_config(config)
        self.remote = await asyncio.to_thread(
            start_remote_daemon, Path(self.tmpdir) / "remote-home", password="pw-remote",
            config=config,
        )
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        # The dispatcher inside the daemon reaches the daemon through its
        # endpoint file (``$KISS_SORCAR_LOCAL``), as a run_agent call does.
        endpoint_file = Path(self.tmpdir) / "sorcar-local.json"
        self.saved_endpoint_env = os.environ.get(LOCAL_ENDPOINT_ENV)
        os.environ[LOCAL_ENDPOINT_ENV] = str(endpoint_file)
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=find_free_port(), certfile=str(certfile),
            keyfile=str(keyfile), url_file=Path(self.tmpdir) / "remote-url.json",
            local_endpoint_file=endpoint_file,
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
        if self.saved_config is None:
            vc._config_path().unlink(missing_ok=True)
        else:
            vc._config_path().write_text(self.saved_config)
        if self.saved_endpoint_env is None:
            os.environ.pop(LOCAL_ENDPOINT_ENV, None)
        else:
            os.environ[LOCAL_ENDPOINT_ENV] = self.saved_endpoint_env
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
        for _ in range(300):
            msg = await self._next()
            if msg.get("type") == reply_type:
                return msg
        raise AssertionError(f"no {reply_type} reply")

    async def _titles(self, machine: str) -> list[str]:
        """The chat titles of *machine*'s history page."""
        reply = await self._ask({"type": "getHistory", "machine": machine}, "history")
        return [str(s["title"]) for s in reply["sessions"]]

    async def _delegate(self, tab: str, late: bool = False) -> tuple[list[dict[str, Any]], str]:
        """Run the delegating task on *tab*; answer the relayed ``new_tab`` as a webview does.

        With *late*, a surface connects while the sub-agent runs and
        what it was told is kept in ``self.late`` (see :meth:`_late_surface`).

        Returns the frames seen until the tab went idle and the
        sub-agent tab id this client opened ("" when no ``new_tab`` came).
        """
        await self._send({
            "type": "submit", "prompt": "Delegate to box", "tabId": tab,
            "model": self.model, "workDir": self.tmpdir,
        })
        seen: list[dict[str, Any]] = []
        child = ""
        for _ in range(2000):
            msg = await self._next()
            seen.append(msg)
            if msg.get("type") == "new_tab" and not child:
                self.assertEqual(msg["parent_tab_id"], tab)
                child = f"{tab}__sub_{msg['task_id']}"
                await self._send({
                    "type": "resumeSession", "taskId": msg["task_id"], "tabId": child,
                })
                if late:
                    self.late = await self._late_surface(tab, msg["task_id"])
            if msg.get("type") == "status" and msg.get("tabId") == tab and not msg["running"]:
                return seen, child
        raise AssertionError("the calling task never went idle")

    async def _late_surface(self, tab: str, task_id: str) -> dict[str, dict[str, Any]]:
        """Connect a surface, send ``ready``, open the running sub-agent's tab as it would.

        Returns the ``new_tab`` it was told (the sub-agent spawned under
        *tab*) and the ``task_events`` its ``resumeSession`` got, by key.
        """
        reader, writer = await open_local_connection(self.server)
        child = f"{tab}__late_{task_id}"
        try:
            writer.write(json.dumps({"type": "ready", "restoredTabs": []}).encode() + b"\n")
            await writer.drain()
            seen: dict[str, dict[str, Any]] = {}
            for _ in range(400):
                line = await asyncio.wait_for(reader.readline(), timeout=90.0)
                msg = json.loads(line.decode("utf-8"))
                kind = str(msg.get("type"))
                if kind == "new_tab" and msg.get("parent_tab_id") == tab and "new_tab" not in seen:
                    seen["new_tab"] = msg
                    writer.write(json.dumps({
                        "type": "resumeSession", "taskId": msg["task_id"], "tabId": child,
                    }).encode() + b"\n")
                    await writer.drain()
                elif kind == "task_events" and msg.get("tabId") == child:
                    seen["task_events"] = msg
                    return seen
            raise AssertionError(f"incomplete late replay: {sorted(seen)}")
        finally:
            writer.close()

    async def test_the_sub_agent_runs_on_the_machine_as_a_nested_tab(self) -> None:
        tab = "tab-caller"
        seen, child = await self._delegate(tab, late=True)
        types = [(m["type"], m.get("tabId") or m.get("tab_id")) for m in seen]
        self.assertTrue(
            child,
            f"no new_tab relayed; the model saw {_DelegatingModel.tool_results}; frames {types}",
        )
        result = next(m for m in seen if m["type"] == "result" and m.get("tabId") == tab)
        self.assertIn(PARENT_SUMMARY, json.dumps(result))
        # The calling model got the sub-agent's result, marked as the box's.
        tool_result = next(
            (t for t in _DelegatingModel.tool_results if CHILD_SUMMARY in t), "",
        )
        self.assertTrue(tool_result, f"no run_agent result among {_DelegatingModel.tool_results}")
        self.assertIn("machine=box", tool_result)
        self.assertIn("success: true", tool_result)
        # Every surface shows the sub-agent as a nested tab of the caller:
        # its transcript replays into the tab opened on new_tab, and its
        # end reaches that tab.
        transcript = next(
            m for m in seen if m["type"] == "task_events" and m.get("tabId") == child
        )
        self.assertEqual(transcript["task"], CHILD_TASK)
        settings = next(e for e in transcript["events"] if e["type"] == "task_settings")
        self.assertTrue(settings["settings"]["is_subagent"])
        self.assertNotEqual(settings["settings"]["work_dir"], self.tmpdir,
                            "the remote's own folder, not the caller's")
        # The caller's model and budget share crossed over; its chat and
        # container, which name nothing on the box, did not.
        self.assertIn("model", settings["settings"]["inherited"])
        self.assertIn("max_budget", settings["settings"]["inherited"])
        self.assertNotIn("chat_id", settings["settings"]["inherited"])
        self.assertNotIn("docker_image", settings["settings"]["inherited"])
        caller = next(
            m for m in seen if m["type"] == "task_settings" and m.get("tabId") == tab
        )
        self.assertNotEqual(settings["settings"]["chat_id"], caller["settings"]["chat_id"])
        self.assertEqual(settings["settings"]["parent_task_id"], caller["settings"]["task_id"])
        # The replay shows the transcript so far; the rest streams live
        # into the same tab, its result included.
        child_frames = [m for m in seen if m.get("tabId") == child]
        self.assertIn(CHILD_SUMMARY, json.dumps(child_frames))
        self.assertIn(("subagentDone", child), types)
        # A surface that connected while the sub-agent ran was told of it
        # too (the same new_tab), and got its transcript on its own tab.
        self.assertEqual(self.late["new_tab"]["task_id"], settings["settings"]["task_id"])
        self.assertEqual(self.late["task_events"]["task"], CHILD_TASK)
        # Its end reached the stream once: nothing was left to announce
        # when the stream was retired.
        self.assertEqual(
            [m["tabId"] for m in seen if m["type"] == "task_events"], [child],
            "the late surface's replay went to it alone",
        )
        # No top-level tab for the sub-agent anywhere: not in the local
        # registry, not in the remote's.
        tabs = await self._ask({"type": "getTabsState"}, "tabs_state")
        self.assertEqual([t["tabId"] for t in tabs["tabs"]], [tab])
        box = next(m for m in mach.load_machines() if m.name == "box")
        remote_tabs = await asyncio.to_thread(
            mach.link_for(box).request, {"type": "getTabsState"}, "tabs_state",
        )
        self.assertEqual(remote_tabs["tabs"], [])
        # The task is the remote daemon's: its history has the row (nested
        # under the local caller's task id), ours has only the caller.
        local_titles = await self._titles(mach.LOCAL_MACHINE)
        self.assertIn("Delegate to box", local_titles)
        self.assertNotIn(CHILD_TASK, local_titles)
        self.assertEqual(
            _remote_rows(self.remote.home), [(CHILD_TASK, caller["settings"]["task_id"])],
        )
        # The dispatcher's tab was unbound when the call returned.
        remote = self.server._server_api.remote_tabs
        for _ in range(40):
            if not remote._tabs:
                break
            await asyncio.sleep(0.25)
        self.assertEqual(remote._tabs, {})

    async def test_a_top_level_api_run_on_a_machine_is_a_published_tab(self) -> None:
        """``daemon_client.run(machine=...)`` from outside a task: a top-level remote tab."""
        from kiss.agents.sorcar import daemon_client

        _DelegatingModel.script = [_finish(CHILD_SUMMARY, pause=2.0)]
        endpoint_file = Path(self.tmpdir) / "sorcar-local.json"
        running = asyncio.get_running_loop().run_in_executor(
            None, functools.partial(
                daemon_client.run, "Top-level on box", machine="box", model=self.model,
                endpoint_file=endpoint_file,
            ),
        )
        # While it runs, every surface shows the dispatcher's tab, bound
        # to the box (the registry entry a local run publishes).
        api_tab = ""
        for _ in range(100):
            tabs = await self._ask({"type": "getTabsState"}, "tabs_state")
            api_tab = next((t["tabId"] for t in tabs["tabs"] if t["tabId"].startswith("api-")), "")
            if api_tab:
                break
            await asyncio.sleep(0.1)
        self.assertTrue(api_tab, "the run's tab was published")
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(api_tab), "box")
        result = await running
        self.assertTrue(result.success)
        self.assertIn(CHILD_SUMMARY, result.text)
        self.assertEqual(_remote_rows(self.remote.home), [("Top-level on box", "")])
        # The call's closeTab took the tab out of the registry and unbound it.
        published = [api_tab]
        for _ in range(100):
            tabs = await self._ask({"type": "getTabsState"}, "tabs_state")
            published = [t["tabId"] for t in tabs["tabs"]]
            if api_tab not in published:
                break
            await asyncio.sleep(0.1)
        self.assertNotIn(api_tab, published)
        self.assertEqual(self.server._server_api.remote_tabs.machine_of(api_tab), "")

    async def test_two_sub_agents_on_the_machine_keep_their_own_tabs(self) -> None:
        """Siblings on one machine: each stream relays its own run's new_tab, not the other's."""
        _DelegatingModel.script = [
            {"pause": 0.0, "calls": [
                ("run_agent", {"task": "First on box", "machine": "box", "wait": "false"}),
                ("run_agent", {"task": "Second on box", "machine": "box"}),
            ]},
            _finish("<p>first done</p>", pause=1.0),
            _finish("<p>second done</p>", pause=3.0),
            {"pause": 0.0, "calls": [("agent_job", {"action": "wait"})]},
            _finish(PARENT_SUMMARY),
        ]
        tab = "tab-two"
        await self._send({
            "type": "submit", "prompt": "Delegate twice", "tabId": tab,
            "model": self.model, "workDir": self.tmpdir,
        })
        seen: list[dict[str, Any]] = []
        children: dict[str, str] = {}  # task id -> the tab this client opened
        for _ in range(4000):
            msg = await self._next()
            seen.append(msg)
            if msg.get("type") == "new_tab" and msg["task_id"] not in children:
                self.assertEqual(msg["parent_tab_id"], tab)
                child = f"{tab}__sub_{msg['task_id']}"
                children[msg["task_id"]] = child
                await self._send({
                    "type": "resumeSession", "taskId": msg["task_id"], "tabId": child,
                })
            if msg.get("type") == "status" and msg.get("tabId") == tab and not msg["running"]:
                break
        self.assertEqual(len(children), 2, f"two new_tabs expected, got {children}")
        # Each child's transcript, result and end reached its own tab: the
        # second's stream outlived the first's retirement.
        for task_id, child in children.items():
            frames = [m for m in seen if m.get("tabId") == child]
            replay = next(m for m in frames if m["type"] == "task_events")
            self.assertEqual(replay["taskId"] if "taskId" in replay else task_id, task_id)
            text = json.dumps(frames)
            self.assertTrue(
                "<p>first done</p>" in text or "<p>second done</p>" in text,
                f"no result on {child}: {[m['type'] for m in frames]}",
            )
            self.assertIn(("subagentDone", child),
                          [(m["type"], m.get("tab_id")) for m in seen])
        results = [t for t in _DelegatingModel.tool_results if "machine=box" in t]
        self.assertTrue(any("first done" in t for t in results), results)
        self.assertTrue(any("second done" in t for t in results), results)
        self.assertEqual(
            sorted(t for t, _ in _remote_rows(self.remote.home)), ["First on box", "Second on box"],
        )

    async def test_a_command_runs_as_the_machines_copy(self) -> None:
        """``agent="sh"``: the remote resolves the command itself (``seaCommand``)."""
        _DelegatingModel.script = [
            _run_agent(task="echo hi", agent="sh", machine="box"),
            _finish(CHILD_SUMMARY, pause=1.0),
            _finish(PARENT_SUMMARY),
        ]
        tab = "tab-sh"
        seen, child = await self._delegate(tab)
        self.assertTrue(child, f"no new_tab; the model saw {_DelegatingModel.tool_results}")
        transcript = next(
            m for m in seen if m["type"] == "task_events" and m.get("tabId") == child
        )
        settings = next(e for e in transcript["events"] if e["type"] == "task_settings")
        self.assertEqual(settings["settings"]["sea"], "sh", settings["settings"])
        self.assertEqual(settings["settings"]["tool_profile"], "bash")
        tool_result = next((t for t in _DelegatingModel.tool_results if "machine=box" in t), "")
        self.assertIn("ran: sh ", tool_result)

    async def test_a_command_the_machine_lacks_is_refused(self) -> None:
        """A SEAS.md folder of this home only: the remote has no such command."""
        from kiss.agents.sorcar.sea_commands import refresh_registry, seas_md_path

        folder = Path(self.tmpdir) / "seas" / "onlyhere"
        folder.mkdir(parents=True)
        (folder / "onlyhere_sea.py").write_text(
            "from kiss.agents.sorcar.sea_commands import BaseSea\n"
            "class OnlyHere(BaseSea):\n"
            "    def description(self):\n        return 'Only registered here.'\n"
        )
        seas_md = seas_md_path()
        saved = seas_md.read_text() if seas_md.exists() else None
        seas_md.write_text(f"{folder.parent}\n")
        refresh_registry()
        try:
            _DelegatingModel.script = [
                _run_agent(task="do it", agent="onlyhere", machine="box"),
                _finish(PARENT_SUMMARY),
            ]
            seen, child = await self._delegate("tab-onlyhere")
            self.assertEqual(child, "", "nothing ran on the box")
            tool_result = next(
                (t for t in _DelegatingModel.tool_results if "onlyhere" in t), "",
            )
            # The remote's refusal is the child's failed result (YAML-quoted).
            self.assertIn("success: false", tool_result)
            self.assertIn("''onlyhere'' is not a registered command on this", tool_result)
            self.assertEqual(_remote_rows(self.remote.home), [])
        finally:
            if saved is None:
                seas_md.unlink(missing_ok=True)
            else:
                seas_md.write_text(saved)
            refresh_registry()

    async def test_this_machines_own_name_runs_the_agent_here(self) -> None:
        """``machine="localhost"`` or this host's IP is an ordinary local call.

        The box of these tests listens on the loopback address, so its
        IP names this machine too: a sub-agent sent there runs here.
        """
        for here, tab in (("localhost", "tab-here"), ("127.0.0.1", "tab-here-ip")):
            _DelegatingModel.script = _default_script(here)
            _DelegatingModel.calls = 0
            _DelegatingModel.tool_results = []
            seen, child = await self._delegate(tab)
            self.assertTrue(child, "the local sub-agent's new_tab was broadcast")
            tool_result = next(
                (t for t in _DelegatingModel.tool_results if CHILD_SUMMARY in t), "",
            )
            self.assertIn("success: true", tool_result)
            self.assertNotIn("machine=", tool_result)
            # It ran here: the box has nothing, no stream was ever bound,
            # and the caller's chat and directory were inherited.
            self.assertEqual(_remote_rows(self.remote.home), [])
            transcript = next(
                m for m in seen if m["type"] == "task_events" and m.get("tabId") == child
            )
            settings = next(e for e in transcript["events"] if e["type"] == "task_settings")
            self.assertEqual(settings["settings"]["work_dir"], self.tmpdir)
            self.assertIn("chat_id", settings["settings"]["inherited"])
            self.assertEqual(self.server._server_api.remote_tabs._tabs, {})

    async def test_an_unregistered_machine_is_refused_before_running(self) -> None:
        _DelegatingModel.script = _default_script("nowhere")
        tab = "tab-nowhere"
        seen, child = await self._delegate(tab)
        self.assertEqual(child, "", "nothing ran anywhere")
        tool_result = next(
            (t for t in _DelegatingModel.tool_results if "not a registered machine" in t), "",
        )
        self.assertIn("'nowhere' is not a registered machine (registered: box)", tool_result)
        self.assertEqual(_remote_rows(self.remote.home), [])


# What the dispatcher decides for a run on a machine, checked on the
# planned command without a daemon.


def _with_registry(machines: list[mach.Machine]) -> list[mach.Machine]:
    """Install *machines* as the registry; return what was there."""
    saved = mach.load_machines()
    mach.save_machines(machines)
    return saved


def test_find_machine_by_name_or_host() -> None:
    saved = _with_registry([
        mach.Machine(name="box", url="https://203.0.113.7:8787", password="x"),
        mach.Machine(name="six", url="https://[fe80::1]:8787", password="x"),
        mach.Machine(name="lab", ssh="alice@lab.example.org:2222"),
        mach.Machine(name="203.0.113.7", ssh="bob@other.example.org"),
    ])
    try:
        found = {
            text: (m.name if m is not None else None)
            for text in ("box", " lab ", "lab.example.org", "LAB.example.org", "fe80::1",
                         "[fe80::1]", "203.0.113.7", "other.example.org", "nowhere", "")
            for m in [mach.find_machine(text)]
        }
        assert found == {
            "box": "box", " lab ": "lab", "lab.example.org": "lab", "LAB.example.org": "lab",
            "fe80::1": "six", "[fe80::1]": "six",
            # A name wins over a host that another machine has.
            "203.0.113.7": "203.0.113.7", "other.example.org": "203.0.113.7",
            "nowhere": None, "": None,
        }
    finally:
        mach.save_machines(saved)
    assert mach.machine_host(mach.Machine(name="bad", url="https://[fe80::1")) == ""


def test_is_local_machine() -> None:
    hostname = socket.gethostname()
    local = ["localhost", "LOCALHOST", "127.0.0.1", "127.1.2.3", "::1", "[::1]", hostname,
             hostname.upper(), mach.local_machine_name(), mach.LOCAL_MACHINE, socket.getfqdn()]
    assert [t for t in local if not mach.is_local_machine(t)] == []
    # Every interface address (DNS or not), in any spelling.
    own = {info[4][0] for info in socket.getaddrinfo(hostname, None)}
    own.update(subprocess.run(
        ["hostname", "-I"], capture_output=True, text=True, check=False,
    ).stdout.split())
    spellings = {str(ipaddress.ip_address(a)) for a in own} | {
        a.exploded for a in map(ipaddress.ip_address, own)
    }
    assert [a for a in spellings if not mach.is_local_machine(a)] == []
    assert not mach.is_local_machine("")
    assert not mach.is_local_machine("nowhere.example.org")
    assert not mach.is_local_machine("203.0.113.9")  # TEST-NET-3, never this host
    assert not mach.is_local_machine("2001:db8::9")


def test_remote_sea_names_the_machines_copy(tmp_path: Path) -> None:
    from kiss.agents.sorcar.agent_dispatch import DEFAULT_AGENT_PATH, remote_sea
    from kiss.agents.sorcar.sea_commands import get_command

    assert remote_sea(DEFAULT_AGENT_PATH) == ("", "")
    cron = get_command("cron")
    assert cron is not None
    assert remote_sea(str(cron)) == ("", "cron")
    own = tmp_path / "own_sea.py"
    own.write_text(
        "from kiss.agents.sorcar.sea_commands import BaseSea\nclass Own(BaseSea):\n    pass\n"
    )
    assert remote_sea(str(own)) == (str(own), "")


def test_plan_run_agent_on_a_machine(tmp_path: Path) -> None:
    from kiss.agents.sorcar.agent_dispatch import plan_run_agent

    saved = _with_registry([mach.Machine(name="box", url="https://203.0.113.7:8787", password="x")])
    try:
        error = plan_run_agent(str(tmp_path), "do it", machine="elsewhere")
        assert error == (
            "Error: 'elsewhere' is not a registered machine (registered: box); register it "
            "with /machine add <name> user@host or /machine add <name> https://host:port "
            "<password>, or pass a registered name or host."
        )
        # The default agent: the machine's plain run in its configured folder.
        plan = plan_run_agent(str(tmp_path), "do it", machine="203.0.113.7")
        assert isinstance(plan, dict)
        assert (plan["machine"], plan["sea_path"], plan["sea_command"], plan["work_dir"]) == (
            "box", "", "", "",
        )
        assert plan["prompt"] == "do it"
        # A command runs as the machine's copy, by name; its own work_dir
        # setting (a path here, cron's scratch directory) is not sent.
        plan = plan_run_agent(str(tmp_path), "list jobs", agent="cron", machine="box")
        assert isinstance(plan, dict)
        assert (plan["prompt"], plan["sea_path"], plan["sea_command"], plan["work_dir"]) == (
            "list jobs", "", "cron", "",
        )
        assert plan["alias"] == ""
        # A work_dir option is a path on the machine, as given...
        plan = plan_run_agent(
            str(tmp_path), "do it", machine="box", options='{"work_dir": "~/jobs"}',
        )
        assert isinstance(plan, dict)
        assert plan["work_dir"] == "~/jobs"
        # ...where the same call here anchors it under the caller's directory;
        # this machine's own name or address is "here" too.
        for here in ("", "localhost", "127.0.0.1", socket.gethostname()):
            plan = plan_run_agent(
                str(tmp_path), "do it", agent="cron", machine=here,
                options='{"work_dir": "jobs"}',
            )
            assert isinstance(plan, str) and "locks work_dir" in plan, (here, plan)
            plan = plan_run_agent(
                str(tmp_path), "do it", machine=here, options='{"work_dir": "jobs"}',
            )
            assert isinstance(plan, dict), (here, plan)
            assert (plan["machine"], plan["sea_command"]) == ("", "")
            assert plan["work_dir"] == str(tmp_path / "jobs")
    finally:
        mach.save_machines(saved)


def test_dispatch_command_on_a_machine_keeps_only_what_crosses() -> None:
    from kiss.agents.sorcar.agent_dispatch import dispatch_command, parse_run_options

    options = parse_run_options('{"chat_id": "c1", "docker_image": "img"}', "", "", "", "")
    cmd = dispatch_command(
        "do it", "", "", "", None, 60.0, options=options, machine="box", sea_command="sh",
    )
    assert (cmd["machine"], cmd["sea_command"]) == ("box", "sh")
    assert cmd["inherit_tools"] is False
    assert (cmd["chat_id"], cmd["docker_image"]) == ("c1", "img")
    assert cmd["provenance"] == {"chat_id": "explicit", "docker_image": "explicit"}


def test_format_dispatch_result_names_the_machine() -> None:
    from kiss.agents.sorcar.agent_dispatch import format_dispatch_result
    from kiss.agents.sorcar.daemon_client import TaskResult

    result = TaskResult(text="ok", success=True, cost=0.0, tokens=0, steps=0,
                        chat_id="", task_id="", settings={"model": "m"})
    assert "machine=box\n" in format_dispatch_result(result, 10.0, machine="box")
    assert "machine=" not in format_dispatch_result(result, 10.0)
