# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A remote sub-agent's own fan-out is announced before any surface resumed its tab.

Every chat page answers a relayed ``new_tab`` by opening the sub-agent
tab ``{parent}__sub_{task_id}`` and sending ``resumeSession`` for it,
which is what used to join that tab id to the remote stream.  A
sub-agent that fanned out at once (a reviewer, a nested
``run_parallel``) names that derived id as the ``parent_tab_id`` of
its children's ``new_tab`` (``subagent_parent_tab_id_of``), and
:meth:`RemoteTab._relay_frame` dropped those frames while the id was
not joined yet, so no surface opened the nested tabs (a local run
broadcasts every ``new_tab`` to every client).  The same held for a
``subagentDone`` addressed to a derived id.

The reproduction is a surface that gets the ``new_tab`` of the first
sub-agent and never resumes it (the slowest possible client): the
nested sub-agent's ``new_tab`` must still arrive, under that derived
parent id, and so must each sub-agent's ``subagentDone`` for the
derived tab the surface opened.  ``_notify_subagent_done`` used to
address only the registered viewers plus the synthetic ``api-…`` tab,
so such a surface learned ``isDone`` only from the ``openSubagentTab``
answering its late ``resumeSession``, which marks a tab finished but
never closes it: the tab stayed open there and closed everywhere else.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from kiss.agents.sorcar import machines as mach
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import open_local_connection
from kiss.tests.remote_daemon import start_remote_daemon


class _NestedFanOutModel(BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint: the first two calls fan out, the rest finish.

    Call 1 (the task) and call 2 (its sub-agent) each answer with a
    ``run_parallel`` of one sub-task; every later call is a ``finish``.
    """

    calls = 0

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        _NestedFanOutModel.calls += 1
        if _NestedFanOutModel.calls <= 2:
            function = {
                "name": "run_parallel",
                "arguments": json.dumps({"tasks": json.dumps(["Say hi from below"])}),
            }
        else:
            function = {
                "name": "finish",
                "arguments": json.dumps({"success": True, "summary_in_html": "<p>hi</p>"}),
            }
        call = {"index": 0, "id": f"call-{_NestedFanOutModel.calls}", "type": "function",
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


class TestRemoteNestedSubagentRelay(IsolatedAsyncioTestCase):
    """One local daemon, one remote daemon whose fake model fans out twice."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        mach.save_machines([])
        _NestedFanOutModel.calls = 0
        self.model_server = HTTPServer(("127.0.0.1", find_free_port()), _NestedFanOutModel)
        self.model_thread = asyncio.get_running_loop().run_in_executor(
            None, self.model_server.serve_forever,
        )
        endpoint = f"http://127.0.0.1:{self.model_server.server_port}/v1"
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
        await self._send({
            "type": "machineAction", "action": "add", "name": "box",
            "address": self.remote.url, "machinePassword": "pw-remote",
        })
        added = await self._until("machines")
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

    async def _next(self) -> dict[str, Any]:
        line = await asyncio.wait_for(self.reader.readline(), timeout=90.0)
        assert line, "local connection closed"
        msg: dict[str, Any] = json.loads(line.decode("utf-8"))
        return msg

    async def _until(self, reply_type: str) -> dict[str, Any]:
        for _ in range(300):
            msg = await self._next()
            if msg.get("type") == reply_type:
                return msg
        raise AssertionError(f"no {reply_type} reply")

    async def test_nested_new_tab_is_relayed_before_any_resume(self) -> None:
        tab = "tab-parent"
        await self._send({
            "type": "submit", "prompt": "Fan out", "tabId": tab,
            "machine": "box", "model": "gpt-4o-mini", "workDir": self.tmpdir,
        })
        announced: list[tuple[str, str]] = []  # (parent_tab_id, task_id)
        done: list[str] = []  # tab ids named by relayed ``subagentDone`` frames
        for _ in range(800):
            msg = await self._next()
            if msg.get("type") == "new_tab":
                announced.append((str(msg["parent_tab_id"]), str(msg["task_id"])))
            elif msg.get("type") == "subagentDone":
                done.append(str(msg["tab_id"]))
            elif msg.get("type") == "status" and msg.get("tabId") == tab and not msg["running"]:
                break
        else:
            raise AssertionError("the parent never went idle")
        self.assertEqual(len(announced), 2, f"both sub-agents announced: {announced}")
        (first_parent, first_task), (nested_parent, nested_task) = announced
        self.assertEqual(first_parent, tab)
        self.assertEqual(
            nested_parent, f"{tab}__sub_{first_task}",
            "the nested sub-agent is announced under the tab every surface opens",
        )
        # This surface never resumed either sub-agent, so it is not a
        # registered viewer of their streams; the completion must still
        # name the derived tab it opened on each ``new_tab``, or the
        # finished tabs would stay open here while a surface that did
        # resume closed them on its ``subagentDone``.
        for parent, task_id in ((first_parent, first_task), (nested_parent, nested_task)):
            self.assertIn(
                f"{parent}__sub_{task_id}", done,
                f"no subagentDone for the tab opened on new_tab; got {done}",
            )
