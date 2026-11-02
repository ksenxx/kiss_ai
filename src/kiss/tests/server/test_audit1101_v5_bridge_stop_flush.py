# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: a stopped bridge-owned run keeps the Bash output it already accepted.

A run the printer bridge registered itself (no server task runner:
the ``/update`` and merge-conflict agents, standalone runs) is freed
by ``JsonPrinter.agent_task_finished`` from ``ChatSorcarAgent.run``'s
``finally``.  ``print(type="bash_stream")`` accepts a fragment that
arrives within 100 ms of the previous flush into a buffer a timer
flushes later; a Stop that lands before that timer unwinds the run,
and ``agent_task_finished`` used to unregister the state and clear
the buffer at once — nobody flushed the accepted text, and after the
unregister nothing could file it under the task anyway.  The runner
flushes a server-owned run's buffer before its terminal ``result``;
the bridge now flushes before it unregisters.

Real pieces throughout: ``ChatSorcarAgent.run`` under a ``JsonPrinter``
against a local streaming OpenAI-compatible endpoint that asks for one
``Bash`` call, the production ``Bash`` tool running a real shell
command, the run's real stop event (the one ``AgentServer._stop_task``
sets), the real agent-state registry and the real sqlite event store.
The Stop is pressed by a trace of the run's own thread at the first
stop check of the ``Bash`` tool's wait that finds the second fragment
accepted into the buffer: the one point the test means, reached
whatever the machine's load, where a thread polling the buffer against
the 100 ms timer could miss it.
"""

from __future__ import annotations

import functools
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from kiss.agents.sorcar import persistence
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.useful_tools import _stop_requested
from kiss.server import agent_state
from kiss.server.json_printer import JsonPrinter
from kiss.tests.agents.sorcar.local_model_server import MODEL
from kiss.tests.agents.sorcar.test_audit1101r2_sorcar_a_append_mode import AppendWhen


class _OneBashCallModel(BaseHTTPRequestHandler):
    """A streaming OpenAI-compatible endpoint: one ``Bash`` call, then ``finish``.

    The ``finish`` bounds a run the Stop never ended: it then returns
    after the command's own 20 s and fails the test on the missing
    ``KeyboardInterrupt`` instead of running the command over and over.
    """

    command = ""
    requests = 0

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        cls = type(self)
        cls.requests += 1
        if cls.requests == 1:
            # A deadline under the auto-detach grace: the command runs
            # in the foreground and streams line by line (the detach
            # path polls its log every 0.2 s, one piece per poll).
            name, arguments = "Bash", {
                "command": cls.command, "description": "stream two fragments",
                "timeout_seconds": 30,
            }
        else:
            name, arguments = "finish", {
                "success": True, "is_continue": False, "summary_in_html": "done",
            }
        call = {
            "index": 0, "id": f"call-{cls.requests}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": MODEL}
        chunks = [
            {**base, "choices": [{
                "index": 0, "delta": {"role": "assistant", "tool_calls": [call]},
                "finish_reason": None,
            }]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {**base, "choices": [], "usage": {
                "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
            }},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@contextmanager
def _serve_one_bash_call(command: str) -> Iterator[str]:
    """Serve :class:`_OneBashCallModel` asking for *command* on a local port."""
    handler = type("Handler", (_OneBashCallModel,), {"command": command})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)


def _pending_bash_output(printer: JsonPrinter, task_id: str) -> list[str]:
    """Return the fragments *printer* holds for *task_id*, accepted but not yet flushed."""
    with printer._bash_lock:
        bs = printer._bash_states.get(task_id)
        return list(bs.buffer) if bs is not None else []


def _tail_pending_at_stop_check(
    printer: JsonPrinter, agent: ChatSorcarAgent, seen: list[str], frame: FrameType,
) -> bool:
    """Whether *frame* is the ``Bash`` wait's stop check with ``TAIL`` accepted but not flushed.

    The wait (``_consume_stream``) calls ``_stop_requested`` right after
    it printed a line and before it waits for the next, so this holds
    first at the moment the test means.  *seen* receives the pending
    fragments then: the Stop lands on them.
    """
    if frame.f_code is not _stop_requested.__code__:
        return False
    pending = _pending_bash_output(printer, agent.last_task_id)
    if "TAIL\n" not in pending:
        return False
    seen.extend(pending)
    return True


def _persisted_output(task_id: str) -> str:
    """Return the ``system_output`` text persisted for *task_id*, in order."""
    persistence._flush_chat_events(task_id)
    rows = persistence._get_db().execute(
        "SELECT event_json FROM events WHERE task_id = ? ORDER BY seq", (task_id,),
    ).fetchall()
    events = [json.loads(row[0]) for row in rows]
    return "".join(e.get("text", "") for e in events if e["type"] == "system_output")


def test_stopped_bridge_run_persists_its_buffered_bash_output(tmp_path: Path) -> None:
    """The fragment accepted right before the Stop lands in the task's transcript.

    The shell prints ``HEAD`` and ``TAIL`` in one write, so the two
    lines reach the tool's wait back to back on the run's thread:
    ``HEAD`` is flushed at once (nothing was flushed in the last 100 ms)
    and ``TAIL`` right after it is accepted into the buffer, timer
    pending.  The wait's next stop check finds ``TAIL`` pending and the
    trace sets the run's stop event there; the wait sees the Stop,
    kills the shell (by then ``exec``'d into its ``sleep``) and drains
    the pipe, and the run unwinds on the Stop's ``KeyboardInterrupt``
    long before the buffer's timer fires, with ``TAIL`` still pending.
    """
    # ``exec``: the Stop's group kill lands the instant the lines are out,
    # and a child the shell is forking right then can be missed by the
    # kill (and hold the pipe through the drain's 5 s); the exec'd sleep
    # is the one process of the group.
    command = "printf 'HEAD\\nTAIL\\n'; exec sleep 20"
    printer = JsonPrinter()
    stop = threading.Event()
    # The run's thread is bound to its stop event before the agent is
    # built (the agent captures it at construction for its tools), the
    # way the task runner and the task-update thread bind theirs.
    printer._thread_local.stop_event = stop
    seen_pending: list[str] = []
    try:
        agent = ChatSorcarAgent("bridge-stop-flush")
        tail_pending = functools.partial(_tail_pending_at_stop_check, printer, agent, seen_pending)
        with _serve_one_bash_call(command) as url, AppendWhen(tail_pending, stop.set) as pressed:
            with pytest.raises(KeyboardInterrupt):
                agent.run(
                    "stream two fragments, then get stopped",
                    model_name=MODEL,
                    model_config={"base_url": url, "api_key": "local"},
                    printer=printer,
                    work_dir=str(tmp_path),
                    tool_profile="bash",
                    use_memory=False,
                    is_parallel=False,
                    max_budget=1.0,
                )
    finally:
        printer._thread_local.stop_event = None
    assert pressed.made == 1, "the Stop was never pressed"
    # The Stop landed while TAIL was accepted but not yet flushed.
    assert seen_pending == ["TAIL\n"]
    task_id = agent.last_task_id
    assert task_id
    # Both accepted fragments were persisted under the task...
    assert _persisted_output(task_id) == "HEAD\nTAIL\n"
    # ...and the run's per-task state is freed all the same.
    assert agent_state.get(task_id) is None
    assert task_id not in printer._bash_states
    assert task_id not in printer._recordings
