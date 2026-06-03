# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Reproduces sub-agent ``new_tab`` mis-routing under worker-thread reuse.

When ``ChatSorcarAgent._run_tasks_parallel`` uses a ``ThreadPoolExecutor``
whose worker threads are reused across multiple sub-agents (e.g.
``max_workers=1``), the very first thing each sub-agent's
``ChatSorcarAgent.run`` does is ``broadcast({"type": "new_tab", ...})``
— BEFORE it sets ``printer._thread_local.task_id`` to the new
sub-agent's task key.  On a reused worker thread, that thread-local
still carries the PREVIOUS sub-agent's task key, so
``JsonPrinter._inject_task_id`` stamps the new_tab event with
the WRONG ``taskId`` (and ``WebPrinter.broadcast`` would then route
it through the previous tab's stream, recording / persisting it under
the previous task).  In the user-visible behaviour the freshly
spawned sub-agent's panels (including its ``result`` event) end up
wired through the wrong tab's stream.

The test forces worker reuse with ``max_workers=1`` (a single worker
thread runs all three sub-agents sequentially), captures every
broadcast post-``_inject_task_id``, and asserts that the injected
``taskId`` on every ``new_tab`` payload matches that payload's own
``task_id`` field (i.e. the new sub-agent's own task), never a
previously-completed sub-agent's task id.

Uses the real ``_FinishHandler`` HTTP server pattern from
``test_chat_parallel_integration.py`` (no mocks/patches).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _add_task
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.json_printer import JsonPrinter


def _finish_response(model: str = "gpt-4o-mini") -> dict:
    """OpenAI chat-completion body that calls ``finish``."""
    return {
        "id": "chatcmpl-fin",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_fin",
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "arguments": json.dumps(
                                    {"success": "true", "summary": "done"}
                                ),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class _FinishHandler(BaseHTTPRequestHandler):
    """Returns a ``finish`` tool call for every POST request."""

    def do_POST(self) -> None:  # noqa: N802
        cl = int(self.headers.get("Content-Length", 0))
        if cl:
            self.rfile.read(cl)
        body = json.dumps(_finish_response()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _start_server() -> tuple[ThreadingHTTPServer, str]:
    """Start a local HTTP server and return ``(server, base_url)``."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FinishHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


def _redirect(tmpdir: str) -> tuple:
    """Point the persistence module at a temp directory."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved: tuple) -> None:
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


class _RecordingPrinter(JsonPrinter):
    """Captures every broadcast event AFTER ``_inject_task_id`` runs.

    Records each event in the order it was emitted so tests can
    inspect both the injected ``taskId`` and the payload's own
    ``task_id`` (for ``new_tab`` events).
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self._capture_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        event = self._inject_task_id(event)
        with self._capture_lock:
            self.events.append({**event, "_thread": threading.current_thread().name})
        with self._lock:
            self._record_event(event)
        self._persist_event(event)


class TestNewTabStaleTaskId:
    """Sub-agent ``new_tab`` must never carry a stale prior sub-agent's taskId.

    Reproduces the production worker-thread reuse pattern: a single
    ``ThreadPoolExecutor`` worker thread runs all three sub-agents
    sequentially (via ``max_workers=1``).  Without the fix, sub-agent
    N's ``new_tab`` broadcast inherits sub-agent N-1's
    ``printer._thread_local.task_id`` and is mis-stamped.
    """

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        self.srv, self.url = _start_server()
        _RunningAgentState.running_agent_states.clear()
        ChatSorcarAgent.running_agents.clear()

    def teardown_method(self) -> None:
        self.srv.shutdown()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        _RunningAgentState.running_agent_states.clear()
        ChatSorcarAgent.running_agents.clear()

    def test_new_tab_taskid_matches_subagent_not_previous(self) -> None:
        """Each ``new_tab`` event's injected ``taskId`` must be its own task id.

        The ``new_tab`` payload's snake_case ``task_id`` field carries
        the freshly-allocated task_history id for the new sub-agent;
        the injected camelCase ``taskId`` (used for fan-out / recording
        / persistence routing) must equal the same value (or be empty,
        meaning a global system event).  It must NEVER be a previous
        sub-agent's task id.
        """
        # Set up the parent agent's persisted task row.
        parent_task_id, parent_chat_id = _add_task(
            "parent task", chat_id="", extra={"model": "gpt-4o-mini"},
        )

        printer = _RecordingPrinter()
        model_config = {"base_url": self.url, "api_key": "test-key"}

        # Simulate the production flow: a ThreadPoolExecutor with
        # ``max_workers=1`` running three sub-agents in sequence on
        # the SAME worker thread.  Each sub-agent has
        # ``_subagent_info`` set so its ``run()`` will broadcast
        # ``new_tab`` (and thereby exercise the
        # ``_inject_task_id``-on-stale-tl.task_id codepath).
        def _run_subagent(idx: int) -> str:
            sub = ChatSorcarAgent(f"sub-{idx}")
            sub.resume_chat_by_id(parent_chat_id)
            sub._subagent_info = {"parent_task_id": parent_task_id}
            return sub.run(
                prompt_template=f"sub-task-{idx}",
                model_name="gpt-4o-mini",
                model_config=model_config,
                work_dir=self.tmpdir,
                printer=printer,
                is_parallel=True,
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            results = list(pool.map(_run_subagent, range(3)))

        assert len(results) == 3

        # Collect every new_tab event captured on the printer.
        new_tab_events = [e for e in printer.events if e.get("type") == "new_tab"]
        assert len(new_tab_events) == 3, (
            f"Expected 3 new_tab events, got {len(new_tab_events)}: "
            f"{new_tab_events}"
        )

        # The injected ``taskId`` on each new_tab event must equal the
        # payload's own ``task_id`` (or be empty/missing — meaning the
        # event was treated as a global system broadcast).  Any other
        # value means the event was mis-stamped with a stale
        # sub-agent's task id.
        offenders: list[dict[str, Any]] = []
        for ev in new_tab_events:
            payload_task_id = ev.get("task_id")
            injected = ev.get("taskId", "")
            if injected != "" and str(injected) != str(payload_task_id):
                offenders.append(ev)
        assert not offenders, (
            "Sub-agent new_tab broadcast(s) were stamped with a STALE "
            "task_id from a previous sub-agent on the reused worker "
            f"thread.  Offenders: {offenders}.  All new_tab events: "
            f"{new_tab_events}."
        )

        # The three new_tab payloads must carry three DISTINCT task ids.
        payload_ids = [int(ev["task_id"]) for ev in new_tab_events]
        assert len(set(payload_ids)) == 3, (
            f"Expected 3 distinct sub-agent task ids in new_tab payloads, "
            f"got {payload_ids}"
        )

        # And the parent task's recording must NOT contain any
        # sub-agent's new_tab event (would happen if the worker thread
        # had inherited the parent task_id).
        parent_recording = printer._recordings.get(str(parent_task_id), [])
        parent_new_tabs = [
            e for e in parent_recording if e.get("type") == "new_tab"
        ]
        assert not parent_new_tabs, (
            "A sub-agent's new_tab broadcast was recorded under the "
            f"parent task's recording: {parent_new_tabs}"
        )
