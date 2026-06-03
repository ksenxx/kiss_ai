# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: sub-agent result panels must NOT appear in the parent
agent's chat webview.

Reproduces the bug where running ``run_parallel`` caused each sub-agent's
``result`` event to be delivered to the parent tab (via the WebPrinter
fan-out) in addition to — or instead of — the sub-agent's own tab.
User-visible symptom: after the parent's result panel, the parent's chat
scroll area also rendered one result panel per sub-agent, duplicating
content that belongs exclusively in each sub-agent's tab.

Root cause: ``ChatSorcarAgent._run_tasks_parallel`` broadcasts
``openSubagentTab`` / ``subagentDone`` events with ``tabId=""``
(targeted-style, empty string) or forgets to emit ``subagentDone``
entirely, causing the fan-out or the frontend to mis-route the
sub-agent result event to the parent tab.

The fix ensures sub-agent ``result`` events carry the sub-agent's own
``taskId`` so the ``WebPrinter`` fan-out routes them exclusively to
subscribers of the sub-agent's task — never to the parent's subscriber
tab.  The test verifies:

1. Every sub-agent ``result`` event has a ``taskId`` different from
   the parent's ``taskId``.
2. No ``result`` event with a sub-agent's ``taskId`` appears in the
   set of events fanned-out to the parent's subscriber tab.
3. The parent's persisted events (loaded from the DB) contain exactly
   one ``result`` — the parent's own.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _load_chat_events_by_task_id,
)
from kiss.agents.vscode.json_printer import JsonPrinter

# ---------------------------------------------------------------------------
# Fake OpenAI server that returns run_parallel → finish
# ---------------------------------------------------------------------------

_CALL_COUNTER: dict[str, int] = {}
_CALL_LOCK = threading.Lock()


def _next_call_id(key: str) -> int:
    with _CALL_LOCK:
        n = _CALL_COUNTER.get(key, 0)
        _CALL_COUNTER[key] = n + 1
        return n


def _run_parallel_response() -> dict:
    """Response that calls run_parallel with 3 arithmetic tasks."""
    return {
        "id": "chatcmpl-rp",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_rp",
                            "type": "function",
                            "function": {
                                "name": "run_parallel",
                                "arguments": json.dumps({
                                    "tasks": json.dumps([
                                        "Compute 2+3. Reply with just the number.",
                                        "Compute 7*8. Reply with just the number.",
                                        "Compute 10-4. Reply with just the number.",
                                    ]),
                                }),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _finish_response(summary: str = "done") -> dict:
    """Response that calls finish."""
    return {
        "id": "chatcmpl-fin",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
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
                                    {"success": "true", "summary": summary},
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


def _sse_wrap(resp: dict) -> bytes:
    """Wrap a non-streaming response as SSE chunks for streaming mode."""
    # Emit the full response as a single SSE data line, then [DONE].
    payload = json.dumps(resp)
    return (
        f"data: {payload}\n\n"
        f"data: [DONE]\n\n"
    ).encode()


class _Handler(BaseHTTPRequestHandler):
    """HTTP handler: first call returns run_parallel, rest return finish.

    Handles both streaming (``stream: true``) and non-streaming requests.
    """

    def do_POST(self) -> None:  # noqa: N802
        cl = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(cl) if cl else b""
        # Parse the request to figure out which agent is calling
        try:
            req = json.loads(raw)
        except Exception:
            req = {}

        is_stream = req.get("stream", False)
        messages = req.get("messages", [])

        # Detect whether this is a sub-agent or the parent.
        msg_text = json.dumps(messages)
        is_sub = "Compute" in msg_text and "run_parallel" not in msg_text
        # Also detect if run_parallel tool result is already present
        has_rp_result = any(
            m.get("role") == "tool" for m in messages
        )

        if is_sub:
            resp = _finish_response("sub-agent-result")
        elif has_rp_result:
            # Parent's second call (after run_parallel returns)
            resp = _finish_response("parent-result")
        else:
            # Parent's first call
            resp = _run_parallel_response()

        if is_stream:
            # Convert to streaming SSE format
            choice = resp["choices"][0]
            msg = choice["message"]
            # First chunk: role
            c1 = {
                "id": resp["id"],
                "object": "chat.completion.chunk",
                "model": resp["model"],
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": None},
                    "finish_reason": None,
                }],
            }
            # Second chunk: tool_calls or content
            delta2: dict[str, Any] = {}
            if msg.get("tool_calls"):
                delta2["tool_calls"] = msg["tool_calls"]
            if msg.get("content"):
                delta2["content"] = msg["content"]
            c2 = {
                "id": resp["id"],
                "object": "chat.completion.chunk",
                "model": resp["model"],
                "choices": [{
                    "index": 0,
                    "delta": delta2,
                    "finish_reason": choice["finish_reason"],
                }],
            }
            # Usage chunk
            c3 = {
                "id": resp["id"],
                "object": "chat.completion.chunk",
                "model": resp["model"],
                "choices": [],
                "usage": resp.get("usage", {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                }),
            }
            lines = [
                f"data: {json.dumps(c1)}\n\n",
                f"data: {json.dumps(c2)}\n\n",
                f"data: {json.dumps(c3)}\n\n",
                "data: [DONE]\n\n",
            ]
            body = "".join(lines).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        else:
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _start_server() -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


# ---------------------------------------------------------------------------
# Capturing printer
# ---------------------------------------------------------------------------


class _CapturePrinter(JsonPrinter):
    """Records every broadcast event with its injected taskId."""

    def __init__(self) -> None:
        super().__init__()
        self.all_events: list[dict[str, Any]] = []
        self._cap_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        event = self._inject_task_id(event)
        with self._cap_lock:
            self.all_events.append(dict(event))
        # Delegate to parent for recording + persistence side effects
        with self._lock:
            self._record_event(event)
        self._persist_event(event)


# ---------------------------------------------------------------------------
# DB redirect helpers
# ---------------------------------------------------------------------------


def _redirect(tmpdir: str) -> tuple:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore(saved: tuple) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSubagentResultNotInParent:
    """Sub-agent result events must not leak into the parent tab."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)
        self.srv, self.url = _start_server()

    def teardown_method(self) -> None:
        self.srv.shutdown()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_parent_persisted_events_have_no_subagent_results(self) -> None:
        """The parent's persisted chat events must not include sub-agent
        result events.  Each sub-agent's result belongs exclusively in
        the sub-agent's own task_history row.
        """
        model_config: dict[str, Any] = {
            "base_url": self.url,
            "api_key": "test-key",
        }

        printer = _CapturePrinter()

        parent = ChatSorcarAgent("parent")
        parent.run(
            prompt_template=(
                "Use run_parallel to compute three arithmetic expressions."
            ),
            model_name="gpt-4o-mini",
            model_config=model_config,
            work_dir=self.tmpdir,
            printer=printer,
            is_parallel=True,
        )

        parent_task_id = parent._last_task_id
        assert parent_task_id is not None

        # Load the parent's persisted events from the DB
        loaded = _load_chat_events_by_task_id(parent_task_id)
        assert loaded is not None
        raw_events = loaded.get("events") or []  # type: ignore[union-attr]
        parent_events: list[Any] = list(raw_events)  # type: ignore[call-overload]

        # Count result events in the parent's persisted events
        parent_result_events = [
            e for e in parent_events if isinstance(e, dict) and e.get("type") == "result"
        ]

        # The parent should have exactly ONE result event (its own).
        # If sub-agent results leaked, there would be 4 (1 parent + 3 subs).
        assert len(parent_result_events) == 1, (
            f"Expected exactly 1 result event in parent's persisted events, "
            f"got {len(parent_result_events)}.  Result events: "
            f"{parent_result_events}"
        )

        # The parent's result must contain "parent-result"
        parent_result = parent_result_events[0]
        summary = parent_result.get("summary", "") or parent_result.get("text", "")
        assert "parent-result" in summary, (
            f"Parent result should contain 'parent-result', got: {summary}"
        )

    def test_broadcast_result_events_carry_correct_task_id(self) -> None:
        """Every sub-agent result event broadcast by the printer must carry
        the sub-agent's own taskId, NOT the parent's taskId.  This
        guarantees the WebPrinter fan-out routes them to the sub-agent
        tab, not the parent tab.
        """
        model_config: dict[str, Any] = {
            "base_url": self.url,
            "api_key": "test-key",
        }

        printer = _CapturePrinter()

        parent = ChatSorcarAgent("parent")
        parent.run(
            prompt_template=(
                "Use run_parallel to compute three arithmetic expressions."
            ),
            model_name="gpt-4o-mini",
            model_config=model_config,
            work_dir=self.tmpdir,
            printer=printer,
            is_parallel=True,
        )

        parent_task_id = parent._last_task_id
        assert parent_task_id is not None
        parent_task_key = str(parent_task_id)

        # Collect all result events from broadcasts
        result_events = [
            e for e in printer.all_events if e.get("type") == "result"
        ]

        # There should be 4 result events total: 1 parent + 3 sub-agents
        assert len(result_events) >= 4, (
            f"Expected at least 4 result events (1 parent + 3 subs), "
            f"got {len(result_events)}"
        )

        # Exactly one result should have the parent's taskId
        parent_results = [
            e for e in result_events
            if e.get("taskId") == parent_task_key
        ]
        assert len(parent_results) == 1, (
            f"Expected exactly 1 result with parent taskId={parent_task_key}, "
            f"got {len(parent_results)}"
        )

        # The remaining results should all have DIFFERENT taskIds
        sub_results = [
            e for e in result_events
            if e.get("taskId") != parent_task_key
        ]
        assert len(sub_results) >= 3, (
            f"Expected at least 3 sub-agent results with non-parent taskId, "
            f"got {len(sub_results)}"
        )
        for sr in sub_results:
            assert sr.get("taskId"), (
                f"Sub-agent result event missing taskId: {sr}"
            )
            assert sr["taskId"] != parent_task_key, (
                f"Sub-agent result has parent's taskId: {sr}"
            )

    def test_no_result_event_fanned_out_to_parent_tab(self) -> None:
        """Simulates WebPrinter fan-out: no sub-agent result event
        should be stamped with the parent's tab id.

        This test subscribes a tab to the parent task (mimicking the
        VS Code server's ``_subscribe_tab_id`` flow) and verifies that
        the fan-out targets for sub-agent result events never include
        the parent tab.
        """
        model_config: dict[str, Any] = {
            "base_url": self.url,
            "api_key": "test-key",
        }

        printer = _CapturePrinter()

        parent = ChatSorcarAgent("parent")
        # Simulate the VS Code server subscribing the parent tab to the
        # parent's task.  We do this by passing ``_subscribe_tab_id``.
        parent_tab_id = "parent-tab-XYZ"
        parent.run(
            prompt_template=(
                "Use run_parallel to compute three arithmetic expressions."
            ),
            model_name="gpt-4o-mini",
            model_config=model_config,
            work_dir=self.tmpdir,
            printer=printer,
            is_parallel=True,
            _subscribe_tab_id=parent_tab_id,
        )

        parent_task_id = parent._last_task_id
        assert parent_task_id is not None
        parent_task_key = str(parent_task_id)

        # Collect all result events
        result_events = [
            e for e in printer.all_events if e.get("type") == "result"
        ]

        # Check fan-out targets for each result event
        for ev in result_events:
            task_id = ev.get("taskId", "")
            if task_id == parent_task_key:
                # Parent result: fan-out should include parent tab
                targets = printer._fanout_targets(task_id)
                assert parent_tab_id in targets, (
                    f"Parent result should fan out to parent tab "
                    f"{parent_tab_id}, got targets={targets}"
                )
            else:
                # Sub-agent result: fan-out should NOT include parent tab
                targets = printer._fanout_targets(task_id)
                assert parent_tab_id not in targets, (
                    f"Sub-agent result (taskId={task_id}) should NOT "
                    f"fan out to parent tab {parent_tab_id}, "
                    f"but targets={targets}"
                )

    def test_subagent_result_events_persisted_to_own_rows(self) -> None:
        """Each sub-agent's result event must be persisted under the
        sub-agent's own task_history row, not the parent's.
        """
        model_config: dict[str, Any] = {
            "base_url": self.url,
            "api_key": "test-key",
        }

        printer = _CapturePrinter()

        parent = ChatSorcarAgent("parent")
        parent.run(
            prompt_template=(
                "Use run_parallel to compute three arithmetic expressions."
            ),
            model_name="gpt-4o-mini",
            model_config=model_config,
            work_dir=self.tmpdir,
            printer=printer,
            is_parallel=True,
        )

        parent_task_id = parent._last_task_id
        assert parent_task_id is not None

        # Find sub-agent rows in the DB
        import time
        # Give async persistence a moment to flush
        time.sleep(0.5)
        th._flush_chat_events()

        db = th._get_db()
        rows = db.execute(
            "SELECT id, extra FROM task_history "
            "WHERE COALESCE(extra, '') LIKE '%\"subagent\"%' "
            "ORDER BY id ASC"
        ).fetchall()
        sub_rows = [{"id": r[0], "extra": r[1]} for r in rows]
        assert len(sub_rows) == 3, (
            f"Expected 3 sub-agent rows, got {len(sub_rows)}"
        )

        # Each sub-agent row should have a result event
        for row in sub_rows:
            loaded = _load_chat_events_by_task_id(row["id"])
            assert loaded is not None
            raw_evts = loaded.get("events") or []  # type: ignore[union-attr]
            events: list[Any] = list(raw_evts)  # type: ignore[call-overload]
            result_evts = [e for e in events if e.get("type") == "result"]
            assert len(result_evts) >= 1, (
                f"Sub-agent row {row['id']} should have at least 1 result "
                f"event, got {len(result_evts)}: events={events}"
            )
