# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration test: sub-agent events must never be delivered to the parent
tab via WebPrinter fan-out.

Reproduces the bug where sub-agent result panels appeared after
"Suggested next" in the parent chat webview.  The test captures every
event copy stamped with the parent tab's ``tabId`` during fan-out and
asserts that no sub-agent result event reaches the parent tab.

The test exercises the full event pipeline:
  ChatSorcarAgent → JsonPrinter.broadcast → _inject_task_id
  → _record_event → _persist_event → _fanout_targets → per-tab copies.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.vscode.json_printer import JsonPrinter

# ---------------------------------------------------------------------------
# Fake OpenAI chat server
# ---------------------------------------------------------------------------


def _run_parallel_response() -> dict:
    return {
        "id": "chatcmpl-rp",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_rp",
                    "type": "function",
                    "function": {
                        "name": "run_parallel",
                        "arguments": json.dumps({
                            "tasks": json.dumps([
                                "Compute 2+3. Reply with just the number.",
                                "Compute 7*8. Reply with just the number.",
                            ]),
                        }),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _finish_response(summary: str = "done") -> dict:
    return {
        "id": "chatcmpl-fin",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_fin",
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": json.dumps(
                            {"success": "true", "summary": summary},
                        ),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class _Handler(BaseHTTPRequestHandler):
    """HTTP handler: first call returns run_parallel, rest return finish."""

    def do_POST(self) -> None:  # noqa: N802
        cl = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(cl) if cl else b""
        try:
            req = json.loads(raw)
        except Exception:
            req = {}
        is_stream = req.get("stream", False)
        messages = req.get("messages", [])
        # Sub-agent requests are identified by their TASK prompt (a
        # user-role message containing "Compute").  Only user-role
        # content is inspected: sub-agents inherit the parent's
        # ``model_config`` (budget-distribution fix), so their system
        # prompt — which mentions run_parallel — also reaches this
        # server, and a whole-conversation heuristic would misroute
        # them into infinite nested run_parallel spawning.
        user_text = " ".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "user"
        )
        is_sub = "Compute" in user_text
        has_rp_result = any(m.get("role") == "tool" for m in messages)

        if is_sub:
            resp = _finish_response("sub-agent-result")
        elif has_rp_result:
            resp = _finish_response("parent-result")
        else:
            resp = _run_parallel_response()

        if is_stream:
            choice = resp["choices"][0]
            msg = choice["message"]
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
            default_usage = {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
            c3 = {
                "id": resp["id"],
                "object": "chat.completion.chunk",
                "model": resp["model"],
                "choices": [],
                "usage": resp.get("usage", default_usage),
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
# Printer that captures per-tab fan-out events
# ---------------------------------------------------------------------------


class _FanoutCapturePrinter(JsonPrinter):
    """Captures every event stamped with a specific tab id during fan-out.

    Simulates the WebPrinter fan-out: after recording and persistence,
    stamps one copy per subscribed tab and appends it to the per-tab
    capture list.
    """

    def __init__(self) -> None:
        super().__init__()
        self.per_tab_events: dict[str, list[dict[str, Any]]] = {}
        self._cap_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record, persist, and simulate fan-out."""
        if "tabId" in event:
            # Targeted system event — deliver verbatim.
            tab_id = event.get("tabId", "")
            if tab_id:
                with self._cap_lock:
                    self.per_tab_events.setdefault(tab_id, []).append(dict(event))
            return

        event = self._inject_task_id(event)

        if not event.get("taskId"):
            return

        with self._lock:
            self._record_event(event)
        self._persist_event(event)

        for tab_id in self._fanout_targets(event.get("taskId")):
            stamped = {**event, "tabId": tab_id}
            with self._cap_lock:
                self.per_tab_events.setdefault(tab_id, []).append(stamped)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _redirect(tmpdir: str) -> tuple:
    from pathlib import Path

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


class TestSubagentEventsAfterFollowup:
    """No sub-agent result events must appear in the parent tab's event
    stream — especially not after a followup_suggestion.
    """

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

    def test_no_subagent_result_in_parent_fanout(self) -> None:
        """Events fanned out to the parent tab must never include a
        result event from any sub-agent.
        """
        printer = _FanoutCapturePrinter()
        parent_tab_id = "parent-tab-001"

        parent = ChatSorcarAgent("parent")
        parent.run(
            prompt_template="Use run_parallel to compute two arithmetic expressions.",
            model_name="gpt-4o-mini",
            model_config={"base_url": self.url, "api_key": "test-key"},
            work_dir=self.tmpdir,
            printer=printer,
            is_parallel=True,
            _subscribe_tab_id=parent_tab_id,
        )

        parent_task_key = str(parent._last_task_id)

        # Events delivered to the parent tab via fan-out
        parent_events = printer.per_tab_events.get(parent_tab_id, [])

        # Extract result events
        result_events = [
            e for e in parent_events if e.get("type") == "result"
        ]

        # The parent tab should receive exactly ONE result: its own
        assert len(result_events) == 1, (
            f"Expected 1 result in parent tab fan-out, got {len(result_events)}: "
            f"{result_events}"
        )
        assert result_events[0].get("taskId") == parent_task_key, (
            f"Parent tab result should have parent taskId={parent_task_key}, "
            f"got taskId={result_events[0].get('taskId')}"
        )

    def test_no_subagent_events_after_parent_result(self) -> None:
        """After the parent's result event, no sub-agent output events
        should appear in the parent tab's fan-out stream.
        """
        printer = _FanoutCapturePrinter()
        parent_tab_id = "parent-tab-002"

        parent = ChatSorcarAgent("parent")
        parent.run(
            prompt_template="Use run_parallel to compute two arithmetic expressions.",
            model_name="gpt-4o-mini",
            model_config={"base_url": self.url, "api_key": "test-key"},
            work_dir=self.tmpdir,
            printer=printer,
            is_parallel=True,
            _subscribe_tab_id=parent_tab_id,
        )

        parent_task_key = str(parent._last_task_id)
        parent_events = printer.per_tab_events.get(parent_tab_id, [])

        # Find the index of the parent's result event
        result_idx = None
        for i, ev in enumerate(parent_events):
            if ev.get("type") == "result" and ev.get("taskId") == parent_task_key:
                result_idx = i
                break

        assert result_idx is not None, "Parent result event not found"

        # All events after the result should belong to the parent task
        post_result = parent_events[result_idx + 1:]
        for ev in post_result:
            task_id = ev.get("taskId", "")
            assert task_id == parent_task_key or task_id == "", (
                f"Post-result event in parent tab has non-parent "
                f"taskId={task_id}: {ev}"
            )

    def test_subagent_results_reach_subagent_tabs(self) -> None:
        """Each sub-agent's result event must be fanned out to the
        sub-agent's own subscriber tab.
        """
        printer = _FanoutCapturePrinter()
        parent_tab_id = "parent-tab-003"

        parent = ChatSorcarAgent("parent")
        parent.run(
            prompt_template="Use run_parallel to compute two arithmetic expressions.",
            model_name="gpt-4o-mini",
            model_config={"base_url": self.url, "api_key": "test-key"},
            work_dir=self.tmpdir,
            printer=printer,
            is_parallel=True,
            _subscribe_tab_id=parent_tab_id,
        )

        parent_task_key = str(parent._last_task_id)

        # Collect sub-agent tab events (any tab that is not the parent)
        sub_tab_ids = [
            tid for tid in printer.per_tab_events
            if tid != parent_tab_id and tid != ""
        ]

        # Each sub-agent tab should have a result event
        for tid in sub_tab_ids:
            tab_events = printer.per_tab_events[tid]
            results = [e for e in tab_events if e.get("type") == "result"]
            assert len(results) >= 1, (
                f"Sub-agent tab {tid} has no result event"
            )
            for r in results:
                assert r.get("taskId") != parent_task_key, (
                    f"Sub-agent tab {tid} has a result with parent's "
                    f"taskId={parent_task_key}"
                )
