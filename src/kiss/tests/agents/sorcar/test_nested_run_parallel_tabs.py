# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E test: nested ``run_parallel`` sub-agents must open frontend tabs.

Reproduces the bug where a sub-agent (spawned by ``run_parallel``)
that itself calls ``run_parallel`` produced nested sub-agents whose
``new_tab`` broadcasts carried the backend-synthetic registry key
(``task-<parent>__sub_<i>``) as ``parent_tab_id``.  No frontend tab
ever carries that id — the webview allocates a RANDOM tab id for each
sub-agent tab (``createBackgroundSubagentTab`` in media/main.js) — so
the frontend's guard::

    if (ev.parent_tab_id && !tabs.find(t => t.id === ev.parent_tab_id))
      break;

silently dropped every nested ``new_tab`` event and the nested
sub-agents never opened any tabs.

The test drives the REAL pipeline end-to-end:

  ChatSorcarAgent.run → run_parallel tool → _run_tasks_parallel
  → nested ChatSorcarAgent.run → run_parallel → _run_tasks_parallel
  → ``new_tab`` broadcasts → simulated main.js tab handling.

The LLM is a local fake OpenAI-compatible HTTP server that
deterministically instructs the root agent to fan out one middle
task, and the middle agent to fan out two leaf tasks.  The printer
subclass replicates the webview's ``case 'new_tab':`` logic exactly:
drop when ``parent_tab_id`` is unknown, otherwise allocate a random
frontend tab id and subscribe it to the task stream (the
``resumeSession`` round-trip).

No mocks, patches, fakes, or test doubles of KISS code.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.json_printer import JsonPrinter

MIDDLE_MARKER = "MIDDLE-FANOUT-TASK"
LEAF_MARKER = "LEAF-TASK"

# Chronological log of (action, last_user_prefix, has_tool_result) per
# fake-server request — dumped on assertion failure for diagnosis.
_REQUEST_LOG: list[str] = []
_REQUEST_LOG_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Fake OpenAI chat server: root → run_parallel([middle]),
# middle → run_parallel([leaf, leaf]), leaf → finish.
# ---------------------------------------------------------------------------


def _tool_call_response(name: str, arguments: dict[str, Any]) -> dict:
    return {
        "id": f"chatcmpl-{name}",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call_{name}_{uuid.uuid4().hex[:6]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        },
    }


def _pick_response(messages: list[dict[str, Any]]) -> dict:
    """Deterministic 3-level dispatch keyed on the current task prompt."""
    last_user = ""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(c.get("text", "")) for c in content
                    if isinstance(c, dict)
                )
            last_user = str(content)
    has_tool_result = any(m.get("role") == "tool" for m in messages)

    if LEAF_MARKER in last_user:
        action = "finish-leaf"
        resp = _tool_call_response(
            "finish", {"success": "true", "summary": "leaf-done"},
        )
    elif has_tool_result:
        action = "finish-fanout"
        resp = _tool_call_response(
            "finish", {"success": "true", "summary": "fanout-done"},
        )
    elif MIDDLE_MARKER in last_user:
        action = "run_parallel-leaves"
        resp = _tool_call_response("run_parallel", {
            "tasks": json.dumps([
                f"{LEAF_MARKER} one: reply done.",
                f"{LEAF_MARKER} two: reply done.",
            ]),
        })
    else:
        action = "run_parallel-middle"
        resp = _tool_call_response("run_parallel", {
            "tasks": json.dumps(
                [f"{MIDDLE_MARKER}: fan out two leaf tasks."],
            ),
        })
    with _REQUEST_LOG_LOCK:
        _REQUEST_LOG.append(
            f"{action} tool_result={has_tool_result} "
            f"last_user={last_user[-120:]!r}",
        )
    return resp


class _Handler(BaseHTTPRequestHandler):
    """OpenAI-compatible handler implementing the 3-level script."""

    def do_POST(self) -> None:  # noqa: N802
        cl = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(cl) if cl else b""
        try:
            req = json.loads(raw)
        except Exception:
            req = {}
        resp = _pick_response(req.get("messages", []))

        if req.get("stream", False):
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
            delta2: dict[str, Any] = {"tool_calls": msg["tool_calls"]}
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
            c3 = {
                "id": resp["id"],
                "object": "chat.completion.chunk",
                "model": resp["model"],
                "choices": [],
                "usage": resp["usage"],
            }
            body = "".join(
                f"data: {json.dumps(c)}\n\n" for c in (c1, c2, c3)
            ).encode() + b"data: [DONE]\n\n"
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
# Printer simulating the webview's ``case 'new_tab':`` handler
# ---------------------------------------------------------------------------


class _FrontendSimPrinter(JsonPrinter):
    """Replicates media/main.js tab handling for ``new_tab`` events.

    Mirrors the webview exactly:

    * Drop the event when ``parent_tab_id`` names no locally-known tab.
    * Otherwise allocate a RANDOM frontend tab id
      (``createBackgroundSubagentTab``) and post ``resumeSession``,
      which the daemon answers by ``subscribe_tab(task_id, tab_id)``.
    """

    def __init__(self, root_tab_id: str) -> None:
        super().__init__()
        self.tabs: set[str] = {root_tab_id}
        self.accepted_new_tabs: list[dict[str, Any]] = []
        self.dropped_new_tabs: list[dict[str, Any]] = []
        self._sim_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record the event and simulate the frontend's tab logic."""
        event = self._inject_task_id(event)
        with self._lock:
            self._record_event(event)
        if event.get("type") != "new_tab":
            return
        parent_tab_id = event.get("parent_tab_id", "")
        with self._sim_lock:
            if parent_tab_id and parent_tab_id not in self.tabs:
                self.dropped_new_tabs.append(dict(event))
                return
            frontend_tab_id = f"fe-{uuid.uuid4().hex[:8]}"
            self.tabs.add(frontend_tab_id)
            self.accepted_new_tabs.append(dict(event))
        # resumeSession round-trip: the daemon subscribes the freshly
        # allocated frontend tab to the sub-agent's task stream.
        self.subscribe_tab(event.get("task_id"), frontend_tab_id)


# ---------------------------------------------------------------------------
# DB redirection helpers
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
# Test
# ---------------------------------------------------------------------------


class TestNestedRunParallelTabs:
    """Nested run_parallel sub-agents must open frontend tabs."""

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

    def test_nested_subagents_open_tabs(self) -> None:
        """Every nested sub-agent ``new_tab`` event must be accepted
        by a webview that owns the (grand)parent tab.

        Tree: root (frontend tab ``frontend-root``) → 1 middle
        sub-agent → 2 leaf sub-agents.  The simulated frontend must
        end up with 4 tabs: root + middle + 2 leaves.  Before the
        fix, the two leaf ``new_tab`` events carried the middle's
        backend-synthetic registry key as ``parent_tab_id`` and were
        dropped.
        """
        root_tab_id = "frontend-root"
        printer = _FrontendSimPrinter(root_tab_id)

        parent = ChatSorcarAgent("nested-tab-parent")
        parent._chat_id = uuid.uuid4().hex
        # Mirror the VS Code server: pre-register the top-level agent
        # under its REAL frontend tab id before run-start.
        root_state = _RunningAgentState(
            root_tab_id,
            "gpt-4o-mini",
            agent=parent,  # type: ignore[arg-type]
            chat_id=parent._chat_id,
            is_task_active=True,
        )
        _RunningAgentState.register(root_tab_id, root_state)
        try:
            parent.run(
                prompt_template="Fan out the work to sub-agents.",
                model_name="gpt-4o-mini",
                model_config={"base_url": self.url, "api_key": "test-key"},
                work_dir=self.tmpdir,
                printer=printer,
                is_parallel=True,
                _subscribe_tab_id=root_tab_id,
            )
        finally:
            _RunningAgentState.unregister(root_tab_id)

        # The middle + both leaves each broadcast exactly one new_tab.
        all_new_tabs = printer.accepted_new_tabs + printer.dropped_new_tabs
        assert len(all_new_tabs) == 3, (
            f"Expected 3 new_tab broadcasts (1 middle + 2 leaves), got "
            f"{len(all_new_tabs)}: {all_new_tabs}\n"
            f"Fake-server request log:\n" + "\n".join(_REQUEST_LOG)
        )

        # THE bug: nested (leaf) new_tab events were dropped because
        # their parent_tab_id was the backend-synthetic registry key.
        assert printer.dropped_new_tabs == [], (
            f"Nested sub-agent new_tab events were dropped by the "
            f"frontend guard (parent_tab_id unknown to the webview): "
            f"{printer.dropped_new_tabs}"
        )

        # Frontend ends with root + middle + 2 leaf tabs.
        assert len(printer.tabs) == 4, (
            f"Expected 4 frontend tabs (root + middle + 2 leaves), "
            f"got {len(printer.tabs)}: {printer.tabs}"
        )

        # The middle tab is a direct child of the root frontend tab.
        middle_events = [
            e for e in printer.accepted_new_tabs
            if e.get("parent_tab_id") == root_tab_id
        ]
        assert len(middle_events) == 1, (
            f"Expected exactly 1 middle new_tab parented to "
            f"{root_tab_id!r}, got: {printer.accepted_new_tabs}"
        )

        # Both leaves are parented to a REAL frontend tab (the random
        # id the webview allocated for the middle sub-agent), never to
        # a backend-synthetic ``task-...__sub_N`` key.
        leaf_events = [
            e for e in printer.accepted_new_tabs
            if e.get("parent_tab_id") != root_tab_id
        ]
        assert len(leaf_events) == 2, (
            f"Expected 2 leaf new_tab events, got: {leaf_events}"
        )
        for ev in leaf_events:
            pid = ev.get("parent_tab_id", "")
            assert pid.startswith("fe-"), (
                f"Leaf new_tab parent_tab_id {pid!r} is not a frontend "
                f"tab id — nested tab would be dropped by the webview"
            )
            assert pid in printer.tabs
