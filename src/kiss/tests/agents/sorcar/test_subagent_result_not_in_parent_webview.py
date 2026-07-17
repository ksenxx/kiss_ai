# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Webview-level integration test: sub-agent result panels MUST NOT render
in the parent agent's chat tab.

This test reproduces the user-visible bug from the screenshot where, after
running ``run_parallel`` from a parent chat, the parent tab's chat scroll
area displays the parent's own ``Result`` panel + ``SUGGESTED NEXT``
followed by one or more sub-agents' ``Result`` panels (each with its own
``Tokens: …`` / ``Cost: …`` footer).

The reproduction wires together:

1. A real :class:`kiss.agents.sorcar.chat_sorcar_agent.ChatSorcarAgent`
   parent which calls ``run_parallel`` to spawn three sub-agents
   (driven by a fake OpenAI HTTP server).
2. A :class:`_FakeWebPrinter` that subclasses
   :class:`kiss.server.web_server.WebPrinter` and overrides
   :meth:`_send_to_ws_clients` to (a) capture every per-tab post-fan-out
   payload that would have been sent over the WebSocket, and (b) mimic
   the frontend's ``new_tab`` round-trip by allocating a fresh
   sub-tab uuid and calling :meth:`subscribe_tab` synchronously so
   later sub-agent events have a real subscriber to fan out to.
3. A small in-memory port of the webview's default-case dispatcher
   from ``media/main.js`` (the ``processOutputEvent`` /
   ``processOutputEventForBgTab`` branch).  The port walks the
   captured payloads and records, for each ``result`` event, which
   tab id ended up "rendering" it — either the active tab (current
   focus) or a background tab matched by id.

The test asserts: **no sub-agent ``result`` payload ends up rendered
into the parent tab**, regardless of whether the parent tab is the
active tab at the moment the payload arrives or sits in the
background.  The user-visible bug is exactly the violation of this
property.
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
from kiss.server.web_server import WebPrinter

# ---------------------------------------------------------------------------
# Fake OpenAI server: parent → run_parallel, sub-agents + parent's 2nd call → finish.
# ---------------------------------------------------------------------------


def _run_parallel_response() -> dict[str, Any]:
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
                                        "Compute 2+3.",
                                        "Compute 7*8.",
                                        "Compute 10-4.",
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


def _finish_response(summary: str) -> dict[str, Any]:
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


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        cl = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(cl) if cl else b""
        try:
            req = json.loads(raw)
        except Exception:
            req = {}
        is_stream = bool(req.get("stream", False))
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
            resp = _finish_response("SUB-RESULT")
        elif has_rp_result:
            resp = _finish_response("PARENT-RESULT")
        else:
            resp = _run_parallel_response()

        if is_stream:
            choice = resp["choices"][0]
            msg = choice["message"]
            delta2: dict[str, Any] = {}
            if msg.get("tool_calls"):
                delta2["tool_calls"] = msg["tool_calls"]
            if msg.get("content"):
                delta2["content"] = msg["content"]
            chunks = [
                {
                    "id": resp["id"],
                    "object": "chat.completion.chunk",
                    "model": resp["model"],
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": None},
                        "finish_reason": None,
                    }],
                },
                {
                    "id": resp["id"],
                    "object": "chat.completion.chunk",
                    "model": resp["model"],
                    "choices": [{
                        "index": 0,
                        "delta": delta2,
                        "finish_reason": choice["finish_reason"],
                    }],
                },
                {
                    "id": resp["id"],
                    "object": "chat.completion.chunk",
                    "model": resp["model"],
                    "choices": [],
                    "usage": resp["usage"],
                },
            ]
            body = (
                "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
                + "data: [DONE]\n\n"
            ).encode()
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

    def log_message(self, *args: Any, **kwargs: Any) -> None:  # noqa: A002
        pass


def _start_server() -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


# ---------------------------------------------------------------------------
# Fake WebPrinter: capture per-tab post-fanout payloads + mimic frontend's
# ``new_tab`` round-trip synchronously so later sub-agent broadcasts have a
# real subscriber to fan out to.
# ---------------------------------------------------------------------------


class _FakeWebPrinter(WebPrinter):
    """A ``WebPrinter`` that captures WS payloads instead of sending them.

    On every ``new_tab`` system event we synthesise the round-trip the
    real frontend performs (``createNewTab`` → ``resumeSession`` → server
    ``subscribe_tab``).  Without this round-trip the sub-agent's
    subsequent broadcasts would silently drop (no subscribers exist for
    the sub-agent's ``task_id``), and the test would miss any
    misrouting that happens during the live stream.
    """

    def __init__(self) -> None:
        super().__init__()
        # All per-tab payloads after fan-out, in the order
        # ``_send_to_ws_clients`` would have shipped them.
        self.wire: list[dict[str, Any]] = []
        # Mapping from sub-agent ``task_id`` -> allocated sub-tab uuid
        # (mimics frontend's ``createNewTab`` allocating a uuid).
        self._sub_tabs: dict[str, str] = {}
        self._wire_lock = threading.Lock()

    def _send_to_ws_clients(self, data: str) -> None:
        """Capture every payload that would have been sent over the WS.

        Also performs the synchronous frontend round-trip for
        ``new_tab`` events by allocating a fresh sub-tab uuid and
        calling :meth:`subscribe_tab` so the sub-agent's subsequent
        broadcasts have a real fan-out target.  This is intentionally
        synchronous: the sub-agent thread is blocked inside its
        ``broadcast`` call (which transitively called us), so the
        subscription is in place before any further sub-agent event
        is fanned out.
        """
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return
        with self._wire_lock:
            self.wire.append(payload)
        if payload.get("type") == "new_tab":
            task_id = payload.get("task_id")
            if task_id is not None:
                sub_tab_id = uuid.uuid4().hex
                self._sub_tabs[str(task_id)] = sub_tab_id
                self.subscribe_tab(task_id, sub_tab_id)


# ---------------------------------------------------------------------------
# In-memory port of media/main.js default-case dispatch.
#
# Mirrors the routing rules in main.js:
#
#   default:
#     if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
#       const bgTab = findTabByEvt(ev);
#       if (bgTab) processOutputEventForBgTab(ev, bgTab);
#       break;
#     }
#     processOutputEvent(ev);  // appends to active tab
#
# We additionally model the side effects of the ``new_tab`` handler
# (creates a fresh tab and switches focus) so the simulated
# ``activeTabId`` evolves the same way the real webview's does.
# ---------------------------------------------------------------------------


def _simulate_webview_dispatch(
    wire: list[dict[str, Any]],
    sub_tabs: dict[str, str],
    parent_tab_id: str,
    tab_current_task_ids: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Replay *wire* through a port of ``main.js`` dispatch.

    Returns a ``tab_id -> list[event]`` map of which tab each payload
    would have ended up "rendering" into.  Background renders and
    active renders are both attributed to the destination tab; the
    test asserts on the parent tab's bucket.

    Models the defensive guard added in main.js's default dispatch:
    a ``result`` / ``usage_info`` event whose ``taskId`` does not
    match the active tab's ``currentTaskId`` (when both are set) is
    dropped, never rendered onto the active tab.  ``task_events``
    handler also stamps ``currentTaskId`` on tabs.

    Args:
        wire: Captured per-tab post-fan-out payloads.
        sub_tabs: ``task_id -> sub_tab_uuid`` map.
        parent_tab_id: The parent tab's id.
        tab_current_task_ids: Optional initial ``tab_id ->
            currentTaskId`` seed (e.g. when the parent tab has
            already been associated with its task via ``task_events``
            BEFORE the wire snippet replayed here begins).
    """
    tabs: set[str] = {parent_tab_id}
    active = parent_tab_id
    rendered: dict[str, list[dict[str, Any]]] = {parent_tab_id: []}
    current_task: dict[str, str] = dict(tab_current_task_ids or {})

    for ev in wire:
        t = ev.get("type")
        if t == "new_tab":
            task_id = ev.get("task_id")
            if task_id is None:
                continue
            sub_tab = sub_tabs.get(str(task_id))
            if sub_tab is None:
                continue
            tabs.add(sub_tab)
            active = sub_tab  # createNewTab() switches focus
            rendered.setdefault(sub_tab, [])
            continue
        if t == "openSubagentTab":
            sub_tab = ev.get("tab_id")
            if isinstance(sub_tab, str) and sub_tab:
                tabs.add(sub_tab)
                rendered.setdefault(sub_tab, [])
            continue
        if t == "subagentDone":
            continue
        if t == "task_events":
            te_tab_id = ev.get("tabId") or active
            te_task_id = ev.get("task_id")
            if te_task_id is not None:
                current_task[te_tab_id] = str(te_task_id)
            continue
        ev_tab = ev.get("tabId")
        if ev_tab is not None and ev_tab != "" and ev_tab != active:
            if ev_tab in tabs:
                rendered.setdefault(ev_tab, []).append(ev)
            # Else: silently dropped (no matching tab exists).
            continue
        # Active-tab dispatch.  Apply defensive guard from main.js:
        # drop a misrouted result / usage_info event when its taskId
        # does not match the active tab's currentTaskId.
        if t in ("result", "usage_info"):
            ev_task = ev.get("taskId")
            act_task = current_task.get(active)
            if ev_task and act_task and str(ev_task) != str(act_task):
                # Dropped by guard — do not render.
                continue
        rendered.setdefault(active, []).append(ev)
    return rendered


# ---------------------------------------------------------------------------
# DB redirect (avoid clobbering user state).
# ---------------------------------------------------------------------------


def _redirect(tmpdir: str) -> tuple[Any, Any, Any]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    from pathlib import Path

    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore(saved: tuple[Any, Any, Any]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSubagentResultNotInParentWebview:
    """Sub-agent ``result`` payloads must not render in the parent tab."""

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

    def _run_parent(self, parent_tab_id: str) -> tuple[
        _FakeWebPrinter, ChatSorcarAgent,
    ]:
        printer = _FakeWebPrinter()
        parent = ChatSorcarAgent("parent")
        parent.run(
            prompt_template=(
                "Use run_parallel to compute three arithmetic expressions."
            ),
            model_name="gpt-4o-mini",
            model_config={"base_url": self.url, "api_key": "test-key"},
            work_dir=self.tmpdir,
            printer=printer,
            is_parallel=True,
            _subscribe_tab_id=parent_tab_id,
        )
        return printer, parent

    def test_parent_tab_renders_only_parent_result(self) -> None:
        """The parent tab's simulated DOM must contain exactly one
        ``result`` event — the parent's own ``PARENT-RESULT``.  Any
        sub-agent ``SUB-RESULT`` ending up there is the bug.
        """
        parent_tab_id = "parent-tab-AAA"
        printer, parent = self._run_parent(parent_tab_id)
        parent_task_key = str(parent._last_task_id)
        assert parent._last_task_id is not None

        rendered = _simulate_webview_dispatch(
            printer.wire, printer._sub_tabs, parent_tab_id,
        )
        parent_bucket = rendered.get(parent_tab_id, [])
        parent_results = [
            e for e in parent_bucket if e.get("type") == "result"
        ]

        # The parent tab must show exactly ONE result panel, and that
        # panel must be the parent's own (carrying the parent's taskId).
        assert len(parent_results) == 1, (
            f"Parent tab should render exactly 1 result panel, got "
            f"{len(parent_results)}: "
            f"{[r.get('text') or r.get('summary') for r in parent_results]}"
        )
        only = parent_results[0]
        assert only.get("taskId") == parent_task_key, (
            f"Parent tab's only result must carry the parent's taskId "
            f"{parent_task_key}; got taskId={only.get('taskId')}"
        )
        text = only.get("text") or only.get("summary") or ""
        assert "PARENT-RESULT" in text, (
            f"Parent tab's result panel should be 'PARENT-RESULT', got: "
            f"{text!r}"
        )

    def test_misrouted_subagent_result_dropped_by_guard(self) -> None:
        """Defensive-guard regression test: even when a sub-agent's
        ``result`` event is artificially injected onto the parent
        tab's WS stream (the user-visible symptom in the bug
        screenshot), the frontend guard introduced in main.js'
        default dispatch MUST drop it instead of rendering it onto
        the parent's DOM.

        This test reproduces the SYMPTOM directly (a sub-agent's
        result event arriving with ``tabId == parent_tab_id``) and
        verifies the guard catches it.
        """
        parent_tab_id = "parent-tab-CCC"
        parent_task_key = "9001"
        sub_task_key = "9002"

        # Wire mirroring what the bug would look like: parent's
        # task_events stamps currentTaskId, parent's result panel
        # renders, then SUGGESTED NEXT, then a MIS-ROUTED sub-agent
        # ``result`` event with the parent's tabId but a different
        # taskId (the leaked sub-agent's task id).
        wire: list[dict[str, Any]] = [
            {
                "type": "task_events",
                "tabId": parent_tab_id,
                "task_id": int(parent_task_key),
                "events": [],
                "task": "parent task",
            },
            {
                "type": "result",
                "tabId": parent_tab_id,
                "taskId": parent_task_key,
                "text": "PARENT-RESULT",
                "total_tokens": 38517,
                "cost": "$0.2148",
            },
            {
                "type": "followup_suggestion",
                "tabId": parent_tab_id,
                "taskId": parent_task_key,
                "text": "SUGGESTED NEXT",
            },
            # The bug: a sub-agent's result reaches the parent's WS
            # stream tagged with parent_tab_id (whatever the root
            # cause may be).  Without the guard, the frontend would
            # render this on the parent tab's DOM, producing the
            # duplicate Result panel the user reported.
            {
                "type": "result",
                "tabId": parent_tab_id,
                "taskId": sub_task_key,
                "text": "SUB-RESULT-LEAK",
                "total_tokens": 7577,
                "cost": "$0.0497",
            },
        ]

        rendered = _simulate_webview_dispatch(
            wire=wire,
            sub_tabs={},
            parent_tab_id=parent_tab_id,
        )

        parent_results = [
            e for e in rendered.get(parent_tab_id, [])
            if e.get("type") == "result"
        ]
        # Exactly the parent's own result; the leaked sub-agent
        # result must be dropped by the defensive guard.
        assert len(parent_results) == 1, (
            f"Defensive guard failed: parent tab rendered "
            f"{len(parent_results)} result panels (expected 1).  "
            f"Texts: "
            f"{[r.get('text') for r in parent_results]}"
        )
        assert parent_results[0].get("text") == "PARENT-RESULT", (
            f"Parent tab's only result must be the parent's own; "
            f"got: {parent_results[0].get('text')!r}"
        )

    def test_guard_does_not_drop_legitimate_parent_result(self) -> None:
        """Negative-control: the guard must NOT drop the parent's
        own ``result`` event (taskId matches active tab's
        currentTaskId).  This guards against a too-aggressive guard
        that breaks normal rendering.
        """
        parent_tab_id = "parent-tab-DDD"
        parent_task_key = "12345"

        wire: list[dict[str, Any]] = [
            {
                "type": "task_events",
                "tabId": parent_tab_id,
                "task_id": int(parent_task_key),
                "events": [],
                "task": "parent task",
            },
            {
                "type": "result",
                "tabId": parent_tab_id,
                "taskId": parent_task_key,
                "text": "PARENT-OK",
                "total_tokens": 1000,
                "cost": "$0.01",
            },
        ]
        rendered = _simulate_webview_dispatch(
            wire=wire,
            sub_tabs={},
            parent_tab_id=parent_tab_id,
        )
        parent_results = [
            e for e in rendered.get(parent_tab_id, [])
            if e.get("type") == "result"
        ]
        assert len(parent_results) == 1
        assert parent_results[0].get("text") == "PARENT-OK"
