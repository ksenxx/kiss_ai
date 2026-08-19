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
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.server.web_server import WebPrinter
from kiss.tests.agents.sorcar.test_subagent_result_not_in_parent_webview import (  # noqa: F401
    _finish_response,
    _Handler,
    _redirect,
    _restore,
    _run_parallel_response,
    _simulate_webview_dispatch,
    _start_server,
)


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
        self.wire: list[dict[str, Any]] = []
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

        def _on_alloc(task_id: Any, chat_id: str) -> None:
            # Mirror task_runner._on_run_task_id_allocated: attach the
            # launching UI tab to the task (also subscribes the tab).
            printer.register_task_ui(str(task_id), parent_tab_id)

        parent.run(
            prompt_template=(
                "Use run_parallel to compute three arithmetic expressions."
            ),
            model_name="gpt-4o-mini",
            model_config={"base_url": self.url, "api_key": "test-key"},
            work_dir=self.tmpdir,
            printer=printer,
            is_parallel=True,
            _on_task_id_allocated=_on_alloc,
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
