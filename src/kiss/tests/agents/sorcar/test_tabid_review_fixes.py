# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the tabId routing fixes (multi-model review).

Two server-side root causes found by the claude-fable-5 / kimi-k3 /
gpt-5.6-sol review sequence:

* **Base-printer parity (B4).**  ``WebPrinter.broadcast`` treats every
  event that carries an explicit ``tabId`` as a transient targeted
  system event — never recorded or persisted — EXCEPT task-scoped
  ``prompt`` / ``result`` events, whose durable copy is recorded and
  persisted with the ``tabId`` STRIPPED (replay re-stamps events with
  the subscribing viewer's own tab id).  The base
  ``JsonPrinter.broadcast`` (production-reachable: ``VSCodeServer()``
  defaults to it and the CLI's ``RecordingConsolePrinter`` subclasses
  it) recorded + persisted EVERY event after taskId injection: a
  viewer-targeted transient ``clear`` (a display type) leaked into the
  recording/DB, and durable prompt/result copies kept a stale frontend
  tab id.

* **Duplicate ``tabId`` wire members (B5).**  ``_fanout_stamped`` (and
  the talk fan-outs) serialise the whole event dict and splice one
  ``"tabId": <target>`` member per subscriber.  An event that already
  carries a ``tabId`` key (reachable: ``_broadcast_subagent_done``
  emits ``{"tab_id": ..., "tabId": ""}``, which the CLI daemon bridge
  forwards verbatim and ``_relay_cli_event`` hands to
  ``_fanout_stamped``) produced JSON with TWO ``"tabId"`` members —
  ambiguous, non-interoperable, and only routed correctly because
  JSON parsers happen to keep the last member.
"""

import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server.json_printer import JsonPrinter
from kiss.server.web_server import WebPrinter


class _WireCapturingWebPrinter(WebPrinter):
    """A real ``WebPrinter`` that captures raw WS payload strings."""

    def __init__(self) -> None:
        super().__init__()
        self.wire: list[str] = []
        self._wire_lock = threading.Lock()

    def _send_to_ws_clients(self, data: str) -> None:
        with self._wire_lock:
            self.wire.append(data)


class _AgentStub:
    """Minimal stand-in carrying the ``_last_task_id`` persistence key."""

    def __init__(self, task_id: str) -> None:
        self._last_task_id = task_id


def _redirect(tmpdir: str):
    """Redirect the DB to a temp dir and reset the singleton connection."""
    old = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return old


def _restore(saved):
    (th._DB_PATH, th._db_conn, th._KISS_DIR) = saved


class _PersistenceHarness:
    """Shared temp-DB setup/teardown for both printer test classes.

    Subclasses declare their own concrete ``printer`` attribute (base
    ``JsonPrinter`` vs ``_WireCapturingWebPrinter``) in their
    ``setup_method``; ``Any`` here lets each subclass narrow it.
    """

    printer: Any

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        th._flush_chat_events()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self):
        th._flush_chat_events()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _register_task(self, task: str) -> str:
        """Create a real task row and register its agent with the printer."""
        task_id, _ = th._add_task(task, chat_id="chat-1")
        th._flush_chat_events()
        with self.printer._lock:
            self.printer._persist_agents[str(task_id)] = _AgentStub(task_id)
        return str(task_id)

    def _persisted_events(self, task_id: str) -> list:
        th._flush_chat_events()
        loaded = th._load_chat_events_by_task_id(task_id)
        if not loaded:
            return []
        events = loaded["events"]
        assert isinstance(events, list)
        return events


class TestBaseJsonPrinterTabIdParity(_PersistenceHarness):
    """B4: base ``JsonPrinter.broadcast`` must mirror WebPrinter's
    explicit-tabId semantics (transient system events; tabId-stripped
    durable prompt/result copies)."""

    printer: JsonPrinter

    def setup_method(self):
        super().setup_method()
        self.printer = JsonPrinter()

    def _start_task_thread_context(self, task_id: str) -> None:
        self.printer._thread_local.task_id = task_id
        self.printer.start_recording()

    def test_tab_stamped_clear_is_transient(self):
        """A viewer-targeted ``clear`` (tabId-stamped, display type)
        must NOT be recorded or persisted — it is a transient routing
        event ``_subscribe_chat_viewers`` sends to reset one tab."""
        task_id = self._register_task("parity clear")
        self._start_task_thread_context(task_id)
        self.printer.broadcast(
            {"type": "clear", "chat_id": "chat-1", "tabId": "viewer-tab-1"}
        )
        recorded = self.printer.stop_recording()
        assert recorded == [], (
            "tab-stamped transient 'clear' leaked into the recording: "
            f"{recorded}"
        )
        assert self._persisted_events(task_id) == [], (
            "tab-stamped transient 'clear' was persisted"
        )

    def test_tab_stamped_prompt_recorded_without_tabid(self):
        """The durable copy of an injected-prompt echo (tabId + taskId)
        must be recorded and persisted WITHOUT the stale tabId."""
        task_id = self._register_task("parity prompt")
        self._start_task_thread_context(task_id)
        self.printer.broadcast(
            {
                "type": "prompt",
                "text": "queued follow-up",
                "tabId": "launcher-tab",
                "taskId": task_id,
            }
        )
        recorded = self.printer.stop_recording()
        assert len(recorded) == 1, f"expected 1 recorded prompt, got {recorded}"
        assert recorded[0]["type"] == "prompt"
        assert "tabId" not in recorded[0], (
            "durable prompt copy kept the stale tabId in the recording"
        )
        persisted = self._persisted_events(task_id)
        assert len(persisted) == 1
        assert persisted[0]["type"] == "prompt"
        assert "tabId" not in persisted[0], (
            "durable prompt copy kept the stale tabId in the DB"
        )

    def test_tab_stamped_result_persisted_without_tabid(self):
        """A tab-stamped terminal ``result`` carrying a task-stream
        taskId gets a tabId-stripped durable copy (same defensive net
        as WebPrinter)."""
        task_id = self._register_task("parity result")
        self._start_task_thread_context(task_id)
        self.printer.broadcast(
            {
                "type": "result",
                "text": "Task stopped by user",
                "success": False,
                "tabId": "launcher-tab",
                "taskId": task_id,
            }
        )
        recorded = self.printer.stop_recording()
        assert len(recorded) == 1
        assert "tabId" not in recorded[0]
        persisted = self._persisted_events(task_id)
        assert len(persisted) == 1
        assert persisted[0]["text"] == "Task stopped by user"
        assert "tabId" not in persisted[0]

    def test_tab_stamped_result_without_taskid_is_transient(self):
        """A tab-stamped result with NO taskId (early failure before a
        row exists) stays transient in the base class too."""
        task_id = self._register_task("parity transient result")
        self._start_task_thread_context(task_id)
        # NOTE: no ``taskId`` on the event; the thread-local must not
        # be used to smuggle a transient tab-scoped event into the
        # durable stream (matches WebPrinter, whose tabId branch never
        # consults the thread-local).
        self.printer.broadcast(
            {
                "type": "result",
                "text": "No model available.",
                "success": False,
                "tabId": "launcher-tab",
            }
        )
        assert self.printer.stop_recording() == []
        assert self._persisted_events(task_id) == []

    def test_taskid_only_events_unchanged(self):
        """The regular (no explicit tabId) path still records+persists."""
        task_id = self._register_task("parity normal")
        self._start_task_thread_context(task_id)
        self.printer.broadcast({"type": "prompt", "text": "hello"})
        self.printer.broadcast({"type": "result", "text": "done", "success": True})
        recorded = self.printer.stop_recording()
        assert [e["type"] for e in recorded] == ["prompt", "result"]
        persisted = self._persisted_events(task_id)
        assert [e["type"] for e in persisted] == ["prompt", "result"]


class TestFanoutSingleTabIdStamp(_PersistenceHarness):
    """B5: the fan-out wire payload must contain exactly ONE ``tabId``
    member per copy, even when the source event already carried one."""

    printer: _WireCapturingWebPrinter

    def setup_method(self):
        super().setup_method()
        self.printer = _WireCapturingWebPrinter()

    def test_fanout_strips_preexisting_tabid(self):
        """``subagentDone`` relayed from the CLI carries ``tabId: ""``;
        the fanned-out copy must carry exactly one tabId — the
        subscriber's."""
        task_id = self._register_task("fanout dup")
        self.printer.subscribe_tab(task_id, "viewer-1")
        # Exactly what _relay_cli_event receives from the CLI bridge
        # for sorcar_agent._broadcast_subagent_done.
        event = {
            "type": "subagentDone",
            "tab_id": f"task-{task_id}__sub_0",
            "tabId": "",
            "taskId": task_id,
            "ts": 1234,
        }
        self.printer._fanout_stamped(event)
        assert len(self.printer.wire) == 1
        raw = self.printer.wire[0]
        assert raw.count('"tabId"') == 1, (
            f"wire payload has duplicate tabId members: {raw}"
        )
        parsed = json.loads(raw)
        assert parsed["tabId"] == "viewer-1"
        assert parsed["tab_id"] == f"task-{task_id}__sub_0"
        assert parsed["type"] == "subagentDone"

    def test_fanout_talk_cli_origin_strips_preexisting_tabid(self):
        """The CLI-origin talk fan-out must also stamp exactly one
        tabId per copy."""
        task_id = self._register_task("fanout talk dup")
        self.printer.subscribe_tab(task_id, "viewer-1")
        captured_wss: list[str] = []
        captured_uds: list[str] = []
        self.printer._send_to_wss_clients = captured_wss.append  # type: ignore[method-assign]
        self.printer._send_to_uds_writers = captured_uds.append  # type: ignore[method-assign]
        event = {
            "type": "talk",
            "text": "hello",
            "talkId": "talk-1",
            "tabId": "",
            "taskId": task_id,
        }
        self.printer._fanout_talk_cli_origin(event)
        assert len(captured_wss) == 1 and len(captured_uds) == 1
        for raw in (*captured_wss, *captured_uds):
            assert raw.count('"tabId"') == 1, (
                f"talk wire payload has duplicate tabId members: {raw}"
            )
            assert json.loads(raw)["tabId"] == "viewer-1"

    def test_fanout_talk_cli_origin_without_tabid_unchanged(self):
        """A CLI-origin talk event with NO pre-existing tabId is
        stamped normally (one tabId per copy, payload intact)."""
        task_id = self._register_task("fanout talk clean")
        self.printer.subscribe_tab(task_id, "viewer-1")
        captured: list[str] = []
        self.printer._send_to_wss_clients = captured.append  # type: ignore[method-assign]
        self.printer._send_to_uds_writers = captured.append  # type: ignore[method-assign]
        self.printer._fanout_talk_cli_origin(
            {"type": "talk", "text": "hi", "talkId": "t2", "taskId": task_id}
        )
        assert len(captured) == 2
        for raw in captured:
            assert raw.count('"tabId"') == 1
            parsed = json.loads(raw)
            assert parsed["tabId"] == "viewer-1"
            assert parsed["text"] == "hi"

    def test_normal_fanout_unchanged(self):
        """Events without a pre-existing tabId are stamped normally."""
        task_id = self._register_task("fanout normal")
        self.printer.subscribe_tab(task_id, "viewer-1")
        self.printer.subscribe_tab(task_id, "viewer-2")
        self.printer._fanout_stamped(
            {"type": "text_delta", "text": "hi", "taskId": task_id}
        )
        assert len(self.printer.wire) == 2
        tabs = set()
        for raw in self.printer.wire:
            assert raw.count('"tabId"') == 1
            parsed = json.loads(raw)
            assert parsed["text"] == "hi"
            tabs.add(parsed["tabId"])
        assert tabs == {"viewer-1", "viewer-2"}
