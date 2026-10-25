# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Files attached to a message sent while a task runs reach the agent.

``appendUserMessage`` carries ``attachments`` (the submit wire shape:
base64 ``data`` + ``mimeType``).  The handler decodes them onto the
owner state's ``pending_user_attachments`` (one list per message); the
pre-step drain injects the texts as ``User says: ...`` messages and
then ONE ``user`` message carrying all the files, which the model
adapters render as image / file content parts.  Files queued after the
agent's last step ride along with the leftover prompts the runner
re-submits as the tab's next run, each message with its own.

Real ``AgentServer``, ``JsonPrinter``, ``SorcarAgent`` and
``AnthropicModel`` (no request is made); no mocks.
"""

from __future__ import annotations

import base64
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model import Attachment
from kiss.server import agent_state
from kiss.server.agent_state import AgentState
from kiss.server.json_printer import JsonPrinter
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import clear_agent_states, clear_tab_states
from kiss.tests.server.parallel_agent_harness import IsolatedKissHome

PNG = bytes([137, 80, 78, 71, 13, 10, 26, 10]) + b"\x00" * 8
PDF = b"%PDF-1.4 steer"


def _wire(data: bytes, mime: str, name: str) -> dict[str, str]:
    return {"name": name, "mimeType": mime, "data": base64.b64encode(data).decode("ascii")}


def _server() -> tuple[AgentServer, list[dict[str, Any]]]:
    server = AgentServer()
    events: list[dict[str, Any]] = []
    server.printer.broadcast = events.append  # type: ignore[assignment]
    return server, events


def _live_state(task_id: str, tab_id: str) -> AgentState:
    st = AgentState(task_id, tab_id=tab_id, server_owned=True)
    st.is_task_active = True
    agent_state.register(st)
    return st


class TestQueueAndDrain:
    def setup_method(self) -> None:
        clear_agent_states()

    def teardown_method(self) -> None:
        clear_agent_states()

    def test_message_files_are_queued_and_shown_to_the_model(self) -> None:
        server, _events = _server()
        st = _live_state("task-att", "tab-att")

        server._cmd_append_user_message({
            "tabId": "tab-att",
            "prompt": "use this screenshot",
            "attachments": [_wire(PNG, "image/png", "shot.png")],
        })
        server._cmd_append_user_message({"tabId": "tab-att", "prompt": "and this"})
        server._cmd_append_user_message({
            "tabId": "tab-att",
            "prompt": "plus the spec",
            "attachments": [_wire(PDF, "application/pdf", "spec.pdf")],
        })
        assert st.pending_user_messages == ["use this screenshot", "and this", "plus the spec"]
        assert [
            [(a.mime_type, a.data) for a in files] for files in st.pending_user_attachments
        ] == [[("image/png", PNG)], [], [("application/pdf", PDF)]]

        model = AnthropicModel("claude-steer-test", api_key="test-key")
        model.initialize("the task")
        agent = SorcarAgent("drain")
        agent.printer = server.printer
        server.printer._thread_local.task_id = "task-att"
        try:
            agent._drain_pending_user_messages(model)
        finally:
            server.printer._thread_local.task_id = None

        roles = [m["role"] for m in model.conversation]
        assert roles == ["user", "user", "user", "user", "user"]
        texts = [m["content"] for m in model.conversation[1:4]]
        assert texts == [
            "User says: use this screenshot. Take the message into account and finish your task.",
            "User says: and this. Take the message into account and finish your task.",
            "User says: plus the spec. Take the message into account and finish your task.",
        ]
        assert all("attachments" not in m for m in model.conversation[:4])
        files = model.conversation[4]
        assert files["content"] == "[files the user attached to the message(s) above]"
        assert [(a.mime_type, a.data) for a in files["attachments"]] == [
            ("image/png", PNG), ("application/pdf", PDF),
        ]
        # The adapter turns the message into real content blocks.
        rendered = model._normalize_message_for_api(files)
        kinds = [b["type"] for b in rendered[0]["content"]]
        assert kinds == ["image", "document", "text"]
        # Drained: a second drain injects nothing.
        assert st.pending_user_messages == []
        assert st.pending_user_attachments == []
        before = len(model.conversation)
        server.printer._thread_local.task_id = "task-att"
        try:
            agent._drain_pending_user_messages(model)
        finally:
            server.printer._thread_local.task_id = None
        assert len(model.conversation) == before

    def test_text_only_messages_add_no_file_message(self) -> None:
        server, _events = _server()
        _live_state("task-txt", "tab-txt")
        server._cmd_append_user_message({"tabId": "tab-txt", "prompt": "just text"})
        server._cmd_append_user_message({
            "tabId": "tab-txt", "prompt": "empty list", "attachments": [],
        })
        model = AnthropicModel("claude-steer-test", api_key="test-key")
        model.initialize("the task")
        agent = SorcarAgent("drain")
        agent.printer = server.printer
        server.printer._thread_local.task_id = "task-txt"
        try:
            agent._drain_pending_user_messages(model)
        finally:
            server.printer._thread_local.task_id = None
        assert [m["content"] for m in model.conversation[1:]] == [
            "User says: just text. Take the message into account and finish your task.",
            "User says: empty list. Take the message into account and finish your task.",
        ]
        assert all("attachments" not in m for m in model.conversation)

    def test_task_followup_keeps_its_files_for_the_running_agent(self) -> None:
        server, _events = _server()
        st = _live_state("task-tag", "tab-tag")
        server._cmd_append_user_message({
            "tabId": "tab-tag",
            "prompt": "<task>later, with this</task>",
            "attachments": [_wire(PNG, "image/png", "shot.png")],
        })
        assert st.queued_followup_tasks == ["later, with this"]
        assert st.pending_user_messages == []
        assert st.pending_user_attachments == []
        # The files belong to the queued task, one entry per block.
        assert [[a.data for a in files] for files in st.queued_followup_files] == [[PNG]]
        server._cmd_append_user_message({
            "tabId": "tab-tag",
            "prompt": "<task>one</task><task>two</task>",
            "attachments": [_wire(PDF, "application/pdf", "spec.pdf")],
        })
        assert st.queued_followup_tasks == ["later, with this", "one", "two"]
        assert [[a.data for a in files] for files in st.queued_followup_files] == [
            [PNG], [PDF], [PDF],
        ]

    def test_run_on_a_busy_tab_keeps_the_message_files(self) -> None:
        """A ``run`` typed while the tab's task runs is a steering message, files included."""
        server, _events = _server()
        st = _live_state("task-busy", "tab-busy")
        st.task_thread = threading.current_thread()
        try:
            server._cmd_run({
                "type": "run",
                "tabId": "tab-busy",
                "prompt": "while you are at it",
                "workDir": "/tmp",
                "attachments": [_wire(PNG, "image/png", "shot.png")],
            })
        finally:
            st.task_thread = None
        assert st.pending_user_messages == ["while you are at it"]
        assert [
            [(a.mime_type, a.data) for a in files] for files in st.pending_user_attachments
        ] == [[("image/png", PNG)]]

    def test_malformed_attachment_entries_are_skipped(self) -> None:
        server, _events = _server()
        st = _live_state("task-bad", "tab-bad")
        server._cmd_append_user_message({
            "tabId": "tab-bad",
            "prompt": "broken",
            "attachments": [
                {"mimeType": "image/png", "data": 12},
                _wire(PDF, "application/pdf", "x"),
            ],
        })
        assert [[a.mime_type for a in files] for files in st.pending_user_attachments] == [
            ["application/pdf"],
        ]
        server._cmd_append_user_message({
            "tabId": "tab-bad", "prompt": "not a list", "attachments": "nope",
        })
        assert [[a.mime_type for a in files] for files in st.pending_user_attachments] == [
            ["application/pdf"], [],
        ]

    def test_daemon_client_puts_attachments_on_the_wire_only_when_given(self) -> None:
        from kiss.agents.sorcar import daemon_client

        plain = daemon_client.run_command("q", tab_id="t", run_token="r")
        assert "attachments" not in plain
        empty = daemon_client.run_command("q", tab_id="t", run_token="r", attachments=[])
        assert "attachments" not in empty
        wire = _wire(PNG, "image/png", "shot.png")
        cmd = daemon_client.run_command("q", tab_id="t", run_token="r", attachments=[wire])
        assert cmd["attachments"] == [wire]

    def test_add_message_to_conversation_attachments_argument(self) -> None:
        model = AnthropicModel("claude-steer-test", api_key="test-key")
        model.initialize("hi")
        model.add_message_to_conversation("user", "plain")
        model.add_message_to_conversation("user", "with file", attachments=[])
        att = Attachment(data=PNG, mime_type="image/png")
        model.add_message_to_conversation("user", "with file", attachments=[att])
        assert "attachments" not in model.conversation[1]
        assert "attachments" not in model.conversation[2]
        assert model.conversation[3]["attachments"] == [att]


_TAB = "tab-steer-redispatch"
_UNKNOWN_MODEL = "steer-attachments-no-such-model"


class _RecordingRunsServer(AgentServer):
    """A real server that keeps every ``run`` command it is given."""

    def __init__(self, printer: JsonPrinter) -> None:
        super().__init__(printer=printer)
        self.runs: list[dict[str, Any]] = []

    def _cmd_run(self, cmd: dict[str, Any]) -> None:
        self.runs.append(dict(cmd))
        super()._cmd_run(cmd)


class _LateFilePrinter(JsonPrinter):
    """Real printer that sends a message with a file once the run has started."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.server: AgentServer | None = None
        self._sent = False

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))
        super().broadcast(event)
        if (
            event.get("type") == "status"
            and event.get("running") is True
            and not self._sent
            and self.server is not None
        ):
            self._sent = True
            self.server._cmd_append_user_message({
                "tabId": _TAB,
                "prompt": "also look at this",
                "attachments": [_wire(PNG, "image/png", "late.png")],
            })


@pytest.fixture
def _home() -> Iterator[IsolatedKissHome]:
    home = IsolatedKissHome("kiss-steer-attachments-")
    try:
        clear_tab_states(_TAB)
        yield home
    finally:
        clear_tab_states(_TAB)
        home.cleanup()


def _wait_idle() -> None:
    deadline = time.time() + 60.0
    while time.time() < deadline:
        with agent_state.STATE_LOCK:
            st = agent_state.find_by_tab(_TAB)
            thread = st.task_thread if st is not None else None
        if thread is not None:
            thread.join(timeout=max(0.0, deadline - time.time()))
            continue
        time.sleep(0.3)
        with agent_state.STATE_LOCK:
            st = agent_state.find_by_tab(_TAB)
            if st is None or st.task_thread is None:
                return
    raise AssertionError("tab never went idle")


def test_leftover_message_files_ride_along_with_the_redispatched_run(
    tmp_path: Path, _home: IsolatedKissHome,
) -> None:
    """A file sent after the agent's last step is the follow-up run's attachment."""
    printer = _LateFilePrinter()
    server = _RecordingRunsServer(printer)
    printer.server = server
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    server._cmd_run({
        "type": "run",
        "tabId": _TAB,
        "prompt": "first run",
        "workDir": str(work_dir),
        "model": _UNKNOWN_MODEL,
        "classifyTasks": False,
        "attachments": [_wire(PDF, "application/pdf", "first.pdf")],
    })
    _wait_idle()
    results = [e for e in printer.events if e.get("type") == "result"]
    assert len(results) == 2, results
    assert len(server.runs) == 2
    followup = server.runs[1]
    assert followup["prompt"] == "also look at this"
    # The follow-up carries the late message's file, not the first run's.
    assert [(a["mimeType"], base64.b64decode(a["data"])) for a in followup["attachments"]] == [
        ("image/png", PNG),
    ]
    st = agent_state.find_by_tab(_TAB)
    assert st is not None
    assert st.pending_user_messages == []
    assert st.pending_user_attachments == []
