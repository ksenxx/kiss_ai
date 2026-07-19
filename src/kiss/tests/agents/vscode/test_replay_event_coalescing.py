# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for replay-payload coalescing and fan-out encoding.

Persisted chat streams store one ``events`` row per streamed token, so
a long task accumulates tens of thousands of tiny ``thinking_delta`` /
``text_delta`` rows.  The server must coalesce consecutive same-type
delta events (concatenating their ``text``) before shipping the replay
payload to the webview — in ``task_events`` (history click / tab
reattach / persisted sub-agent tabs) and ``adjacent_task_events``
(prev/next navigation).  Rendering is identical because the frontend
replay loop simply appends each delta's ``text``.

Also pins the :class:`WebPrinter` fan-out wire format: the event is
JSON-serialised once and the per-tab ``tabId`` stamp is spliced into
the serialised string, which must decode to exactly
``{**event, "tabId": tab_id}`` for every subscribed tab.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.server.server import VSCodeServer
from kiss.server.web_server import WebPrinter


def _redirect(tmpdir: str) -> tuple[Path, object, Path]:
    """Redirect the persistence DB to a temp dir; return saved state."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _make_server() -> tuple[VSCodeServer, list[dict]]:
    """Create a VSCodeServer whose broadcasts go into an in-memory list."""
    server = VSCodeServer()
    events: list[dict] = []
    lock = threading.Lock()

    def capture(event: dict) -> None:
        ev = server.printer._inject_task_id(event)
        with server.printer._lock:
            server.printer._record_event(ev)
        with lock:
            events.append(ev)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


# The raw per-token stream a finished task would have persisted: three
# thinking tokens, three text tokens, a tool call boundary, two more
# text tokens, and the result.
_RAW_STREAM: list[dict] = [
    {"type": "prompt", "text": "do the thing"},
    {"type": "thinking_start"},
    {"type": "thinking_delta", "text": "th"},
    {"type": "thinking_delta", "text": "ink"},
    {"type": "thinking_delta", "text": "ing"},
    {"type": "thinking_end"},
    {"type": "text_delta", "text": "he"},
    {"type": "text_delta", "text": "ll"},
    {"type": "text_delta", "text": "o"},
    {"type": "tool_call", "name": "Bash", "command": "ls"},
    {"type": "text_delta", "text": "wor"},
    {"type": "text_delta", "text": "ld"},
    {"type": "text_end"},
    {"type": "result", "text": "done", "step_count": 1},
]

# What the webview must receive: consecutive same-type deltas merged,
# every boundary event preserved in order.
_COALESCED_TYPES: list[str] = [
    "prompt",
    "thinking_start",
    "thinking_delta",
    "thinking_end",
    "text_delta",
    "tool_call",
    "text_delta",
    "text_end",
    "result",
]


def _persist_stream(task_id: str, stream: list[dict]) -> None:
    """Persist *stream* one event per row, mirroring live streaming."""
    for ev in stream:
        th._queue_chat_event(dict(ev), task_id=task_id)
    th._flush_chat_events()


def _assert_coalesced(events: list[dict]) -> None:
    """Assert *events* is the coalesced form of ``_RAW_STREAM``."""
    types = [e.get("type") for e in events]
    assert types == _COALESCED_TYPES, f"got types={types}"
    deltas = [e for e in events if e.get("type") == "thinking_delta"]
    assert deltas[0]["text"] == "thinking"
    texts = [e["text"] for e in events if e.get("type") == "text_delta"]
    assert texts == ["hello", "world"]
    # The merged event keeps the metadata of the first fragment,
    # including the injected ``_timestamp``.
    assert "_timestamp" in deltas[0]


class TestReplayCoalescing:
    """Server-side replay payloads merge consecutive delta events."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect(self.tmpdir)

    def teardown_method(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_replay_session_coalesces_consecutive_deltas(self) -> None:
        chat_id = "chat-coalesce-1"
        task_id, _ = th._add_task("streamed task", chat_id=chat_id)
        _persist_stream(task_id, _RAW_STREAM)

        server, events = _make_server()
        server._replay_session(chat_id=chat_id, tab_id="tab-1")

        replays = [e for e in events if e.get("type") == "task_events"]
        assert len(replays) == 1
        assert replays[0]["tabId"] == "tab-1"
        _assert_coalesced(replays[0]["events"])

    def test_adjacent_task_events_coalesced(self) -> None:
        chat_id = "chat-coalesce-2"
        first_id, _ = th._add_task("first task", chat_id=chat_id)
        _persist_stream(first_id, _RAW_STREAM)
        time.sleep(0.01)
        second_id, _ = th._add_task("second task", chat_id=chat_id)
        th._append_chat_event(
            {"type": "text_delta", "text": "second"}, task_id=second_id,
        )

        server, events = _make_server()
        server._get_adjacent_task(chat_id, second_id, "prev", tab_id="tab-2")

        adj = [e for e in events if e.get("type") == "adjacent_task_events"]
        assert len(adj) == 1
        assert adj[0]["tabId"] == "tab-2"
        assert adj[0]["task_id"] == first_id
        _assert_coalesced(adj[0]["events"])

    def test_adjacent_task_without_result_sends_empty_events(self) -> None:
        server, events = _make_server()
        server._get_adjacent_task("no-such-chat", None, "prev", tab_id="t")

        adj = [e for e in events if e.get("type") == "adjacent_task_events"]
        assert len(adj) == 1
        assert adj[0]["events"] == []
        # Genuine "no adjacent row" contract: the frontend distinguishes
        # end-of-chat (task '' AND task_id None) from a real row with an
        # empty trajectory (real task_id, events []).
        assert adj[0]["task"] == ""
        assert adj[0]["task_id"] is None

    def test_adjacent_task_with_empty_trajectory_keeps_task_id(self) -> None:
        """A real adjacent row whose trajectory recorded NO events must
        reply with its real ``task_id``/title and ``events: []`` — NOT
        the end-of-chat shape (task '', task_id None).  The frontend
        relies on this to scroll PAST short/empty tasks instead of
        latching noPrevTask/noNextTask on them."""
        chat_id = "chat-empty-traj"
        first_id, _ = th._add_task("short empty task", chat_id=chat_id)
        # No events persisted for first_id — empty trajectory.
        time.sleep(0.01)
        second_id, _ = th._add_task("current task", chat_id=chat_id)
        _persist_stream(second_id, _RAW_STREAM)

        server, events = _make_server()
        server._get_adjacent_task(chat_id, second_id, "prev", tab_id="t2")

        adj = [e for e in events if e.get("type") == "adjacent_task_events"]
        assert len(adj) == 1
        assert adj[0]["task_id"] == first_id
        assert adj[0]["task"] == "short empty task"
        assert adj[0]["events"] == []

    def test_persisted_subagent_task_events_coalesced(self) -> None:
        chat_id = "chat-coalesce-3"
        parent_id, _ = th._add_task("parent task", chat_id=chat_id)
        sub_id, _ = th._add_task("sub task", chat_id=chat_id)
        _persist_stream(sub_id, _RAW_STREAM)
        th._save_task_extra(
            {
                "model": "test-model",
                "work_dir": "/tmp",
                "version": "test",
                "tokens": 0,
                "cost": 0.0,
                "is_parallel": False,
                "is_worktree": False,
                "subagent": {"parent_task_id": parent_id},
            },
            task_id=sub_id,
        )

        server, events = _make_server()
        server._open_persisted_subagent_tabs(
            parent_task_id=parent_id, parent_tab_id="tab-parent",
        )

        sub_replays = [e for e in events if e.get("type") == "task_events"]
        assert len(sub_replays) == 1
        assert sub_replays[0]["task_id"] == sub_id
        _assert_coalesced(sub_replays[0]["events"])

    def test_single_deltas_pass_through_unchanged(self) -> None:
        """A stream with no consecutive same-type deltas is untouched."""
        chat_id = "chat-coalesce-4"
        task_id, _ = th._add_task("tiny task", chat_id=chat_id)
        stream = [
            {"type": "text_delta", "text": "only"},
            {"type": "text_end"},
            {"type": "result", "text": "done"},
        ]
        _persist_stream(task_id, stream)

        server, events = _make_server()
        server._replay_session(chat_id=chat_id, tab_id="tab-4")

        replays = [e for e in events if e.get("type") == "task_events"]
        assert len(replays) == 1
        got = replays[0]["events"]
        assert [e.get("type") for e in got] == [
            "text_delta", "text_end", "result",
        ]
        assert got[0]["text"] == "only"


class TestFanoutSingleSerialization:
    """The spliced fan-out payload decodes to ``{**event, "tabId": tab}``
    for every subscribed tab, over a real UDS transport."""

    def test_fanout_wire_payload_is_json_equivalent(self) -> None:
        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        tmpdir = tempfile.mkdtemp()
        sock_path = str(Path(tmpdir) / "fanout.sock")
        received: list[str] = []
        got_lines = threading.Event()

        async def _handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
        ) -> None:
            while True:
                line = await reader.readline()
                if not line:
                    break
                received.append(line.decode("utf-8"))
                if len(received) >= 2:
                    got_lines.set()

        server: asyncio.Server | None = None
        writer: asyncio.StreamWriter | None = None
        try:
            server = asyncio.run_coroutine_threadsafe(
                asyncio.start_unix_server(_handler, path=sock_path), loop,
            ).result(timeout=5)

            async def _connect() -> asyncio.StreamWriter:
                _reader, w = await asyncio.open_unix_connection(
                    sock_path, limit=16 * 1024 * 1024,
                )
                return w

            writer = asyncio.run_coroutine_threadsafe(
                _connect(), loop,
            ).result(timeout=5)

            printer = WebPrinter()
            printer._loop = loop
            printer.add_uds_writer(writer)
            printer.subscribe_tab("42", "tab-A")
            printer.subscribe_tab("42", "tab-B")

            event = {
                "type": "text_delta",
                "text": 'uni\u00e7ode "quoted" \\ back\nslash',
            }

            def _emit() -> None:
                printer._thread_local.task_id = "42"
                printer.broadcast(event)

            t = threading.Thread(target=_emit)
            t.start()
            t.join(timeout=5)
            assert not t.is_alive()

            assert got_lines.wait(timeout=5), f"received={received}"
            decoded = sorted(
                (json.loads(line) for line in received),
                key=lambda d: str(d.get("tabId")),
            )
            # ``broadcast`` stamps the event's emission time (``ts``,
            # ms since epoch — see ``stamp_event_ts``) once, BEFORE
            # serialization, so both fan-out copies must carry the
            # identical stamp.
            ts_a = decoded[0].pop("ts")
            ts_b = decoded[1].pop("ts")
            assert isinstance(ts_a, int) and ts_a > 0
            assert ts_b == ts_a
            expected_base = {
                "type": "text_delta",
                "text": 'uni\u00e7ode "quoted" \\ back\nslash',
                "taskId": "42",
            }
            assert decoded[0] == {**expected_base, "tabId": "tab-A"}
            assert decoded[1] == {**expected_base, "tabId": "tab-B"}
        finally:
            async def _shutdown() -> None:
                # Close the client first so the handler's ``readline``
                # sees EOF and the handler task exits cleanly before
                # the loop stops.
                if writer is not None:
                    writer.close()
                    await writer.wait_closed()
                if server is not None:
                    server.close()
                    await server.wait_closed()
                loop.stop()

            asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            loop_thread.join(timeout=5)
            loop.close()
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    import unittest

    unittest.main()
