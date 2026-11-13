# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A replayed transcript leaves its pictures out; the page fetches each on demand.

A ``tool_result``'s screenshots travel as base64 (``images[].b64``),
which deflate cannot shrink: they were most of the bytes of a replayed
transcript on a slow network, for pictures the page shows only once
their panel is open and scrolled to.  A replay now carries each
picture as ``{mime, path, lazy: {taskId, index}}``
(``server._lazy_replay_images``), ``getReplayImage`` answers with the
picture (``AgentServer.replay_image`` reads the live recording while
the task runs, else the events table) and a bound remote tab's fetch
is forwarded to the machine (``sidebar_panels.machine_replay_image_reply``).

Real ``AgentServer`` with a real database; a real daemon and a real
link for the forwarded fetch; no mocks.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest import TestCase

import kiss.agents.sorcar.machines as mach
import kiss.agents.sorcar.persistence as th
import kiss.core.user_config as vc
from kiss.server import sidebar_panels
from kiss.server.server import AgentServer, _coalesced_replay_events, _replay_image
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import find_free_port
from kiss.tests.local_ws import _generate_self_signed_cert
from kiss.tests.persistence_helpers import close_db

_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 8).decode()
_SVG_B64 = base64.b64encode(b'<svg xmlns="http://www.w3.org/2000/svg"/>').decode()


def _transcript() -> list[dict[str, Any]]:
    """A persisted stream with two picture-carrying results around a plain one."""
    return [
        {"type": "tool_call", "tool": "screenshot", "input": {"file_path": "a.png"}},
        {
            "type": "tool_result", "content": "saved a.png", "is_error": False,
            "tool_name": "screenshot",
            "images": [
                {"b64": _PNG_B64, "mime": "image/png", "path": "a.png"},
                {"b64": _SVG_B64, "mime": "image/svg+xml", "path": "d.svg"},
            ],
        },
        {"type": "tool_call", "tool": "Bash", "input": {"command": "ls"}},
        {"type": "tool_result", "content": "a.png", "is_error": False, "tool_name": "Bash"},
        {"type": "tool_call", "tool": "screenshot", "input": {"file_path": "b.png"}},
        {
            "type": "tool_result", "content": "saved b.png", "is_error": False,
            "tool_name": "screenshot",
            "images": [{"b64": _PNG_B64, "mime": "image/png", "path": "b.png"}],
        },
        {"type": "result", "summary": "done", "success": True},
    ]


def _pictures(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every ``images`` entry of *events*, in order."""
    return [im for ev in events for im in (ev.get("images") or [])]


class _DatabaseCase(TestCase):
    """A temporary history database and an ``AgentServer`` whose broadcasts are captured."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-lazy-img-"))
        self._saved_persistence = (th._DB_PATH, th._KISS_DIR)
        kiss_dir = self.tmpdir / ".kiss"
        kiss_dir.mkdir()
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        self.server = AgentServer()
        self.server.work_dir = str(self.tmpdir)
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        original = self.server.printer.broadcast
        self.server.printer.broadcast = self._capturing(original)  # type: ignore[assignment]

    def _capturing(self, original: Any) -> Any:
        """A broadcast that records every event, then forwards it."""
        def capture(event: dict[str, Any]) -> None:
            with self._lock:
                self.events.append(dict(event))
            original(event)
        return capture

    def tearDown(self) -> None:
        close_db()
        th._DB_PATH, th._KISS_DIR = self._saved_persistence
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _persist(self, task: str, stream: list[dict[str, Any]]) -> tuple[str, str]:
        """Add a finished task with *stream* as its transcript; return (task_id, chat_id)."""
        task_id, chat_id = th._add_task(task)
        for ev in stream:
            th._queue_chat_event(dict(ev), task_id=task_id)
        th._flush_chat_events()
        th._save_task_result(result="done", task_id=task_id)
        return task_id, chat_id

    def _replayed(self, kind: str = "task_events") -> dict[str, Any]:
        """The last captured *kind* frame."""
        with self._lock:
            frames = [e for e in self.events if e.get("type") == kind]
        self.assertTrue(frames, f"no {kind} broadcast")
        return frames[-1]


class TestLazyReplayImages(_DatabaseCase):
    """A history click's replay and the fetch that follows it."""

    def test_replay_carries_handles_and_fetch_returns_each_picture(self) -> None:
        task_id, chat_id = self._persist("take screenshots", _transcript())
        self.server._handle_command(
            {"type": "resumeSession", "chatId": chat_id, "taskId": task_id, "tabId": "tab-1"},
        )
        replay = self._replayed()
        self.assertEqual(str(replay["task_id"]), task_id)
        handles = _pictures(replay["events"])
        self.assertEqual(
            handles,
            [
                {"mime": "image/png", "path": "a.png", "lazy": {"taskId": task_id, "index": 0}},
                {"mime": "image/svg+xml", "path": "d.svg", "lazy": {"taskId": task_id, "index": 1}},
                {"mime": "image/png", "path": "b.png", "lazy": {"taskId": task_id, "index": 2}},
            ],
        )
        self.assertNotIn(_PNG_B64, json.dumps(replay))
        # The fetch finds each picture by the handle's index.
        fetched = [self.server.replay_image(task_id, i) for i in range(3)]
        self.assertEqual(
            [(im and im["b64"], im and im["mime"]) for im in fetched],
            [(_PNG_B64, "image/png"), (_SVG_B64, "image/svg+xml"), (_PNG_B64, "image/png")],
        )
        self.assertIsNone(self.server.replay_image(task_id, 3))
        self.assertIsNone(self.server.replay_image(task_id, -1))
        self.assertIsNone(self.server.replay_image("no-such-task", 0))
        # The rest of the transcript is untouched, and the plain
        # result's events are the same objects (no copy for them).
        kinds = [ev.get("type") for ev in replay["events"] if ev.get("type") != "task_settings"]
        self.assertEqual(kinds, [ev["type"] for ev in _transcript()])

    def test_adjacent_task_replay_uses_handles_too(self) -> None:
        first_id, chat_id = self._persist("first", _transcript())
        time.sleep(0.01)  # the adjacent walk orders the chat's tasks by timestamp
        second_id, _ = th._add_task("second", chat_id=chat_id)
        th._append_chat_event(
            {"type": "result", "summary": "ok", "success": True}, task_id=second_id,
        )
        th._save_task_result(result="ok", task_id=second_id)
        self.server._get_adjacent_task(chat_id, second_id, "prev", tab_id="tab-1")
        adjacent = self._replayed("adjacent_task_events")
        self.assertEqual(str(adjacent["task_id"]), first_id)
        self.assertEqual(
            [im["lazy"]["index"] for im in _pictures(adjacent["events"])], [0, 1, 2],
        )
        self.assertNotIn(_PNG_B64, json.dumps(adjacent))

    def test_live_recording_is_read_while_the_task_runs(self) -> None:
        """A running task's replay reads the printer's recording; so does the fetch."""
        printer = self.server.printer
        task_id, _chat_id = th._add_task("live")
        printer.ensure_recording_for_task(task_id)
        try:
            for ev in _transcript()[:2]:
                printer.broadcast({**ev, "taskId": task_id})
            # The recording is the copy a running task's replay reads.
            recording = printer.peek_recording_for_task(task_id)
            self.assertEqual(len(_pictures(recording)), 2)
            stripped = _coalesced_replay_events(recording, task_id)
            self.assertEqual(
                [im["lazy"]["index"] for im in _pictures(stripped)], [0, 1],
            )
            # Stripping copied the events: the recording keeps its bytes.
            live = _pictures(printer.peek_recording_for_task(task_id))
            self.assertEqual(live[0]["b64"], _PNG_B64)
            found = self.server.replay_image(task_id, 1)
            self.assertIsNotNone(found)
            assert found is not None
            self.assertEqual(found["b64"], _SVG_B64)
        finally:
            printer.cleanup_task(task_id)

    def test_events_without_task_keep_their_pictures(self) -> None:
        """Without a task id (a share export, an older caller) nothing is stripped."""
        events = _coalesced_replay_events([dict(ev) for ev in _transcript()])
        self.assertEqual([im["b64"] for im in _pictures(events)], [_PNG_B64, _SVG_B64, _PNG_B64])
        self.assertIsNone(_replay_image("not a list", 0))
        self.assertIsNone(_replay_image([{"images": "junk"}, {"images": [None, {"b64": ""}]}], 0))


class TestFetchOverTheDaemonAndTheLink(TestCase):
    """``getReplayImage`` over a real daemon, directly and forwarded to a machine."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-lazy-img-daemon-"))
        self._saved_persistence = (th._DB_PATH, th._KISS_DIR)
        kiss_dir = self.tmpdir / ".kiss"
        kiss_dir.mkdir()
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "history.db"
        self._saved_cfg = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = self.tmpdir / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"
        certfile, keyfile = self.tmpdir / "cert.pem", self.tmpdir / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = find_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=self.port, certfile=str(certfile), keyfile=str(keyfile),
            url_file=self.tmpdir / "remote-url.json",
            local_endpoint_file=self.tmpdir / "sorcar-local.json",
            work_dir=str(self.tmpdir),
        )
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self.server.start_async(), self.loop).result(60)
        mach.save_machines([])
        self.machine = mach.add_machine("box", f"https://127.0.0.1:{self.port}", "")
        self.task_id, _ = th._add_task("screens")
        for ev in _transcript():
            th._queue_chat_event(dict(ev), task_id=self.task_id)
        th._flush_chat_events()
        th._save_task_result(result="done", task_id=self.task_id)

    def tearDown(self) -> None:
        try:
            mach.save_machines([])
            asyncio.run_coroutine_threadsafe(self.server.stop_async(), self.loop).result(60)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(timeout=30)
            self.loop.close()
            close_db()
            th._DB_PATH, th._KISS_DIR = self._saved_persistence
            vc.CONFIG_DIR, vc.CONFIG_PATH = self._saved_cfg
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_link_request_answers_the_picture(self) -> None:
        link = mach.link_for(self.machine)
        reply = link.request(
            {"type": "getReplayImage", "taskId": self.task_id, "index": 2, "token": "rimg-7"},
            "replayImage", timeout=30,
        )
        self.assertEqual(
            {k: reply.get(k) for k in ("taskId", "index", "token", "mime", "b64")},
            {"taskId": self.task_id, "index": 2, "token": "rimg-7",
             "mime": "image/png", "b64": _PNG_B64},
        )
        missing = link.request(
            {"type": "getReplayImage", "taskId": self.task_id, "index": 9, "token": "rimg-8"},
            "replayImage", timeout=30,
        )
        self.assertEqual(missing.get("token"), "rimg-8")
        self.assertNotIn("b64", missing)
        self.assertIn("error", missing)
        bad_index = link.request(
            {"type": "getReplayImage", "taskId": self.task_id, "index": "zero"},
            "replayImage", timeout=30,
        )
        self.assertEqual(bad_index.get("index"), -1)
        self.assertIn("error", bad_index)
        # The history database unreadable (a directory in its place):
        # the picture still gets its answer, or the page would show it
        # as loading for good.
        close_db()
        th._DB_PATH = self.tmpdir / "db-in-the-way"
        th._DB_PATH.mkdir()
        unreadable = link.request(
            {"type": "getReplayImage", "taskId": self.task_id, "index": 0, "token": "rimg-10"},
            "replayImage", timeout=10,
        )
        self.assertEqual(unreadable.get("token"), "rimg-10")
        self.assertNotIn("b64", unreadable)
        self.assertIn("Failed to load the picture", unreadable["error"])

    def test_forwarded_fetch_comes_back_stamped_with_the_machine(self) -> None:
        cmd = {
            "type": "getReplayImage", "taskId": self.task_id, "index": 1,
            "token": "rimg-9", "machine": "box", "connId": "c1",
        }
        reply = sidebar_panels.machine_replay_image_reply("box", cmd)
        self.assertEqual(reply.get("machine"), "box")
        self.assertEqual(reply.get("b64"), _SVG_B64)
        self.assertEqual(reply.get("token"), "rimg-9")
        # An unknown machine answers with the handle and the error.
        refused = sidebar_panels.machine_replay_image_reply("nowhere", cmd)
        self.assertEqual(
            {k: refused.get(k) for k in ("type", "taskId", "index", "token", "machine")},
            {"type": "replayImage", "taskId": self.task_id, "index": 1,
             "token": "rimg-9", "machine": "nowhere"},
        )
        self.assertIn("error", refused)
        self.assertNotIn("b64", refused)
