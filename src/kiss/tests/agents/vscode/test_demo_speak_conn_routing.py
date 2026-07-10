# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: ``demoSpeak`` synthesis replies route to the requesting connection.

The demo-mode chat webview asks the daemon to synthesize a replayed
utterance with the same GPT audio model the live ``talk`` tool uses
(``demoSpeak`` -> ``_cmd_demo_speak`` -> ``demoSpeakAudio``).  Every
client command arrives stamped with the requesting connection's
``connId`` (see ``RemoteAccessServer._dispatch_client_command``), and
``WebPrinter.broadcast`` delivers any event carrying a non-empty
``connId`` ONLY to that connection — the established request/reply
routing every other reply-style handler (``models``, ``files``,
``history``, ``cliInfo``, ...) already uses.

Bug: ``_cmd_demo_speak`` dropped the command's ``connId``, so each
``demoSpeakAudio`` reply (a large base64 MP3) was broadcast to EVERY
connected client (every VS Code window, every remote browser tab)
instead of only the demo-playing webview that asked — and a ``reqId``
collision across two webviews could resolve the wrong tab's waiter.

These tests drive the REAL ``VSCodeServer._handle_command`` (no mocks
of production code).  The synthesized clip comes from the production
per-daemon ``_DEMO_SPEAK_CACHE`` — pre-populating it is exactly the
state a second demo replay of the same history runs against, and it
also proves the handler keys synthesis by the verbatim
``(text, language, emotion)`` triple it received.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import pytest

from kiss.agents.vscode import commands
from kiss.agents.vscode.server import VSCodeServer

CLIP_B64 = "QUJDREVGRw=="
TEXT = "Hello from the demo replay."
LANGUAGE = "en-US"
EMOTION = "warm"


def _snapshot_cache() -> dict[tuple[str, str, str], tuple[str, str]]:
    """Copy the production demo-speech cache under its lock."""
    with commands._DEMO_SPEAK_CACHE_LOCK:
        return dict(commands._DEMO_SPEAK_CACHE)


def _restore_cache(
    snapshot: dict[tuple[str, str, str], tuple[str, str]],
) -> None:
    """Restore the production demo-speech cache to *snapshot*."""
    with commands._DEMO_SPEAK_CACHE_LOCK:
        commands._DEMO_SPEAK_CACHE.clear()
        commands._DEMO_SPEAK_CACHE.update(snapshot)


@pytest.fixture(autouse=True)
def _isolated_demo_speak_cache() -> Iterator[None]:
    """Snapshot/restore the module-level clip cache around every test
    so seeded entries never leak into other test files."""
    snapshot = _snapshot_cache()
    try:
        yield
    finally:
        _restore_cache(snapshot)


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]]]:
    """Build a real ``VSCodeServer`` with a capturing broadcast printer."""
    server = VSCodeServer()
    captured: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            captured.append(dict(event))

    server.printer.broadcast = capture  # type: ignore[method-assign]
    return server, captured


def _wait_for_reply(
    events: list[dict[str, Any]], req_id: str, timeout: float = 5.0,
) -> dict[str, Any]:
    """Return the ``demoSpeakAudio`` reply for *req_id*, polling until
    the handler's synthesis thread broadcasts it or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for ev in events:
            if ev.get("type") == "demoSpeakAudio" and ev.get("reqId") == req_id:
                return ev
        time.sleep(0.01)
    raise AssertionError(f"no demoSpeakAudio reply for {req_id!r}; got {events}")


def _prime_cache(text: str, language: str, emotion: str) -> None:
    """Seed the production demo-speech cache for one utterance triple."""
    with commands._DEMO_SPEAK_CACHE_LOCK:
        commands._DEMO_SPEAK_CACHE.clear()
        commands._DEMO_SPEAK_CACHE[(text, language, emotion)] = (
            CLIP_B64, "audio/mpeg",
        )


def test_demo_speak_reply_targets_requesting_connection() -> None:
    """The ``demoSpeakAudio`` reply must carry the request's ``connId``
    so ``WebPrinter.broadcast`` delivers it ONLY to the connection that
    asked — exactly like every other request/reply handler."""
    _prime_cache(TEXT, LANGUAGE, EMOTION)
    server, events = _make_server()
    server._handle_command({
        "type": "demoSpeak",
        "reqId": "req-conn-1",
        "text": TEXT,
        "language": LANGUAGE,
        "emotion": EMOTION,
        "tabId": "tab-1",
        "connId": "conn-A",
    })
    reply = _wait_for_reply(events, "req-conn-1")
    assert reply.get("connId") == "conn-A", (
        "demoSpeakAudio must be stamped with the requesting connection's "
        f"connId for targeted delivery; got {reply}"
    )
    assert reply["audioB64"] == CLIP_B64
    assert reply["audioMime"] == "audio/mpeg"
    assert reply["tabId"] == "tab-1"


def test_demo_speak_uses_verbatim_synthesis_key() -> None:
    """The handler must key the clip by the exact ``(text, language,
    emotion)`` triple it received — the same arguments the live talk
    tool passes to ``synthesize_talk_audio`` — so a cached clip of the
    identical utterance is returned verbatim."""
    _prime_cache(TEXT, LANGUAGE, EMOTION)
    server, events = _make_server()
    server._handle_command({
        "type": "demoSpeak",
        "reqId": "req-key-1",
        "text": TEXT,
        "language": LANGUAGE,
        "emotion": EMOTION,
        "tabId": "tab-2",
        "connId": "conn-B",
    })
    reply = _wait_for_reply(events, "req-key-1")
    assert reply["audioB64"] == CLIP_B64, (
        "a cached clip for the identical (text, language, emotion) triple "
        f"must be returned without re-synthesis; got {reply}"
    )


def test_demo_speak_empty_text_reply_still_targeted() -> None:
    """A blank utterance degrades to an empty-clip reply (the webview
    then applies its live-talk fallback rules) — but the reply must
    still be stamped with the requester's ``connId``."""
    server, events = _make_server()
    server._handle_command({
        "type": "demoSpeak",
        "reqId": "req-empty-1",
        "text": "   ",
        "language": "",
        "emotion": "",
        "tabId": "tab-3",
        "connId": "conn-C",
    })
    reply = _wait_for_reply(events, "req-empty-1")
    assert reply["audioB64"] == ""
    assert reply.get("connId") == "conn-C", (
        "even a failed/empty synthesis reply must route only to the "
        f"requesting connection; got {reply}"
    )


def test_demo_speak_cold_cache_calls_shared_synth_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a cold cache the handler must synthesize through THE shared
    production function the live ``talk`` tool uses —
    ``kiss.agents.vscode.speech_synthesis.synthesize_talk_audio`` —
    passing the verbatim ``(text, language, emotion)`` triple, and ship
    the returned clip in the reply.  The function is substituted only
    at the external-API boundary (it performs a paid GPT-audio network
    call); everything else in the pipeline is production code."""
    from kiss.agents.vscode import speech_synthesis

    calls: list[tuple[str, str, str]] = []

    def record_and_return(
        text: str, language: str = "", emotion: str = "",
    ) -> tuple[str, str]:
        calls.append((text, language, emotion))
        return CLIP_B64, "audio/mpeg"

    monkeypatch.setattr(
        speech_synthesis, "synthesize_talk_audio", record_and_return,
    )
    with commands._DEMO_SPEAK_CACHE_LOCK:
        commands._DEMO_SPEAK_CACHE.clear()
    server, events = _make_server()
    server._handle_command({
        "type": "demoSpeak",
        "reqId": "req-cold-1",
        "text": TEXT,
        "language": LANGUAGE,
        "emotion": EMOTION,
        "tabId": "tab-5",
        "connId": "conn-D",
    })
    reply = _wait_for_reply(events, "req-cold-1")
    assert calls == [(TEXT, LANGUAGE, EMOTION)], (
        "the daemon must call the shared synthesize_talk_audio exactly "
        f"once with the verbatim utterance triple; got {calls}"
    )
    assert reply["audioB64"] == CLIP_B64
    assert reply.get("connId") == "conn-D"
    # The synthesized clip must now be cached under the same triple.
    with commands._DEMO_SPEAK_CACHE_LOCK:
        cached = commands._DEMO_SPEAK_CACHE.get((TEXT, LANGUAGE, EMOTION))
    assert cached == (CLIP_B64, "audio/mpeg")


def test_demo_speak_without_conn_id_broadcasts_plain() -> None:
    """Back-compat: a request with no ``connId`` (e.g. a direct
    single-client ``VSCodeServer`` with a plain ``JsonPrinter``) must
    produce a reply WITHOUT a connId stamp, so ``WebPrinter.broadcast``
    falls back to the tab-targeted system-event path."""
    _prime_cache(TEXT, LANGUAGE, EMOTION)
    server, events = _make_server()
    server._handle_command({
        "type": "demoSpeak",
        "reqId": "req-nc-1",
        "text": TEXT,
        "language": LANGUAGE,
        "emotion": EMOTION,
        "tabId": "tab-4",
    })
    reply = _wait_for_reply(events, "req-nc-1")
    assert "connId" not in reply, (
        f"no connId on the request means none on the reply; got {reply}"
    )
    assert reply["audioB64"] == CLIP_B64


class TestDemoSpeakTwoWindowIsolation(IsolatedAsyncioTestCase):
    """Full-wire E2E: two UDS connections (= two VS Code windows) on
    one real ``RemoteAccessServer`` daemon.  Window A's ``demoSpeak``
    reply — a large base64 clip — must arrive on A's socket (with the
    ``connId`` routing stamp stripped from the wire) and NEVER on B's.
    """

    async def asyncSetUp(self) -> None:
        import kiss.agents.sorcar.persistence as th
        from kiss.agents.vscode.web_server import (
            RemoteAccessServer,
            _generate_self_signed_cert,
        )

        self._cache_snapshot = _snapshot_cache()
        self.tmpdir = tempfile.mkdtemp(prefix="kiss_demo_speak_uds_")
        self._th = th
        self._saved_persistence = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()
        self._writers: list[asyncio.StreamWriter] = []

    async def asyncTearDown(self) -> None:
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        await self.server.stop_async()
        if self._th._db_conn is not None:
            self._th._db_conn.close()
        (
            self._th._DB_PATH,
            self._th._db_conn,
            self._th._KISS_DIR,
        ) = self._saved_persistence
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        _restore_cache(self._cache_snapshot)

    async def _connect(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open one UDS connection (simulates one VS Code window)."""
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path), limit=16 * 1024 * 1024,
        )
        self._writers.append(writer)
        return reader, writer

    async def _send(
        self, writer: asyncio.StreamWriter, cmd: dict[str, Any],
    ) -> None:
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _drain_until(
        self,
        reader: asyncio.StreamReader,
        type_name: str,
        seen: list[str] | None = None,
        max_events: int = 100,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Read events until one of *type_name* arrives, recording every
        event type passed over in *seen*."""
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "UDS closed unexpectedly"
            msg = json.loads(line.decode("utf-8"))
            assert isinstance(msg, dict)
            if seen is not None:
                seen.append(str(msg.get("type", "")))
            if msg.get("type") == type_name:
                return msg
        raise AssertionError(f"no {type_name!r} within {max_events} events")

    async def test_demo_speak_clip_reaches_only_requesting_window(
        self,
    ) -> None:
        """demoSpeak over the real daemon wire: window A gets the clip
        (connId stripped), window B never sees a demoSpeakAudio."""
        _prime_cache(TEXT, LANGUAGE, EMOTION)
        reader_a, writer_a = await self._connect()
        reader_b, writer_b = await self._connect()

        await self._send(writer_a, {
            "type": "demoSpeak",
            "reqId": "wire-req-1",
            "text": TEXT,
            "language": LANGUAGE,
            "emotion": EMOTION,
            "tabId": "tab-A",
        })
        reply = await self._drain_until(reader_a, "demoSpeakAudio")
        self.assertEqual(reply.get("reqId"), "wire-req-1")
        self.assertEqual(reply.get("audioB64"), CLIP_B64)
        # The routing stamp is internal; it must never reach the wire.
        self.assertNotIn("connId", reply)

        # Probe window B: its probe response is answered directly, so
        # any leaked demoSpeakAudio would already be queued before it.
        await self._send(writer_b, {"type": "activeTasksQuery"})
        seen_b: list[str] = []
        await self._drain_until(reader_b, "activeTasksResponse", seen_b)
        self.assertNotIn(
            "demoSpeakAudio", seen_b,
            "window B must never receive window A's synthesis reply",
        )
