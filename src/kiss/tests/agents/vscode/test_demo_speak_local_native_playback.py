# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the daemon plays demo-replay speech clips for local webviews.

Regression tests for silent demo-mode speech: when a task is loaded in
demo mode, the FIRST replayed utterance played but every subsequent one
(later talks in the same demo, or any talk after loading another task
in demo mode) stayed silent.

Root cause: a local VS Code chat webview cannot reliably play a clip
in-page — Chromium's autoplay policy rejects ``Audio.play()`` in a
webview unless the user interacted with it seconds earlier
(microsoft/vscode#197937 / #178642, closed as not actionable).  The
first clip after the history-row click may still ride the fresh user
activation; every later ``play()`` is rejected and the webview's
rejection handler silently degrades (the demo advances without sound).
Live ``talk`` events already solve this: ``WebPrinter._fanout_talk``
plays the clip natively on the daemon's machine and stamps local UDS
webview copies ``muted``.  The demo-mode ``demoSpeakAudio`` reply path
(``WebPrinter.broadcast``'s targeted ``connId`` delivery) lacked that
arbitration — these tests pin it down.

These tests run a REAL ``RemoteAccessServer`` with a UDS listener and
REAL UDS client connections (no mocks of production code), mirroring
``test_talk_daemon_local_playback.py``; the daemon's audio player is a
REAL child process substituted via ``KISS_SORCAR_PLAY_CMD`` whose
marker files carry the exact bytes it played.  The synthesized clips
come from the production per-daemon ``_DEMO_SPEAK_CACHE`` (pre-seeding
it is exactly the state a repeat demo replay runs against and avoids a
paid GPT-audio network call).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import shutil
import socket
import ssl
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar import cli_talk
from kiss.agents.vscode import commands
from kiss.agents.vscode.vscode_config import load_config
from kiss.agents.vscode.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
)

# Tiny-but-real MP3-looking byte strings; the fake player only copies
# bytes, so any payload works — what matters is byte-exact delivery.
MP3_BYTES_1 = b"ID3\x03\x00demo-clip-one-" + bytes(range(64))
MP3_B64_1 = base64.b64encode(MP3_BYTES_1).decode("ascii")
MP3_BYTES_2 = b"ID3\x03\x00demo-clip-two-" + bytes(range(64, 128))
MP3_B64_2 = base64.b64encode(MP3_BYTES_2).decode("ascii")

TEXT_1 = "First replayed demo utterance."
TEXT_2 = "Second replayed demo utterance."
LANGUAGE = "en-US"
EMOTION = "warm"


def _pick_free_port() -> int:
    """Reserve and immediately release one localhost TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    """Return a client SSL context accepting the test self-signed cert."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    """Point the sorcar persistence layer at a private temp dir."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    """Undo :func:`_redirect_persistence`."""
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _write_player(tmp_path: Path, marker_dir: Path) -> Path:
    """Write a python script standing in for the daemon's audio player.

    The script receives the audio file path as its LAST argument and
    writes one unique JSON marker file into *marker_dir* recording the
    audio bytes (base64) it read.
    """
    marker_dir.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "fake_player.py"
    script.write_text(
        "import base64, json, os, sys, uuid\n"
        f"marker_dir = {str(marker_dir)!r}\n"
        "path = sys.argv[-1]\n"
        "with open(path, 'rb') as fh:\n"
        "    data = fh.read()\n"
        "marker = os.path.join(marker_dir, uuid.uuid4().hex + '.json')\n"
        "with open(marker, 'w') as fh:\n"
        "    json.dump({'audio_b64':"
        " base64.b64encode(data).decode('ascii')}, fh)\n"
    )
    return script


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


def _seed_cache() -> None:
    """Seed the production clip cache with both test utterances."""
    with commands._DEMO_SPEAK_CACHE_LOCK:
        commands._DEMO_SPEAK_CACHE.clear()
        commands._DEMO_SPEAK_CACHE[(TEXT_1, LANGUAGE, EMOTION)] = (
            MP3_B64_1, "audio/mpeg",
        )
        commands._DEMO_SPEAK_CACHE[(TEXT_2, LANGUAGE, EMOTION)] = (
            MP3_B64_2, "audio/mpeg",
        )


class TestDemoSpeakLocalNativePlayback(IsolatedAsyncioTestCase):
    """Demo-replay clips must play natively for local UDS webviews."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss_demo_native_")
        self.saved = _redirect_persistence(self.tmpdir)
        self._cache_snapshot = _snapshot_cache()
        _seed_cache()
        self.marker_dir = Path(self.tmpdir) / "markers"
        player = _write_player(Path(self.tmpdir), self.marker_dir)
        self.saved_play_cmd = os.environ.get("KISS_SORCAR_PLAY_CMD")
        os.environ["KISS_SORCAR_PLAY_CMD"] = (
            f"{shlex.quote(sys.executable)} {shlex.quote(str(player))}"
        )
        cli_talk.reset_shared_player_for_tests()
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.port = _pick_free_port()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
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
        if self.saved_play_cmd is None:
            os.environ.pop("KISS_SORCAR_PLAY_CMD", None)
        else:
            os.environ["KISS_SORCAR_PLAY_CMD"] = self.saved_play_cmd
        cli_talk.reset_shared_player_for_tests()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        _restore_cache(self._cache_snapshot)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(
        self, tab_id: str,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open one UDS webview client and announce ``ready``."""
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path), limit=16 * 1024 * 1024,
        )
        self._writers.append(writer)
        for cmd in (
            {"type": "setWorkDir", "workDir": self.tmpdir},
            {"type": "ready", "tabId": tab_id, "workDir": self.tmpdir},
        ):
            writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
        await writer.drain()
        return reader, writer

    async def _connect_wss(self) -> Any:
        """Open and authenticate one remote WSS browser connection."""
        ws = await connect(
            f"wss://127.0.0.1:{self.port}/ws",
            ssl=_no_verify_ssl(),
            max_size=16 * 1024 * 1024,
        )
        password = str(load_config().get("remote_password", "") or "")
        await ws.send(json.dumps({"type": "auth", "password": password}))
        for _ in range(3):
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            msg = json.loads(raw)
            assert isinstance(msg, dict)
            if msg.get("type") == "auth_ok":
                return ws
            if msg.get("type") != "auth_required":
                raise AssertionError(f"unexpected auth response: {msg}")
            await ws.send(json.dumps({"type": "auth", "password": password}))
        raise AssertionError("WSS auth did not complete")

    async def _send(
        self, writer: asyncio.StreamWriter, cmd: dict[str, Any],
    ) -> None:
        """Send one JSON command line over the UDS socket."""
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _drain_until(
        self,
        reader: asyncio.StreamReader,
        type_name: str,
        req_id: str | None = None,
        max_events: int = 200,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Read UDS events until one of *type_name* (and *req_id*) arrives."""
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "UDS closed unexpectedly"
            msg = json.loads(line.decode("utf-8"))
            assert isinstance(msg, dict)
            if msg.get("type") == type_name and (
                req_id is None or msg.get("reqId") == req_id
            ):
                return msg
        raise AssertionError(f"no {type_name!r} within {max_events} events")

    async def _ws_drain_until(
        self,
        ws: Any,
        type_name: str,
        req_id: str | None = None,
        max_events: int = 200,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Read WSS events until one of *type_name* (and *req_id*) arrives."""
        for _ in range(max_events):
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            msg = json.loads(raw)
            assert isinstance(msg, dict)
            if msg.get("type") == type_name and (
                req_id is None or msg.get("reqId") == req_id
            ):
                return msg
        raise AssertionError(f"no {type_name!r} within {max_events} events")

    def _wait_markers(self, count: int, timeout: float = 5.0) -> list[dict]:
        """Wait for *count* fake-player marker files; return their JSON."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            markers = sorted(self.marker_dir.glob("*.json"))
            if len(markers) >= count:
                return [json.loads(m.read_text()) for m in markers]
            time.sleep(0.05)
        markers = sorted(self.marker_dir.glob("*.json"))
        return [json.loads(m.read_text()) for m in markers]

    def _demo_speak(self, req_id: str, text: str) -> dict[str, Any]:
        """Build one ``demoSpeak`` request command."""
        return {
            "type": "demoSpeak",
            "reqId": req_id,
            "text": text,
            "language": LANGUAGE,
            "emotion": EMOTION,
            "tabId": "demo-tab",
        }

    async def test_every_demo_clip_plays_natively_and_mutes_webview(
        self,
    ) -> None:
        """THE SILENT-DEMO BUG: every replayed utterance must be heard.

        A local UDS webview replaying a demo asks the daemon for one
        clip per utterance.  The daemon must play EACH clip natively on
        its own speakers (byte-exact) and stamp EACH reply ``muted`` —
        an unmuted reply makes the webview attempt ``Audio.play()``,
        get autoplay-rejected (no fresh user gesture), and silently
        skip the utterance, which is exactly the reported bug: only
        the first demo speech (riding the history-click gesture) was
        ever heard.
        """
        tab = "webtab-" + uuid.uuid4().hex[:8]
        reader, writer = await self._connect(tab)

        await self._send(writer, self._demo_speak("demo-req-1", TEXT_1))
        reply1 = await self._drain_until(
            reader, "demoSpeakAudio", "demo-req-1",
        )
        self.assertEqual(reply1.get("audioB64"), MP3_B64_1)
        self.assertEqual(
            reply1.get("text"), TEXT_1,
            "demoSpeakAudio should carry text so daemon-side native "
            "playback can fall back to TTS if the clip player fails",
        )
        self.assertTrue(
            reply1.get("muted"),
            "the local UDS webview's demoSpeakAudio reply must be muted "
            "while the daemon plays the clip natively — an unmuted copy "
            "is autoplay-rejected in the webview and the utterance goes "
            f"silent; got {reply1.keys()}",
        )
        markers = await asyncio.to_thread(self._wait_markers, 1)
        self.assertEqual(
            len(markers), 1,
            "the daemon never played the first demo clip natively",
        )
        self.assertEqual(
            base64.b64decode(markers[0]["audio_b64"]), MP3_BYTES_1,
            "first clip bytes were corrupted",
        )

        # The SUBSEQUENT utterance — the one the reported bug silenced —
        # must also be played natively and arrive muted.
        await self._send(writer, self._demo_speak("demo-req-2", TEXT_2))
        reply2 = await self._drain_until(
            reader, "demoSpeakAudio", "demo-req-2",
        )
        self.assertEqual(reply2.get("audioB64"), MP3_B64_2)
        self.assertTrue(
            reply2.get("muted"),
            "the second demo utterance's reply must be muted too — the "
            "daemon owns playback for every clip, not just the first",
        )
        played = {
            m["audio_b64"]
            for m in await asyncio.to_thread(self._wait_markers, 2)
        }
        self.assertEqual(
            played, {MP3_B64_1, MP3_B64_2},
            "the daemon must natively play BOTH demo clips byte-exact "
            "(the reported bug: every clip after the first stayed silent)",
        )

    async def test_clips_for_second_demo_task_also_play_natively(
        self,
    ) -> None:
        """Loading another task in demo mode must speak natively too.

        The second half of the reported bug: after one demo finishes,
        loading ANOTHER task in demo mode produced no sound at all.
        Each demo replay requests clips over the same UDS connection
        with fresh reqIds — the daemon must play every one of them.
        """
        tab = "webtab-" + uuid.uuid4().hex[:8]
        reader, writer = await self._connect(tab)

        # Demo replay of task A.
        await self._send(writer, self._demo_speak("taskA-req-1", TEXT_1))
        await self._drain_until(reader, "demoSpeakAudio", "taskA-req-1")
        # Demo replay of task B (new replay, new reqId, same utterance
        # text as a cached clip — the repeat-replay state).
        await self._send(writer, self._demo_speak("taskB-req-1", TEXT_1))
        reply = await self._drain_until(
            reader, "demoSpeakAudio", "taskB-req-1",
        )
        self.assertTrue(
            reply.get("muted"),
            "the second demo task's clip reply must be muted while the "
            "daemon plays it natively",
        )
        markers = await asyncio.to_thread(self._wait_markers, 2)
        self.assertEqual(
            len(markers), 2,
            "the daemon must play the clip once per demo replay (talkId "
            "dedupe is keyed by reqId, which is fresh per request)",
        )

    async def test_remote_wss_demo_clip_stays_playable_and_does_not_play_locally(
        self,
    ) -> None:
        """A remote WSS browser is another device: keep its copy audible.

        The daemon-side speakers are only a substitute for LOCAL UDS
        VS Code webviews that share this machine and are subject to
        VS Code webview autoplay rejection.  A WSS browser may be on a
        phone/laptop elsewhere; playing on the daemon machine would be
        inaudible there and muting the reply would make the remote
        demo silent.  Therefore WSS replies must stay unmuted and must
        not spawn the local player.
        """
        async with await self._connect_wss() as ws:
            await ws.send(json.dumps(self._demo_speak("wss-req-1", TEXT_1)))
            reply = await self._ws_drain_until(
                ws, "demoSpeakAudio", "wss-req-1",
            )
        self.assertEqual(reply.get("audioB64"), MP3_B64_1)
        self.assertEqual(reply.get("text"), TEXT_1)
        self.assertFalse(
            reply.get("muted"),
            "a remote WSS demoSpeakAudio reply must remain playable on "
            "that remote device; only local UDS webviews are muted",
        )
        await asyncio.sleep(0.3)
        self.assertEqual(
            len(list(self.marker_dir.glob("*.json"))), 0,
            "the daemon must not play a WSS remote browser's demo clip "
            "on local speakers",
        )

    async def test_empty_synthesis_reply_is_not_played_or_muted(
        self,
    ) -> None:
        """A failed/blank synthesis reply carries no clip: nothing to
        play natively, and the reply must NOT be muted (the webview's
        waiter degrades to silence on empty ``audioB64`` on its own).
        """
        tab = "webtab-" + uuid.uuid4().hex[:8]
        reader, writer = await self._connect(tab)

        await self._send(writer, self._demo_speak("empty-req-1", "   "))
        reply = await self._drain_until(
            reader, "demoSpeakAudio", "empty-req-1",
        )
        self.assertEqual(reply.get("audioB64"), "")
        self.assertFalse(reply.get("muted"))
        await asyncio.sleep(0.3)
        self.assertEqual(
            len(list(self.marker_dir.glob("*.json"))), 0,
            "the daemon must not spawn a player for an empty clip",
        )
