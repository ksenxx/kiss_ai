# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for remote-webapp voice transcription.

Bug being reproduced — the remote webapp recognized the "Sorcar" wake
word but never transcribed the speech that followed.  In webview mode
the VS Code extension host runs a local listener that captures and
translates post-wake speech; in browser mode nothing did: the page's
``voice.js`` only flashed the mic button on a wake, captured no audio,
and the server had no handler to transcribe it.  The fix makes the
page capture the utterance and ship it as ``{"type":
"voiceTranscribe", "audio": <base64 16kHz mono s16le PCM>}`` over the
WebSocket, and adds a server-side handler that translates the audio
with the same KISS transcription agent the local listener uses
(:func:`kiss.agents.vscode.voice_wake.transcribe_pcm`) and replies
with ``{"type": "voiceSpeech", "text": ..., "speaker": ...,
"language": ...}`` — ``language`` being the agent-reported tag of the
spoken language (``null`` when unknown).

These tests spin up the real :class:`RemoteAccessServer` on a free
port and talk to it over a real ``wss://`` connection — no mocks.
The transcription test uses ACTUAL VOICE: a sentence synthesized with
the macOS TTS engine (``say``) is sent as PCM and the real gpt-audio
API must return its text.  Before the fix, a ``voiceTranscribe``
message got no ``voiceSpeech`` reply at all (it fell through to the
generic webview-command translator), so every test here timed out.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import socket
import ssl
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import ClientConnection, connect

import kiss.agents.sorcar.persistence as th
import kiss.agents.vscode.vscode_config as vc
from kiss.agents.vscode.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
)

HAVE_MAC_TTS = bool(shutil.which("say")) and bool(shutil.which("afconvert"))
HAVE_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    """Point the sorcar persistence layer at a per-test directory."""
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


def _find_free_port() -> int:
    """Return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _no_verify_ssl() -> ssl.SSLContext:
    """Return an SSL client context that skips certificate verification."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _tts_pcm_base64(directory: Path, text: str) -> str:
    """Speak *text* with macOS TTS and return base64 16kHz s16le PCM.

    This is the exact payload the browser-mode voice.js posts after
    capturing the utterance that follows the wake word: the capture
    endpoints on ~2s of trailing silence and INCLUDES those silent
    blocks, so the same trailing silence is appended here (the server
    now trims trailing silence before the transcription-agent call —
    long silent padding empirically makes gpt-audio deny hearing any
    audio — so the padded payload also exercises that trimming).
    """
    aiff = directory / "speech.aiff"
    wav = directory / "speech.wav"
    subprocess.run(["say", text, "-o", str(aiff)], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
         str(aiff), str(wav)],
        check=True,
    )
    with wave.open(str(wav), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
    pcm += b"\x00" * (2 * 2 * 16000)  # 2s of trailing silence
    return base64.b64encode(pcm).decode("ascii")


class WebVoiceTranscribeTest(IsolatedAsyncioTestCase):
    """A remote-web client's post-wake speech must be transcribed."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-web-voice-")
        self.saved = _redirect_persistence(self.tmpdir)
        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)

        self.port = _find_free_port()
        self.url = f"wss://127.0.0.1:{self.port}/ws"
        self.ctx = _no_verify_ssl()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=Path(self.tmpdir) / "sorcar.sock",
        )
        await self.server.start_async()
        self._sockets: list[ClientConnection] = []

    async def asyncTearDown(self) -> None:
        for ws in self._sockets:
            try:
                await ws.close()
            except Exception:
                pass
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        vc.CONFIG_DIR = self._orig_cfg_dir
        vc.CONFIG_PATH = self._orig_cfg_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect_ok(self) -> ClientConnection:
        """Open + successfully authenticate one WSS connection."""
        ws = await connect(self.url, ssl=self.ctx, max_size=64 * 1024 * 1024)
        self._sockets.append(ws)
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        self.assertEqual(resp["type"], "auth_ok")
        return ws

    async def _voice_speech_reply(
        self, ws: ClientConnection, audio_b64: str, timeout: float,
    ) -> dict[str, Any]:
        """Send a ``voiceTranscribe`` and await its ``voiceSpeech`` reply.

        Unrelated broadcast messages arriving on the same socket are
        skipped.  Fails the test when no reply arrives in *timeout*
        seconds — the reproduced bug: the server never answered.
        """
        await ws.send(
            json.dumps({"type": "voiceTranscribe", "audio": audio_b64}),
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            self.assertGreater(
                remaining, 0,
                "no voiceSpeech reply to voiceTranscribe (the reproduced "
                "bug: post-wake speech is never transcribed)",
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            msg = json.loads(raw)
            if isinstance(msg, dict) and msg.get("type") == "voiceSpeech":
                return cast(dict[str, Any], msg)

    async def test_empty_audio_replies_empty_voice_speech(self) -> None:
        """Silence (empty capture) must reply with an empty text.

        The page uses the reply to clear its transcribing indicator,
        so even a no-op transcription must be answered.  No gpt-audio
        call is made for empty PCM, so this runs offline.
        """
        ws = await self._connect_ok()
        msg = await self._voice_speech_reply(ws, "", timeout=15)
        self.assertEqual(msg["text"], "")
        self.assertIsNone(msg["speaker"])
        self.assertIsNone(msg["language"])

    async def test_undecodable_audio_replies_empty_voice_speech(self) -> None:
        """Malformed base64 must degrade to an empty transcription."""
        ws = await self._connect_ok()
        msg = await self._voice_speech_reply(
            ws, "!!!not-base64$$$", timeout=15,
        )
        self.assertEqual(msg["text"], "")
        self.assertIsNone(msg["speaker"])
        self.assertIsNone(msg["language"])

    async def test_oversized_audio_replies_empty_voice_speech(self) -> None:
        """An absurdly large payload is dropped, not transcribed."""
        ws = await self._connect_ok()
        oversized = base64.b64encode(b"\x00" * (4 * 1024 * 1024)).decode()
        msg = await self._voice_speech_reply(ws, oversized, timeout=15)
        self.assertEqual(msg["text"], "")
        self.assertIsNone(msg["speaker"])
        self.assertIsNone(msg["language"])

    @unittest.skipUnless(
        HAVE_MAC_TTS, "requires macOS `say` and `afconvert`",
    )
    @unittest.skipUnless(
        HAVE_OPENAI_KEY, "requires OPENAI_API_KEY (real gpt-audio call)",
    )
    async def test_actual_voice_is_transcribed(self) -> None:
        """ACTUAL VOICE: real spoken audio must come back as its text.

        Synthesizes a sentence with the macOS TTS engine — the same
        real-audio technique the wake-listener tests use — sends it
        exactly as browser-mode voice.js does after a wake, and
        requires the real gpt-audio translation to return the words.
        """
        ws = await self._connect_ok()
        audio_b64 = _tts_pcm_base64(
            Path(self.tmpdir), "open the readme file",
        )
        msg = await self._voice_speech_reply(ws, audio_b64, timeout=180)
        text = msg["text"].lower()
        self.assertIn("readme", text)
        self.assertIn("open", text)
        self.assertEqual(msg["language"], "en")
        speaker = msg["speaker"]
        self.assertTrue(
            speaker is None or (isinstance(speaker, int) and speaker >= 1),
            f"speaker must be None or a positive int, got {speaker!r}",
        )


if __name__ == "__main__":
    unittest.main()
