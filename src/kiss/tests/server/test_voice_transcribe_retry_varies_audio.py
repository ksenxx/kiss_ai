# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""E2E: a refused transcription is retried with DIFFERENT audio bytes.

Reproduces the 2026-10-01 failure of
``test_web_voice_transcribe.py::test_actual_voice_is_transcribed``:
gpt-audio answered real speech with "Please provide the audio, and I
will transcribe and translate it accordingly." on BOTH attempts.  The
API routes identical requests to the same replica, which at
temperature 0 repeats the identical refusal, so a byte-identical
retry could never recover.  ``transcribe_pcm`` now appends
:data:`RETRY_EXTRA_TAIL_SECONDS` of silence on the retry.

The tests drive the real ``transcribe_pcm`` against a local stand-in
of the chat-completions endpoint that records every request body and
scripts its replies (refusal first, transcript second); the attached
WAVs prove what audio each attempt carried.
"""

from __future__ import annotations

import base64
import http.server
import io
import json
import math
import os
import struct
import threading
import unittest
import wave
from collections.abc import Iterator
from typing import Any

from kiss.server.voice_wake import (
    RETRY_EXTRA_TAIL_SECONDS,
    SAMPLE_RATE,
    TRAILING_SILENCE_KEEP_SECONDS,
    transcribe_pcm,
)

REFUSAL = (
    "Please provide the audio, and I will transcribe and translate it "
    "accordingly."
)
TRANSCRIPT = "en\nOpen the readme file."


class ScriptedOpenAiServer:
    """Local chat-completions stand-in answering each POST from a script."""

    def __init__(self, replies: list[str]) -> None:
        self.request_bodies: list[bytes] = []
        self._replies: Iterator[str] = iter(replies)
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — http.server API
                length = int(self.headers.get("Content-Length", "0"))
                server.request_bodies.append(self.rfile.read(length))
                body = json.dumps({
                    "id": "chatcmpl-scripted",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "gpt-audio",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": next(server._replies),
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — http.server API
                pass

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Shut the server down and join its thread."""
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def _attached_pcm(request_body: bytes) -> bytes:
    """Return the s16le PCM of the single ``input_audio`` WAV in a request."""
    datas: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            audio = node.get("input_audio")
            if isinstance(audio, dict) and isinstance(audio.get("data"), str):
                datas.append(audio["data"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(request_body))
    assert len(datas) == 1, f"expected one audio part, got {len(datas)}"
    with wave.open(io.BytesIO(base64.b64decode(datas[0])), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE
        return wf.readframes(wf.getnframes())


def _sine_pcm(seconds: float) -> bytes:
    """Loud 440 Hz sine, 16 kHz mono s16le — every block counts as speech."""
    return b"".join(
        struct.pack("<h", int(20000 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE)))
        for i in range(int(SAMPLE_RATE * seconds))
    )


def _silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * seconds)


class TestRetryVariesAudio(unittest.TestCase):
    """``transcribe_pcm`` against a scripted local API."""

    def _transcribe(
        self, replies: list[str], pcm: bytes,
    ) -> tuple[dict[str, Any], ScriptedOpenAiServer]:
        server = ScriptedOpenAiServer(replies)
        overrides = {
            "OPENAI_BASE_URL": f"http://127.0.0.1:{server.port}/v1",
            "OPENAI_API_KEY": "sk-kiss-retry-test",
            "KISS_VOICE_AUDIO_TIMEOUT": "20",
        }
        saved = {key: os.environ.get(key) for key in overrides}
        os.environ.update(overrides)
        try:
            return transcribe_pcm(pcm), server
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            server.close()

    def test_refused_first_attempt_is_retried_with_longer_audio(self) -> None:
        """The retry carries the utterance plus a second of silence, and wins."""
        speech = _sine_pcm(1.0)
        result, server = self._transcribe([REFUSAL, TRANSCRIPT], speech + _silence(2.0))
        self.assertEqual(result, {"text": "Open the readme file.", "language": "en"})
        self.assertEqual(len(server.request_bodies), 2)
        first = _attached_pcm(server.request_bodies[0])
        second = _attached_pcm(server.request_bodies[1])
        self.assertEqual(first, speech + _silence(TRAILING_SILENCE_KEEP_SECONDS))
        self.assertEqual(second, first + _silence(RETRY_EXTRA_TAIL_SECONDS))

    def test_two_refusals_report_no_speech(self) -> None:
        """Both attempts refused: no hallucinated command is forwarded."""
        result, server = self._transcribe([REFUSAL, REFUSAL], _sine_pcm(1.0))
        self.assertEqual(result, {"text": "", "language": None})
        self.assertEqual(len(server.request_bodies), 2)

    def test_accepted_first_attempt_makes_one_request(self) -> None:
        """A transcript on the first attempt is returned without a retry."""
        result, server = self._transcribe([TRANSCRIPT, REFUSAL], _sine_pcm(1.0))
        self.assertEqual(result["text"], "Open the readme file.")
        self.assertEqual(len(server.request_bodies), 1)


if __name__ == "__main__":
    unittest.main()
