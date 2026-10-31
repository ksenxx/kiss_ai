# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audio an Anthropic conversation carries raw is transcribed once, not on every request.

Anthropic takes no audio, so audio goes in as a Whisper transcript.  The
conversation keeps the audio raw, as an ``attachments`` entry (the files
a user attaches to a queued follow-up, or a conversation handed over
from Gemini) or as an OpenAI ``input_audio`` part (handed over from an
OpenAI model), and the adapter converts it again for every request.  So
every agent step and every prompt-cache keep-alive ping paid for one
more Whisper call (not stop-aware, up to 120 s per attempt), and a
transcript that came back different changed the prompt bytes and so
missed the prompt cache.

Real local endpoints stand in for both APIs: the Anthropic Messages
endpoint (``ANTHROPIC_BASE_URL``) answers two keep-alive pings, and the
Whisper endpoint (``OPENAI_BASE_URL``) numbers its transcripts, so a
second transcription would show in the second ping's prompt.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.model import Attachment
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.tests.core import local_anthropic_server as anthropic_server

_FIRST_TRANSCRIPT = "[Audio transcription]\ntranscript 1"


class _NumberingWhisper(BaseHTTPRequestHandler):
    """A Whisper endpoint answering the n-th transcription with ``transcript n``."""

    paths: list[str] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        self.paths.append(self.path)
        body = f"transcript {len(self.paths)}".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture
def whisper(monkeypatch: pytest.MonkeyPatch) -> Generator[list[str]]:
    """Serve a local Whisper endpoint for one test; yield the paths it was asked for."""
    paths: list[str] = []
    handler = type("_Whisper", (_NumberingWhisper,), {"paths": paths})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-local")
    try:
        yield paths
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def _clip() -> bytes:
    """Audio bytes no other test uses (transcripts are remembered per clip)."""
    return b"RIFF\x24\x00\x00\x00WAVE" + uuid.uuid4().bytes


def _transcripts(request: dict[str, Any]) -> list[str]:
    """The audio transcription blocks of one recorded Messages request."""
    return [
        block["text"]
        for message in request["messages"]
        if isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "text" and block["text"].startswith("[Audio transcription]")
    ]


def _ping_twice(
    monkeypatch: pytest.MonkeyPatch, conversation: list[dict[str, Any]] | None, clip: bytes,
) -> list[dict[str, Any]]:
    """Send two keep-alive pings over a conversation carrying *clip*; return the requests.

    With *conversation* ``None`` the clip comes as the files a user attached
    to a queued follow-up message; otherwise *conversation* is handed over
    the way ``KISSAgent`` does on a model switch.
    """
    with anthropic_server.serve([]) as (url, requests):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        model = AnthropicModel(anthropic_server.MODEL, api_key="local")
        model.initialize("Summarize the recording.")
        if conversation is None:
            model.add_message_to_conversation(
                "user",
                "[files the user attached to the message(s) above]",
                attachments=[Attachment(clip, "audio/wav")],
            )
        else:
            model.conversation = conversation
        model.keep_prompt_cache_warm({})
        model.keep_prompt_cache_warm({})
    return list(requests)


def test_audio_a_user_attached_mid_task_is_transcribed_once(
    monkeypatch: pytest.MonkeyPatch, whisper: list[str],
) -> None:
    """An attachment message: one Whisper call, the same transcript in both prompts."""
    requests = _ping_twice(monkeypatch, None, _clip())

    assert whisper == ["/v1/audio/transcriptions"], f"{len(whisper)} Whisper calls for one clip"
    assert [_transcripts(request) for request in requests] == [[_FIRST_TRANSCRIPT]] * 2


def test_audio_handed_over_from_an_openai_model_is_transcribed_once(
    monkeypatch: pytest.MonkeyPatch, whisper: list[str],
) -> None:
    """An OpenAI ``input_audio`` part handed over: one Whisper call, a stable prompt."""
    clip = _clip()
    source = OpenAICompatibleModel("gpt-audio", base_url="http://127.0.0.1:9", api_key="unused")
    source.initialize("What does the recording say?", attachments=[Attachment(clip, "audio/wav")])
    assert any(
        part.get("type") == "input_audio" for part in source.conversation[-1]["content"]
    )

    requests = _ping_twice(monkeypatch, list(source.conversation), clip)

    assert whisper == ["/v1/audio/transcriptions"], f"{len(whisper)} Whisper calls for one clip"
    assert [_transcripts(request) for request in requests] == [[_FIRST_TRANSCRIPT]] * 2
