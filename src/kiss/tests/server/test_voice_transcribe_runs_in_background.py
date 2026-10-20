# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a dictation in flight does not hold up the same page's next command.

A connection's commands are dispatched one after another, and
``voiceTranscribe`` is a network round-trip to the transcription model
(plus, on first use, the speaker model's download).  Serviced inline,
one dictation held the page's next ``stop`` / ``submit`` /
``activeTasksQuery`` until the transcript came back (V2-6); the reply
now goes out from a background task, like the app probes.

Real daemon over its local WSS endpoint, one real client; the
transcription API is a local HTTP stand-in that answers after a delay,
so the ordering is observable.
"""

from __future__ import annotations

import asyncio
import base64
import http.server
import json
import os
import threading
import time
from typing import Any, cast

from kiss.tests.server.test_voice_dual_check import _tone_pcm
from kiss.tests.server.two_client_harness import TwoClientHarness

HOLD = 4.0
"""Seconds the transcription API sits on the request before answering."""


class _HoldingOpenAiServer(http.server.ThreadingHTTPServer):
    """A chat-completions endpoint that answers an empty transcript after ``HOLD`` seconds."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _HoldingHandler)
        self.hits = 0


class _HoldingHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 — http.server API
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        cast(_HoldingOpenAiServer, self.server).hits += 1
        time.sleep(HOLD)
        body = json.dumps({
            "id": "chatcmpl-held", "object": "chat.completion", "created": 0,
            "model": "gpt-audio",
            "choices": [{
                "index": 0, "finish_reason": "stop",
                # An empty transcript: nothing to submit, and no
                # speaker identification (the speaker model download).
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"language": "en", "text": ""}),
                },
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — http.server API
        pass


class VoiceTranscribeInBackgroundTest(TwoClientHarness):
    """``activeTasksQuery`` sent right after a dictation is answered first."""

    def setUp(self) -> None:
        super().setUp()
        self.api = _HoldingOpenAiServer()
        threading.Thread(target=self.api.serve_forever, daemon=True).start()
        self.addCleanup(self.api.server_close)
        self.addCleanup(self.api.shutdown)
        saved = {k: os.environ.get(k) for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY")}
        os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{self.api.server_address[1]}/v1"
        os.environ["OPENAI_API_KEY"] = "sk-kiss-background-voice-test"
        self.addCleanup(self._restore_env, saved)

    @staticmethod
    def _restore_env(saved: dict[str, str | None]) -> None:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_active_tasks_query_is_answered_before_the_transcript(self) -> None:
        client = self.open_client("a")
        client.ready()
        self.wait_event(client, "models")

        started = time.monotonic()
        client.send({
            "type": "voiceTranscribe", "tabId": client.tab_id,
            "audio": base64.b64encode(_tone_pcm(1.0)).decode("ascii"),
        })
        client.send({"type": "activeTasksQuery"})
        self.wait_event(client, "activeTasksResponse")
        self.assertLess(
            time.monotonic() - started, HOLD / 2,
            "activeTasksQuery waited for the transcription",
        )
        self.assertEqual(client.of_type("voiceSpeech"), [])

        speech = self.wait_event(client, "voiceSpeech", timeout=60.0)
        self.assertEqual(speech["text"], "")
        self.assertGreaterEqual(self.api.hits, 1, "the transcription API was never asked")
        replies = [e for e in client.events if e["type"] in ("activeTasksResponse", "voiceSpeech")]
        self.assertEqual(replies[0]["type"], "activeTasksResponse")

    def test_a_later_dictation_is_answered_after_an_earlier_one(self) -> None:
        """The page's ``voiceSpeech`` replies keep the order of its ``voiceTranscribe`` sends.

        ``voice.js`` pairs each reply with the OLDEST of its outstanding
        rounds, so a second dictation answered first (here one with no
        audio, which needs no transcription) would take the first
        round's conversation and hand its own to the first's transcript.
        """
        client = self.open_client("a")
        client.ready()
        self.wait_event(client, "models")

        started = time.monotonic()
        client.send({
            "type": "voiceTranscribe", "tabId": client.tab_id,
            "audio": base64.b64encode(_tone_pcm(1.0)).decode("ascii"),
        })
        client.send({"type": "voiceTranscribe", "tabId": client.tab_id, "audio": ""})
        client.send({"type": "activeTasksQuery"})
        self.wait_event(client, "activeTasksResponse")
        self.wait_for(lambda: self.api.hits >= 1, what="the first dictation reached the API")
        time.sleep(HOLD / 4)
        self.assertEqual(
            client.of_type("voiceSpeech"), [],
            "the second dictation was answered before the first",
        )
        self.wait_for(
            lambda: len(client.of_type("voiceSpeech")) == 2, timeout=60.0,
            what="both dictations answered",
        )
        self.assertGreaterEqual(time.monotonic() - started, HOLD)

    def test_a_dictation_arriving_while_the_daemon_stops_is_dropped_unrun(self) -> None:
        """Once the background replies are cancelled at shutdown, a dictation spawns none.

        ``stop_async`` cancels the background replies before it closes
        the listeners, so a frame can still arrive in between; it must
        not start a transcription the shutdown will not wait for.  The
        same step is taken here on the live daemon, then the dictation
        is sent: the transcription API is never asked and the page's
        later ``ping`` is still answered.
        """
        client = self.open_client("a")
        client.ready()
        self.wait_event(client, "models")
        api = self.server._server_api
        loop = self.server._loop
        assert loop is not None
        asyncio.run_coroutine_threadsafe(api.cancel_background_replies(), loop).result(10)
        client.send({
            "type": "voiceTranscribe", "tabId": client.tab_id,
            "audio": base64.b64encode(_tone_pcm(1.0)).decode("ascii"),
        })
        client.send({"type": "ping"})
        self.wait_event(client, "pong")
        time.sleep(0.5)
        self.assertEqual(self.api.hits, 0, "a transcription started during the shutdown")
        self.assertEqual(client.of_type("voiceSpeech"), [])
        self.assertEqual(api._pending_voice, {})
