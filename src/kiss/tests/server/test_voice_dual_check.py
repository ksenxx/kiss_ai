"""Tests for the DUAL "Hey Sorcar" wake check of ``transcribe_pcm``.

The wake word is verified twice: locally in the browser (Vosk grammar
decoding in ``media/voice.js``) and again from the transcript.  The
browser prepends the wake word's own audio to the captured utterance
(``wakePrefixed``), so a genuine wake's transcript begins with
something that sounds like "Hey Sorcar";
:func:`kiss.server.speech.split_wake_prefix` confirms and cuts that
prefix, and an unconfirmed transcript is rejected as a false wake
(empty ``text``).

The end-to-end tests call the REAL :func:`transcribe_pcm` — real
:class:`KISSAgent`, real OpenAI client, real HTTP — against a local
canned server standing in for the chat-completions endpoint with a
fixed reply, so the tests can steer the transcript exactly:

- a transcript that starts with the wake word yields the text with
  the wake word cut;
- a transcript WITHOUT the wake word is rejected: empty ``text`` (before
  the dual check, whatever the STT hallucinated was submitted as the
  user's command);
- a non-English transcript without the wake word is accepted only when
  both browser recognizers heard it (``wakeAgreed``), because gpt-audio
  may drop the English wake word while translating;
- the audio attached to the API request is the WAV of the PCM handed
  in, proving the capture itself is what gets transcribed.

The prefix-gate functions themselves are exercised directly with the
transcript shapes gpt-audio produces for "Hey Sorcar" ("Hey Sorcar,",
"Hey Sarkar.", "Hey soccer") plus the spellings it could fall back to.
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
from typing import Any

from kiss.server.speech import (
    SAMPLE_RATE,
    levenshtein,
    sounds_like_wake_word,
    split_wake_prefix,
    transcribe_pcm,
)


class TestLevenshtein(unittest.TestCase):
    """The edit-distance helper of the fuzzy wake-prefix match."""

    def test_identical(self) -> None:
        self.assertEqual(levenshtein("sorcar", "sorcar"), 0)

    def test_empty_vs_word(self) -> None:
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("abc", ""), 3)

    def test_substitutions_insertions_deletions(self) -> None:
        self.assertEqual(levenshtein("sorcar", "sorkar"), 1)
        self.assertEqual(levenshtein("sorcar", "sorca"), 1)
        self.assertEqual(levenshtein("sorcar", "sorrcar"), 1)
        self.assertEqual(levenshtein("kitten", "sitting"), 3)


class TestSoundsLikeWakeWord(unittest.TestCase):
    """Transcribed words that must (not) count as the wake word."""

    def test_known_transcription_aliases(self) -> None:
        for word in ["sorcar", "Sorcar", "SORCAR,", "Sarkar", "sorkar!",
                     "sarcar", "Soccer"]:
            self.assertTrue(sounds_like_wake_word(word), word)

    def test_joined_two_word_aliases(self) -> None:
        # Any split whose letters collapse to a listed spelling counts.
        for words in ["sir car", "Sore car", "sar car,", "Sor car"]:
            self.assertTrue(sounds_like_wake_word(words), words)

    def test_fuzzy_near_misses_match(self) -> None:
        # Plausible one-off STT spellings within edit distance 1.
        for word in ["sorcer", "sorcars", "sorca", "zorcar", "sourcar"]:
            self.assertTrue(sounds_like_wake_word(word), word)

    def test_ordinary_words_do_not_match(self) -> None:
        # Two edits from "sorcar" already reach everyday words.
        for word in ["", "the", "hey", "sorry", "solar", "sonar", "oscar",
                     "sugar", "saucer", "car", "abracadabra", "please",
                     "!!!", "42"]:
            self.assertFalse(sounds_like_wake_word(word), word)

    def test_ordinary_word_pairs_never_fuzzy_match(self) -> None:
        # The fuzzy branch must not apply to joined pairs, only the
        # alias list may: "so car" is one edit from the wake word.
        for words in ["so car", "our car", "sorry car", "the car"]:
            self.assertFalse(sounds_like_wake_word(words), words)


class TestSplitWakePrefix(unittest.TestCase):
    """Confirming and cutting the wake word at the transcript's start."""

    def test_plain_prefix_is_cut(self) -> None:
        self.assertEqual(
            split_wake_prefix("Hey Sorcar open the door"),
            (True, "open the door"),
        )

    def test_punctuated_prefix_is_cut(self) -> None:
        self.assertEqual(
            split_wake_prefix("Hey Sorcar, what is the weather like today?"),
            (True, "what is the weather like today?"),
        )
        self.assertEqual(
            split_wake_prefix("Hey, Sorcar. Run the tests, please."),
            (True, "Run the tests, please."),
        )
        self.assertEqual(
            split_wake_prefix("Hey Sorcar! open the door"),
            (True, "open the door"),
        )

    def test_unspaced_dash_does_not_join_the_command_word(self) -> None:
        # Punctuation separates words: "Sorcar—run" must not be
        # matched as one token that swallows the verb.
        self.assertEqual(
            split_wake_prefix("Hey Sorcar—run the tests"),
            (True, "run the tests"),
        )
        self.assertEqual(
            split_wake_prefix("Hey Sorcar-open the door"),
            (True, "open the door"),
        )

    def test_alias_prefixes_are_cut(self) -> None:
        for prefix in ["Hey Sarkar", "Hey soccer,", "Hey Sorkar.", "Hey Sorcer"]:
            confirmed, rest = split_wake_prefix(f"{prefix} run the tests")
            self.assertTrue(confirmed, prefix)
            self.assertEqual(rest, "run the tests", prefix)

    def test_two_word_alias_prefix_is_cut(self) -> None:
        self.assertEqual(
            split_wake_prefix("Hey sir car, run the tests"),
            (True, "run the tests"),
        )

    def test_noise_before_the_wake_word_is_cut_too(self) -> None:
        # The pre-wake ring can pick up a breath or a stray word before
        # the wake phrase; gpt-audio may also drop the "Hey".
        self.assertEqual(
            split_wake_prefix("Okay, hey Sorcar, do it"),
            (True, "do it"),
        )
        self.assertEqual(split_wake_prefix("Sorcar, do it"), (True, "do it"))

    def test_single_word_match_does_not_eat_the_next_word(self) -> None:
        # Span 1 is tried before span 2 at every position.
        self.assertEqual(
            split_wake_prefix("Hey Sorcar car wash hours"),
            (True, "car wash hours"),
        )

    def test_wake_word_only_yields_empty_rest(self) -> None:
        self.assertEqual(split_wake_prefix("Hey Sorcar."), (True, ""))
        self.assertEqual(split_wake_prefix("Hey Sorcar"), (True, ""))

    def test_content_after_the_cut_is_preserved_verbatim(self) -> None:
        # Exactly ONE wake prefix is cut: a genuine command mentioning
        # the name right after the wake phrase keeps its words.
        self.assertEqual(
            split_wake_prefix("Hey Sorcar, Sorcar is the name of the app"),
            (True, "Sorcar is the name of the app"),
        )

    def test_transcript_without_wake_word_is_rejected(self) -> None:
        for text in [
            "open the door",
            "the weather is nice today",
            "please provide the audio",
            "I bought a candelabra for the abstract algebra lecture",
            "",
        ]:
            confirmed, rest = split_wake_prefix(text)
            self.assertFalse(confirmed, text)
            self.assertEqual(rest, text.strip(), text)

    def test_everyday_hey_phrases_and_the_old_wake_word_are_rejected(self) -> None:
        # Sound-alikes the browser's grammar force-fits onto the phrase,
        # and "Abracadabra", must not confirm a wake.
        for text in [
            # A paused "Hey, so the car." wakes the browser (grammar
            # "hey sir car", free "hey so the car"); this check drops it.
            "Hey, so the car.",
            "Hey so, the car broke down again",
            "Hey Sarah, can you call me",
            "Hey sir, the car is here",
            "Sorry, car trouble today",
            "Abracadabra, open the door",
        ]:
            confirmed, _rest = split_wake_prefix(text)
            self.assertFalse(confirmed, text)

    def test_trailing_alias_wake_within_the_window_is_cut(self) -> None:
        # A trailing-alias wake ("... hey Sorcar") can carry several
        # ring words before the wake word; the 8-word window must reach
        # it.
        self.assertEqual(
            split_wake_prefix(
                "Could you please help me now hey Sorcar, open the door"
            ),
            (True, "open the door"),
        )

    def test_wake_word_beyond_the_scan_window_is_rejected(self) -> None:
        confirmed, _rest = split_wake_prefix(
            "one two three four five six seven eight Sorcar said nothing"
        )
        self.assertFalse(confirmed)

    def test_earlier_sound_alike_words_do_not_shadow_the_wake_word(self) -> None:
        # Words sharing sounds with the name come first, but none of
        # them is within one edit of it, so the cut happens at the name.
        self.assertEqual(
            split_wake_prefix("Sorry, solar car, hey Sorcar, open the door"),
            (True, "open the door"),
        )


class CannedOpenAiServer:
    """A local HTTP stand-in for the chat-completions endpoint.

    Real HTTP served to the real OpenAI client; every POST is answered
    with one fixed assistant reply, and each request body is kept for
    inspection (the WAV attached proves what audio was transcribed).
    """

    def __init__(self, reply_text: str) -> None:
        self.request_bodies: list[bytes] = []
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — http.server API
                length = int(self.headers.get("Content-Length", "0"))
                server.request_bodies.append(self.rfile.read(length))
                body = json.dumps({
                    "id": "chatcmpl-canned",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "gpt-audio",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": server._reply_text,
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

        self._reply_text = reply_text
        self._httpd = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), Handler
        )
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """Shut the server down and join its thread."""
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def _attached_wav_pcm(request_body: bytes) -> bytes:
    """Return the s16le PCM of the ``input_audio`` WAV in a request."""
    payload = json.loads(request_body)
    datas: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            audio = node.get("input_audio")
            if isinstance(audio, dict) and isinstance(
                audio.get("data"), str
            ):
                datas.append(audio["data"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    assert len(datas) == 1, f"expected one audio part, got {len(datas)}"
    wav_bytes = base64.b64decode(datas[0])
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getnchannels() == 1
        return wf.readframes(wf.getnframes())


def _tone_pcm(seconds: float) -> bytes:
    """Return *seconds* of a clearly audible 440Hz tone (s16le PCM)."""
    count = int(seconds * SAMPLE_RATE)
    frames = [
        int(12000 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE))
        for i in range(count)
    ]
    return struct.pack(f"<{count}h", *frames)


class TestDualCheckEndToEnd(unittest.TestCase):
    """The real transcription agent against a canned transcription API."""

    def _transcribe(
        self,
        reply_text: str,
        pcm: bytes,
        expect_wake_prefix: bool = True,
        wake_agreed: bool = False,
    ) -> tuple[dict[str, Any], list[bytes]]:
        """Run ``transcribe_pcm`` on *pcm* with the API answering *reply_text*."""
        canned = CannedOpenAiServer(reply_text)
        saved = {k: os.environ.get(k) for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY")}
        os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{canned.port}/v1"
        os.environ["OPENAI_API_KEY"] = "sk-kiss-dual-check-test"
        try:
            result = transcribe_pcm(
                pcm,
                expect_wake_prefix=expect_wake_prefix,
                wake_agreed=wake_agreed,
            )
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            canned.close()
        return result, canned.request_bodies

    def test_confirmed_wake_prefix_is_cut_from_the_speech(self) -> None:
        result, bodies = self._transcribe(
            "en\nHey Sorcar, please open the editor.", _tone_pcm(1.0),
        )
        self.assertEqual(result, {"text": "please open the editor.", "language": "en"})
        self.assertEqual(len(bodies), 1)

    def test_unconfirmed_transcript_is_rejected_as_false_wake(self) -> None:
        result, _bodies = self._transcribe(
            "en\nplease open the editor.", _tone_pcm(1.0),
        )
        self.assertEqual(result, {"text": "", "language": None})

    def test_without_the_flag_the_transcript_is_kept_verbatim(self) -> None:
        result, _bodies = self._transcribe(
            "en\nplease open the editor.", _tone_pcm(1.0),
            expect_wake_prefix=False,
        )
        self.assertEqual(result, {"text": "please open the editor.", "language": "en"})

    def test_non_english_speech_needs_both_browser_recognizers(self) -> None:
        # gpt-audio may drop the English wake word while it
        # TRANSLATES non-English speech, so an unconfirmed transcript
        # with a non-English language tag is still reported when both
        # browser recognizers heard the word (wakeAgreed).
        result, _bodies = self._transcribe(
            "fr\nHello everyone.", _tone_pcm(1.0), wake_agreed=True,
        )
        self.assertEqual(result, {"text": "Hello everyone.", "language": "fr"})
        # Without that agreement it is a false wake, like the Russian
        # and Bengali background talk accepted on 2026-11-08.
        result, _bodies = self._transcribe("ru\nOh, my toy fell down.", _tone_pcm(1.0))
        self.assertEqual(result, {"text": "", "language": None})
        # Agreement never rescues an English transcript without the word.
        result, _bodies = self._transcribe(
            "en\nPlease open the editor.", _tone_pcm(1.0), wake_agreed=True,
        )
        self.assertEqual(result, {"text": "", "language": None})

    def test_the_handed_in_audio_is_what_gets_transcribed(self) -> None:
        pcm = _tone_pcm(1.0)
        _result, bodies = self._transcribe("en\nHey Sorcar, open it.", pcm)
        self.assertEqual(_attached_wav_pcm(bodies[0]), pcm)

    def test_silent_audio_never_reaches_the_api(self) -> None:
        result, bodies = self._transcribe(
            "en\nHey Sorcar, open it.", b"\x00\x00" * SAMPLE_RATE,
        )
        self.assertEqual(result, {"text": "", "language": None})
        self.assertEqual(bodies, [])


if __name__ == "__main__":
    unittest.main()
