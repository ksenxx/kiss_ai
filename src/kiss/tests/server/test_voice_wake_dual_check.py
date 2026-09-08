# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the DUAL "Sorcar" wake check.

The wake word is now verified twice: locally (Vosk grammar decoding)
and again from the transcript.  The listener prepends the wake word's
own audio — a rolling pre-wake ring kept by
:class:`kiss.server.voice_wake.WakeSession` — to the captured
utterance, so a genuine wake's transcript begins with something that
sounds like "Sorcar"; :func:`kiss.server.voice_wake.split_wake_prefix`
confirms and cuts that prefix, and an unconfirmed transcript is
rejected as a false wake (``NO_SPEECH``).

The end-to-end tests drive the REAL listener subprocess
(``kiss.server.voice_wake --wav``) with REAL macOS TTS audio of
"Sorcar" plus a command.  The transcription API is a local canned
HTTP server standing in for the OpenAI endpoint — real HTTP, real
OpenAI client, fixed reply — so the tests can steer the transcript
exactly:

- a transcript that starts with the wake word yields ``SPEECH`` with
  the wake word cut;
- a transcript WITHOUT the wake word is rejected: ``NO_SPEECH``, no
  ``SPEECH`` line at all (before the dual check, whatever the STT
  hallucinated was submitted as the user's command);
- the audio attached to the API request is inspected to prove the
  wake word's audio really was prepended to the capture.

The prefix-gate functions themselves are exercised directly with the
transcript shapes gpt-audio produces for "Sorcar" (observed: "Sorcar",
"soccer", "circa", "Sarkar", "sir car", ...).
"""

from __future__ import annotations

import base64
import http.server
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from typing import Any

from kiss.server.voice_wake import (
    SAMPLE_RATE,
    block_rms,
    levenshtein,
    sounds_like_wake_word,
    split_wake_prefix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

HAVE_MAC_TTS = bool(shutil.which("say")) and bool(shutil.which("afconvert"))


class TestLevenshtein(unittest.TestCase):
    """The edit-distance helper of the fuzzy wake-prefix match."""

    def test_identical(self) -> None:
        self.assertEqual(levenshtein("sorcar", "sorcar"), 0)

    def test_empty_vs_word(self) -> None:
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("abc", ""), 3)

    def test_substitutions_insertions_deletions(self) -> None:
        self.assertEqual(levenshtein("sorcar", "sarkar"), 2)
        self.assertEqual(levenshtein("sorcar", "soccer"), 2)
        self.assertEqual(levenshtein("sorcar", "socar"), 1)
        self.assertEqual(levenshtein("kitten", "sitting"), 3)


class TestSoundsLikeWakeWord(unittest.TestCase):
    """Transcribed words that must (not) count as the wake word."""

    def test_known_transcription_aliases(self) -> None:
        for word in ["sorcar", "Sorcar", "SORCAR,", "soccer", "circa",
                     "sarkar", "sorkar", "sorger", "sorcerer"]:
            self.assertTrue(sounds_like_wake_word(word), word)

    def test_joined_two_word_aliases(self) -> None:
        for words in ["sir car", "sore car", "sar car", "so car",
                      "saw car"]:
            self.assertTrue(sounds_like_wake_word(words), words)

    def test_fuzzy_near_misses_match(self) -> None:
        # Plausible one-off STT spellings within edit distance 2.
        for word in ["sorcarr", "sarcar", "zorcar", "sokar", "sircar"]:
            self.assertTrue(sounds_like_wake_word(word), word)

    def test_ordinary_words_do_not_match(self) -> None:
        for word in ["", "the", "so", "car", "sir", "weather", "please",
                     "sugar", "solaris", "corsair", "!!!", "42"]:
            self.assertFalse(sounds_like_wake_word(word), word)

    def test_ordinary_word_pairs_never_fuzzy_match(self) -> None:
        # "ourcar" is 2 edits from "sorcar": the fuzzy branch must not
        # apply to joined pairs, only the measured alias list may.
        for words in ["our car", "your car", "the car", "so what"]:
            self.assertFalse(sounds_like_wake_word(words), words)


class TestSplitWakePrefix(unittest.TestCase):
    """Confirming and cutting the wake word at the transcript's start."""

    def test_plain_prefix_is_cut(self) -> None:
        self.assertEqual(
            split_wake_prefix("Sorcar open the door"),
            (True, "open the door"),
        )

    def test_punctuated_prefix_is_cut(self) -> None:
        self.assertEqual(
            split_wake_prefix("Sorcar, open the door."),
            (True, "open the door."),
        )

    def test_alias_prefixes_are_cut(self) -> None:
        for prefix in ["Soccer", "Circa,", "Sarkar", "Sorcerer,"]:
            confirmed, rest = split_wake_prefix(f"{prefix} run the tests")
            self.assertTrue(confirmed, prefix)
            self.assertEqual(rest, "run the tests", prefix)

    def test_two_word_alias_prefix_is_cut(self) -> None:
        self.assertEqual(
            split_wake_prefix("Sir car, run the tests"),
            (True, "run the tests"),
        )

    def test_noise_before_the_wake_word_is_cut_too(self) -> None:
        # The pre-wake ring can pick up a breath or the "hey there" of
        # a trailing-alias wake before the wake word itself.
        self.assertEqual(
            split_wake_prefix("Hey there Sorcar, do it"),
            (True, "do it"),
        )

    def test_single_word_alias_does_not_eat_the_next_word(self) -> None:
        # Span 1 is tried before span 2: "Sorcar so" must not be
        # consumed as one fuzzy two-word alias.
        self.assertEqual(
            split_wake_prefix("Sorcar so run the tests"),
            (True, "so run the tests"),
        )

    def test_wake_word_only_yields_empty_rest(self) -> None:
        self.assertEqual(split_wake_prefix("Sorcar."), (True, ""))

    def test_clipped_onset_counts_only_as_the_first_word(self) -> None:
        # Measured live: gpt-audio merged the wake word into the
        # sentence and transcribed "Sorcar, what is the weather" as
        # "So, what is the weather".
        self.assertEqual(
            split_wake_prefix("So, what is the weather like today?"),
            (True, "what is the weather like today?"),
        )
        self.assertEqual(
            split_wake_prefix("Sir, run the build"),
            (True, "run the build"),
        )
        confirmed, _rest = split_wake_prefix("it is so nice outside")
        self.assertFalse(confirmed)

    def test_full_alias_is_preferred_over_a_clipped_onset(self) -> None:
        # "So car" is the whole wake word; cutting only its "So" would
        # leak "car" into the command.
        self.assertEqual(
            split_wake_prefix("So car run the tests"),
            (True, "run the tests"),
        )

    def test_content_after_the_cut_is_preserved_verbatim(self) -> None:
        # Exactly ONE wake prefix is cut: a genuine command mentioning
        # "soccer" right after the wake word keeps its words (the old
        # loop-stripping behavior lost them).
        self.assertEqual(
            split_wake_prefix("Sorcar soccer is my favorite sport"),
            (True, "soccer is my favorite sport"),
        )

    def test_transcript_without_wake_word_is_rejected(self) -> None:
        for text in [
            "open the door",
            "the weather is nice today",
            "please provide the audio",
            "",
        ]:
            confirmed, rest = split_wake_prefix(text)
            self.assertFalse(confirmed, text)
            self.assertEqual(rest, text.strip(), text)

    def test_trailing_alias_wake_within_the_window_is_cut(self) -> None:
        # A trailing-alias wake ("... Sorcar") can carry several ring
        # words before the wake word; the 8-word window must reach it.
        self.assertEqual(
            split_wake_prefix(
                "Could you please help me now Sorcar, open the door"
            ),
            (True, "open the door"),
        )

    def test_wake_word_beyond_the_scan_window_is_rejected(self) -> None:
        confirmed, _rest = split_wake_prefix(
            "one two three four five six seven eight sorcar said nothing"
        )
        self.assertFalse(confirmed)

    def test_exact_wake_word_outranks_an_earlier_sound_alike(self) -> None:
        # "Sorry" (a known mishearing alias) and "solar" (a fuzzy
        # near-miss) come first, but the real wake word follows: the
        # cut must happen at "Sorcar", not at the weaker match.
        self.assertEqual(
            split_wake_prefix("Sorry, I was late Sorcar, open the door"),
            (True, "open the door"),
        )
        self.assertEqual(
            split_wake_prefix("The solar panel hums Sorcar, open the door"),
            (True, "open the door"),
        )

    def test_everyday_pair_phrases_are_rejected(self) -> None:
        confirmed, _rest = split_wake_prefix("Our car needs service")
        self.assertFalse(confirmed)


class CannedOpenAiServer:
    """A local HTTP stand-in for the chat-completions endpoint.

    Real HTTP served to the real OpenAI client inside the listener
    subprocess; every POST is answered with one fixed assistant reply,
    and each request body is kept for inspection (the WAV the listener
    attached proves what audio was transcribed).
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
    import io

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getnchannels() == 1
        return wf.readframes(wf.getnframes())


def _tts_wav(directory: Path, name: str, text: str) -> Path:
    """Synthesize *text* into a 16kHz mono 16-bit WAV via macOS TTS."""
    aiff = directory / f"{name}.aiff"
    wav = directory / f"{name}.wav"
    subprocess.run(["say", text, "-o", str(aiff)], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
         str(aiff), str(wav)],
        check=True,
    )
    return wav


def _wav_frames(path: Path) -> int:
    """Return the frame count of a WAV file."""
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes()


@unittest.skipUnless(HAVE_MAC_TTS, "requires macOS `say` and `afconvert`")
class TestDualCheckEndToEnd(unittest.TestCase):
    """The real listener, real wake audio, canned transcription API."""

    _tmp: tempfile.TemporaryDirectory[str]
    wav: Path
    speech_frames: int
    two_round_wav: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        tmpdir = Path(cls._tmp.name)
        # 1s lead-in silence fills the pre-wake ring before "Sorcar";
        # 1.5s after it satisfies the strict wake pause, and 2.5s at
        # the end exceeds END_SILENCE_SECONDS so the capture closes.
        cls.wav = _tts_wav(
            tmpdir, "utterance",
            "[[slnc 1000]] Sorcar [[slnc 1500]] "
            "please open the editor [[slnc 2500]]",
        )
        cls.speech_frames = _wav_frames(
            _tts_wav(tmpdir, "speech-only", "please open the editor")
        )
        # Two full wake+command rounds: the 4s gap lets the first
        # capture endpoint (2s) and the wake cooldown (2s) expire.
        cls.two_round_wav = _tts_wav(
            tmpdir, "two-rounds",
            "[[slnc 1000]] Sorcar [[slnc 1500]] "
            "please open the editor [[slnc 4000]] "
            "Sorcar [[slnc 1500]] "
            "please open the editor [[slnc 2500]]",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _run_listener(
        self, reply_text: str, wav: Path | None = None
    ) -> tuple[subprocess.CompletedProcess[str], CannedOpenAiServer]:
        """Run the listener on the class WAV against a canned API."""
        canned = CannedOpenAiServer(reply_text)
        try:
            env = dict(os.environ)
            env.update({
                "OPENAI_BASE_URL": f"http://127.0.0.1:{canned.port}/v1",
                "OPENAI_API_KEY": "sk-kiss-dual-check-test",
            })
            proc = subprocess.run(
                [
                    "uv", "run", "python", "-m",
                    "kiss.server.voice_wake",
                    "--wav", str(wav if wav is not None else self.wav),
                ],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except BaseException:
            canned.close()
            raise
        return proc, canned

    def test_confirmed_wake_prefix_is_cut_from_the_speech(self) -> None:
        proc, canned = self._run_listener(
            "en\nSorcar, please open the editor."
        )
        try:
            detail = f"stdout={proc.stdout!r}\nstderr={proc.stderr[-2000:]}"
            lines = proc.stdout.splitlines()
            self.assertIn("WAKE", lines, msg=detail)
            speech = [ln for ln in lines if ln.startswith("SPEECH ")]
            self.assertEqual(len(speech), 1, msg=detail)
            payload = json.loads(speech[0][len("SPEECH "):])
            self.assertEqual(
                payload["text"], "please open the editor.", msg=detail
            )
            self.assertEqual(payload["language"], "en", msg=detail)
            self.assertNotIn("NO_SPEECH", lines, msg=detail)
        finally:
            canned.close()

    def test_unconfirmed_transcript_is_rejected_as_false_wake(self) -> None:
        proc, canned = self._run_listener("en\nplease open the editor.")
        try:
            detail = f"stdout={proc.stdout!r}\nstderr={proc.stderr[-2000:]}"
            lines = proc.stdout.splitlines()
            self.assertIn("WAKE", lines, msg=detail)
            self.assertIn("NO_SPEECH", lines, msg=detail)
            self.assertEqual(
                [ln for ln in lines if ln.startswith("SPEECH ")],
                [],
                msg=detail,
            )
            self.assertIn("not confirmed", proc.stderr, msg=detail)
        finally:
            canned.close()

    def test_non_english_speech_keeps_the_single_local_check(self) -> None:
        # gpt-audio reliably drops the English wake word while it
        # TRANSLATES non-English speech, so an unconfirmed transcript
        # with a non-English language tag must still be reported.
        proc, canned = self._run_listener("fr\nHello everyone.")
        try:
            detail = f"stdout={proc.stdout!r}\nstderr={proc.stderr[-2000:]}"
            lines = proc.stdout.splitlines()
            speech = [ln for ln in lines if ln.startswith("SPEECH ")]
            self.assertEqual(len(speech), 1, msg=detail)
            payload = json.loads(speech[0][len("SPEECH "):])
            self.assertEqual(payload["text"], "Hello everyone.", msg=detail)
            self.assertEqual(payload["language"], "fr", msg=detail)
        finally:
            canned.close()

    def test_second_round_gets_a_fresh_wake_preamble(self) -> None:
        # The pre-wake ring is cleared when a capture starts and must
        # refill before the next wake: both rounds' API requests carry
        # audible wake audio at their head and both are confirmed.
        proc, canned = self._run_listener(
            "en\nSorcar, please open the editor.", wav=self.two_round_wav,
        )
        try:
            detail = f"stdout={proc.stdout!r}\nstderr={proc.stderr[-2000:]}"
            lines = proc.stdout.splitlines()
            self.assertEqual(lines.count("WAKE"), 2, msg=detail)
            speech = [ln for ln in lines if ln.startswith("SPEECH ")]
            self.assertEqual(len(speech), 2, msg=detail)
            for line in speech:
                payload = json.loads(line[len("SPEECH "):])
                self.assertEqual(
                    payload["text"], "please open the editor.", msg=detail
                )
            self.assertEqual(len(canned.request_bodies), 2, msg=detail)
            for body in canned.request_bodies:
                pcm = _attached_wav_pcm(body)
                head = pcm[: 2 * 2 * SAMPLE_RATE]
                block = 2 * 4000
                self.assertGreaterEqual(
                    max(
                        block_rms(head[i:i + block])
                        for i in range(0, len(head), block)
                    ),
                    0.01,
                    msg=detail,
                )
        finally:
            canned.close()

    def test_wake_word_audio_is_prepended_to_the_capture(self) -> None:
        proc, canned = self._run_listener("en\nSorcar, please open the editor.")
        try:
            detail = f"stdout={proc.stdout!r}\nstderr={proc.stderr[-2000:]}"
            self.assertEqual(len(canned.request_bodies), 1, msg=detail)
            pcm = _attached_wav_pcm(canned.request_bodies[0])
            # The capture alone starts at the command speech (leading
            # silence is never recorded); only the prepended pre-wake
            # ring can push the attached audio well past the command's
            # own length.
            self.assertGreaterEqual(
                len(pcm) // 2,
                self.speech_frames + int(1.5 * SAMPLE_RATE),
                msg=detail,
            )
            # And the wake word itself must be audible in that ring:
            # the first two seconds carry loud speech.
            head = pcm[: 2 * 2 * SAMPLE_RATE]
            block = 2 * 4000
            louds = [
                block_rms(head[i:i + block])
                for i in range(0, len(head), block)
            ]
            self.assertGreaterEqual(max(louds), 0.01, msg=detail)
        finally:
            canned.close()


if __name__ == "__main__":
    unittest.main()
