# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for KISS-agent-based voice transcription.

The post-wake speech translation is done by a KISS (Sorcar) agent —
:class:`kiss.core.kiss_agent.KISSAgent` running the ``gpt-audio``
model non-agentically with the utterance attached as audio — instead
of a hand-rolled OpenAI client call.  The agent must return BOTH the
English text and the language of the speech, and the listener's
``SPEECH`` protocol line must carry that language.

Real audio, real speech models, real GPT calls — no mocks:

- ``TestParseTranscriptionReply`` pins down every branch of the
  agent-reply parser (strict JSON, fenced JSON, prose-wrapped JSON,
  non-JSON fallbacks, junk language values) with real model-output
  shapes.

- ``TestTranscribeAgentDirect`` synthesizes speech with the macOS TTS
  engine and calls :func:`transcribe_pcm` directly, asserting the
  returned English text and detected language for English and French
  speech, that questions are transcribed rather than answered, and
  that empty audio and a failing API degrade to empty results without
  raising.

- ``TestListenerSpeechLanguage`` streams "Sorcar" + a sentence through
  the actual wake-word listener subprocess and asserts the ``SPEECH``
  line's JSON payload includes ``text``, ``speaker`` and ``language``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from kiss.agents.vscode.voice_wake import (
    SAMPLE_RATE,
    parse_transcription_reply,
    transcribe_pcm,
    translate_pcm_to_english,
    trim_trailing_silence,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]

HAVE_MAC_TTS = bool(shutil.which("say")) and bool(shutil.which("afconvert"))
HAVE_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))


def _has_mac_voice(voice: str) -> bool:
    """Return True when the macOS TTS *voice* is installed."""
    if not HAVE_MAC_TTS:
        return False
    listing = subprocess.run(
        ["say", "-v", "?"], capture_output=True, text=True, check=True
    ).stdout
    return any(line.split()[:1] == [voice] for line in listing.splitlines())


def _tts_wav(directory: Path, name: str, text: str, voice: str | None = None) -> Path:
    """Synthesize *text* into a 16kHz mono 16-bit WAV via macOS TTS."""
    aiff = directory / f"{name}.aiff"
    wav = directory / f"{name}.wav"
    cmd = ["say", text, "-o", str(aiff)]
    if voice is not None:
        cmd[1:1] = ["-v", voice]
    subprocess.run(cmd, check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
         str(aiff), str(wav)],
        check=True,
    )
    return wav


def _wav_pcm(wav: Path) -> bytes:
    """Return the raw PCM frames of a 16kHz mono 16-bit WAV file."""
    with wave.open(str(wav), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _tts_pcm(directory: Path, name: str, text: str, voice: str | None = None) -> bytes:
    """Synthesize *text* and return its raw 16kHz mono s16le PCM."""
    return _wav_pcm(_tts_wav(directory, name, text, voice))


def _concat_wavs(
    out: Path, parts: list[Path], gap_seconds: float, tail_seconds: float
) -> Path:
    """Concatenate WAV *parts* with silence gaps into a 16kHz mono WAV."""
    gap = b"\x00\x00" * int(SAMPLE_RATE * gap_seconds)
    tail = b"\x00\x00" * int(SAMPLE_RATE * tail_seconds)
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for i, part in enumerate(parts):
            with wave.open(str(part), "rb") as rf:
                wf.writeframes(rf.readframes(rf.getnframes()))
            wf.writeframes(tail if i == len(parts) - 1 else gap)
    return out


def _sine_pcm(seconds: float, frequency: float = 440.0) -> bytes:
    """Return loud synthetic sine-wave PCM (16kHz mono s16le)."""
    frames = int(SAMPLE_RATE * seconds)
    return b"".join(
        struct.pack(
            "<h",
            int(20000 * math.sin(2 * math.pi * frequency * i / SAMPLE_RATE)),
        )
        for i in range(frames)
    )


class TestParseTranscriptionReply(unittest.TestCase):
    """Every branch of the transcription-agent reply parser.

    The agent is asked for two lines (language tag, then English
    text) — the empirically reliable format for gpt-audio — but the
    parser also accepts JSON object replies and degrades to plain
    text.  The two-line reply shapes below (trailing spaces on the
    tag line etc.) were observed in live gpt-audio probes.
    """

    def test_two_lines(self) -> None:
        text, language = parse_transcription_reply(
            "fr\nHello, can you open the window please?"
        )
        self.assertEqual(text, "Hello, can you open the window please?")
        self.assertEqual(language, "fr")

    def test_two_lines_with_trailing_spaces_on_tag(self) -> None:
        # Live gpt-audio replies often pad the tag line: "en  \n...".
        text, language = parse_transcription_reply(
            "en  \nWhat is the capital of France?"
        )
        self.assertEqual(text, "What is the capital of France?")
        self.assertEqual(language, "en")

    def test_two_lines_with_decorated_tag(self) -> None:
        text, language = parse_transcription_reply(
            "[fr]\nOpen the window."
        )
        self.assertEqual(text, "Open the window.")
        self.assertEqual(language, "fr")

    def test_multi_line_text_is_preserved(self) -> None:
        text, language = parse_transcription_reply(
            "en\nFirst sentence.\nSecond sentence."
        )
        self.assertEqual(text, "First sentence.\nSecond sentence.")
        self.assertEqual(language, "en")

    def test_lone_tag_line_is_plain_text(self) -> None:
        # A single line — even one followed by trailing whitespace —
        # is never treated as a language tag: only a tag with actual
        # text after it parses as the two-line shape.
        for reply in ("en", "en\n", "en \n  "):
            text, language = parse_transcription_reply(reply)
            self.assertEqual(text, "en", reply)
            self.assertIsNone(language, reply)

    def test_first_line_not_a_tag_falls_back(self) -> None:
        reply = "Hello there.\nSecond line."
        text, language = parse_transcription_reply(reply)
        self.assertEqual(text, reply)
        self.assertIsNone(language)

    def test_fenced_two_lines(self) -> None:
        text, language = parse_transcription_reply(
            "```\nes\nGood morning.\n```"
        )
        self.assertEqual(text, "Good morning.")
        self.assertEqual(language, "es")

    def test_strict_json(self) -> None:
        text, language = parse_transcription_reply(
            '{"text": "Hello there.", "language": "en"}'
        )
        self.assertEqual(text, "Hello there.")
        self.assertEqual(language, "en")

    def test_fenced_json(self) -> None:
        text, language = parse_transcription_reply(
            '```json\n{"text": "Open the file.", "language": "fr"}\n```'
        )
        self.assertEqual(text, "Open the file.")
        self.assertEqual(language, "fr")

    def test_bare_fenced_json(self) -> None:
        text, language = parse_transcription_reply(
            '```\n{"text": "Run the tests.", "language": "de"}\n```'
        )
        self.assertEqual(text, "Run the tests.")
        self.assertEqual(language, "de")

    def test_json_with_surrounding_prose(self) -> None:
        text, language = parse_transcription_reply(
            'Here is the transcription:\n'
            '{"text": "Good morning.", "language": "es"}\nDone.'
        )
        self.assertEqual(text, "Good morning.")
        self.assertEqual(language, "es")

    def test_language_tag_is_normalized(self) -> None:
        text, language = parse_transcription_reply(
            '{"text": "Hi.", "language": " EN-US "}'
        )
        self.assertEqual(text, "Hi.")
        self.assertEqual(language, "en-us")

    def test_non_string_language_is_dropped(self) -> None:
        text, language = parse_transcription_reply(
            '{"text": "Hi.", "language": 5}'
        )
        self.assertEqual(text, "Hi.")
        self.assertIsNone(language)

    def test_junk_language_is_dropped(self) -> None:
        for junk in ("", "   ", "not a language tag!", "e", "x" * 40):
            _, language = parse_transcription_reply(
                json.dumps({"text": "Hi.", "language": junk})
            )
            self.assertIsNone(language, junk)

    def test_missing_text_key(self) -> None:
        text, language = parse_transcription_reply('{"language": "en"}')
        self.assertEqual(text, "")
        self.assertEqual(language, "en")

    def test_non_string_text_is_dropped(self) -> None:
        text, language = parse_transcription_reply(
            '{"text": 42, "language": "en"}'
        )
        self.assertEqual(text, "")
        self.assertEqual(language, "en")

    def test_json_array_falls_back_to_plain_text(self) -> None:
        reply = '["not", "a", "dict"]'
        text, language = parse_transcription_reply(reply)
        self.assertEqual(text, reply)
        self.assertIsNone(language)

    def test_plain_text_reply_falls_back(self) -> None:
        text, language = parse_transcription_reply("Just the spoken words.")
        self.assertEqual(text, "Just the spoken words.")
        self.assertIsNone(language)

    def test_invalid_json_braces_fall_back(self) -> None:
        reply = "set {a: 1} and {b: 2end"
        text, language = parse_transcription_reply(reply)
        self.assertEqual(text, reply)
        self.assertIsNone(language)

    def test_empty_reply(self) -> None:
        text, language = parse_transcription_reply("")
        self.assertEqual(text, "")
        self.assertIsNone(language)

    def test_whitespace_reply(self) -> None:
        text, language = parse_transcription_reply("   \n  ")
        self.assertEqual(text, "")
        self.assertIsNone(language)


class TestTrimTrailingSilence(unittest.TestCase):
    """Trailing-silence trimming applied before the agent call.

    The endpointed capture carries ~2s of trailing silence, which
    empirically flips gpt-audio into denying it heard any audio
    (0/3 padded vs 3/3 trimmed on the same speech), so the PCM is
    trimmed to the last loud block plus a short tail.
    """

    def test_empty_pcm(self) -> None:
        self.assertEqual(trim_trailing_silence(b""), b"")

    def test_all_silence_trims_to_nothing(self) -> None:
        self.assertEqual(
            trim_trailing_silence(b"\x00\x00" * (3 * SAMPLE_RATE)), b""
        )

    def test_loud_audio_is_unchanged(self) -> None:
        pcm = _sine_pcm(1.0)
        self.assertEqual(trim_trailing_silence(pcm), pcm)

    def test_trailing_silence_is_trimmed(self) -> None:
        speech = _sine_pcm(1.0)
        padded = speech + b"\x00\x00" * (2 * SAMPLE_RATE)
        trimmed = trim_trailing_silence(padded)
        # Every speech byte survives, plus at most a short tail.
        self.assertEqual(trimmed[: len(speech)], speech)
        self.assertLess(len(trimmed), len(speech) + SAMPLE_RATE * 2)

    def test_leading_silence_is_preserved(self) -> None:
        pcm = b"\x00\x00" * SAMPLE_RATE + _sine_pcm(0.5)
        self.assertEqual(trim_trailing_silence(pcm), pcm)


@unittest.skipUnless(
    HAVE_MAC_TTS and HAVE_OPENAI_KEY, "needs macOS TTS and OPENAI_API_KEY"
)
class TestTranscribeAgentDirect(unittest.TestCase):
    """The KISS transcription agent on real synthesized speech."""

    def test_english_speech_text_and_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pcm = _tts_pcm(
                Path(tmp), "en", "The weather is nice today."
            )
            result = transcribe_pcm(pcm)
        self.assertIn("weather", result["text"].lower())
        self.assertEqual(result["language"], "en")

    def test_french_speech_is_translated_and_detected(self) -> None:
        if not _has_mac_voice("Thomas"):
            self.skipTest("French macOS voice 'Thomas' not installed")
        with tempfile.TemporaryDirectory() as tmp:
            pcm = _tts_pcm(
                Path(tmp),
                "fr",
                "Bonjour, pouvez-vous ouvrir la fenêtre s'il vous plaît?",
                voice="Thomas",
            )
            result = transcribe_pcm(pcm)
        self.assertTrue(result["language"].startswith("fr"), result)
        text = result["text"].lower()
        self.assertIn("window", text)
        # Translated to English, not echoed in French.
        self.assertNotIn("fenêtre", text)

    def test_question_is_transcribed_not_answered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pcm = _tts_pcm(
                Path(tmp), "q", "What is the capital of France?"
            )
            result = transcribe_pcm(pcm)
        text = result["text"].lower()
        self.assertIn("capital of france", text)
        self.assertNotIn("paris", text)
        self.assertEqual(result["language"], "en")

    def test_translate_wrapper_returns_text_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pcm = _tts_pcm(Path(tmp), "w", "Delete the temporary files.")
        self.assertIn(
            "temporary files", translate_pcm_to_english(pcm).lower()
        )


class TestTranscribeAgentDegraded(unittest.TestCase):
    """Failure paths of the transcription agent (no TTS needed)."""

    def test_empty_pcm_skips_the_agent(self) -> None:
        self.assertEqual(
            transcribe_pcm(b""), {"text": "", "language": None}
        )

    def test_api_failure_returns_empty_result(self) -> None:
        # A bogus API key in a subprocess: the agent call must fail
        # fast (bounded by KISS_VOICE_AUDIO_TIMEOUT) and degrade to an
        # empty result instead of raising or hanging.
        script = (
            "import json, math, struct\n"
            "from kiss.agents.vscode.voice_wake import ("
            "SAMPLE_RATE, transcribe_pcm)\n"
            "pcm = b''.join(struct.pack('<h', int(20000 * math.sin("
            "2 * math.pi * 440 * i / SAMPLE_RATE)))"
            " for i in range(SAMPLE_RATE))\n"
            "print(json.dumps(transcribe_pcm(pcm)))\n"
        )
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = "sk-invalid-not-a-real-key"
        env["KISS_VOICE_AUDIO_TIMEOUT"] = "20"
        proc = subprocess.run(
            ["uv", "run", "python", "-c", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            json.loads(proc.stdout.strip().splitlines()[-1]),
            {"text": "", "language": None},
        )


@unittest.skipUnless(
    HAVE_MAC_TTS and HAVE_OPENAI_KEY, "needs macOS TTS and OPENAI_API_KEY"
)
class TestListenerSpeechLanguage(unittest.TestCase):
    """Full listener subprocess: SPEECH payload carries the language."""

    def test_speech_payload_has_text_speaker_and_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wake = _tts_wav(tmp_path, "wake", "Sorcar")
            speech = _tts_wav(
                tmp_path, "speech", "Please summarize the latest changes."
            )
            wav = _concat_wavs(
                tmp_path / "combined.wav",
                [wake, speech],
                gap_seconds=0.6,
                tail_seconds=2.5,
            )
            proc = subprocess.run(
                [
                    "uv", "run", "python", "-m",
                    "kiss.agents.vscode.voice_wake", "--wav", str(wav),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=600,
                env=dict(os.environ),
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = proc.stdout.splitlines()
        self.assertIn("WAKE", lines, proc.stdout)
        payloads = [
            json.loads(line[len("SPEECH "):])
            for line in lines
            if line.startswith("SPEECH ")
        ]
        self.assertEqual(len(payloads), 1, proc.stdout)
        payload = payloads[0]
        self.assertEqual(
            set(payload), {"text", "speaker", "language"}, payload
        )
        self.assertIn("summarize", payload["text"].lower())
        self.assertEqual(payload["language"], "en")
        self.assertTrue(
            payload["speaker"] is None
            or (isinstance(payload["speaker"], int) and payload["speaker"] >= 1)
        )


if __name__ == "__main__":
    unittest.main()
