# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for translating speech after the "Sorcar" wake word.

Real audio, real speech models, real GPT translation — no mocks.  The
translation is a single ``gpt-audio`` chat-completions call (audio
content part first, dictation instruction text after, in one user
message), so the tests also pin down the dictation semantics:

- ``test_translate_speech_after_wake`` speaks "Sorcar" followed by a
  sentence with the macOS TTS engine, streams the audio through the
  actual wake-word listener (``kiss.agents.vscode.voice_wake``), and
  asserts that the listener emits a ``SPEECH`` line whose payload is
  the GPT-translated English text.

- ``test_question_is_transcribed_not_answered`` guards against the
  gpt-audio failure mode of *answering* the dictated speech instead of
  transcribing it.

- ``test_no_speech_after_wake`` checks that silence after the wake
  word produces a ``NO_SPEECH`` line and no GPT call output.

- ``test_translate_error_is_not_fatal`` runs with a bogus OpenAI key
  and asserts the listener reports the failure without crashing.

- ``TestSpeechCapture`` drives the real endpointing logic with raw PCM
  (loud/silent blocks) covering every capture branch, and exercises the
  transcript post-processing (preamble/quote stripping, wake-word
  prefix removal) with real model-output shapes observed in probes.
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
    BLOCK_SIZE,
    SAMPLE_RATE,
    SpeechCapture,
    clean_transcript,
    pcm_to_wav_bytes,
    strip_leading_wake_word,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]

HAVE_MAC_TTS = bool(shutil.which("say")) and bool(shutil.which("afconvert"))
HAVE_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))

BLOCK_SECONDS = BLOCK_SIZE / SAMPLE_RATE


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


def _concat_wavs(
    out: Path,
    parts: list[Path],
    gap_seconds: float,
    tail_seconds: float | None = None,
) -> Path:
    """Concatenate WAV *parts* with *gap_seconds* of silence between
    them and *tail_seconds* (default: *gap_seconds*) after the last,
    writing a 16kHz mono 16-bit WAV to *out*."""
    gap = b"\x00\x00" * int(SAMPLE_RATE * gap_seconds)
    if tail_seconds is None:
        tail_seconds = gap_seconds
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


def _run_listener(wav: Path, env_overrides: dict[str, str] | None = None):
    """Run the real wake listener module against *wav*; return the
    completed process."""
    env = dict(os.environ)
    env.update(env_overrides or {})
    return subprocess.run(
        [
            "uv", "run", "python", "-m",
            "kiss.agents.vscode.voice_wake", "--wav", str(wav),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )


def _speech_payloads(stdout: str) -> list[str]:
    """Extract the JSON payloads of all SPEECH lines in *stdout*."""
    return [
        json.loads(line[len("SPEECH "):])
        for line in stdout.splitlines()
        if line.startswith("SPEECH ")
    ]


def _loud_block() -> bytes:
    """One audio block of a clearly audible 440Hz tone (s16le)."""
    frames = [
        int(12000 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE))
        for i in range(BLOCK_SIZE)
    ]
    return struct.pack(f"<{BLOCK_SIZE}h", *frames)


def _silent_block() -> bytes:
    """One audio block of pure silence (s16le)."""
    return b"\x00\x00" * BLOCK_SIZE


@unittest.skipUnless(HAVE_MAC_TTS, "requires macOS `say` and `afconvert`")
@unittest.skipUnless(HAVE_OPENAI_KEY, "requires OPENAI_API_KEY")
class TestVoiceTranslateFromWav(unittest.TestCase):
    """Speech following 'Sorcar' is GPT-translated into English."""

    def test_translate_speech_after_wake(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            wake = _tts_wav(tmpdir, "wake", "Sorcar")
            speech = _tts_wav(
                tmpdir, "speech", "hello world, please fix the parser bug",
            )
            wav = _concat_wavs(
                tmpdir / "combined.wav", [wake, speech], gap_seconds=1.5,
            )
            proc = _run_listener(wav)
        self.assertIn("WAKE", proc.stdout.split(), msg=proc.stderr[-2000:])
        # The UI shows a yellow "transcribing" flash driven by this line.
        self.assertIn(
            "TRANSCRIBING", proc.stdout.split(),
            msg=proc.stdout + proc.stderr[-2000:],
        )
        payloads = _speech_payloads(proc.stdout)
        self.assertEqual(len(payloads), 1, msg=proc.stdout + proc.stderr[-2000:])
        text = payloads[0].lower()
        self.assertIn("hello", text, msg=proc.stdout + proc.stderr[-2000:])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])

    def test_translate_non_english_speech_to_english(self) -> None:
        # Use a French macOS voice when available so the GPT model must
        # actually translate; otherwise skip.
        voices = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, check=True,
        ).stdout
        french = None
        for line in voices.splitlines():
            if "fr_FR" in line or "fr-FR" in line:
                french = line.split()[0]
                break
        if not french:
            self.skipTest("no French macOS TTS voice installed")
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            wake = _tts_wav(tmpdir, "wake", "Sorcar")
            aiff = tmpdir / "fr.aiff"
            subprocess.run(
                ["say", "-v", french, "Bonjour tout le monde",
                 "-o", str(aiff)],
                check=True,
            )
            fr_wav = tmpdir / "fr.wav"
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                 str(aiff), str(fr_wav)],
                check=True,
            )
            wav = _concat_wavs(
                tmpdir / "combined.wav", [wake, fr_wav], gap_seconds=1.5,
            )
            proc = _run_listener(wav)
        payloads = _speech_payloads(proc.stdout)
        self.assertEqual(len(payloads), 1, msg=proc.stdout + proc.stderr[-2000:])
        text = payloads[0].lower()
        self.assertIn("hello", text, msg=proc.stdout + proc.stderr[-2000:])
        self.assertNotIn("bonjour", text, msg=proc.stdout)

    def test_question_is_transcribed_not_answered(self) -> None:
        # gpt-audio's known failure mode is answering the dictated
        # speech ("Paris.") instead of transcribing it; the dictation
        # prompt must make it output the words verbatim.
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            wake = _tts_wav(tmpdir, "wake", "Sorcar")
            speech = _tts_wav(
                tmpdir, "speech", "What is the capital of France?",
            )
            wav = _concat_wavs(
                tmpdir / "combined.wav", [wake, speech], gap_seconds=1.5,
            )
            proc = _run_listener(wav)
        payloads = _speech_payloads(proc.stdout)
        self.assertEqual(len(payloads), 1, msg=proc.stdout + proc.stderr[-2000:])
        text = payloads[0].lower()
        self.assertIn("capital", text, msg=proc.stdout)
        self.assertIn("france", text, msg=proc.stdout)
        self.assertNotIn("paris", text, msg=proc.stdout)

    def test_speech_cut_off_at_end_of_file_is_still_translated(self) -> None:
        # The file ends while the capture is still waiting for trailing
        # silence; the listener must flush and translate what it heard.
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            wake = _tts_wav(tmpdir, "wake", "Sorcar")
            speech = _tts_wav(tmpdir, "speech", "hello world")
            wav = _concat_wavs(
                tmpdir / "combined.wav",
                [wake, speech],
                gap_seconds=1.5,
                tail_seconds=0.3,
            )
            proc = _run_listener(wav)
        payloads = _speech_payloads(proc.stdout)
        self.assertEqual(len(payloads), 1, msg=proc.stdout + proc.stderr[-2000:])
        self.assertIn(
            "hello", payloads[0].lower(),
            msg=proc.stdout + proc.stderr[-2000:],
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])

    def test_no_speech_after_wake(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            wake = _tts_wav(tmpdir, "wake", "Sorcar")
            wav = _concat_wavs(
                tmpdir / "combined.wav", [wake], gap_seconds=8.0,
            )
            proc = _run_listener(wav)
        lines = proc.stdout.split()
        self.assertIn("WAKE", lines, msg=proc.stderr[-2000:])
        self.assertIn("NO_SPEECH", lines, msg=proc.stdout + proc.stderr[-2000:])
        # Pure silence never reaches the GPT call, so the UI must not
        # be told a transcription started.
        self.assertNotIn("TRANSCRIBING", lines)
        self.assertEqual(_speech_payloads(proc.stdout), [])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])

    def test_translate_error_is_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            wake = _tts_wav(tmpdir, "wake", "Sorcar")
            speech = _tts_wav(tmpdir, "speech", "hello world")
            wav = _concat_wavs(
                tmpdir / "combined.wav", [wake, speech], gap_seconds=1.5,
            )
            proc = _run_listener(
                wav, env_overrides={"OPENAI_API_KEY": "sk-invalid-key"},
            )
        lines = proc.stdout.split()
        self.assertIn("WAKE", lines, msg=proc.stderr[-2000:])
        self.assertIn("NO_SPEECH", lines, msg=proc.stdout + proc.stderr[-2000:])
        # Speech was captured, so the transcribing indicator fired even
        # though the GPT call itself failed.
        self.assertIn("TRANSCRIBING", lines)
        self.assertEqual(_speech_payloads(proc.stdout), [])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])
        self.assertIn("translation failed", proc.stderr)


class TestSpeechCapture(unittest.TestCase):
    """The real endpointing logic, driven block-by-block with raw PCM.

    Semantics under test (see ``SpeechCapture``):

    - Leading silence after the wake word is ignored.
    - Speech immediately after the wake word is captured; users do not
      need to pause after saying "Sorcar".
    - Capture ends after trailing silence, no-speech timeout, or a
      hard maximum duration.
    """

    def test_capture_ends_after_trailing_silence(self) -> None:
        capture = SpeechCapture()
        loud, silent = _loud_block(), _silent_block()
        self.assertIsNone(capture.feed(silent))  # leading silence ignored
        self.assertIsNone(capture.feed(loud))  # speech begins immediately
        result = None
        blocks = 0
        while result is None:
            result = capture.feed(silent)
            blocks += 1
            self.assertLess(blocks, 100, "capture never ended on silence")
        self.assertAlmostEqual(
            blocks * BLOCK_SECONDS,
            SpeechCapture.END_SILENCE_SECONDS,
            delta=BLOCK_SECONDS,
        )
        self.assertEqual(len(result), len(loud) + blocks * len(silent))

    def test_immediate_short_speech_is_not_dropped(self) -> None:
        capture = SpeechCapture()
        loud, silent = _loud_block(), _silent_block()
        self.assertIsNone(capture.feed(loud))
        result = None
        blocks = 0
        while result is None:
            result = capture.feed(silent)
            blocks += 1
            self.assertLess(blocks, 100, "capture never ended on silence")
        self.assertTrue(result.startswith(loud))

    def test_no_speech_times_out_to_empty_capture(self) -> None:
        capture = SpeechCapture()
        silent = _silent_block()
        result = None
        blocks = 0
        while result is None:
            result = capture.feed(silent)
            blocks += 1
            self.assertLess(blocks, 200, "capture never timed out")
        self.assertEqual(result, b"")  # no speech detected
        self.assertGreaterEqual(
            blocks * BLOCK_SECONDS,
            SpeechCapture.NO_SPEECH_TIMEOUT_SECONDS,
        )

    def test_capture_stops_at_max_duration(self) -> None:
        capture = SpeechCapture()
        loud = _loud_block()
        result = None
        blocks = 0
        while result is None:
            result = capture.feed(loud)
            blocks += 1
            self.assertLess(blocks, 400, "capture never hit max duration")
        self.assertEqual(len(result), blocks * len(loud))
        self.assertLessEqual(
            blocks * BLOCK_SECONDS,
            SpeechCapture.MAX_CAPTURE_SECONDS + BLOCK_SECONDS,
        )

    def test_flush_returns_speech_heard_so_far(self) -> None:
        capture = SpeechCapture()
        loud = _loud_block()
        self.assertIsNone(capture.feed(loud))
        self.assertEqual(capture.flush(), loud)

    def test_flush_without_speech_is_empty(self) -> None:
        capture = SpeechCapture()
        self.assertIsNone(capture.feed(_silent_block()))
        self.assertEqual(capture.flush(), b"")

    def test_strip_leading_wake_word_aliases(self) -> None:
        self.assertEqual(strip_leading_wake_word("Sorcar, fix it"), "fix it")
        self.assertEqual(
            strip_leading_wake_word("sir car please fix it"),
            "please fix it",
        )
        self.assertEqual(strip_leading_wake_word("fix Sorcar bug"), "fix Sorcar bug")
        # gpt-audio has been observed transcribing "Sorcar" as "Sorger".
        self.assertEqual(
            strip_leading_wake_word("Sorger, fix the failing test"),
            "fix the failing test",
        )

    def test_clean_transcript_strips_preamble_and_quotes(self) -> None:
        # Real gpt-audio output shape observed in stress probes.
        self.assertEqual(
            clean_transcript(
                'Sure. Here is the transcription of the speech:\n\n'
                '"Fix the failing test in the parser."'
            ),
            "Fix the failing test in the parser.",
        )
        self.assertEqual(
            clean_transcript("Here's the translation: Fix the bug."),
            "Fix the bug.",
        )
        self.assertEqual(
            clean_transcript('"Hello world."'),
            "Hello world.",
        )
        self.assertEqual(
            clean_transcript("Fix the bug in the main file."),
            "Fix the bug in the main file.",
        )
        self.assertEqual(clean_transcript("  \n "), "")

    def test_pcm_to_wav_bytes_roundtrip(self) -> None:
        pcm = _loud_block()
        wav_bytes = pcm_to_wav_bytes(pcm)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(wav_bytes)
            f.flush()
            with wave.open(f.name, "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), SAMPLE_RATE)
                self.assertEqual(wf.readframes(wf.getnframes()), pcm)


if __name__ == "__main__":
    unittest.main()
