# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for speaker identification of post-wake speech.

Real audio, real speech models, real GPT translation — no mocks.
After the "Sorcar" wake word, the listener captures the utterance,
translates it to English AND identifies the speaker with the Vosk
speaker-identification model (x-vector embeddings compared by cosine
distance).  Each distinct voice gets a unique number starting from 1;
the same voice keeps its number across utterances.  The listener
reports both on stdout as one JSON object per utterance::

    SPEECH {"text": "<english text>", "speaker": <int or null>}

- ``TestSpeakerRegistry`` drives the real embedding registry with raw
  vectors, covering every assignment branch (first speaker, distinct
  speaker, repeat speaker, degenerate zero vector).
- ``TestSpeakerIdFromWav`` synthesizes three utterances with two
  different macOS TTS voices (A, B, A), streams them through the real
  listener process and asserts the speakers are numbered [1, 2, 1]
  while the texts are the translated task descriptions.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any

from kiss.agents.vscode.voice_wake import (
    SAMPLE_RATE,
    SpeakerRegistry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]

HAVE_MAC_TTS = bool(shutil.which("say")) and bool(shutil.which("afconvert"))
HAVE_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))


def _tts_wav(directory: Path, name: str, text: str, voice: str | None = None) -> Path:
    """Synthesize *text* into a 16kHz mono 16-bit WAV via macOS TTS."""
    aiff = directory / f"{name}.aiff"
    wav = directory / f"{name}.wav"
    cmd = ["say", text, "-o", str(aiff)]
    if voice:
        cmd[1:1] = ["-v", voice]
    subprocess.run(cmd, check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
         str(aiff), str(wav)],
        check=True,
    )
    return wav


def _concat_wavs(out: Path, parts: list[Path], gap_seconds: float) -> Path:
    """Concatenate WAV *parts* with *gap_seconds* of silence between
    and after them, writing a 16kHz mono 16-bit WAV to *out*."""
    gap = b"\x00\x00" * int(SAMPLE_RATE * gap_seconds)
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for part in parts:
            with wave.open(str(part), "rb") as rf:
                wf.writeframes(rf.readframes(rf.getnframes()))
            wf.writeframes(gap)
    return out


def _run_listener(
    wav: Path, models_dir: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the real wake listener module against *wav*."""
    cmd = [
        "uv", "run", "python", "-m",
        "kiss.agents.vscode.voice_wake", "--wav", str(wav),
    ]
    if models_dir is not None:
        cmd += ["--models-dir", str(models_dir)]
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        env=dict(os.environ),
    )


def _speech_payloads(stdout: str) -> list[Any]:
    """Extract the JSON payloads of all SPEECH lines in *stdout*."""
    return [
        json.loads(line[len("SPEECH "):])
        for line in stdout.splitlines()
        if line.startswith("SPEECH ")
    ]


def _two_english_voices() -> tuple[str, str] | None:
    """Pick two clearly different installed English macOS voices."""
    voices = subprocess.run(
        ["say", "-v", "?"], capture_output=True, text=True, check=True,
    ).stdout
    # Prefer well-known voices with very different timbre.
    preferred = ["Samantha", "Daniel", "Karen", "Moira", "Rishi", "Alex",
                 "Fred", "Tessa"]
    installed = []
    for line in voices.splitlines():
        if "en_" in line or "en-" in line:
            installed.append(line.split()[0])
    chosen = [v for v in preferred if v in installed]
    if len(chosen) >= 2:
        return chosen[0], chosen[1]
    if len(installed) >= 2:
        return installed[0], installed[1]
    return None


class TestSpeakerRegistry(unittest.TestCase):
    """The real embedding registry, driven with raw x-vector shapes."""

    def test_first_speaker_gets_number_one(self) -> None:
        registry = SpeakerRegistry()
        self.assertEqual(registry.identify([1.0, 0.0, 0.0]), 1)

    def test_distinct_voice_gets_next_number(self) -> None:
        registry = SpeakerRegistry()
        self.assertEqual(registry.identify([1.0, 0.0, 0.0]), 1)
        # Orthogonal vector: cosine distance 1.0 — clearly a new voice.
        self.assertEqual(registry.identify([0.0, 1.0, 0.0]), 2)
        # A third distinct voice keeps counting up.
        self.assertEqual(registry.identify([0.0, 0.0, 1.0]), 3)

    def test_same_voice_keeps_its_number(self) -> None:
        registry = SpeakerRegistry()
        self.assertEqual(registry.identify([1.0, 0.0, 0.0]), 1)
        self.assertEqual(registry.identify([0.0, 1.0, 0.0]), 2)
        # Nearly identical to speaker 1's embedding (tiny angle).
        self.assertEqual(registry.identify([0.99, 0.01, 0.0]), 1)
        # And speaker 2 again.
        self.assertEqual(registry.identify([0.01, 0.99, 0.0]), 2)

    def test_exact_repeat_embedding_matches(self) -> None:
        registry = SpeakerRegistry()
        vec = [0.3, -0.5, 0.8, 0.1]
        self.assertEqual(registry.identify(vec), 1)
        self.assertEqual(registry.identify(list(vec)), 1)

    def test_zero_vector_is_a_new_speaker_not_a_crash(self) -> None:
        registry = SpeakerRegistry()
        self.assertEqual(registry.identify([1.0, 0.0]), 1)
        # Degenerate all-zero embedding must not divide by zero; it
        # cannot match anything, so it becomes a new speaker.
        self.assertEqual(registry.identify([0.0, 0.0]), 2)
        # And a later real vector still matches its own speaker.
        self.assertEqual(registry.identify([1.0, 0.0]), 1)

    def test_empty_vector_is_a_new_speaker_not_a_crash(self) -> None:
        registry = SpeakerRegistry()
        self.assertEqual(registry.identify([]), 1)
        self.assertEqual(registry.identify([1.0, 0.0]), 2)


@unittest.skipUnless(HAVE_MAC_TTS, "requires macOS `say` and `afconvert`")
@unittest.skipUnless(HAVE_OPENAI_KEY, "requires OPENAI_API_KEY")
class TestSpeakerIdFromWav(unittest.TestCase):
    """Distinct voices get distinct numbers; repeats keep theirs."""

    # Keyword expected in each utterance's transcript, in spoken order.
    _EXPECTED_KEYWORDS = ("parser", "test", "documentation")

    @classmethod
    def _stt_flaked(cls, payloads: list[Any]) -> str | None:
        """Return why *payloads* shows an STT-backend flake, else None.

        The gpt-audio backend occasionally hallucinates a reply
        instead of transcribing an utterance (observed
        nondeterministically, even at temperature 0: "Please provide
        the audio, and I will transcribe and translate it
        accordingly.").  A hallucinated transcript loses its keyword,
        and a persistent refusal degrades to NO_SPEECH (a missing
        payload); both warrant a retry of the whole end-to-end run.
        Everything else (wake counts, payload shapes, speaker
        numbers) is asserted strictly on every run.
        """
        if len(payloads) != len(cls._EXPECTED_KEYWORDS):
            return f"expected 3 SPEECH payloads, got {len(payloads)}"
        for keyword, payload in zip(cls._EXPECTED_KEYWORDS, payloads):
            if not isinstance(payload, dict):
                return None
            text = payload.get("text")
            if not isinstance(text, str) or keyword not in text.lower():
                return f"{keyword!r} not found in transcript {text!r}"
        return None

    def test_two_voices_numbered_in_order_and_repeat_matches(self) -> None:
        pair = _two_english_voices()
        if not pair:
            self.skipTest("fewer than two English macOS TTS voices installed")
        voice_a, voice_b = pair
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            parts = []
            for i, (voice, text) in enumerate([
                (voice_a, "fix the parser bug"),
                (voice_b, "run all the tests"),
                (voice_a, "update the documentation"),
            ]):
                wake = _tts_wav(tmpdir, f"wake{i}", "Sorcar", voice)
                speech = _tts_wav(tmpdir, f"speech{i}", text, voice)
                utterance = _concat_wavs(
                    tmpdir / f"utterance{i}.wav", [wake, speech],
                    gap_seconds=1.0,
                )
                parts.append(utterance)
            wav = _concat_wavs(
                tmpdir / "combined.wav", parts, gap_seconds=3.0,
            )
            # Retry-tolerant against STT hallucinations only (see
            # _stt_flaked): wake detection and speaker identification
            # run locally and are asserted strictly on EVERY attempt;
            # a genuine regression therefore fails all three runs.
            flake_details: list[str] = []
            for _attempt in range(3):
                proc = _run_listener(wav)
                detail = proc.stdout + proc.stderr[-2000:]
                self.assertEqual(
                    proc.stdout.split().count("WAKE"), 3, msg=detail
                )
                self.assertEqual(
                    proc.returncode, 0, msg=proc.stderr[-2000:]
                )
                payloads = _speech_payloads(proc.stdout)
                flake = self._stt_flaked(payloads)
                if flake is not None:
                    flake_details.append(f"{flake}\n{detail}")
                    continue
                for payload in payloads:
                    self.assertIsInstance(payload, dict, msg=detail)
                    self.assertIsInstance(payload["text"], str, msg=detail)
                    self.assertIsInstance(
                        payload["speaker"], int, msg=detail
                    )
                speakers = [p["speaker"] for p in payloads]
                self.assertEqual(speakers, [1, 2, 1], msg=detail)
                return
        self.fail(
            "STT backend flaked on all 3 runs:\n"
            + "\n---\n".join(flake_details)
        )

    def test_single_utterance_payload_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            wake = _tts_wav(tmpdir, "wake", "Sorcar")
            speech = _tts_wav(tmpdir, "speech", "hello world")
            wav = _concat_wavs(
                tmpdir / "combined.wav", [wake, speech], gap_seconds=1.5,
            )
            proc = _run_listener(wav)
        detail = proc.stdout + proc.stderr[-2000:]
        payloads = _speech_payloads(proc.stdout)
        self.assertEqual(len(payloads), 1, msg=detail)
        payload = payloads[0]
        self.assertIsInstance(payload, dict, msg=detail)
        self.assertIn("hello", payload["text"].lower(), msg=detail)
        self.assertEqual(payload["speaker"], 1, msg=detail)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])

    def test_broken_speaker_model_degrades_to_null_speaker(self) -> None:
        # A corrupt/unloadable speaker model must not break translation:
        # the SPEECH payload arrives with speaker null and the failure
        # is reported on stderr.  The wake model is reused from the
        # default cache (symlinked), while the spk model directory
        # exists but is empty, so SpkModel construction fails.
        from kiss.agents.vscode.voice_wake import (
            DEFAULT_MODELS_DIR,
            MODEL_NAME,
            SPK_MODEL_NAME,
            ensure_model,
        )

        ensure_model(DEFAULT_MODELS_DIR)
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            models_dir = tmpdir / "models"
            models_dir.mkdir()
            (models_dir / MODEL_NAME).symlink_to(
                DEFAULT_MODELS_DIR / MODEL_NAME
            )
            (models_dir / SPK_MODEL_NAME).mkdir()  # empty => load fails
            wake = _tts_wav(tmpdir, "wake", "Sorcar")
            speech = _tts_wav(tmpdir, "speech", "hello world")
            wav = _concat_wavs(
                tmpdir / "combined.wav", [wake, speech], gap_seconds=1.5,
            )
            proc = _run_listener(wav, models_dir=models_dir)
        detail = proc.stdout + proc.stderr[-2000:]
        payloads = _speech_payloads(proc.stdout)
        self.assertEqual(len(payloads), 1, msg=detail)
        payload = payloads[0]
        self.assertIsInstance(payload, dict, msg=detail)
        self.assertIn("hello", payload["text"].lower(), msg=detail)
        self.assertIsNone(payload["speaker"], msg=detail)
        self.assertIn("speaker identification failed", proc.stderr, msg=detail)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])


class TestSpeakerIdentifierLocal(unittest.TestCase):
    """The real x-vector extractor, driven directly (local models only)."""

    def test_empty_pcm_yields_no_speaker(self) -> None:
        from kiss.agents.vscode.voice_wake import (
            DEFAULT_MODELS_DIR,
            SpeakerIdentifier,
        )

        identifier = SpeakerIdentifier(DEFAULT_MODELS_DIR)
        self.assertIsNone(identifier.speaker_of(b""))


if __name__ == "__main__":
    unittest.main()
