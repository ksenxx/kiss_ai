"""End-to-end tests for speaker identification of dictated speech.

Real audio, real speech models — no mocks.  The daemon identifies the
speaker of every ``voiceTranscribe`` utterance with the Vosk
speaker-identification model (x-vector embeddings compared by cosine
distance; :class:`kiss.server.speech.SpeakerIdentifier`).  Each
distinct voice gets a unique number starting from 1; the same voice
keeps its number across utterances.

- ``TestSpeakerRegistry`` drives the real embedding registry with raw
  vectors, covering every assignment branch (first speaker, distinct
  speaker, repeat speaker, degenerate zero vector).
- ``TestSpeakerIdFromWav`` synthesizes three utterances with two
  different macOS TTS voices (A, B, A), feeds them to the real
  identifier and asserts the speakers are numbered [1, 2, 1].
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from kiss.server.speech import SpeakerIdentifier, SpeakerRegistry, default_models_dir

HAVE_MAC_TTS = bool(shutil.which("say")) and bool(shutil.which("afconvert"))


def _tts_pcm(directory: Path, name: str, text: str, voice: str | None = None) -> bytes:
    """Synthesize *text* via macOS TTS and return its 16kHz mono s16le PCM."""
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
    with wave.open(str(wav), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _two_english_voices() -> tuple[str, str] | None:
    """Pick two clearly different installed English macOS voices."""
    voices = subprocess.run(
        ["say", "-v", "?"], capture_output=True, text=True, check=True,
    ).stdout
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
        self.assertEqual(registry.identify([0.0, 1.0, 0.0]), 2)
        self.assertEqual(registry.identify([0.0, 0.0, 1.0]), 3)

    def test_same_voice_keeps_its_number(self) -> None:
        registry = SpeakerRegistry()
        self.assertEqual(registry.identify([1.0, 0.0, 0.0]), 1)
        self.assertEqual(registry.identify([0.0, 1.0, 0.0]), 2)
        self.assertEqual(registry.identify([0.99, 0.01, 0.0]), 1)
        self.assertEqual(registry.identify([0.01, 0.99, 0.0]), 2)

    def test_exact_repeat_embedding_matches(self) -> None:
        registry = SpeakerRegistry()
        vec = [0.3, -0.5, 0.8, 0.1]
        self.assertEqual(registry.identify(vec), 1)
        self.assertEqual(registry.identify(list(vec)), 1)

    def test_zero_vector_is_a_new_speaker_not_a_crash(self) -> None:
        registry = SpeakerRegistry()
        self.assertEqual(registry.identify([1.0, 0.0]), 1)
        self.assertEqual(registry.identify([0.0, 0.0]), 2)
        self.assertEqual(registry.identify([1.0, 0.0]), 1)

    def test_empty_vector_is_a_new_speaker_not_a_crash(self) -> None:
        registry = SpeakerRegistry()
        self.assertEqual(registry.identify([]), 1)
        self.assertEqual(registry.identify([1.0, 0.0]), 2)


@unittest.skipUnless(HAVE_MAC_TTS, "requires macOS `say` and `afconvert`")
class TestSpeakerIdFromWav(unittest.TestCase):
    """Distinct voices get distinct numbers; repeats keep theirs."""

    def test_two_voices_numbered_in_order_and_repeat_matches(self) -> None:
        pair = _two_english_voices()
        if not pair:
            self.skipTest("fewer than two English macOS TTS voices installed")
        voice_a, voice_b = pair
        identifier = SpeakerIdentifier(default_models_dir())
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            speakers = [
                identifier.speaker_of(_tts_pcm(tmpdir, f"speech{i}", text, voice))
                for i, (voice, text) in enumerate([
                    (voice_a, "fix the parser bug"),
                    (voice_b, "run all the tests"),
                    (voice_a, "update the documentation"),
                ])
            ]
        self.assertEqual(speakers, [1, 2, 1])


class TestSpeakerIdentifierLocal(unittest.TestCase):
    """The real x-vector extractor, driven directly (local models only)."""

    def test_empty_pcm_yields_no_speaker(self) -> None:
        identifier = SpeakerIdentifier(default_models_dir())
        self.assertIsNone(identifier.speaker_of(b""))


    def test_broken_speaker_model_raises(self) -> None:
        # The daemon latches speaker identification off on the first
        # failure (``_identify_voice_speaker``); the identifier itself
        # must surface the broken model rather than hide it.
        from kiss.server.speech import MODEL_NAME, SPK_MODEL_NAME, ensure_model

        ensure_model(default_models_dir())
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            models_dir.mkdir()
            (models_dir / MODEL_NAME).symlink_to(default_models_dir() / MODEL_NAME)
            (models_dir / SPK_MODEL_NAME).mkdir()
            with self.assertRaises(Exception):
                SpeakerIdentifier(models_dir)


if __name__ == "__main__":
    unittest.main()
