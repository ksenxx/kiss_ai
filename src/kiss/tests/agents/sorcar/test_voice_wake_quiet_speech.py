# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: softly spoken "Sorcar" must wake the listener.

Reproduces a real bug: users had to speak the wake word LOUDLY to
trigger it.  Quiet, breathy speech carries a soft onset that the
grammar-constrained recognizer decodes as a brief leading ``[unk]``
before the alias — the macOS "Whisper" voice deterministically decodes
as ``[unk] sore car`` with a ~60ms ``[unk]`` span — and the strict
whole-utterance matcher rejected it, so only a loud, clean ``sore car``
decode ever woke the listener.

Real audio and the real Vosk model, no mocks:

- ``test_soft_breathy_sorcar_wakes_every_time`` synthesizes "Sorcar"
  three times with the macOS *Whisper* voice (soft breathy speech) and
  streams the audio through the real :class:`WakeDetector` at the
  default sensitivity: every utterance must wake.  Before the fix the
  first utterance (the ``[unk] sore car`` decode) was rejected.

- ``test_sentences_with_word_prefixes_still_never_wake`` streams
  ordinary sentences whose spoken-word prefixes decode to long
  ``[unk]`` spans (~0.5s and up, measured) through the same detector
  and asserts none of them wakes: the leading-noise acceptance must
  not weaken the anti-false-positive design.

- ``TestLeadingNoiseGate`` pins the gate's contract with the exact
  word timings measured live from Vosk decodes.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from kiss.agents.vscode.voice_wake import (
    BLOCK_SIZE,
    DEFAULT_MODELS_DIR,
    MAX_LEADING_NOISE_SECONDS,
    WakeDetector,
    ensure_model,
    wake_with_leading_noise,
)

HAVE_MAC_TTS = bool(shutil.which("say")) and bool(shutil.which("afconvert"))
HAVE_WHISPER_VOICE = HAVE_MAC_TTS and "Whisper" in subprocess.run(
    ["say", "-v", "?"], capture_output=True, text=True
).stdout


def _tts_pcm(directory: Path, name: str, text: str, voice: str | None) -> bytes:
    """Synthesize *text* to 16kHz mono s16le PCM with the macOS TTS."""
    aiff = directory / f"{name}.aiff"
    wav = directory / f"{name}.wav"
    cmd = ["say"] + (["-v", voice] if voice else []) + [text, "-o", str(aiff)]
    subprocess.run(cmd, check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
         str(aiff), str(wav)],
        check=True,
    )
    with wave.open(str(wav), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _count_wakes(pcm: bytes) -> int:
    """Stream PCM through a real default-sensitivity WakeDetector."""
    detector = WakeDetector(ensure_model(DEFAULT_MODELS_DIR))
    return sum(
        1
        for start in range(0, len(pcm), 2 * BLOCK_SIZE)
        if detector.feed(pcm[start:start + 2 * BLOCK_SIZE])
    )


class TestLeadingNoiseGate(unittest.TestCase):
    """wake_with_leading_noise accepts brief noise, nothing more."""

    def test_measured_soft_sorcar_decode_wakes(self) -> None:
        # The exact decode of whispered "Sorcar" measured live: a 60ms
        # breathy-onset [unk] then the alias.  This is the bug's shape.
        self.assertTrue(
            wake_with_leading_noise([
                {"word": "[unk]", "start": 0.0, "end": 0.06, "conf": 0.51},
                {"word": "sore", "start": 0.07, "end": 0.33, "conf": 1.0},
                {"word": "car", "start": 0.39, "end": 0.72, "conf": 1.0},
            ])
        )

    def test_long_unk_prefix_rejects(self) -> None:
        # Spoken-word prefixes decode to [unk] spans of ~0.5s and up
        # ("hey there" 0.61s measured); they must stay rejected.
        self.assertFalse(
            wake_with_leading_noise([
                {"word": "[unk]", "start": 0.0, "end": 0.61},
                {"word": "sore", "start": 0.65, "end": 0.95},
                {"word": "car", "start": 0.98, "end": 1.3},
            ])
        )

    def test_multiple_short_unks_sum_against_the_budget(self) -> None:
        words = [
            {"word": "[unk]", "start": 0.0, "end": 0.2},
            {"word": "[unk]", "start": 0.25, "end": 0.45},
            {"word": "sir", "start": 0.5, "end": 0.8},
            {"word": "car", "start": 0.85, "end": 1.2},
        ]
        self.assertGreater(0.4, MAX_LEADING_NOISE_SECONDS)
        self.assertFalse(wake_with_leading_noise(words))
        words[1]["end"] = 0.3  # total 0.25s, within the budget
        self.assertTrue(wake_with_leading_noise(words))

    def test_anything_after_the_alias_rejects(self) -> None:
        # "yes sir the car is ready" decodes [unk] sir car [unk]; the
        # utterance does not END with the alias so it must not wake.
        self.assertFalse(
            wake_with_leading_noise([
                {"word": "[unk]", "start": 0.0, "end": 0.21},
                {"word": "sir", "start": 0.24, "end": 0.51},
                {"word": "car", "start": 0.66, "end": 0.93},
                {"word": "[unk]", "start": 0.93, "end": 1.35},
            ])
        )

    def test_non_alias_tail_rejects(self) -> None:
        # "he wrecked his car" decodes [unk] car; "car" alone is not
        # an alias.
        self.assertFalse(
            wake_with_leading_noise([
                {"word": "[unk]", "start": 0.0, "end": 0.06},
                {"word": "car", "start": 0.1, "end": 0.45},
            ])
        )

    def test_alias_without_noise_prefix_is_not_this_gates_job(self) -> None:
        # A clean alias is matches_wake's case; this gate demands at
        # least one leading [unk] so it never widens exact matching.
        self.assertFalse(
            wake_with_leading_noise([
                {"word": "sore", "start": 0.0, "end": 0.3},
                {"word": "car", "start": 0.35, "end": 0.7},
            ])
        )

    def test_missing_timings_reject(self) -> None:
        # Without numeric start/end the noise span is unknowable; the
        # gate only opens on evidence.
        self.assertFalse(
            wake_with_leading_noise([
                {"word": "[unk]"},
                {"word": "sore", "start": 0.07, "end": 0.33},
                {"word": "car", "start": 0.39, "end": 0.72},
            ])
        )

    def test_empty_and_missing_word_lists_reject(self) -> None:
        self.assertFalse(wake_with_leading_noise(None))
        self.assertFalse(wake_with_leading_noise([]))
        self.assertFalse(wake_with_leading_noise([
            {"word": "[unk]", "start": 0.0, "end": 0.06},
        ]))


@unittest.skipUnless(
    HAVE_WHISPER_VOICE,
    "requires macOS `say` (with the Whisper voice) and `afconvert`",
)
class TestQuietSpeechWake(unittest.TestCase):
    """Soft breathy 'Sorcar' wakes; word-prefixed sentences never do."""

    def test_soft_breathy_sorcar_wakes_every_time(self) -> None:
        # The Whisper voice renders soft breathy speech; its first
        # utterance decodes as '[unk] sore car' (60ms breathy-onset
        # [unk], deterministic).  Every one of the three utterances
        # must wake — before the fix only two did, which is exactly
        # the "I have to speak loudly" bug.
        with tempfile.TemporaryDirectory() as tmp:
            pcm = _tts_pcm(
                Path(tmp),
                "soft_sorcar",
                "Sorcar [[slnc 1500]] Sorcar [[slnc 1500]] "
                "Sorcar [[slnc 1500]]",
                "Whisper",
            )
        self.assertEqual(_count_wakes(pcm), 3)

    def test_sentences_with_word_prefixes_still_never_wake(self) -> None:
        # Spoken words before an alias-sounding tail decode to long
        # [unk] spans; the leading-noise budget must reject them all.
        with tempfile.TemporaryDirectory() as tmp:
            pcm = _tts_pcm(
                Path(tmp),
                "sentences",
                " [[slnc 800]] ".join([
                    "he wrecked his car",
                    "she sells a car",
                    "yes sir the car is ready",
                    "hey there sorcar",
                ]),
                None,
            )
        self.assertEqual(_count_wakes(pcm), 0)


if __name__ == "__main__":
    unittest.main()
