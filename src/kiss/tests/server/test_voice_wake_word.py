# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the "Sorcar" voice wake word.

Real audio, real speech models, no mocks:

- ``test_wake_from_wav`` speaks "Sorcar" with the macOS TTS engine and
  streams the audio through the actual Python wake-word listener
  (``kiss.server.voice_wake``) used by the VS Code extension.

- ``test_no_wake_from_alias_sentences`` speaks everyday sentences that
  contain wake-alias-sounding words ("soccer", "circa", "sir ... car",
  "so called") mid-sentence or at the start of continuous speech and
  asserts the listener never fires: detection must not be so
  sensitive that ordinary conversation wakes it.

The Chromium mic-browser test (which executes the in-page
``media/voice.js`` listener) stays behind in
``kiss.tests.agents.vscode.test_voice_wake_word``; everything here
depends only on ``kiss.server.voice_wake``, which is why the file
moved from tests/agents/vscode to tests/server.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

from kiss.server.voice_wake import matches_wake, words_confident

PROJECT_ROOT = Path(__file__).resolve().parents[4]

HAVE_MAC_TTS = bool(shutil.which("say")) and bool(shutil.which("afconvert"))


class TestWakeMatchingStrictness(unittest.TestCase):
    """The wake predicates reject everything but an isolated, confident
    alias — the over-sensitivity fix in its distilled form."""

    def test_exact_alias_matches(self) -> None:
        self.assertTrue(matches_wake("sore car"))
        self.assertTrue(matches_wake("  Sore   Car  "))
        self.assertTrue(matches_wake("sir car"))

    def test_common_words_are_not_aliases(self) -> None:
        self.assertFalse(matches_wake("soccer"))
        self.assertFalse(matches_wake("circa"))
        self.assertFalse(matches_wake("so car"))
        self.assertFalse(matches_wake("saw car"))

    def test_alias_in_context_never_matches(self) -> None:
        self.assertFalse(matches_wake("[unk] sir car [unk]"))
        self.assertFalse(matches_wake("[unk] sore car [unk]"))
        self.assertFalse(matches_wake("sir car [unk]"))
        self.assertFalse(matches_wake("[unk] sar car"))
        self.assertFalse(matches_wake("sore car [unk]"))
        self.assertFalse(matches_wake("so"))
        self.assertFalse(matches_wake(""))

    def test_word_confidence_gate(self) -> None:
        human = [{"conf": 0.53, "word": "sir"}, {"conf": 1.0, "word": "car"}]
        garbage = [{"conf": 1.0, "word": "sore"}, {"conf": 0.2, "word": "car"}]
        self.assertTrue(words_confident(human))
        self.assertFalse(words_confident(garbage))
        self.assertTrue(words_confident(None))
        self.assertTrue(words_confident([]))
        self.assertTrue(words_confident([{"word": "sore"}]))
        self.assertTrue(words_confident([{"conf": 250.0, "word": "sore"}]))


def _make_sorcar_wav(directory: Path) -> Path:
    """Synthesize a 16kHz mono 16-bit WAV that says "Sorcar" three times."""
    aiff = directory / "sorcar.aiff"
    wav = directory / "sorcar.wav"
    subprocess.run(
        [
            "say",
            "Sorcar [[slnc 1200]] Sorcar [[slnc 1200]] Sorcar [[slnc 1200]]",
            "-o",
            str(aiff),
        ],
        check=True,
    )
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
         str(aiff), str(wav)],
        check=True,
    )
    return wav


def _make_alias_sentences_wav(directory: Path) -> Path:
    """Synthesize sentences with alias-sounding words that must not wake.

    Covers both mid-sentence alias words (decoded by the grammar as
    ``[unk] soccer [unk]`` etc.) and utterances that *start* with an
    alias-sounding word followed by continuous speech.
    """
    aiff = directory / "sentences.aiff"
    wav = directory / "sentences.wav"
    text = " [[slnc 800]] ".join(
        [
            "I watched the soccer game yesterday with my friends",
            "yes sir the car is ready to go",
            "that painting is from circa nineteen twenty",
            "soccer is my favorite sport",
            "so called experts say otherwise",
            "sir can you help me please",
            "I am so careful when I drive the car",
        ]
    )
    subprocess.run(["say", text, "-o", str(aiff)], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
         str(aiff), str(wav)],
        check=True,
    )
    return wav


def _free_port() -> int:
    """Return an OS-assigned free TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@unittest.skipUnless(HAVE_MAC_TTS, "requires macOS `say` and `afconvert`")
class TestVoiceWakeFromWav(unittest.TestCase):
    """The Python wake listener detects spoken 'Sorcar' in real audio."""

    def test_wake_from_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav = _make_sorcar_wav(Path(tmp))
            proc = subprocess.run(
                [
                    "uv", "run", "python", "-m",
                    "kiss.server.voice_wake", "--wav", str(wav),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=600,
            )
        lines = proc.stdout.split()
        self.assertIn("READY", lines, msg=proc.stderr[-2000:])
        self.assertIn("WAKE", lines, msg=proc.stderr[-2000:])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])


@unittest.skipUnless(HAVE_MAC_TTS, "requires macOS `say` and `afconvert`")
class TestNoFalseWakeFromWav(unittest.TestCase):
    """Ordinary speech containing alias-sounding words never wakes."""

    def test_no_wake_from_alias_sentences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav = _make_alias_sentences_wav(Path(tmp))
            proc = subprocess.run(
                [
                    "uv", "run", "python", "-m",
                    "kiss.server.voice_wake", "--wav", str(wav),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=600,
            )
        lines = proc.stdout.split()
        self.assertIn("READY", lines, msg=proc.stderr[-2000:])
        self.assertNotIn("WAKE", lines, msg=proc.stdout)
        self.assertEqual(proc.returncode, 1, msg=proc.stdout)


if __name__ == "__main__":
    unittest.main()
