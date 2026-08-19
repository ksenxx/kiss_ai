# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end real-voice tests for the wake-word sensitivity setting.

Real audio (macOS TTS), real speech models, no mocks.  The sensitivity
slider (0..100, default 80) must ACTUALLY change how eagerly the
"Sorcar" wake word fires, in both listener implementations:

- The Python listener (``kiss.server.voice_wake``) used by the
  VS Code extension host accepts ``--sensitivity N``:

  * Spoken "Sorcar" wakes at the default sensitivity.
  * Spoken "soccer" force-fits onto an alias with word confidences of
    only ~0.55-0.69 (measured live); it wakes at the default
    sensitivity but a LOW sensitivity raises the confidence gate above
    the force-fit scores and rejects it.
  * Spoken "hey there Sorcar" decodes to ``[unk] sore car`` — the
    alias at the END of the utterance.  Strict whole-utterance
    matching rejects it at a LOW sensitivity (< 75), while the
    default (80) accepts a trailing alias and wakes.
  * Ordinary sentences containing alias-sounding words never wake at
    the default sensitivity.

The browser (remote webapp) slider test, which executes the in-page
``media/voice.js`` listener in a real Chromium, stays behind in
``kiss.tests.agents.vscode.test_voice_wake_sensitivity``; everything
here depends only on ``kiss.server.voice_wake``, which is why the
file moved from tests/agents/vscode to tests/server.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]

HAVE_MAC_TTS = bool(shutil.which("say")) and bool(shutil.which("afconvert"))


def _say_wav(directory: Path, name: str, text: str) -> Path:
    """Synthesize *text* as a 16kHz mono 16-bit WAV via macOS TTS."""
    aiff = directory / f"{name}.aiff"
    wav = directory / f"{name}.wav"
    subprocess.run(["say", text, "-o", str(aiff)], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
         str(aiff), str(wav)],
        check=True,
    )
    return wav


def _run_listener(
    wav: Path, *extra_args: str
) -> subprocess.CompletedProcess[str]:
    """Stream *wav* through the real Python wake listener CLI."""
    return subprocess.run(
        [
            "uv", "run", "python", "-m",
            "kiss.server.voice_wake", "--wav", str(wav),
            *extra_args,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


@unittest.skipUnless(HAVE_MAC_TTS, "requires macOS `say` and `afconvert`")
class TestSensitivityCliRealVoice(unittest.TestCase):
    """--sensitivity changes what real spoken audio wakes the listener."""

    tmpdir: Path
    sorcar_wav: Path
    soccer_wav: Path
    hey_wav: Path
    sentences_wav: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = Path(tempfile.mkdtemp())
        cls.sorcar_wav = _say_wav(
            cls.tmpdir,
            "sorcar",
            "Sorcar [[slnc 1500]] Sorcar [[slnc 1500]] Sorcar [[slnc 1500]]",
        )
        cls.soccer_wav = _say_wav(
            cls.tmpdir,
            "soccer",
            "soccer [[slnc 1500]] soccer [[slnc 1500]] soccer [[slnc 1500]]",
        )
        cls.hey_wav = _say_wav(
            cls.tmpdir,
            "hey",
            "hey there [[slnc 300]] Sorcar [[slnc 1500]] "
            "hey there [[slnc 300]] Sorcar [[slnc 1500]]",
        )
        cls.sentences_wav = _say_wav(
            cls.tmpdir,
            "sentences",
            " [[slnc 800]] ".join(
                [
                    "I watched the soccer game yesterday with my friends",
                    "yes sir the car is ready to go",
                    "soccer is my favorite sport",
                    "so called experts say otherwise",
                ]
            ),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_sorcar_wakes_at_default_sensitivity(self) -> None:
        proc = _run_listener(self.sorcar_wav, "--sensitivity", "80")
        lines = proc.stdout.split()
        self.assertIn("READY", lines, msg=proc.stderr[-2000:])
        self.assertIn("WAKE", lines, msg=proc.stderr[-2000:])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])

    def test_low_sensitivity_rejects_sound_alike(self) -> None:
        wakes = _run_listener(self.soccer_wav, "--sensitivity", "80")
        self.assertIn("WAKE", wakes.stdout.split(),
                      msg=wakes.stderr[-2000:])
        self.assertEqual(wakes.returncode, 0, msg=wakes.stderr[-2000:])

        rejects = _run_listener(self.soccer_wav, "--sensitivity", "10")
        self.assertIn("READY", rejects.stdout.split(),
                      msg=rejects.stderr[-2000:])
        self.assertNotIn("WAKE", rejects.stdout.split(),
                         msg=rejects.stdout)
        self.assertEqual(rejects.returncode, 1, msg=rejects.stdout)

    @pytest.mark.slow
    def test_high_sensitivity_wakes_on_trailing_alias(self) -> None:
        strict = _run_listener(self.hey_wav, "--sensitivity", "50")
        self.assertIn("READY", strict.stdout.split(),
                      msg=strict.stderr[-2000:])
        self.assertNotIn("WAKE", strict.stdout.split(), msg=strict.stdout)
        self.assertEqual(strict.returncode, 1, msg=strict.stdout)

        eager = _run_listener(self.hey_wav, "--sensitivity", "80")
        self.assertIn("WAKE", eager.stdout.split(),
                      msg=eager.stderr[-2000:])
        self.assertEqual(eager.returncode, 0, msg=eager.stderr[-2000:])

    def test_sentences_never_wake_at_default_sensitivity(self) -> None:
        proc = _run_listener(self.sentences_wav, "--sensitivity", "80")
        lines = proc.stdout.split()
        self.assertIn("READY", lines, msg=proc.stderr[-2000:])
        self.assertNotIn("WAKE", lines, msg=proc.stdout)
        self.assertEqual(proc.returncode, 1, msg=proc.stdout)

    def test_invalid_sensitivity_is_rejected(self) -> None:
        for bad in ("150", "-5", "abc", "nan"):
            proc = _run_listener(self.sorcar_wav, "--sensitivity", bad)
            self.assertEqual(
                proc.returncode, 2,
                msg=f"--sensitivity {bad}: {proc.stderr[-500:]}",
            )


def _free_port() -> int:
    """Return an OS-assigned free TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    unittest.main()
