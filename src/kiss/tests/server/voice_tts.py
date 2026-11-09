# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""macOS text-to-speech helpers shared by the real-voice tests.

The browser wake-word tests under ``kiss.tests.webapp`` feed a spoken
"Hey Sorcar" into Chromium's fake microphone; the WAV comes from the
macOS ``say`` + ``afconvert`` pair, so those tests skip unless
:data:`HAVE_MAC_TTS`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

HAVE_MAC_TTS = bool(shutil.which("say")) and bool(shutil.which("afconvert"))


def _say_wav(
    directory: Path, name: str, text: str, voice: str | None = None
) -> Path:
    """Synthesize *text* as a 16kHz mono 16-bit WAV via macOS TTS."""
    aiff = directory / f"{name}.aiff"
    wav = directory / f"{name}.wav"
    voice_args = ["-v", voice] if voice else []
    subprocess.run(["say", *voice_args, text, "-o", str(aiff)], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
         str(aiff), str(wav)],
        check=True,
    )
    return wav


def _make_sorcar_wav(directory: Path, phrase: str = "Hey Sorcar") -> Path:
    """Synthesize a 16kHz mono 16-bit WAV that says *phrase* three times."""
    return _say_wav(
        directory, "sorcar", " [[slnc 1200]] ".join([phrase] * 3) + " [[slnc 1200]]",
    )
