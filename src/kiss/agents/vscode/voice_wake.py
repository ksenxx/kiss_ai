# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Always-on local "Sorcar" wake-word listener.

Runs the lightweight offline Vosk small English model
(``vosk-model-small-en-us-0.15``, ~40MB, Apache-2.0) against the
microphone and prints one line per event on stdout so a supervising
process (the VS Code extension host) can react:

- ``READY`` — model loaded and microphone open; listening has begun.
- ``WAKE``  — the wake word "Sorcar" was heard.

Recognition is grammar-constrained: the recognizer only searches for a
small set of phrases that sound like "Sorcar" plus the mandatory
``[unk]`` catch-all (without ``[unk]`` the Kaldi WFST search stalls on
out-of-grammar audio).  "sorcar" itself is not in the model vocabulary,
so in-vocabulary phonetic aliases act as the trigger.

Everything runs locally; no audio ever leaves the machine.

Usage::

    python -m kiss.agents.vscode.voice_wake            # listen on the mic
    python -m kiss.agents.vscode.voice_wake --wav f.wav  # feed a WAV file
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
import urllib.request
import wave
import zipfile
from pathlib import Path

WAKE_ALIASES = [
    "sorcar",
    "soccer",
    "circa",
    "sir car",
    "sore car",
    "so car",
    "saw car",
    "sar car",
]

MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_ZIP_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
DEFAULT_MODELS_DIR = Path.home() / ".kiss" / "models"
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # frames per audio block (250ms at 16kHz)
COOLDOWN_SECONDS = 2.0


def ensure_model(models_dir: Path) -> Path:
    """Return the local Vosk model directory, downloading it on first use.

    Args:
        models_dir: Directory that caches downloaded models.

    Returns:
        Path to the unpacked model directory.
    """
    model_dir = models_dir / MODEL_NAME
    if model_dir.is_dir():
        return model_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    zip_path = models_dir / f"{MODEL_NAME}.zip"
    print(f"downloading {MODEL_ZIP_URL} ...", file=sys.stderr, flush=True)
    tmp = zip_path.with_suffix(".tmp")
    urllib.request.urlretrieve(MODEL_ZIP_URL, tmp)
    tmp.replace(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(models_dir)
    zip_path.unlink(missing_ok=True)
    return model_dir


def matches_wake(text: str) -> bool:
    """Return True when *text* contains any wake-word alias."""
    normalized = " ".join(text.lower().split())
    return any(alias in normalized for alias in WAKE_ALIASES)


class WakeDetector:
    """Feeds raw 16kHz mono s16le audio into Vosk and detects the wake word.

    Checks both partial results (for low latency) and final results,
    with a cooldown so one utterance cannot fire twice.
    """

    def __init__(self, model_dir: Path) -> None:
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)
        grammar = json.dumps([*WAKE_ALIASES, "[unk]"])
        self._recognizer = KaldiRecognizer(
            Model(str(model_dir)), SAMPLE_RATE, grammar
        )
        self._last_wake = 0.0

    def feed(self, data: bytes) -> bool:
        """Process one audio block; return True when the wake word fired."""
        if self._recognizer.AcceptWaveform(data):
            text = json.loads(self._recognizer.Result()).get("text", "")
        else:
            text = json.loads(self._recognizer.PartialResult()).get(
                "partial", ""
            )
        if not matches_wake(text):
            return False
        now = time.monotonic()
        if now - self._last_wake < COOLDOWN_SECONDS:
            return False
        self._last_wake = now
        # Reset so leftover partial text cannot re-trigger after cooldown.
        self._recognizer.Reset()
        return True


def run_wav(detector: WakeDetector, wav_path: Path) -> int:
    """Stream a WAV file through the detector (test/offline mode).

    The file must be 16kHz mono 16-bit PCM — the same format the
    microphone path uses.

    Returns:
        Process exit code: 0 when the wake word was detected, 1 otherwise.
    """
    with wave.open(str(wav_path), "rb") as wf:
        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getframerate() != SAMPLE_RATE
        ):
            print(
                f"error: {wav_path} must be {SAMPLE_RATE}Hz mono 16-bit PCM, "
                f"got {wf.getframerate()}Hz {wf.getnchannels()}ch "
                f"{8 * wf.getsampwidth()}-bit",
                file=sys.stderr,
                flush=True,
            )
            return 2
        print("READY", flush=True)
        detected = False
        while True:
            data = wf.readframes(BLOCK_SIZE)
            if not data:
                break
            if detector.feed(data):
                print("WAKE", flush=True)
                detected = True
    return 0 if detected else 1


def run_mic(detector: WakeDetector) -> int:
    """Listen on the default microphone forever, printing WAKE events.

    Returns:
        Process exit code (only reached on stream failure).
    """
    import sounddevice

    blocks: queue.Queue[bytes] = queue.Queue()

    def on_audio(
        indata: bytes, _frames: int, _time_info: object, _status: object
    ) -> None:
        # ``indata`` is a CFFI buffer; copy it before the driver reuses it.
        blocks.put(bytes(indata))

    with sounddevice.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=on_audio,
    ):
        print("READY", flush=True)
        while True:
            if detector.feed(blocks.get()):
                print("WAKE", flush=True)


def main() -> int:
    """CLI entry point for the wake-word listener."""
    parser = argparse.ArgumentParser(description="Sorcar wake-word listener")
    parser.add_argument(
        "--wav",
        type=Path,
        default=None,
        help="Read audio from a 16kHz mono 16-bit WAV file instead of the mic",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory caching downloaded Vosk models",
    )
    args = parser.parse_args()

    detector = WakeDetector(ensure_model(args.models_dir))
    if args.wav is not None:
        return run_wav(detector, args.wav)
    return run_mic(detector)


if __name__ == "__main__":
    sys.exit(main())
