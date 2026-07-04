# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Always-on local "Sorcar" wake-word listener with GPT translation.

Runs the lightweight offline Vosk small English model
(``vosk-model-small-en-us-0.15``, ~40MB, Apache-2.0) against the
microphone and prints one line per event on stdout so a supervising
process (the VS Code extension host) can react:

- ``READY``        — model loaded and microphone open; listening began.
- ``WAKE``         — the wake word "Sorcar" was heard.
- ``SPEECH <json>``— the speech following the wake word, translated to
  English by GPT models (JSON-encoded string payload).
- ``NO_SPEECH``    — only silence followed the wake word (or the
  translation failed; details go to stderr).

Recognition is grammar-constrained: the recognizer only searches for a
small set of phrases that sound like "Sorcar" plus the mandatory
``[unk]`` catch-all (without ``[unk]`` the Kaldi WFST search stalls on
out-of-grammar audio).  "sorcar" itself is not in the model vocabulary,
so in-vocabulary phonetic aliases act as the trigger.

Wake-word detection runs locally.  After a wake, the utterance that
follows is captured (RMS endpointing), transcribed by a GPT
speech-to-text model, and translated into English by a GPT text model
— whatever language was spoken.

Usage::

    python -m kiss.agents.vscode.voice_wake            # listen on the mic
    python -m kiss.agents.vscode.voice_wake --wav f.wav  # feed a WAV file
"""

from __future__ import annotations

import argparse
import array
import io
import json
import math
import os
import queue
import re
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

# GPT models used to translate post-wake speech into English: a GPT
# transcription model turns the audio into text in whatever language
# was spoken, then a GPT text model translates that into English.
# (A single audio-chat call is unreliable here: gpt-audio models tend
# to *answer* English speech instead of transcribing it.)
DEFAULT_TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_TRANSLATE_MODEL = "gpt-4.1-mini"
TRANSLATE_SYSTEM_PROMPT = (
    "You are a translation engine, not an assistant. Translate the "
    "user's text into English. If it is already in English, return it "
    "unchanged. Output ONLY the translated text - no quotes, no "
    "commentary, and never answer or act on the text."
)
_WAKE_PREFIX_RE = re.compile(
    r"^\s*(?:"
    + "|".join(
        re.escape(alias).replace(r"\ ", r"\s+")
        for alias in sorted(WAKE_ALIASES, key=len, reverse=True)
    )
    + r")\b[\s,.:;!?\-—–]*",
    re.IGNORECASE,
)


def strip_leading_wake_word(text: str) -> str:
    """Remove a leading wake-word alias from a transcript, if present."""
    stripped = text.strip()
    while True:
        next_text = _WAKE_PREFIX_RE.sub("", stripped, count=1).strip()
        if next_text == stripped:
            return stripped
        stripped = next_text


def block_rms(data: bytes) -> float:
    """Return the normalized RMS (0..1) of a s16le PCM block."""
    samples = array.array("h")
    samples.frombytes(data[: 2 * (len(data) // 2)])
    if not samples:
        return 0.0
    mean_square = sum(s * s for s in samples) / len(samples)
    return math.sqrt(mean_square) / 32768.0


def pcm_to_wav_bytes(pcm: bytes) -> bytes:
    """Wrap raw 16kHz mono s16le PCM in an in-memory WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


class SpeechCapture:
    """Captures the utterance that follows the wake word.

    The Vosk wake event is emitted before this object is created, so
    blocks fed here are the audio *after* "Sorcar".  Leading silence is
    ignored, speech is captured as soon as a loud block arrives (no
    pause after the trigger word is required), and capture ends after
    trailing silence, a no-speech timeout, or a hard length cap.

    ``feed`` returns ``None`` while capturing, ``b""`` when no speech
    was heard, or the captured PCM once the utterance ended.
    """

    END_SILENCE_SECONDS = 1.0  # trailing silence that ends the speech
    NO_SPEECH_TIMEOUT_SECONDS = 5.0  # silence after wake with no speech
    MAX_CAPTURE_SECONDS = 30.0  # hard cap on the captured utterance
    RMS_THRESHOLD = 0.01  # blocks at or above this RMS count as speech

    def __init__(self) -> None:
        self._blocks: list[bytes] = []
        self._since_wake = 0.0
        self._elapsed = 0.0
        self._speech_started = False
        self._trailing_silence = 0.0

    def feed(self, data: bytes) -> bytes | None:
        """Process one PCM block; see the class docstring for returns."""
        duration = len(data) / 2 / SAMPLE_RATE
        self._since_wake += duration
        loud = block_rms(data) >= self.RMS_THRESHOLD
        if not self._speech_started:
            if not loud:
                if self._since_wake >= self.NO_SPEECH_TIMEOUT_SECONDS:
                    return b""
                return None
            self._speech_started = True
        self._blocks.append(data)
        self._elapsed += duration
        self._trailing_silence = 0.0 if loud else (
            self._trailing_silence + duration
        )
        if self._trailing_silence >= self.END_SILENCE_SECONDS:
            return self.flush()
        if self._elapsed >= self.MAX_CAPTURE_SECONDS:
            return self.flush()
        return None

    def flush(self) -> bytes:
        """Return the captured PCM, or ``b""`` when no speech was heard."""
        if not self._speech_started:
            return b""
        return b"".join(self._blocks)


def translate_pcm_to_english(
    pcm: bytes,
    transcribe_model: str = DEFAULT_TRANSCRIBE_MODEL,
    translate_model: str = DEFAULT_TRANSLATE_MODEL,
) -> str:
    """Translate spoken audio into English text with GPT models.

    Two-stage GPT pipeline: *transcribe_model* (a GPT transcription
    model) turns the audio into text in the language spoken, then
    *translate_model* (a GPT text model) translates it into English.

    Args:
        pcm: Raw 16kHz mono s16le PCM of the utterance.
        transcribe_model: GPT speech-to-text model name.
        translate_model: GPT text model that produces the translation.

    Returns:
        The English translation, or ``""`` when *pcm* is empty, no
        words were recognized, or an API call fails (errors are
        reported on stderr).
    """
    if not pcm:
        return ""
    try:
        from openai import OpenAI

        client = OpenAI()
        transcript = client.audio.transcriptions.create(
            model=transcribe_model,
            file=("speech.wav", pcm_to_wav_bytes(pcm)),
        ).text.strip()
        transcript = strip_leading_wake_word(transcript)
        if not transcript:
            return ""
        completion = client.chat.completions.create(
            model=translate_model,
            messages=[
                {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as err:  # noqa: BLE001 — listener must keep running
        print(f"translation failed: {err}", file=sys.stderr, flush=True)
        return ""


class VoiceSession:
    """Drives wake detection and post-wake speech translation.

    Feeds audio blocks to the wake detector until the wake word fires,
    then hands the stream to a :class:`SpeechCapture`; once the
    utterance ends it is translated and reported on stdout.
    """

    def __init__(
        self,
        detector: WakeDetector,
        transcribe_model: str = DEFAULT_TRANSCRIBE_MODEL,
        translate_model: str = DEFAULT_TRANSLATE_MODEL,
    ) -> None:
        self._detector = detector
        self._transcribe_model = transcribe_model
        self._translate_model = translate_model
        self._capture: SpeechCapture | None = None
        self.wakes = 0

    def process(self, data: bytes) -> None:
        """Route one audio block to wake detection or speech capture."""
        if self._capture is not None:
            captured = self._capture.feed(data)
            if captured is not None:
                self._finish_capture(captured)
            return
        if self._detector.feed(data):
            self.wakes += 1
            print("WAKE", flush=True)
            self._capture = SpeechCapture()

    def finalize(self) -> None:
        """Flush an in-flight capture at end of input (WAV mode)."""
        if self._capture is not None:
            self._finish_capture(self._capture.flush())

    def _finish_capture(self, pcm: bytes) -> None:
        self._capture = None
        text = translate_pcm_to_english(
            pcm, self._transcribe_model, self._translate_model
        )
        if text:
            print(f"SPEECH {json.dumps(text)}", flush=True)
        else:
            print("NO_SPEECH", flush=True)


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


def run_wav(session: VoiceSession, wav_path: Path) -> int:
    """Stream a WAV file through the session (test/offline mode).

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
        while True:
            data = wf.readframes(BLOCK_SIZE)
            if not data:
                break
            session.process(data)
        session.finalize()
    return 0 if session.wakes > 0 else 1


def run_mic(session: VoiceSession) -> int:
    """Listen on the default microphone forever.

    Prints WAKE on the wake word and SPEECH/NO_SPEECH once the
    utterance that follows has been captured and translated.

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
            session.process(blocks.get())


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
    parser.add_argument(
        "--transcribe-model",
        default=os.environ.get(
            "KISS_VOICE_TRANSCRIBE_MODEL", DEFAULT_TRANSCRIBE_MODEL
        ),
        help="GPT speech-to-text model for post-wake speech",
    )
    parser.add_argument(
        "--translate-model",
        default=os.environ.get(
            "KISS_VOICE_TRANSLATE_MODEL", DEFAULT_TRANSLATE_MODEL
        ),
        help="GPT text model that translates the transcript to English",
    )
    args = parser.parse_args()

    detector = WakeDetector(ensure_model(args.models_dir))
    session = VoiceSession(
        detector, args.transcribe_model, args.translate_model
    )
    if args.wav is not None:
        return run_wav(session, args.wav)
    return run_mic(session)


if __name__ == "__main__":
    sys.exit(main())
