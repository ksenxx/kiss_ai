# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests: the wake word keeps working after a
transcription.

Reproduces (and proves fixed) the bug where saying "Sorcar" after a
transcription did nothing until the mic was toggled off and on: the
gpt-audio translation call used to run on the audio loop, so a slow or
stalled HTTPS request blocked wake detection for its whole duration
(up to the OpenAI client's 600s default timeout).

Real audio, real speech models, real listener process, no mocks:

- The macOS TTS engine speaks "Sorcar", then a sentence, then "Sorcar"
  again into one WAV file.
- The real Python listener (``kiss.server.voice_wake`` — the
  process the VS Code extension spawns) streams that WAV.
- Its ``OPENAI_BASE_URL`` points at a local *stalling* HTTP server
  that accepts the translation request and never answers, emulating
  the hung network call that deafened the listener.

With the old blocking code the listener sits inside the stalled API
call when the second "Sorcar" plays and never fires; with the fix the
translation runs on a background thread, so the second WAKE is
reported even while the first transcription is still in flight.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]

HAVE_MAC_TTS = bool(shutil.which("say")) and bool(shutil.which("afconvert"))

STALL_TIMEOUT_SECONDS = 8.0


def _make_wake_speech_wake_wav(directory: Path) -> Path:
    """Synthesize: "Sorcar" — pause — a sentence — long pause — "Sorcar".

    The 1.5s pause after each "Sorcar" satisfies strict wake detection
    (~200ms quiet gate) and gives SpeechCapture leading silence; the
    2.5s pause after the sentence exceeds END_SILENCE_SECONDS (2.0s),
    ending the capture and starting the translation; the trailing
    1.5s pause lets the second wake fire on a partial result.
    """
    aiff = directory / "wake-speech-wake.aiff"
    wav = directory / "wake-speech-wake.wav"
    text = (
        "Sorcar [[slnc 1500]] "
        "please fix the parser bug in the compiler [[slnc 2500]] "
        "Sorcar [[slnc 1500]]"
    )
    subprocess.run(["say", text, "-o", str(aiff)], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
         str(aiff), str(wav)],
        check=True,
    )
    return wav


class StallingHttpServer:
    """A TCP server that accepts connections and never responds.

    Emulates a hung translation API: the OpenAI client connects, sends
    the request, and waits until its own timeout fires.  Every
    connection (the client retries once) is held open silently.
    """

    def __init__(self) -> None:
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = int(self._sock.getsockname()[1])
        self.connections = 0
        self._held: list[socket.socket] = []
        self._closing = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _addr = self._sock.accept()
            except OSError:
                return
            self.connections += 1
            self._held.append(conn)
            threading.Thread(
                target=self._drain, args=(conn,), daemon=True
            ).start()

    @staticmethod
    def _drain(conn: socket.socket) -> None:
        try:
            while conn.recv(65536):
                pass
        except OSError:
            pass

    def close(self) -> None:
        """Shut the listener and every held connection down."""
        self._closing = True
        try:
            self._sock.close()
        except OSError:
            pass
        for conn in self._held:
            try:
                conn.close()
            except OSError:
                pass
        self._thread.join(timeout=5)


class ClosingHttpServer:
    """A TCP server that accepts and immediately closes connections.

    Gives the OpenAI client a deterministic instant failure — unlike a
    "dead port", which another local process could bind between the
    test freeing it and the client connecting.
    """

    def __init__(self) -> None:
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = int(self._sock.getsockname()[1])
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _addr = self._sock.accept()
            except OSError:
                return
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        """Shut the listener down."""
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(timeout=5)


@unittest.skipUnless(HAVE_MAC_TTS, "requires macOS `say` and `afconvert`")
class TestWakeWordSurvivesStalledTranscription(unittest.TestCase):
    """Saying "Sorcar" again works while a transcription is stalled."""

    @pytest.mark.slow
    def test_second_wake_fires_during_stalled_transcription(self) -> None:
        stall = StallingHttpServer()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                wav = _make_wake_speech_wake_wav(Path(tmp))
                env = dict(os.environ)
                env.update(
                    {
                        "OPENAI_BASE_URL": (
                            f"http://127.0.0.1:{stall.port}/v1"
                        ),
                        "OPENAI_API_KEY": "sk-kiss-e2e-stall-test",
                        "KISS_VOICE_AUDIO_TIMEOUT": str(
                            STALL_TIMEOUT_SECONDS
                        ),
                    }
                )
                proc = subprocess.run(
                    [
                        "uv", "run", "python", "-m",
                        "kiss.server.voice_wake", "--wav", str(wav),
                    ],
                    cwd=PROJECT_ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=max(120, STALL_TIMEOUT_SECONDS * 6),
                )
        finally:
            stall.close()

        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        detail = f"stdout={lines!r}\nstderr={proc.stderr[-2000:]}"
        self.assertIn("READY", lines, msg=detail)

        wake_count = lines.count("WAKE")
        self.assertEqual(wake_count, 2, msg=detail)

        self.assertIn("TRANSCRIBING", lines, msg=detail)
        self.assertGreaterEqual(stall.connections, 1, msg=detail)

        second_wake_idx = [i for i, ln in enumerate(lines)
                           if ln == "WAKE"][1]
        no_speech_idxs = [i for i, ln in enumerate(lines)
                          if ln == "NO_SPEECH"]
        self.assertTrue(no_speech_idxs, msg=detail)
        self.assertLess(second_wake_idx, no_speech_idxs[0], msg=detail)

        self.assertEqual(len(no_speech_idxs), wake_count, msg=detail)

        self.assertEqual(proc.returncode, 0, msg=detail)


def _tts_pcm(directory: Path, name: str, text: str) -> bytes:
    """Return 16kHz mono s16le PCM of *text* spoken by macOS TTS."""
    import wave

    aiff = directory / f"{name}.aiff"
    wav = directory / f"{name}.wav"
    subprocess.run(["say", text, "-o", str(aiff)], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
         str(aiff), str(wav)],
        check=True,
    )
    with wave.open(str(wav), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _silence(seconds: float) -> bytes:
    """Return *seconds* of 16kHz mono s16le digital silence."""
    return b"\x00\x00" * int(seconds * 16000)


@unittest.skipUnless(HAVE_MAC_TTS, "requires macOS `say` and `afconvert`")
class TestWakeCooldownSurvivesCapture(unittest.TestCase):
    """The wake cooldown clock keeps ticking during speech capture.

    Regression test for a review finding: the detector's audio clock
    only advanced for blocks it decoded, so the multi-second speech
    capture froze it.  A "Sorcar" spoken right after a capture ended
    then looked like it was within 2s of the previous wake and its
    low-latency partial trigger was suppressed; when the user went on
    speaking their next command (so no exact-alias final result could
    rescue it), the wake was lost entirely — the wake word "stopped
    working" right after a transcription.
    """

    def test_wake_right_after_capture_with_continued_speech(self) -> None:
        from kiss.server.voice_wake import (
            DEFAULT_MODELS_DIR,
            WakeDetector,
            WakeSession,
            ensure_model,
        )

        with tempfile.TemporaryDirectory() as tmp:
            sorcar = _tts_pcm(Path(tmp), "sorcar", "Sorcar")
            speech = _tts_pcm(
                Path(tmp), "speech",
                "please fix the parser bug in the compiler now",
            )

        stream = (
            sorcar + _silence(1.0)
            + speech + _silence(2.2)
            + sorcar + _silence(0.4)
            + speech
        )

        refuse = ClosingHttpServer()
        overrides = {
            "OPENAI_BASE_URL": f"http://127.0.0.1:{refuse.port}/v1",
            "OPENAI_API_KEY": "sk-kiss-e2e-cooldown-test",
            "KISS_VOICE_AUDIO_TIMEOUT": "5.0",
        }
        saved = {key: os.environ.get(key) for key in overrides}
        os.environ.update(overrides)
        try:
            session = WakeSession(
                WakeDetector(ensure_model(DEFAULT_MODELS_DIR))
            )
            block = 2 * 800
            for start in range(0, len(stream), block):
                session.process(stream[start:start + block])
            session.finalize()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            refuse.close()

        self.assertEqual(session.wakes, 2)


@unittest.skipUnless(HAVE_MAC_TTS, "requires macOS `say` and `afconvert`")
class TestWatchdogSilenceAdvancesSession(unittest.TestCase):
    """A dead mic gap is counted as silence, not a frozen session."""

    def test_dead_gap_closes_capture_and_expires_cooldown(self) -> None:
        from kiss.server.voice_wake import (
            DEFAULT_MODELS_DIR,
            WakeDetector,
            WakeSession,
            ensure_model,
        )

        with tempfile.TemporaryDirectory() as tmp:
            sorcar = _tts_pcm(Path(tmp), "sorcar-dead-gap", "Sorcar")

        session = WakeSession(WakeDetector(ensure_model(DEFAULT_MODELS_DIR)))
        block = 2 * 800

        def feed(pcm: bytes) -> None:
            for start in range(0, len(pcm), block):
                session.process(pcm[start:start + block])

        feed(sorcar + _silence(1.0))
        self.assertEqual(session.wakes, 1)

        session.process_silence(5.0)

        feed(sorcar + _silence(1.0))
        session.finalize()

        self.assertEqual(session.wakes, 2)


class TestAudioTimeoutOverride(unittest.TestCase):
    """KISS_VOICE_AUDIO_TIMEOUT accepts only finite positive values."""

    def _timeout_with(self, raw: str | None) -> float:
        from kiss.server.voice_wake import audio_timeout_seconds

        saved = os.environ.get("KISS_VOICE_AUDIO_TIMEOUT")
        if raw is None:
            os.environ.pop("KISS_VOICE_AUDIO_TIMEOUT", None)
        else:
            os.environ["KISS_VOICE_AUDIO_TIMEOUT"] = raw
        try:
            return audio_timeout_seconds()
        finally:
            if saved is None:
                os.environ.pop("KISS_VOICE_AUDIO_TIMEOUT", None)
            else:
                os.environ["KISS_VOICE_AUDIO_TIMEOUT"] = saved

    def test_valid_override(self) -> None:
        self.assertEqual(self._timeout_with("7.5"), 7.5)

    def test_invalid_values_fall_back_to_default(self) -> None:
        from kiss.server.voice_wake import (
            DEFAULT_AUDIO_TIMEOUT_SECONDS,
        )

        for raw in (None, "", "abc", "-3", "0", "nan", "inf", "-inf"):
            self.assertEqual(
                self._timeout_with(raw),
                DEFAULT_AUDIO_TIMEOUT_SECONDS,
                msg=f"raw={raw!r}",
            )


if __name__ == "__main__":
    unittest.main()
