# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the "Sorcar" voice wake word.

Real audio, real speech models, no mocks:

- ``test_wake_from_wav`` speaks "Sorcar" with the macOS TTS engine and
  streams the audio through the actual Python wake-word listener
  (``kiss.agents.vscode.voice_wake``) used by the VS Code extension.

- ``test_wake_word_mic_browser`` boots the real remote-access web
  server, opens the chat page in Chromium whose *microphone* is fed the
  spoken "Sorcar" audio (Chromium's fake capture device plays the WAV
  through the getUserMedia stack), and asserts that the in-page
  vosk-browser listener fires the wake indicator (the transient
  ``voice-triggered`` flash on the mic button) while never typing the
  literal word "sorcar" into the task input textbox.
"""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

from kiss.agents.vscode.web_server import RemoteAccessServer

PROJECT_ROOT = Path(__file__).resolve().parents[5]

HAVE_MAC_TTS = bool(shutil.which("say")) and bool(shutil.which("afconvert"))


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
                    "kiss.agents.vscode.voice_wake", "--wav", str(wav),
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
class TestVoiceWakeWordMicBrowser(unittest.TestCase):
    """Speaking 'Sorcar' into the (fake-device) microphone of a real
    Chromium visiting the real web app fires the wake indicator and
    never types the literal word 'sorcar' into the input."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.wav = _make_sorcar_wav(self.tmpdir)
        self.port = _free_port()

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self.loop_thread.start()

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            url_file=self.tmpdir / "remote-url.json",
            uds_path=self.tmpdir / "sorcar.sock",
        )
        asyncio.run_coroutine_threadsafe(
            self.server.start_async(), self.loop,
        ).result(timeout=60)

    def tearDown(self) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                self.server.stop_async(), self.loop,
            ).result(timeout=30)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop_thread.join(timeout=10)
            self.loop.close()
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_wake_word_mic_browser(self) -> None:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=[
                    "--use-fake-device-for-media-stream",
                    "--use-fake-ui-for-media-stream",
                    f"--use-file-for-fake-audio-capture={self.wav}",
                    "--autoplay-policy=no-user-gesture-required",
                    "--mute-audio",
                ],
            )
            try:
                context = browser.new_context(ignore_https_errors=True)
                # Pre-enable the voice toggle so voice.js starts the
                # microphone pipeline on page load.
                context.add_init_script(
                    "localStorage.setItem('kissVoiceEnabled', '1');"
                )
                page = context.new_page()
                page.goto(
                    f"https://127.0.0.1:{self.port}/",
                    wait_until="load",
                    timeout=60_000,
                )
                # Record every value the task input ever takes so we
                # can prove the literal wake word is never typed there.
                page.evaluate(
                    "window.__seenInputValues = [];"
                    "const inp = document.getElementById('task-input');"
                    "if (inp) inp.addEventListener('input', () =>"
                    " window.__seenInputValues.push(inp.value));"
                )
                # Generous timeout: on a cold cache the server first
                # downloads the ~40MB Vosk model archive, then the
                # browser fetches and unpacks it inside the WASM
                # recognizer worker before listening starts.  The wake
                # event flashes the transient 'voice-triggered' class
                # on the mic button for 600ms; wait_for_function polls
                # every animation frame, so the flash cannot be missed.
                page.wait_for_function(
                    "document.getElementById('voice-btn')"
                    " && document.getElementById('voice-btn')"
                    ".classList.contains('voice-triggered')",
                    timeout=300_000,
                )
                value = page.evaluate(
                    "document.getElementById('task-input').value"
                )
                self.assertEqual(value, "")
                seen = page.evaluate("window.__seenInputValues")
                self.assertEqual(seen, [])
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
