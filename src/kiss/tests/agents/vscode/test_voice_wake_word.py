# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Chromium mic-browser wake-word test that stays behind from
``kiss.tests.agents.vscode.test_voice_wake_word`` (now
``kiss.tests.server.test_voice_wake_word``): a real Chromium visits
the served chat page, so the in-page vosk listener in
``media/voice.js`` (bundled under ``src/kiss/agents/vscode/media``)
runs for real.  The Python-listener tests, which depend only on
``kiss.server.voice_wake``, moved to tests/server; their WAV/TTS
helpers are imported back from there.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.test_voice_wake_word import (
    HAVE_MAC_TTS,
    _free_port,
    _make_sorcar_wav,
)


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
                context.add_init_script(
                    "localStorage.setItem('kissVoiceEnabled', '1');"
                )
                page = context.new_page()
                page.goto(
                    f"https://127.0.0.1:{self.port}/",
                    wait_until="load",
                    timeout=60_000,
                )
                page.evaluate(
                    "window.__seenInputValues = [];"
                    "const inp = document.getElementById('task-input');"
                    "if (inp) inp.addEventListener('input', () =>"
                    " window.__seenInputValues.push(inp.value));"
                )
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
