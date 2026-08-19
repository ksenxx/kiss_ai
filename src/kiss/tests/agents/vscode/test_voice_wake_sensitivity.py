# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Chromium slider test that stays behind from
``kiss.tests.agents.vscode.test_voice_wake_sensitivity`` (now
``kiss.tests.server.test_voice_wake_sensitivity``): a real Chromium
visits the served chat page, so the in-page listener in
``media/voice.js`` (bundled under ``src/kiss/agents/vscode/media``)
runs for real.  At a stored strict sensitivity (50) the wake indicator
never fires; dragging the settings-panel sensitivity slider to 85
makes the very same audio wake the listener and persists the value in
localStorage.  A fresh profile defaults the slider to 80.  The
Python-listener tests, which depend only on ``kiss.server.voice_wake``,
moved to tests/server; the TTS/port helpers are imported back from
there.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

import pytest

from kiss.tests.server.test_voice_wake_sensitivity import (
    HAVE_MAC_TTS,
    _free_port,
    _say_wav,
)


@unittest.skipUnless(HAVE_MAC_TTS, "requires macOS `say` and `afconvert`")
class TestSensitivitySliderBrowser(unittest.TestCase):
    """Dragging the settings-panel slider changes what real spoken
    audio wakes the in-browser (remote webapp) listener."""

    def setUp(self) -> None:
        from kiss.server.web_server import RemoteAccessServer

        self.tmpdir = Path(tempfile.mkdtemp())
        self.wav = _say_wav(
            self.tmpdir, "hey", "hey there [[slnc 300]] Sorcar [[slnc 1500]]"
        )
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

    @pytest.mark.slow
    def test_slider_changes_browser_wake_sensitivity(self) -> None:
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
                    "localStorage.setItem('kissVoiceSensitivity', '50');"
                )
                page = context.new_page()
                page.goto(
                    f"https://127.0.0.1:{self.port}/",
                    wait_until="load",
                    timeout=60_000,
                )
                slider = page.evaluate(
                    "(() => { const s = document.getElementById("
                    "'cfg-voice-sensitivity'); return s &&"
                    " {value: s.value, min: s.min, max: s.max}; })()"
                )
                self.assertIsNotNone(slider, "sensitivity slider missing")
                self.assertEqual(slider["value"], "50")
                self.assertEqual(slider["min"], "0")
                self.assertEqual(slider["max"], "100")
                page.evaluate(
                    "window.__sawWake = false;"
                    "const btn = document.getElementById('voice-btn');"
                    "new MutationObserver(() => {"
                    "  if (btn.classList.contains('voice-triggered'))"
                    "    window.__sawWake = true;"
                    "}).observe(btn, {attributes: true});"
                )
                page.wait_for_function(
                    "document.getElementById('voice-btn')"
                    ".classList.contains('voice-listening')",
                    timeout=300_000,
                )
                page.wait_for_timeout(10_000)
                self.assertFalse(
                    page.evaluate("window.__sawWake"),
                    "audio must not wake at sensitivity 50",
                )
                page.evaluate(
                    "const s = document.getElementById("
                    "'cfg-voice-sensitivity');"
                    "s.value = '85';"
                    "s.dispatchEvent(new Event('input', {bubbles: true}));"
                )
                self.assertEqual(
                    page.evaluate(
                        "localStorage.getItem('kissVoiceSensitivity')"
                    ),
                    "85",
                    "slider must persist the sensitivity",
                )
                page.wait_for_function(
                    "window.__sawWake === true", timeout=120_000
                )
                fresh = browser.new_context(ignore_https_errors=True)
                fresh_page = fresh.new_page()
                fresh_page.goto(
                    f"https://127.0.0.1:{self.port}/",
                    wait_until="load",
                    timeout=60_000,
                )
                self.assertEqual(
                    fresh_page.evaluate(
                        "document.getElementById("
                        "'cfg-voice-sensitivity').value"
                    ),
                    "80",
                    "a fresh profile must default the slider to 80",
                )
                fresh.close()
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
