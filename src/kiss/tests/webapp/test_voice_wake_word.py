# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Chromium mic-browser wake-word test: a real Chromium visits the
served chat page, so the in-page vosk listener in ``media/voice.js``
(bundled under ``src/kiss/webapp/media``) runs for real.  The WAV/TTS
helpers come from ``kiss.tests.server.voice_tts``.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.server.conftest import find_free_port
from kiss.tests.server.voice_tts import HAVE_MAC_TTS, _make_sorcar_wav, _say_wav


@unittest.skipUnless(HAVE_MAC_TTS, "requires macOS `say` and `afconvert`")
class TestVoiceWakeWordMicBrowser(unittest.TestCase):
    """Speaking 'Hey Sorcar' into the (fake-device) microphone of a real
    Chromium visiting the real web app fires the wake indicator and
    never types the literal wake phrase into the input."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.wav = _make_sorcar_wav(self.tmpdir)
        self.port = find_free_port()

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self.loop_thread.start()

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            url_file=self.tmpdir / "remote-url.json",
            local_endpoint_file=self.tmpdir / "sorcar-local.json",
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
            browser = _launch_chromium_with_mic(pw, self.wav)
            try:
                context = browser.new_context(ignore_https_errors=True)
                context.add_init_script(
                    "localStorage.setItem('kissVoiceEnabled', '1');"
                )
                page = context.new_page()
                goto_retrying_network_change(
                    page,
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

    def test_wake_after_30_seconds_of_listening(self) -> None:
        """Chrome swaps an audio sink that has played 30 s of exact
        silence for a fake one that ran the page's audio processor at
        under half speed, so the recognizers got chopped audio and a
        wake phrase said after the first 30 s of listening was missed.
        The phrase here first plays 35 s after the mic opens."""
        from playwright.sync_api import sync_playwright

        wav = _say_wav(
            self.tmpdir,
            "late",
            "[[slnc 35000]] "
            + " [[slnc 1200]] ".join(["Hey Sorcar"] * 3)
            + " [[slnc 1200]]",
        )
        with sync_playwright() as pw:
            browser = _launch_chromium_with_mic(pw, wav)
            try:
                context = browser.new_context(ignore_https_errors=True)
                context.add_init_script(
                    "localStorage.setItem('kissVoiceEnabled', '1');"
                )
                page = context.new_page()
                goto_retrying_network_change(
                    page,
                    f"https://127.0.0.1:{self.port}/",
                    wait_until="load",
                    timeout=60_000,
                )
                page.wait_for_function(
                    "document.getElementById('voice-btn')"
                    ".classList.contains('voice-listening')",
                    timeout=120_000,
                )
                started = page.evaluate("performance.now()")
                page.wait_for_function(
                    "document.getElementById('voice-btn')"
                    ".classList.contains('voice-triggered')",
                    timeout=70_000,
                )
                woke_after = (page.evaluate("performance.now()") - started) / 1000
                self.assertGreater(woke_after, 30)
            finally:
                browser.close()

    def test_everyday_speech_does_not_wake(self) -> None:
        """Everyday speech that shares the wake grammar's words must
        leave the listener asleep at the most permissive sensitivity
        (100): "sorry car" lacks the "hey", and a sentence that goes on
        after "hey so, the car" ends its grammar final in [unk].  (A
        paused "Hey, so the car." or "hey soccer" does wake the browser;
        the daemon's transcript check rejects the first, see
        test_voice_dual_check.py, and the second sounds the same as the
        wake phrase.)"""
        from playwright.sync_api import sync_playwright

        wav = _say_wav(
            self.tmpdir,
            "everyday",
            " [[slnc 1200]] ".join([
                "sorry car",
                "hey so the car is ready",
                "hey so, the car broke down again",
                "hey sir, the car is here",
                "hey sarah can you call me",
                "I play soccer every Saturday",
            ]) + " [[slnc 1200]]",
        )
        with sync_playwright() as pw:
            browser = _launch_chromium_with_mic(pw, wav)
            try:
                context = browser.new_context(ignore_https_errors=True)
                context.add_init_script(
                    "localStorage.setItem('kissVoiceEnabled', '1');"
                    "localStorage.setItem('kissVoiceSensitivity', '100');"
                )
                page = context.new_page()
                goto_retrying_network_change(
                    page,
                    f"https://127.0.0.1:{self.port}/",
                    wait_until="load",
                    timeout=60_000,
                )
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
                # Chromium loops the WAV, so this covers every phrase
                # (about 12 s of audio) at least once.
                page.wait_for_timeout(20_000)
                self.assertFalse(
                    page.evaluate("window.__sawWake"),
                    "everyday speech must not wake the listener",
                )
            finally:
                browser.close()


def _launch_chromium_with_mic(pw: Any, wav: Path) -> Any:
    """Launch headless Chromium whose fake microphone plays *wav* on a loop."""
    return pw.chromium.launch(
        headless=True,
        args=[
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
            f"--use-file-for-fake-audio-capture={wav}",
            "--autoplay-policy=no-user-gesture-required",
            "--mute-audio",
        ],
    )


if __name__ == "__main__":
    unittest.main()
