# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end real-voice tests for the wake-word sensitivity setting.

Real audio (macOS TTS), real speech models, no mocks.  The sensitivity
slider (0..100, default 85) must ACTUALLY change how eagerly the
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
    default (85) accepts a trailing alias and wakes.
  * Ordinary sentences containing alias-sounding words never wake at
    the default sensitivity.

- The browser (remote webapp) listener in ``media/voice.js``: a real
  Chromium visits the real web app with its microphone fed the spoken
  "hey there Sorcar" audio.  At a stored strict sensitivity (50) the
  wake indicator never fires; dragging the settings-panel sensitivity
  slider to 85 makes the very same audio wake the listener and
  persists the value in localStorage.  A fresh profile defaults the
  slider to 85.
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

PROJECT_ROOT = Path(__file__).resolve().parents[5]

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
        # The short pause before "Sorcar" keeps the TTS from slurring
        # the phrase into a single [unk] (observed with some prosody):
        # this decodes to "[unk] sore car" with conf 1.0 reliably.
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
        # Explicit default: fails before the feature exists because
        # --sensitivity is an unknown argument (argparse exits 2).
        proc = _run_listener(self.sorcar_wav, "--sensitivity", "85")
        lines = proc.stdout.split()
        self.assertIn("READY", lines, msg=proc.stderr[-2000:])
        self.assertIn("WAKE", lines, msg=proc.stderr[-2000:])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])

    def test_low_sensitivity_rejects_sound_alike(self) -> None:
        # "soccer" force-fits onto "sar car"/"sir car" with word
        # confidences ~0.55-0.69 (measured): it wakes at the default
        # sensitivity but not at sensitivity 10, whose confidence gate
        # (0.72) exceeds every force-fit score.
        wakes = _run_listener(self.soccer_wav, "--sensitivity", "85")
        self.assertIn("WAKE", wakes.stdout.split(),
                      msg=wakes.stderr[-2000:])
        self.assertEqual(wakes.returncode, 0, msg=wakes.stderr[-2000:])

        rejects = _run_listener(self.soccer_wav, "--sensitivity", "10")
        self.assertIn("READY", rejects.stdout.split(),
                      msg=rejects.stderr[-2000:])
        self.assertNotIn("WAKE", rejects.stdout.split(),
                         msg=rejects.stdout)
        self.assertEqual(rejects.returncode, 1, msg=rejects.stdout)

    def test_high_sensitivity_wakes_on_trailing_alias(self) -> None:
        # "hey there Sorcar" decodes to "[unk] sore car" (measured):
        # strict whole-utterance matching rejects it below 75; at 85
        # (the default) a trailing alias is accepted and wakes.
        strict = _run_listener(self.hey_wav, "--sensitivity", "50")
        self.assertIn("READY", strict.stdout.split(),
                      msg=strict.stderr[-2000:])
        self.assertNotIn("WAKE", strict.stdout.split(), msg=strict.stdout)
        self.assertEqual(strict.returncode, 1, msg=strict.stdout)

        eager = _run_listener(self.hey_wav, "--sensitivity", "85")
        self.assertIn("WAKE", eager.stdout.split(),
                      msg=eager.stderr[-2000:])
        self.assertEqual(eager.returncode, 0, msg=eager.stderr[-2000:])

    def test_sentences_never_wake_at_default_sensitivity(self) -> None:
        proc = _run_listener(self.sentences_wav, "--sensitivity", "85")
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


@unittest.skipUnless(HAVE_MAC_TTS, "requires macOS `say` and `afconvert`")
class TestSensitivitySliderBrowser(unittest.TestCase):
    """Dragging the settings-panel slider changes what real spoken
    audio wakes the in-browser (remote webapp) listener."""

    def setUp(self) -> None:
        from kiss.server.web_server import RemoteAccessServer

        self.tmpdir = Path(tempfile.mkdtemp())
        # Chromium loops the fake-capture file, so the same "hey there
        # Sorcar" audio keeps playing before and after the slider move.
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
                # Seed a strict sensitivity (50): the default (85)
                # accepts a trailing alias and would wake immediately.
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
                # The settings panel must contain the sensitivity
                # slider, reflecting the stored value.
                slider = page.evaluate(
                    "(() => { const s = document.getElementById("
                    "'cfg-voice-sensitivity'); return s &&"
                    " {value: s.value, min: s.min, max: s.max}; })()"
                )
                self.assertIsNotNone(slider, "sensitivity slider missing")
                self.assertEqual(slider["value"], "50")
                self.assertEqual(slider["min"], "0")
                self.assertEqual(slider["max"], "100")
                # Record every wake flash on the mic button.
                page.evaluate(
                    "window.__sawWake = false;"
                    "const btn = document.getElementById('voice-btn');"
                    "new MutationObserver(() => {"
                    "  if (btn.classList.contains('voice-triggered'))"
                    "    window.__sawWake = true;"
                    "}).observe(btn, {attributes: true});"
                )
                # Wait for listening to start (cold cache: the server
                # downloads the Vosk model, the browser unpacks it in
                # the WASM worker).
                page.wait_for_function(
                    "document.getElementById('voice-btn')"
                    ".classList.contains('voice-listening')",
                    timeout=300_000,
                )
                # At the strict stored sensitivity (50) the looping
                # "hey there Sorcar" audio must NOT wake the listener.
                page.wait_for_timeout(10_000)
                self.assertFalse(
                    page.evaluate("window.__sawWake"),
                    "audio must not wake at sensitivity 50",
                )
                # Drag the slider to 85: the SAME audio must now wake.
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
                # A fresh profile (no stored value) must default the
                # slider to 85.
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
                    "85",
                    "a fresh profile must default the slider to 85",
                )
                fresh.close()
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
