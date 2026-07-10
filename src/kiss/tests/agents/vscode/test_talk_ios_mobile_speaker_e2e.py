# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Real speaker→microphone e2e test: ``talk`` audio on an iPhone webapp.

Reproduces (and guards the fix for) the bug where the agent's ``talk``
tool was silent on the remote mobile webapp on iOS: the webview called
``speechSynthesis.speak()`` from a WebSocket ``message`` event, and iOS
Safari **silently drops** any ``speak()`` unless speech synthesis was
first unlocked by a ``speak()`` call made synchronously inside a
user-gesture handler (``touchend``/``click``/``keydown`` — WebKit's
activation events).  See e.g.
https://stackoverflow.com/questions/67655133 and
https://webkit.org/blog/6784/new-video-policies-for-ios/.

The test exercises the WHOLE pipeline with real audio hardware — no
mocks, no synthetic buffers:

1. The REAL production webview (``media/main.js`` — the exact same file
   the remote mobile webapp serves via ``chat.html``) runs in jsdom
   under a ``speechSynthesis`` environment faithful to iOS Safari's
   documented gating: a ``speak()`` outside a user gesture is silently
   suppressed (no sound, no events, no error) until a ``speak()`` has
   executed inside a gesture handler, which unlocks the session.
2. The driver simulates the user's tap on the mobile webapp (the tap
   that submits their question) and then delivers the agent's ``talk``
   broadcast — which, as in production, arrives OUTSIDE any gesture.
3. Whatever utterances actually reach the (unlocked) speech engine are
   rendered aloud through the device SPEAKERS with macOS ``say``.
4. The device MICROPHONE records the playback and the recording is
   checked to contain audible speech well above the noise floor.

Before the fix, main.js never spoke inside a user gesture, so the talk
utterances were suppressed, nothing was played, and the microphone
heard silence — exactly the reported iPhone symptom.

The suite skips itself when the acoustic loop is unavailable (no input
device, not macOS, jsdom missing, or the calibration playback is
inaudible to the microphone — muted speakers, headphones, etc.).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
VSCODE_DIR = PROJECT_ROOT / "src" / "kiss" / "agents" / "vscode"
JSDOM_PKG = VSCODE_DIR / "node_modules" / "jsdom" / "package.json"

SAMPLE_RATE = 44100

# The reply the agent's ``talk`` tool speaks after the user asks a
# question from their iPhone.
TALK_TEXT = (
    "Alright, I looked into it. "
    "The tests are green and the fix is ready!"
)

# Node driver: runs the REAL production ``media/main.js`` in jsdom
# under an iOS-Safari-faithful speechSynthesis gate, optionally
# simulates the user's tap, dispatches one ``talk`` event exactly like
# the remote webapp's WebSocket layer does, and prints JSON:
# ``{"spoken": [{text, rate, pitch}...], "suppressed": [text...]}``.
NODE_IOS_DRIVER = r"""
'use strict';
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = process.argv[2];
const language = process.argv[3];
const text = process.argv[4];
const simulateTap = process.argv[5] === 'tap';

let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  url: 'https://localhost/',
});
const win = dom.window;
win.Element.prototype.scrollIntoView = function () {};
win.Element.prototype.scrollTo = function () {};
win.HTMLElement.prototype.scrollTo = function () {};
win.requestAnimationFrame = function (cb) { cb(); return 0; };
win.acquireVsCodeApi = function () {
  let state;
  return {postMessage: () => {}, getState: () => state,
          setState: s => { state = s; }};
};

// --- iOS Safari speechSynthesis gating emulation ---
// Documented behavior (webkit.org/blog/6784, stackoverflow 67655133):
//  * the first speak() must execute synchronously inside a handler for
//    a user-gesture event (touchend / click / keydown) — that unlocks
//    speech for the rest of the session;
//  * until unlocked, speak() calls made outside a gesture (timers,
//    network/message events) are suppressed with NO sound, NO events
//    and NO error.
let inUserGesture = 0;
let unlocked = false;
const spoken = [];
const suppressed = [];
win.SpeechSynthesisUtterance = function (t) { this.text = t; this.lang = ''; };
win.speechSynthesis = {
  speaking: false,
  paused: false,
  getVoices: () => [],
  addEventListener: () => {},
  cancel: () => {},
  resume() { this.paused = false; },
  speak: u => {
    if (inUserGesture > 0) unlocked = true;
    if (!unlocked) { suppressed.push(u.text); return; }
    spoken.push(u);
  },
};

win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

// The user taps the mobile webapp (focusing the box / submitting their
// question).  jsdom dispatches synchronously, so every listener runs
// while ``inUserGesture`` is raised — exactly like a real gesture.
function gesture(type) {
  inUserGesture++;
  try {
    win.document.body.dispatchEvent(
        new win.Event(type, {bubbles: true, cancelable: true}));
  } finally {
    inUserGesture--;
  }
}
if (simulateTap) {
  gesture('touchend');
  gesture('click');
}

// The agent's ``talk`` broadcast arrives over the WebSocket long after
// the tap — NOT inside any user-gesture handler.
win.dispatchEvent(new win.MessageEvent('message', {data: {
  type: 'talk', language, text, emotion: '', talkId: 'ios-e2e-talk-1',
  tabId: win._demoApi.getActiveTabId(),
}}));

console.log(JSON.stringify({
  spoken: spoken.map(u => ({text: u.text, rate: u.rate, pitch: u.pitch})),
  suppressed,
}));
"""


def ios_talk_pipeline(text: str, tap: bool) -> dict:
    """Run *text* through the real webview talk pipeline on "iOS".

    Executes the production ``media/main.js`` in jsdom via node under
    the iOS speech gate, optionally simulating the user's tap first.
    Returns ``{"spoken": [{"text", "rate", "pitch"}...],
    "suppressed": [text, ...]}`` — the utterances the engine played
    versus silently dropped.
    """
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("node binary not found on PATH")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(NODE_IOS_DRIVER)
        driver = fh.name
    try:
        proc = subprocess.run(
            [node, driver, str(VSCODE_DIR / "media"), "en-US", text,
             "tap" if tap else "notap"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=VSCODE_DIR,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
                 "NODE_PATH": str(VSCODE_DIR / "node_modules")},
        )
    finally:
        Path(driver).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise AssertionError(
            f"iOS talk driver failed: {proc.stderr}\n{proc.stdout}"
        )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert isinstance(result, dict)
    return result


def say_aloud(utterances: list[dict]) -> None:
    """Play *utterances* in order through the speakers via macOS `say`.

    Empty texts (e.g. the gesture-unlock priming utterance) make no
    sound on a real device and are skipped here too.
    """
    for utt in utterances:
        text = str(utt.get("text", "")).strip()
        if not text:
            continue
        subprocess.run(["say", text], check=True, timeout=120)


def record_while(action) -> np.ndarray:
    """Record the default microphone while *action*() runs.

    Returns the mono float32 recording, including ~0.4s of ambient
    lead-in before the action (usable as the noise floor) and ~0.4s
    of tail.
    """
    import sounddevice as sd

    chunks: list[np.ndarray] = []

    def on_audio(indata, _frames, _time, _status) -> None:
        chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, callback=on_audio
    ):
        time.sleep(0.4)
        action()
        time.sleep(0.4)
    return np.concatenate(chunks)[:, 0]


def frame_rms(audio: np.ndarray) -> np.ndarray:
    """Return the per-frame RMS energy track of *audio* (20ms hop)."""
    frame = int(0.04 * SAMPLE_RATE)
    hop = int(0.02 * SAMPLE_RATE)
    out = []
    for start in range(0, len(audio) - frame, hop):
        window = audio[start:start + frame]
        out.append(float(np.sqrt(np.mean(window * window))))
    return np.asarray(out)


class TestTalkIosMobileSpeakerE2E(unittest.TestCase):
    """The agent's ``talk`` reply must be audible on an iOS device."""

    noise_floor: float

    @classmethod
    def setUpClass(cls) -> None:
        """Calibrate the speaker→microphone loop or skip the suite."""
        if sys.platform != "darwin":
            raise unittest.SkipTest("macOS `say` engine required")
        if shutil.which("node") is None:
            raise unittest.SkipTest("node is not available on PATH")
        if not JSDOM_PKG.is_file():
            raise unittest.SkipTest(
                f"jsdom is not installed under {VSCODE_DIR/'node_modules'}"
                " — run `npm install` there"
            )
        try:
            import sounddevice as sd

            sd.query_devices(kind="input")
        except Exception as exc:  # noqa: BLE001 - any device error
            raise unittest.SkipTest(f"no audio input device: {exc}")
        recording = record_while(
            lambda: subprocess.run(
                ["say", "calibration check one two three"],
                check=True,
                timeout=60,
            )
        )
        lead_in = recording[: int(0.3 * SAMPLE_RATE)]
        cls.noise_floor = float(np.sqrt(np.mean(lead_in * lead_in)))
        speech_rms = float(np.percentile(frame_rms(recording), 95))
        if speech_rms < 3.0 * cls.noise_floor or speech_rms < 1e-4:
            raise unittest.SkipTest(
                "speaker playback is inaudible to the microphone "
                f"(speech RMS {speech_rms:.6f}, "
                f"noise floor {cls.noise_floor:.6f}) — speakers muted "
                "or headphones in use"
            )

    def test_talk_after_user_tap_is_spoken_and_audible(self) -> None:
        """A talk reply on a tapped iPhone webapp must make real sound.

        The user taps the mobile webapp to ask their question; the
        agent's ``talk`` broadcast then arrives over the WebSocket.
        Under iOS gating the webview must have unlocked speech during
        that tap so the reply's utterances actually reach the engine —
        and the speaker→microphone loop must record audible speech.
        Before the fix the utterances were silently suppressed and the
        microphone heard only silence.
        """
        result = ios_talk_pipeline(TALK_TEXT, tap=True)
        audible = [u for u in result["spoken"] if str(u["text"]).strip()]

        # The acoustic reproduction FIRST: play exactly what the iOS
        # device would play and listen with the microphone.  Before
        # the fix ``audible`` was empty (everything suppressed), so
        # nothing was played and this recorded pure silence — the
        # reported iPhone symptom, failing right here.
        recording = record_while(lambda: say_aloud(audible))
        speech_rms = float(np.percentile(frame_rms(recording), 95))
        self.assertGreater(
            speech_rms,
            3.0 * self.noise_floor,
            f"the microphone heard no talk audio (speech RMS "
            f"{speech_rms:.6f} vs noise floor {self.noise_floor:.6f}) "
            f"— iOS suppressed: {result['suppressed']!r}",
        )

        # And the pipeline details: nothing suppressed, both reply
        # sentences reached the speech engine.
        self.assertEqual(
            result["suppressed"],
            [],
            "iOS silently dropped talk utterances — speech synthesis "
            "was never unlocked during the user's tap",
        )
        self.assertGreaterEqual(
            len(audible),
            2,
            f"expected the talk reply's sentences to reach the speech "
            f"engine, got {result['spoken']!r}",
        )
        spoken_text = " ".join(str(u["text"]) for u in audible)
        self.assertIn("tests are green", spoken_text)

    def test_talk_without_any_user_gesture_stays_silent_on_ios(self) -> None:
        """Without ANY user gesture iOS suppresses speech — by design.

        This guards the fidelity of the iOS gate itself: the unlock
        must happen only inside a real user gesture, so a page the
        user never touched cannot start talking (Apple's autoplay
        policy).  The talk event must be suppressed, not played.
        """
        result = ios_talk_pipeline(TALK_TEXT, tap=False)
        self.assertEqual(result["spoken"], [])
        self.assertGreaterEqual(len(result["suppressed"]), 1)


if __name__ == "__main__":
    unittest.main()
