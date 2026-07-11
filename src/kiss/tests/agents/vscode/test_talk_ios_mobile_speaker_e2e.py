# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Real speaker→microphone e2e test: ``talk`` audio on an iPhone webapp.

Guards the CURRENT ``talk``-tool sound contract on iOS: the robotic
Web Speech (``speechSynthesis``) fallback has been removed from the
webview for good.  A ``talk`` event now sounds in exactly one way —
the GPT-synthesized clip it carries (``ev.audioB64``) is played
through an ``Audio`` element (``new window.Audio(dataURL).play()``)
on the mobile page.  When the event carries no playable audio, or the
browser blocks playback (iOS autoplay policy rejecting ``play()``
outside user activation), the utterance degrades to SILENCE and the
talk queue advances — ``speechSynthesis.speak()`` must NEVER be
called, with or without a user gesture.  The old gesture-unlock
primer for Web Speech (the empty utterance spoken on the first tap)
is gone because nothing uses the speech engine anymore.

The test exercises the WHOLE pipeline with real audio hardware — no
mocks of production code, no synthetic buffers:

1. A REAL speech clip is synthesized with macOS ``say -o`` and
   base64-encoded exactly like speech_synthesis.py's GPT clip.
2. The REAL production webview (``media/main.js`` — the exact same
   file the remote mobile webapp serves via ``chat.html``) runs in
   jsdom under a simplified model of iOS Safari's two media states:
   an ``Audio`` element whose ``play()`` either succeeds (autoplay
   allowed) or rejects with ``NotAllowedError`` (autoplay blocked),
   plus a spy ``speechSynthesis`` that records every call — the
   webview must make none.  Production neither detects gestures nor
   retries a blocked ``play()`` — it attempts playback exactly once
   and degrades to silence — so the driver's tap flag only toggles
   which environment state the stub simulates.
3. The driver optionally simulates the user's tap on the mobile
   webapp and then delivers agent ``talk`` broadcasts — which, as in
   production, arrive over the WebSocket OUTSIDE any gesture.
4. The exact clip the Audio element played (its data URL payload) is
   rendered aloud through the device SPEAKERS with ``afplay``.
5. The device MICROPHONE records the playback and the recording is
   checked to contain audible speech well above the noise floor.

The suite skips itself when the acoustic loop is unavailable (no
input device, not macOS, jsdom missing, or the calibration playback
is inaudible to the microphone — muted speakers, headphones, etc.).
"""

from __future__ import annotations

import base64
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
# under an iOS-Safari-faithful Audio autoplay gate and a
# speechSynthesis spy, optionally simulates the user's tap, delivers
# the scenario's ``talk`` events exactly like the remote webapp's
# WebSocket layer does, and prints JSON:
# ``{"constructed": [src...], "played": [src...],
#    "rejected": [src...], "speechSynthesisCalls": [label...],
#    "utterancesConstructed": N}``.
NODE_IOS_DRIVER = r"""
'use strict';
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = process.argv[2];
const scenario = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));

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

// --- speechSynthesis spy: the webview must NEVER call it ---
// The robotic Web Speech fallback was removed from main.js; every
// method invocation and every SpeechSynthesisUtterance construction
// is recorded and must stay at zero, tap or no tap.
const speechSynthesisCalls = [];
let utterancesConstructed = 0;
win.SpeechSynthesisUtterance = function (t) {
  utterancesConstructed++;
  this.text = t;
  this.lang = '';
};
win.speechSynthesis = {
  speaking: false,
  paused: false,
  getVoices: () => { speechSynthesisCalls.push('getVoices'); return []; },
  addEventListener: () => { speechSynthesisCalls.push('addEventListener'); },
  cancel: () => { speechSynthesisCalls.push('cancel'); },
  resume: () => { speechSynthesisCalls.push('resume'); },
  pause: () => { speechSynthesisCalls.push('pause'); },
  speak: u => {
    speechSynthesisCalls.push('speak:' + (u && u.text));
  },
};

// --- iOS Safari Audio autoplay gate (simplified environment model) ---
// Models the two states of Apple's media policy
// (webkit.org/blog/6784): audio playback is either allowed or
// blocked — a blocked ``play()`` rejects with NotAllowedError.  The
// ``userGestureSeen`` flag only selects which state this stub
// simulates; the production webview neither detects gestures nor
// retries — it attempts ``play()`` exactly once and degrades to
// silence when rejected.  Successful playback fires ``ended``
// asynchronously like a real clip so the talk queue advances.
let userGestureSeen = false;
const constructed = [];
const played = [];
const rejected = [];
function IOSAudio(src) {
  constructed.push(src);
  this.src = src;
  this.onended = null;
  this.onerror = null;
  this.onabort = null;
  const self = this;
  this.play = function () {
    if (!userGestureSeen) {
      rejected.push(src);
      return Promise.reject(new Error('NotAllowedError'));
    }
    played.push(src);
    return new Promise(resolve => {
      resolve();
      setTimeout(() => { if (self.onended) self.onended(); }, 0);
    });
  };
  this.pause = function () {};
}
win.Audio = IOSAudio;

win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

// The user taps the mobile webapp (focusing the box / submitting
// their question) — the stub then simulates the autoplay-allowed
// state that user activation grants on iOS.
function gesture(type) {
  userGestureSeen = true;
  win.document.body.dispatchEvent(
      new win.Event(type, {bubbles: true, cancelable: true}));
}
if (scenario.tap) {
  gesture('touchend');
  gesture('click');
}

// The agent's ``talk`` broadcasts arrive over the WebSocket long
// after any tap — NOT inside any user-gesture handler.
const tabId = win._demoApi.getActiveTabId();
for (const ev of scenario.events) {
  win.dispatchEvent(new win.MessageEvent('message', {data: Object.assign(
      {type: 'talk', emotion: '', tabId: tabId}, ev)}));
}

// Let the serialized talk queue drain (play() resolutions and the
// async 'ended' events are timer/microtask hops), then report.
setTimeout(() => {
  console.log(JSON.stringify({
    constructed: constructed,
    played: played,
    rejected: rejected,
    speechSynthesisCalls: speechSynthesisCalls,
    utterancesConstructed: utterancesConstructed,
  }));
}, 300);
"""


def synthesize_clip(text: str) -> str:
    """Synthesize *text* as a real speech clip and return its base64.

    Uses macOS ``say -o`` to produce an AIFF file — a stand-in with
    identical plumbing to the GPT clip speech_synthesis.py attaches to
    live ``talk`` events as ``audioB64``.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        clip = Path(tmpdir) / "clip.aiff"
        subprocess.run(
            ["say", "-o", str(clip), text], check=True, timeout=120
        )
        return base64.b64encode(clip.read_bytes()).decode("ascii")


def ios_talk_pipeline(events: list[dict], tap: bool) -> dict:
    """Deliver *events* through the real webview talk pipeline on "iOS".

    Executes the production ``media/main.js`` in jsdom via node under
    the iOS Audio autoplay gate and speechSynthesis spy, optionally
    simulating the user's tap first.  Each event dict is merged into a
    ``{"type": "talk"}`` broadcast (supply ``text``, ``talkId`` and
    optionally ``audioB64``/``audioMime``).  Returns
    ``{"constructed": [src...], "played": [src...],
    "rejected": [src...], "speechSynthesisCalls": [...],
    "utterancesConstructed": N}`` — which clips the Audio element
    played or had blocked, and whether Web Speech was ever touched.
    """
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("node binary not found on PATH")
    with tempfile.TemporaryDirectory() as tmpdir:
        driver = Path(tmpdir) / "driver.js"
        driver.write_text(NODE_IOS_DRIVER)
        scenario = Path(tmpdir) / "scenario.json"
        scenario.write_text(json.dumps({"tap": tap, "events": events}))
        proc = subprocess.run(
            [node, str(driver), str(VSCODE_DIR / "media"), str(scenario)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=VSCODE_DIR,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
                 "NODE_PATH": str(VSCODE_DIR / "node_modules")},
        )
    if proc.returncode != 0:
        raise AssertionError(
            f"iOS talk driver failed: {proc.stderr}\n{proc.stdout}"
        )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert isinstance(result, dict)
    return result


def play_data_url_aloud(data_url: str) -> None:
    """Play the audio *data_url* through the speakers via ``afplay``.

    Decodes the base64 payload — the exact bytes the webview's Audio
    element was given — and renders it on the default output device,
    reproducing what the iPhone's speaker plays.
    """
    b64 = data_url.split(";base64,", 1)[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        clip = Path(tmpdir) / "clip.aiff"
        clip.write_bytes(base64.b64decode(b64))
        subprocess.run(["afplay", str(clip)], check=True, timeout=120)


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
    """The agent's ``talk`` clip must be audible on an iOS device."""

    noise_floor: float
    clip_b64: str

    @classmethod
    def setUpClass(cls) -> None:
        """Calibrate the speaker→microphone loop or skip the suite."""
        if sys.platform != "darwin":
            raise unittest.SkipTest("macOS `say`/`afplay` required")
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
        cls.clip_b64 = synthesize_clip(TALK_TEXT)

    def test_talk_after_user_tap_is_spoken_and_audible(self) -> None:
        """A talk clip on a tapped iPhone webapp must make real sound.

        The user taps the mobile webapp to ask their question; the
        agent's ``talk`` broadcasts then arrive over the WebSocket
        carrying the GPT-synthesized clip (``audioB64``).  The webview
        must play exactly that clip through an Audio element — the
        speaker→microphone loop must record audible speech — while a
        talk event WITHOUT audio in between degrades to silence and
        the queue still advances to the next clip.  ``speechSynthesis``
        must never be touched: the robotic fallback is gone.
        """
        clip_url = "data:audio/aiff;base64," + self.clip_b64
        result = ios_talk_pipeline(
            [
                {"text": TALK_TEXT, "talkId": "ios-e2e-talk-1",
                 "audioB64": self.clip_b64, "audioMime": "audio/aiff"},
                # No audioB64: must be silent (no Audio element, no
                # Web Speech) and must NOT stall the talk queue.
                {"text": "this event carries no audio",
                 "talkId": "ios-e2e-talk-2"},
                {"text": TALK_TEXT, "talkId": "ios-e2e-talk-3",
                 "audioB64": self.clip_b64, "audioMime": "audio/aiff"},
            ],
            tap=True,
        )

        # The robotic Web Speech engine must never be touched.
        self.assertEqual(result["speechSynthesisCalls"], [])
        self.assertEqual(result["utterancesConstructed"], 0)

        # Only the two audio-carrying events reach an Audio element,
        # both play the exact GPT clip, and none is autoplay-blocked
        # (the user's tap provided the activation).  The middle
        # audio-less event was silent AND the queue advanced past it —
        # otherwise the second clip would never have played.
        self.assertEqual(result["constructed"], [clip_url, clip_url])
        self.assertEqual(result["played"], [clip_url, clip_url])
        self.assertEqual(result["rejected"], [])

        # The acoustic proof: render the exact clip the Audio element
        # played through the speakers and listen with the microphone.
        # A silent regression (empty/garbled clip) records only noise
        # and fails right here.
        recording = record_while(
            lambda: play_data_url_aloud(result["played"][0])
        )
        speech_rms = float(np.percentile(frame_rms(recording), 95))
        self.assertGreater(
            speech_rms,
            3.0 * self.noise_floor,
            f"the microphone heard no talk audio (speech RMS "
            f"{speech_rms:.6f} vs noise floor {self.noise_floor:.6f})",
        )

    def test_talk_without_any_user_gesture_stays_silent_on_ios(self) -> None:
        """Without ANY user gesture iOS blocks the clip — silence only.

        Apple's autoplay policy rejects ``Audio.play()`` on a page the
        user never touched.  The webview must accept that rejection
        and degrade to SILENCE: no retry through ``speechSynthesis``
        (the robotic fallback is gone for good), no hang — the talk is
        simply skipped.
        """
        clip_url = "data:audio/aiff;base64," + self.clip_b64
        result = ios_talk_pipeline(
            [
                {"text": TALK_TEXT, "talkId": "ios-e2e-talk-nogesture-1",
                 "audioB64": self.clip_b64, "audioMime": "audio/aiff"},
            ],
            tap=False,
        )

        # The clip was handed to an Audio element, iOS blocked it, and
        # nothing was played.
        self.assertEqual(result["constructed"], [clip_url])
        self.assertEqual(result["rejected"], [clip_url])
        self.assertEqual(result["played"], [])

        # And crucially: the blocked clip must NOT fall back to the
        # robotic Web Speech voice — no speechSynthesis call ever.
        self.assertEqual(result["speechSynthesisCalls"], [])
        self.assertEqual(result["utterancesConstructed"], 0)


if __name__ == "__main__":
    unittest.main()
