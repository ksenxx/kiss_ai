# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: the robotic Web Speech system voice must NEVER be used.

The agent's one voice is the GPT-synthesized natural clip
(speech_synthesis.py).  Historically the webview fell back to the
browser's Web Speech API — the robotic "alien voice" — whenever a
``talk`` event carried no audio or clip playback was
blocked/undecodable.  Per product decision (matching voice.js: "a
quiet ack is a far better failure than the loud robotic voice"), that
fallback is removed: when no natural clip can play, the talk degrades
to SILENCE and the talk queue advances so later talks still play.

These tests run the REAL production ``media/main.js`` in jsdom with NO
production code mocked, under a deterministic virtual clock, and
assert for every historical fallback trigger that:

* ``window.speechSynthesis.speak`` is NEVER invoked, and
* the talk queue still advances (no hang, later natural clips still
  play).

Scenarios:

* ``live-silent``  — live ``talk`` event without ``audioB64``;
* ``live-reject``  — live ``talk`` whose clip ``play()`` rejects
  (autoplay policy block / undecodable clip).

Each scenario queues a SECOND talk carrying a good clip and asserts it
plays, proving the silent degradation released the talk queue.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
VSCODE_DIR = PROJECT_ROOT / "src" / "kiss" / "agents" / "vscode"
JSDOM_PKG = VSCODE_DIR / "node_modules" / "jsdom" / "package.json"

TALK_TEXT = (
    "France is my pick to win the twenty twenty six World Cup, and "
    "the betting markets agree with roughly a one third implied "
    "probability."
)

VIRTUAL_CAP_MS = 600_000

NODE_DRIVER = r"""
'use strict';
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = process.argv[2];
const mode = process.argv[3]; // live-silent | live-reject
const capMs = parseInt(process.argv[4], 10);
const talkText = process.argv[5];

let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
// The VS Code webview page (NOT the remote browser chat).
html = html.replace(/class="remote-chat/g, 'class="');

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

// ---- Deterministic virtual clock ------------------------------------
let vnow = 0;
let timerSeq = 0;
const vtimers = new Map(); // id -> {at, cb, every|null}
function schedule(cb, ms, every) {
  const id = ++timerSeq;
  vtimers.set(id, {
    at: vnow + Math.max(0, Number(ms) || 0),
    cb: cb,
    every: every,
  });
  return id;
}
win.setTimeout = (cb, ms) => schedule(cb, ms, null);
win.clearTimeout = id => { vtimers.delete(id); };
win.setInterval = (cb, ms) =>
  schedule(cb, ms, Math.max(1, Number(ms) || 1));
win.clearInterval = id => { vtimers.delete(id); };
/** Errors thrown by page timer callbacks: recorded (never swallowed)
 * and asserted empty by the Python test. */
const timerErrors = [];
/** Fire all timers due up to *target* virtual ms, in time order. */
function advanceTo(target) {
  for (;;) {
    let nextId = 0;
    let next = null;
    for (const [id, t] of vtimers) {
      if (t.at <= target && (next === null || t.at < next.at)) {
        next = t;
        nextId = id;
      }
    }
    if (next === null) break;
    vnow = Math.max(vnow, next.at);
    if (next.every) next.at = vnow + next.every;
    else vtimers.delete(nextId);
    try {
      next.cb();
    } catch (e) {
      timerErrors.push(String((e && (e.stack || e.message)) || e));
    }
  }
  vnow = Math.max(vnow, target);
}
/** Let pending promise chains settle (real microtasks + macrotask). */
function drain() { return new Promise(resolve => setImmediate(resolve)); }

// ---- Web Speech API: fully working engine that records every
// utterance — ANY entry in `spoken` means the robotic fallback ran ----
const spoken = [];
win.SpeechSynthesisUtterance = function (t) { this.text = t; this.lang = ''; };
win.speechSynthesis = {
  speaking: false,
  paused: false,
  getVoices: () => [],
  addEventListener: () => {},
  cancel: () => {},
  pause() { this.paused = true; },
  resume() { this.paused = false; },
  speak: u => {
    spoken.push(u.text);
    win.setTimeout(() => { if (u.onend) u.onend(); }, 0);
  },
};

// ---- Audio element: records natural-clip playback; a clip whose
// source carries the REJECT payload simulates an autoplay-policy
// block / undecodable clip (play() rejects) ----
const clips = [];
const GOOD_CLIP = 'R09PRENMSVA=';   // base64("GOODCLIP")
const REJECT_CLIP = 'UkVKRUNUQ0xJUA=='; // base64("REJECTCLIP")
function FakeAudio(src) {
  this.src = src;
  this.onended = null;
  this.onerror = null;
  this.onabort = null;
  const self = this;
  this.play = function () {
    clips.push(src);
    if (src.indexOf(REJECT_CLIP) !== -1) {
      return Promise.reject(new Error('play() blocked'));
    }
    return new Promise(resolve => {
      resolve();
      win.setTimeout(() => { if (self.onended) self.onended(); }, 0);
    });
  };
  this.pause = function () {};
}
win.Audio = FakeAudio;

// ---- Daemon side ----
win.acquireVsCodeApi = function () {
  let state;
  return {
    getState: () => state,
    setState: s => { state = s; },
    postMessage: () => {},
  };
};

win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

// Live ``talk`` events straight through the window message handler,
// exactly as the extension host delivers them.  The first talk
// triggers the historical fallback (no audio / rejected play); the
// SECOND carries a good clip and must still play — proof the silent
// degradation released the serialized talk queue.
const first = {type: 'talk', text: talkText, language: 'en-US',
               emotion: 'warm', talkId: 'talk-1'};
if (mode === 'live-reject') {
  first.audioB64 = REJECT_CLIP;
  first.audioMime = 'audio/mpeg';
}
win.dispatchEvent(new win.MessageEvent('message', {data: first}));
win.dispatchEvent(new win.MessageEvent('message', {data: {
  type: 'talk', text: 'And that is the full story.',
  language: 'en-US', emotion: 'warm', talkId: 'talk-2',
  audioB64: GOOD_CLIP, audioMime: 'audio/mpeg',
}}));

(async () => {
  // do-while: always advance + drain at least once so promise chains
  // started synchronously above (e.g. a rejected clip play()) settle
  // before the exit condition is evaluated.
  do {
    advanceTo(vnow + 500);
    await drain();
  } while (vnow < capMs && vtimers.size !== 0);
  console.log(JSON.stringify({
    spoken: spoken,
    clips: clips,
    vnow: vnow,
    timerErrors: timerErrors,
  }));
  process.exit(0);
})().catch(e => {
  console.error(e && (e.stack || e.message || String(e)));
  process.exit(1);
});
"""


def run_fallback_driver(mode: str) -> dict:
    """Run the jsdom no-Web-Speech driver and return its JSON result.

    Args:
        mode: Fallback scenario — ``live-silent`` or ``live-reject``.

    Returns:
        ``{"spoken": [...], "clips": [...], "vnow": N, "timerErrors":
        [...]}`` — every Web-Speech utterance, every Audio clip play
        attempt, the final virtual time, and page timer exceptions.
    """
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("node binary not found on PATH")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(NODE_DRIVER)
        driver = fh.name
    try:
        proc = subprocess.run(
            [node, driver, str(VSCODE_DIR / "media"), mode,
             str(VIRTUAL_CAP_MS), TALK_TEXT],
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
            f"no-web-speech driver ({mode}) failed: "
            f"{proc.stderr}\n{proc.stdout}"
        )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert isinstance(result, dict)
    return result


GOOD_CLIP_B64 = "R09PRENMSVA="


@unittest.skipUnless(JSDOM_PKG.exists(), "jsdom not installed")
class TestNoWebSpeechFallback(unittest.TestCase):
    """Web Speech must never speak; degradation is silence + advance."""

    def check_no_web_speech(self, result: dict) -> None:
        """Assert no timer exceptions and zero Web-Speech utterances."""
        assert not result["timerErrors"], (
            f"page timer callbacks threw: {result['timerErrors']}"
        )
        assert not result["spoken"], (
            "the robotic Web Speech system voice was used: "
            f"{result['spoken']}"
        )

    def test_live_talk_without_audio_stays_silent_queue_advances(
        self,
    ) -> None:
        """A live ``talk`` with no synthesized clip must stay SILENT —
        never the robotic Web Speech read — and the queued next talk's
        natural clip must still play (queue released)."""
        result = run_fallback_driver("live-silent")
        self.check_no_web_speech(result)
        assert any(GOOD_CLIP_B64 in c for c in result["clips"]), (
            "silent degradation did not release the talk queue — the "
            f"next talk's natural clip never played: {result}"
        )

    def test_live_talk_rejected_play_stays_silent_queue_advances(
        self,
    ) -> None:
        """A live ``talk`` whose clip ``play()`` rejects (autoplay
        block / undecodable audio) must degrade to silence, not the
        robotic voice, and release the talk queue."""
        result = run_fallback_driver("live-reject")
        self.check_no_web_speech(result)
        assert any(GOOD_CLIP_B64 in c for c in result["clips"]), (
            "rejected-play degradation did not release the talk queue: "
            f"{result}"
        )


if __name__ == "__main__":
    unittest.main()
