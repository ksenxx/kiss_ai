# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: demo replay of a long ``talk`` must NOT degrade to the robotic voice.

Bug reproduced here: the daemon synthesizes demo-replay speech with the
GPT audio model, and one synthesis operation can legitimately take
minutes — ``speech_synthesis.synthesize_talk_audio`` applies
``voice_wake.audio_timeout_seconds()`` (``DEFAULT_AUDIO_TIMEOUT_SECONDS``
= 60s) PER API ATTEMPT, and the OpenAI SDK retries a timed-out request
twice by default, i.e. up to three 60s attempts plus backoff.  But the
webview's ``requestDemoSpeechClip`` waiter used to give up after only
15 seconds (``DEMO_SPEAK_TIMEOUT_MS = 15000``).  A long replayed
``talk`` — e.g. a multi-paragraph answer like the World Cup demo task —
takes the audio model well over 15s to synthesize, so the webview timed
out, resolved ``null`` and read the text with the Web Speech API: the
robotic "alien voice".  The daemon's natural clip then arrived moments
later and was silently discarded.

These tests run the REAL production ``media/main.js`` + ``media/demo.js``
in jsdom (no mocks of production code) under a deterministic virtual
clock, drive a full demo replay of a recorded session containing a
``talk`` tool call, and answer the webview's ``demoSpeak`` request with
a synthesized clip only after a realistic long-synthesis delay:

* reply after 150 virtual seconds (a first 60s attempt timing out and
  the SDK's retry succeeding — inside the daemon's legitimate
  3-attempt operation ceiling) — the demo MUST wait for and play the
  natural clip, never a robotic system voice;
* no reply at all — the demo must eventually degrade to SILENCE
  (never hang, never the robotic Web Speech system voice).
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

# A representative long spoken answer (like the World Cup demo task's
# talk): multi-sentence scripts are exactly what push GPT audio
# synthesis into the multi-attempt regime this test exercises.
TALK_TEXT = (
    "France is my pick to win the twenty twenty six World Cup. "
    "The betting markets make them the clear favorite at roughly a "
    "third implied probability, they are five and oh with a plus "
    "twelve goal differential, and Kylian Mbappe leads the tournament "
    "in goals. Argentina are the defending champions and Messi just "
    "produced an all-time comeback against Egypt, Spain have not "
    "conceded a single goal all tournament, and England survived the "
    "altitude at the Azteca. Keep an eye on Norway too, because "
    "Erling Haaland already stunned Brazil. But when the final kicks "
    "off at MetLife Stadium, I expect France to lift the trophy."
)

# The daemon's PER-ATTEMPT synthesis API timeout
# (DEFAULT_AUDIO_TIMEOUT_SECONDS in voice_wake.py, applied by
# speech_synthesis.synthesize_talk_audio to each API attempt).
DAEMON_PER_ATTEMPT_TIMEOUT_MS = 60_000

# The OpenAI SDK retries a timed-out request twice by default
# (DEFAULT_MAX_RETRIES = 2), so one synthesis operation is up to three
# per-attempt windows plus backoff.
DAEMON_MAX_ATTEMPTS = 3

# Virtual delay of the daemon's demoSpeakAudio reply in the slow-synthesis
# repro: a first 60s attempt times out and the SDK's retry succeeds —
# a legitimate, successful synthesis inside the daemon's 3-attempt
# ceiling, yet far beyond both the webview's former 15s waiter timeout
# and an insufficient 120s one.
SLOW_REPLY_DELAY_MS = 150_000

# Virtual-time ceiling of one driver run.  Must exceed the webview's
# demoSpeak waiter timeout so the never-reply case reaches its fallback.
VIRTUAL_CAP_MS = 600_000

# Node driver: loads chat.html (script tags stripped), installs a
# deterministic virtual clock over the page's setTimeout/setInterval,
# runs the real main.js + demo.js, replays one recorded session with a
# ``talk`` tool call, and answers the webview's ``demoSpeak`` request
# after ``replyDelayMs`` VIRTUAL milliseconds (argv; -1 = never reply).
# Prints JSON: {"spoken": [...], "clips": [...], "vnow": N,
# "replayDone": bool, "timerErrors": [...]}.
NODE_DEMO_DRIVER = r"""
'use strict';
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = process.argv[2];
const replyDelayMs = parseInt(process.argv[3], 10); // -1 = never reply
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
// Replaces the page's timers BEFORE main.js/demo.js load so every
// timeout in production code (demo sleeps, the demoSpeak waiter
// timeout, speech-completion callbacks) runs on virtual time.  This
// makes a "synthesis reply arrives after 30 seconds" scenario execute
// in milliseconds of real time, deterministically.
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
 * and asserted empty by the Python test, so a production JS exception
 * cannot silently fake a pass. */
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

// ---- Web Speech API: unlocked engine that records utterances ----
const spoken = [];
win.SpeechSynthesisUtterance = function (t) { this.text = t; this.lang = ''; };
win.speechSynthesis = {
  speaking: false,
  paused: false,
  getVoices: () => [],
  addEventListener: () => {},
  cancel: () => {},
  resume() { this.paused = false; },
  speak: u => {
    spoken.push(u.text);
    win.setTimeout(() => { if (u.onend) u.onend(); }, 0);
  },
};

// ---- Audio element: records natural-clip playback (full source, so
// the Python test can assert the EXACT daemon clip played) ----
const clips = [];
function FakeAudio(src) {
  this.src = src;
  this.onended = null;
  this.onerror = null;
  this.onabort = null;
  const self = this;
  this.play = function () {
    clips.push(src);
    return new Promise(resolve => {
      resolve();
      win.setTimeout(() => { if (self.onended) self.onended(); }, 0);
    });
  };
  this.pause = function () {};
}
win.Audio = FakeAudio;

// ---- Daemon side: answer resumeSession instantly; answer demoSpeak
// with a clip after replyDelayMs VIRTUAL ms (like a long GPT audio
// synthesis), or never when replyDelayMs < 0 ----
const B64_CLIP = 'QUJDREVGRw=='; // any base64 payload
win.acquireVsCodeApi = function () {
  let state;
  return {
    getState: () => state,
    setState: s => { state = s; },
    postMessage: msg => {
      if (msg.type === 'resumeSession') {
        const tabId = msg.tabId;
        win.setTimeout(() => {
          win.dispatchEvent(new win.MessageEvent('message', {data: {
            type: 'task_events',
            tabId: tabId,
            chat_id: 'chat-1',
            task_id: 'task-1',
            task: 'Demo task',
            extra: {},
            events: [
              {type: 'text_delta', text: 'Answering out loud now. '},
              {type: 'tool_call', name: 'talk', tool_id: 'tc-1',
               arguments: '', extras: {
                 text: talkText, language: 'en-US', emotion: 'warm'}},
              {type: 'tool_result', tool_id: 'tc-1',
               result: 'Spoke to the user.'},
            ],
          }}));
        }, 0);
      } else if (msg.type === 'demoSpeak') {
        if (replyDelayMs < 0) return; // synthesis reply never arrives
        win.setTimeout(() => {
          win.dispatchEvent(new win.MessageEvent('message', {data: {
            type: 'demoSpeakAudio',
            reqId: msg.reqId,
            audioB64: B64_CLIP,
            audioMime: 'audio/mpeg',
            tabId: msg.tabId,
          }}));
        }, replyDelayMs);
      }
    },
  };
};

win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
win.eval(fs.readFileSync(path.join(MEDIA, 'demo.js'), 'utf8'));

const sessions = [{
  id: 'chat-1', task_id: 'task-1', has_events: true,
  preview: 'Demo task',
}];

let replayDone = false;
win._startDemoReplay(sessions, sessions[0]).then(
  () => { replayDone = true; },
  e => {
    console.error(e && (e.stack || e.message || String(e)));
    process.exit(1);
  },
);

(async () => {
  // Pump: advance virtual time in small steps, letting promise chains
  // settle between steps, until the replay finishes or the virtual cap
  // is reached (the cap catches a hung replay).
  while (!replayDone && vnow < capMs) {
    advanceTo(vnow + 500);
    await drain();
  }
  console.log(JSON.stringify({
    spoken: spoken,
    clips: clips,
    vnow: vnow,
    replayDone: replayDone,
    timerErrors: timerErrors,
  }));
  process.exit(0);
})().catch(e => {
  console.error(e && (e.stack || e.message || String(e)));
  process.exit(1);
});
"""


def run_slow_synth_driver(reply_delay_ms: int) -> dict:
    """Run the jsdom slow-synthesis demo driver and return its JSON result.

    Args:
        reply_delay_ms: Virtual milliseconds after which the fake daemon
            sends the ``demoSpeakAudio`` clip reply; ``-1`` means the
            reply never arrives.

    Returns:
        ``{"spoken": [...], "clips": [...], "vnow": N, "replayDone": bool,
        "timerErrors": [...]}`` — Web-Speech utterances, natural Audio
        clip sources, the final virtual time, whether the replay ran to
        completion, and any exceptions thrown by page timer callbacks.
    """
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("node binary not found on PATH")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(NODE_DEMO_DRIVER)
        driver = fh.name
    try:
        proc = subprocess.run(
            [node, driver, str(VSCODE_DIR / "media"),
             str(reply_delay_ms), str(VIRTUAL_CAP_MS), TALK_TEXT],
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
            f"slow-synthesis demo driver failed: {proc.stderr}\n{proc.stdout}"
        )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert isinstance(result, dict)
    return result


@unittest.skipUnless(JSDOM_PKG.exists(), "jsdom not installed")
class TestDemoTalkSlowSynthesis(unittest.TestCase):
    """Long-synthesis demo talks must play the natural clip, not robotic."""

    def test_slow_synthesis_plays_natural_clip_not_robotic_voice(self) -> None:
        """Synthesis reply arrives after 150 virtual seconds (first 60s
        attempt times out, the SDK's retry succeeds — inside the
        daemon's legitimate 3-attempt operation ceiling): the demo MUST
        wait for and play that natural clip — never read the text with
        the robotic Web Speech fallback."""
        assert SLOW_REPLY_DELAY_MS < (
            DAEMON_PER_ATTEMPT_TIMEOUT_MS * DAEMON_MAX_ATTEMPTS
        ), "repro delay must model a successful daemon synthesis"
        result = run_slow_synth_driver(SLOW_REPLY_DELAY_MS)
        assert not result["timerErrors"], (
            f"page timer callbacks threw: {result['timerErrors']}"
        )
        assert result["replayDone"], f"demo replay never finished: {result}"
        robotic = [t for t in result["spoken"] if TALK_TEXT.split()[0] in t]
        assert not robotic, (
            "replayed long talk was read with the robotic Web Speech "
            f"voice instead of the synthesized natural clip: {result}"
        )
        daemon_clips = [c for c in result["clips"] if "QUJDREVGRw==" in c]
        assert daemon_clips, (
            "the daemon's synthesized clip (reply after "
            f"{SLOW_REPLY_DELAY_MS}ms, within "
            f"{DAEMON_MAX_ATTEMPTS} x {DAEMON_PER_ATTEMPT_TIMEOUT_MS}ms) "
            f"was never played: {result}"
        )

    def test_synthesis_reply_never_arrives_stays_silent(self) -> None:
        """No synthesis reply at all (daemon died mid-request): the demo
        must eventually degrade to SILENCE — never hang the replay and
        never use the robotic Web Speech system voice."""
        result = run_slow_synth_driver(-1)
        assert not result["timerErrors"], (
            f"page timer callbacks threw: {result['timerErrors']}"
        )
        assert result["replayDone"], f"demo replay never finished: {result}"
        assert not result["spoken"], (
            "replayed talk with no synthesis reply used the robotic Web "
            f"Speech system voice instead of staying silent: {result}"
        )
        assert not result["clips"], (
            f"no clip reply was ever sent, yet a clip played: {result}"
        )


if __name__ == "__main__":
    unittest.main()
