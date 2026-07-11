# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: demo replay speaks talk/prompt via the SAME mechanism as ``talk``.

Demo-mode replay narrates recorded ``talk`` tool calls and mid-task
user ``prompt`` events aloud.  The live ``talk`` tool's playback
mechanism (main.js ``case 'talk'`` -> ``playTalkEventSound``) is: play
the GPT-synthesized natural clip when the event carries audio
(``ev.audioB64``); otherwise — or when clip playback is
unavailable/blocked — degrade to SILENCE and complete immediately so
the serialized talk queue advances.  The robotic Web Speech
(``speechSynthesis``) fallback has been removed from the project: the
browser system voice must NEVER be used, for live talks or demo
replay alike.

These tests run the REAL production ``media/main.js`` + ``media/demo.js``
in jsdom (no mocks of production code), drive a full demo replay of a
recorded session containing a ``talk`` tool call and a user ``prompt``,
answer the webview's ``demoSpeak``/``resumeSession`` requests exactly
like the daemon does, and assert the demo plays through the same
mechanism as a live ``talk`` event dispatched in the same page:

* a synthesized clip plays through the regular Audio-element path;
* a failed synthesis (empty ``demoSpeakAudio``) or a blocked clip
  ``play()`` degrades to silence — ``speechSynthesis.speak`` is never
  invoked — and the replay still runs to completion (never wedged).
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

TALK_TEXT = "Hello from the demo talk tool."
PROMPT_TEXT = "please also check the tests"
LIVE_TALK_TEXT = "Live talk without audio stays silent."

# Base64 payloads embedded in the fake Audio sources so the Python
# assertions can tell WHICH clip was played/attempted.
GOOD_CLIP_B64 = "R09PRENMSVA="  # base64("GOODCLIP") — play() resolves
REJECT_CLIP_B64 = "UkVKRUNUQ0xJUA=="  # base64("REJECTCLIP") — rejects

# Node driver: loads chat.html (script tags stripped), runs the real
# main.js + demo.js, then
#   1. dispatches LIVE ``talk`` events (baseline: the regular talk
#      mechanism) — one with no audio (must stay silent), in
#      ``blockedplay`` mode one whose clip ``play()`` rejects (must
#      also stay silent), and a final one with a good clip that MUST
#      play, proving the silent degradation released the serialized
#      talk queue;
#   2. runs a full demo replay (``window._startDemoReplay``) of one
#      recorded session containing a ``talk`` tool call and a mid-task
#      user ``prompt``, replying to the webview's ``resumeSession`` and
#      ``demoSpeak`` requests like the daemon does.  The synthesis
#      reply mode comes from argv:
#        * ``nosynth`` — ``demoSpeakAudio`` with empty ``audioB64``
#          (synthesis failed / model unavailable);
#        * ``blockedplay`` — a clip IS returned but ``Audio.play()``
#          rejects (autoplay policy block);
#        * anything else (``withclip``) — a good clip that plays.
# Prints JSON: {"spoken": [...], "clips": [...], "liveSpoken": [...],
# "liveClips": [...], "replayDone": bool}.  ``spoken`` records every
# ``speechSynthesis.speak`` call — ANY entry means the removed robotic
# Web Speech fallback ran.  ``clips`` records every Audio ``play()``
# attempt (including rejected ones).
NODE_DEMO_DRIVER = r"""
'use strict';
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = process.argv[2];
const mode = process.argv[3]; // 'nosynth' | 'blockedplay' | 'withclip'

let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
// The VS Code webview page (NOT the remote browser chat): the
// historical silence bug only manifested when the body lacks the
// 'remote-chat' class.
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

// ---- Web Speech API: fully working engine that records every
// utterance — ANY entry in `spoken` means the removed robotic
// fallback ran (the production code must never call speak) ----
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
    setTimeout(() => { if (u.onend) u.onend(); }, 0);
  },
};

// ---- Audio element: records every clip play() ATTEMPT; a source
// carrying the REJECT payload rejects (autoplay-policy block /
// undecodable clip), everything else plays and ends normally ----
const clips = [];
const GOOD_CLIP = 'R09PRENMSVA=';
const REJECT_CLIP = 'UkVKRUNUQ0xJUA==';
function FakeAudio(src) {
  this.src = src;
  this.onended = null;
  this.onerror = null;
  this.onabort = null;
  const self = this;
  this.play = function () {
    clips.push(src);
    if (src.indexOf(REJECT_CLIP) !== -1) {
      return Promise.reject(new Error('NotAllowedError'));
    }
    return new Promise(resolve => {
      resolve();
      setTimeout(() => { if (self.onended) self.onended(); }, 0);
    });
  };
  this.pause = function () {};
}
win.Audio = FakeAudio;

// ---- Daemon side: answer resumeSession / demoSpeak like server.py
// and commands.py do ----
let repliedTabId = null;
win.acquireVsCodeApi = function () {
  let state;
  return {
    getState: () => state,
    setState: s => { state = s; },
    postMessage: msg => {
      if (msg.type === 'resumeSession') {
        repliedTabId = msg.tabId;
        setTimeout(() => {
          win.dispatchEvent(new win.MessageEvent('message', {data: {
            type: 'task_events',
            tabId: repliedTabId,
            chat_id: 'chat-1',
            task_id: 'task-1',
            task: 'Demo task',
            extra: {},
            events: [
              {type: 'text_delta', text: 'Greeting the user now. '},
              {type: 'tool_call', name: 'talk', tool_id: 'tc-1',
               arguments: '', extras: {
                 text: 'HELLO_TALK', language: 'en-US', emotion: 'warm'}},
              {type: 'tool_result', tool_id: 'tc-1',
               result: 'Spoke to the user.'},
              {type: 'prompt', text: 'HELLO_PROMPT'},
            ],
          }}));
        }, 0);
      } else if (msg.type === 'demoSpeak') {
        setTimeout(() => {
          win.dispatchEvent(new win.MessageEvent('message', {data: {
            type: 'demoSpeakAudio',
            reqId: msg.reqId,
            audioB64: mode === 'nosynth' ? ''
              : (mode === 'blockedplay' ? REJECT_CLIP : GOOD_CLIP),
            audioMime: 'audio/mpeg',
            tabId: msg.tabId,
          }}));
        }, 0);
      }
    },
  };
};

win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
win.eval(fs.readFileSync(path.join(MEDIA, 'demo.js'), 'utf8'));

// ---- Baseline: the regular talk tool mechanism (live events).  A
// talk with no audio (and, in blockedplay mode, one whose clip
// play() rejects) must degrade to SILENCE; the final talk carries a
// good clip and must still play — proof the serialized talk queue
// was released, never wedged. ----
const liveTabId = win._demoApi.getActiveTabId();
win.dispatchEvent(new win.MessageEvent('message', {data: {
  type: 'talk', language: 'en-US', text: process.argv[4],
  emotion: '', talkId: 'live-talk-1',
  tabId: liveTabId,
}}));
if (mode === 'blockedplay') {
  win.dispatchEvent(new win.MessageEvent('message', {data: {
    type: 'talk', language: 'en-US',
    text: 'Live talk with a blocked clip stays silent.',
    emotion: '', talkId: 'live-talk-2', tabId: liveTabId,
    audioB64: REJECT_CLIP, audioMime: 'audio/mpeg',
  }}));
}
win.dispatchEvent(new win.MessageEvent('message', {data: {
  type: 'talk', language: 'en-US',
  text: 'Live talk with a good clip still plays.',
  emotion: '', talkId: 'live-talk-3', tabId: liveTabId,
  audioB64: GOOD_CLIP, audioMime: 'audio/mpeg',
}}));

const liveWait = new Promise(resolve => { setTimeout(resolve, 100); });

liveWait.then(() => {
  const liveSpoken = spoken.slice();
  const liveClips = clips.slice();
  const sessions = [{
    id: 'chat-1', task_id: 'task-1', has_events: true,
    preview: 'Demo task',
  }];
  return win._startDemoReplay(sessions, sessions[0]).then(() => {
    console.log(JSON.stringify({
      liveSpoken: liveSpoken,
      liveClips: liveClips,
      spoken: spoken.slice(liveSpoken.length),
      clips: clips.slice(liveClips.length),
      replayDone: true,
    }));
    // jsdom keeps the node event loop alive (pending timers of the
    // page); the driver's work is done once the JSON is printed.
    process.exit(0);
  });
}).catch(e => {
  console.error(e && (e.stack || e.message || String(e)));
  process.exit(1);
});
"""


def run_demo_driver(mode: str) -> dict:
    """Run the jsdom demo-replay driver and return its JSON result.

    Args:
        mode: ``"nosynth"`` (daemon synthesis fails, empty clip),
            ``"blockedplay"`` (clip returned but ``Audio.play()`` is
            rejected by the autoplay policy) or ``"withclip"``
            (synthesis succeeds and the clip plays normally).

    Returns:
        ``{"liveSpoken": [...], "liveClips": [...], "spoken": [...],
        "clips": [...], "replayDone": bool}`` — Web-Speech utterances
        and Audio clip play attempts of the live baseline talks versus
        the demo replay, plus whether the replay ran to completion.
    """
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("node binary not found on PATH")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(NODE_DEMO_DRIVER)
        driver = fh.name
    try:
        proc = subprocess.run(
            [node, driver, str(VSCODE_DIR / "media"), mode,
             LIVE_TALK_TEXT],
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
            f"demo talk driver failed: {proc.stderr}\n{proc.stdout}"
        )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert isinstance(result, dict)
    return result


@unittest.skipUnless(JSDOM_PKG.exists(), "jsdom not installed")
class TestDemoTalkSameMechanismAsLiveTalk(unittest.TestCase):
    """Demo replay talk/prompt playback == the regular talk mechanism.

    In every scenario the robotic Web Speech system voice must NEVER
    be used (``spoken``/``liveSpoken`` empty) and neither the live
    talk queue nor the demo replay may wedge.
    """

    def check_never_web_speech_and_replay_done(self, result: dict) -> None:
        """Assert zero Web-Speech utterances (live and demo) and that
        the demo replay ran to completion."""
        assert not result["liveSpoken"], (
            "a live talk used the removed robotic Web Speech "
            f"fallback: {result}"
        )
        assert not result["spoken"], (
            "the demo replay used the removed robotic Web Speech "
            f"fallback: {result}"
        )
        assert result["replayDone"], (
            f"demo replay never finished: {result}"
        )

    def test_demo_talk_and_prompt_stay_silent_when_synthesis_fails(
        self,
    ) -> None:
        """No synthesized clip: demo speech must degrade to SILENCE
        exactly like a live ``talk`` event without audio — never the
        robotic Web Speech voice — and the replay must still run to
        completion (queue not wedged)."""
        result = run_demo_driver("nosynth")
        self.check_never_web_speech_and_replay_done(result)
        assert not result["clips"], (
            "synthesis returned no clip, yet the demo attempted clip "
            f"playback: {result}"
        )
        # Live parity: the audio-less talk stayed silent AND released
        # the serialized talk queue — the next talk's good clip played.
        assert any(GOOD_CLIP_B64 in c for c in result["liveClips"]), (
            "silent degradation did not release the live talk queue — "
            f"the next talk's natural clip never played: {result}"
        )

    def test_demo_clip_plays_through_regular_audio_path(self) -> None:
        """Synthesized clip present: the demo must play it through the
        same Audio-element path a live ``talk`` with audio uses (one
        clip per utterance: talk + prompt)."""
        result = run_demo_driver("withclip")
        self.check_never_web_speech_and_replay_done(result)
        assert len(result["clips"]) == 2, (
            f"expected 2 demo clips (talk + prompt): {result}"
        )
        assert all(GOOD_CLIP_B64 in c for c in result["clips"]), (
            f"demo did not play the daemon's synthesized clip: {result}"
        )
        assert any(GOOD_CLIP_B64 in c for c in result["liveClips"]), (
            f"live talk with a good clip did not play it: {result}"
        )

    def test_demo_blocked_clip_degrades_silently_like_live_talk(
        self,
    ) -> None:
        """Clip playback rejected (autoplay block): the demo must
        degrade to SILENCE exactly like a live ``talk`` whose clip
        play() is rejected — never the robotic Web Speech voice — and
        the replay must still run to completion."""
        result = run_demo_driver("blockedplay")
        self.check_never_web_speech_and_replay_done(result)
        # The demo ATTEMPTED both synthesized clips (talk + prompt) —
        # proof it advanced past the first blocked utterance instead
        # of wedging.
        demo_attempts = [
            c for c in result["clips"] if REJECT_CLIP_B64 in c
        ]
        assert len(demo_attempts) == 2, (
            "demo replay did not attempt both blocked clips (talk + "
            f"prompt) — playback wedged after the rejection: {result}"
        )
        # Live parity + ordering: the blocked clip was attempted, then
        # the queued next talk's good clip still played.
        live = result["liveClips"]
        reject_idx = next(
            (i for i, c in enumerate(live) if REJECT_CLIP_B64 in c),
            None,
        )
        good_idx = next(
            (i for i, c in enumerate(live) if GOOD_CLIP_B64 in c),
            None,
        )
        assert reject_idx is not None, (
            f"live blocked clip was never attempted: {result}"
        )
        assert good_idx is not None, (
            "rejected-play degradation did not release the live talk "
            f"queue — the next talk's good clip never played: {result}"
        )
        assert reject_idx < good_idx, (
            f"live talk queue order was not preserved: {result}"
        )


if __name__ == "__main__":
    unittest.main()
