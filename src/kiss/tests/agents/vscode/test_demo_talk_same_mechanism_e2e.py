# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: demo replay speaks talk/prompt via the SAME mechanism as ``talk``.

Demo-mode replay narrates recorded ``talk`` tool calls and mid-task
user ``prompt`` events aloud.  The live ``talk`` tool's playback
mechanism (main.js ``case 'talk'``) is: play the GPT-synthesized clip
when the event carries audio, otherwise — or when clip playback is
unavailable/blocked — read the text with the Web Speech API
(``speakWithSystemVoice``).

Bug: the demo path (``enqueueDemoSpeech``) used DIFFERENT playback
rules — in the VS Code webview it degraded to SILENCE whenever the
daemon's ``demoSpeakAudio`` synthesis reply carried no clip or the
clip's ``play()`` was rejected, so replayed talks and prompt
narrations were never heard even though a live ``talk`` with the very
same (audio-less) event speaks fine through the Web Speech fallback.

These tests run the REAL production ``media/main.js`` + ``media/demo.js``
in jsdom (no mocks of production code), drive a full demo replay of a
recorded session containing a ``talk`` tool call and a user ``prompt``,
answer the webview's ``demoSpeak``/``resumeSession`` requests exactly
like the daemon does, and assert the demo speaks through the same
mechanism as a live ``talk`` event dispatched in the same page.
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
LIVE_TALK_TEXT = "Live talk without audio speaks fine."

# Node driver: loads chat.html (script tags stripped), runs the real
# main.js + demo.js, then
#   1. dispatches one LIVE ``talk`` event with no audio (baseline: the
#      regular talk mechanism, which falls back to the Web Speech API);
#   2. runs a full demo replay (``window._startDemoReplay``) of one
#      recorded session containing a ``talk`` tool call and a mid-task
#      user ``prompt``, replying to the webview's ``resumeSession`` and
#      ``demoSpeak`` requests like the daemon does.  The synthesis
#      reply mode comes from argv:
#        * ``nosynth`` — ``demoSpeakAudio`` with empty ``audioB64``
#          (synthesis failed / model unavailable);
#        * ``blockedplay`` — a clip IS returned but ``Audio.play()``
#          rejects (autoplay policy block), forcing the fallback path.
# Prints JSON: {"spoken": [...], "clips": [...], "liveSpoken": [...],
# "liveClips": [...]}.
NODE_DEMO_DRIVER = r"""
'use strict';
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = process.argv[2];
const mode = process.argv[3]; // 'nosynth' | 'blockedplay'

let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
// The VS Code webview page (NOT the remote browser chat): the silence
// bug only manifested when the body lacks the 'remote-chat' class.
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
    // Finish asynchronously like a real engine so the talk queue's
    // completion plumbing (utter.onend) is exercised.
    setTimeout(() => { if (u.onend) u.onend(); }, 0);
  },
};

// ---- Audio element: records clip playback; play() optionally
// rejects to emulate an autoplay-policy block ----
const clips = [];
function FakeAudio(src) {
  this.src = src;
  this.onended = null;
  this.onerror = null;
  this.onabort = null;
  const self = this;
  this.play = function () {
    if (mode === 'blockedplay') {
      return Promise.reject(new Error('NotAllowedError'));
    }
    clips.push(src.slice(0, 40));
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
const B64_CLIP = 'QUJDREVGRw=='; // any base64 payload
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
            audioB64: mode === 'nosynth' ? '' : B64_CLIP,
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

// ---- Baseline: the regular talk tool mechanism (live event, no
// audio -> Web Speech fallback) ----
win.dispatchEvent(new win.MessageEvent('message', {data: {
  type: 'talk', language: 'en-US', text: process.argv[4],
  emotion: '', talkId: 'live-talk-1',
  tabId: win._demoApi.getActiveTabId(),
}}));

const liveWait = new Promise(resolve => { setTimeout(resolve, 50); });

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
        mode: ``"nosynth"`` (daemon synthesis fails, empty clip) or
            ``"blockedplay"`` (clip returned but ``Audio.play()`` is
            rejected by the autoplay policy).

    Returns:
        ``{"liveSpoken": [...], "liveClips": [...], "spoken": [...],
        "clips": [...]}`` — Web-Speech utterances and Audio clips of
        the live baseline talk versus the demo replay.
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
    """Demo replay talk/prompt playback == the regular talk mechanism."""

    def test_demo_talk_and_prompt_speak_when_synthesis_fails(self) -> None:
        """No synthesized clip: demo speech must fall back to the Web
        Speech API exactly like a live ``talk`` event without audio —
        never degrade to silence."""
        result = run_demo_driver("nosynth")
        live = " ".join(result["liveSpoken"])
        assert LIVE_TALK_TEXT.rstrip(".!") .split()[0] in live, (
            f"baseline live talk did not speak: {result}"
        )
        demo = " ".join(result["spoken"])
        assert "HELLO_TALK" in demo, (
            "replayed talk tool call was NOT spoken via the regular "
            f"talk mechanism (Web Speech fallback): {result}"
        )
        assert "User says HELLO_PROMPT" in demo, (
            "replayed user prompt was NOT narrated via the regular "
            f"talk mechanism (Web Speech fallback): {result}"
        )

    def test_demo_clip_plays_through_regular_audio_path(self) -> None:
        """Synthesized clip present: the demo must play it through the
        same Audio-element path a live ``talk`` with audio uses (one
        clip per utterance: talk + prompt)."""
        # Any mode other than 'nosynth'/'blockedplay' means: synthesis
        # succeeds and the clip plays normally.
        result = run_demo_driver("withclip")
        assert len(result["clips"]) == 2, (
            f"expected 2 demo clips (talk + prompt): {result}"
        )
        assert not result["spoken"], (
            f"clip playback must not also invoke Web Speech: {result}"
        )

    def test_demo_blocked_clip_falls_back_like_live_talk(self) -> None:
        """Clip playback rejected (autoplay block): the demo must fall
        back to the Web Speech API exactly like a live ``talk`` whose
        clip play() is rejected — never to silence."""
        result = run_demo_driver("blockedplay")
        demo = " ".join(result["spoken"])
        assert "HELLO_TALK" in demo, (
            "replayed talk with a blocked clip was NOT spoken through "
            f"the regular talk fallback: {result}"
        )
        assert "User says HELLO_PROMPT" in demo, (
            "replayed prompt with a blocked clip was NOT narrated "
            f"through the regular talk fallback: {result}"
        )


if __name__ == "__main__":
    unittest.main()
