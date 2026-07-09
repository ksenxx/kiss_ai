// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for the "Working on it." acknowledgement
// played by media/voice.js after a voice-dictated task is submitted.
//
// Bug (the "alien voice"): in VS Code webview mode voice.js played the
// GPT-synthesized ack clip with `new Audio(...).play()`.  A dictated
// task involves no recent click inside the webview, so Chromium's
// autoplay policy rejects the play() promise (microsoft/vscode#197937)
// and voice.js fell back to the Web Speech API system voice — a loud,
// high-pitched robotic "Working on it." on EVERY spoken task, even
// after agent `talk` clips were fixed to play natively on the daemon.
//
// Required behaviour verified here:
//
//  1. webview mode: the ack is delegated to the extension host with a
//     `{type: 'voiceAck'}` post (the host plays media/working-on-it.mp3
//     natively on the machine's speakers — no autoplay policy there);
//     the webview itself never constructs an Audio element and NEVER
//     speaks with the robotic Web Speech voice.
//  2. browser mode (remote chat page): the Audio element still plays
//     the clip (a real browser tab allows it after the user's
//     interaction), but a rejected play() now degrades to SILENCE —
//     never to the robotic Web Speech voice.  No voiceAck post is sent
//     (the daemon may be on another machine).
//  3. browser mode without an Audio API: silent, no robotic fallback,
//     no crash.
//
// Runs the real media/voice.js in a real jsdom document — no mocks for
// the code under test (Audio / speechSynthesis are the browser
// environment, which jsdom does not provide).  Run with:
//
//     node test/voiceAckNoRoboticFallback.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const VOICE_JS_PATH = path.join(__dirname, '..', 'media', 'voice.js');

let passed = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.message}`);
  }
}

/** Let pending promise rejections / microtasks settle. */
function flush() {
  return new Promise(resolve => setTimeout(resolve, 25));
}

/**
 * Build a fresh jsdom window running the real media/voice.js with the
 * given __VOICE__ config, a recording Audio stub whose play() is
 * rejected (the webview autoplay policy), a recording Web Speech stub,
 * and a recorder for 'kiss-voice-post' bridge messages.
 *
 * withAudio=false omits the Audio API entirely (e.g. a stripped-down
 * embedder) so the no-Audio degradation path can be exercised.
 */
function makeWindow(cfg, {withAudio = true} = {}) {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body>' +
      '<button id="voice-btn" class="toggle-btn"></button>' +
      '<textarea id="task-input"></textarea>' +
      '</body></html>',
    {runScripts: 'dangerously', url: 'https://localhost/'},
  );
  const win = dom.window;
  win.__VOICE__ = cfg;

  const record = {
    audioUrls: [],
    audioPlays: 0,
    spoken: [],
    posts: [],
  };

  if (withAudio) {
    win.Audio = function Audio(url) {
      record.audioUrls.push(String(url));
      this.play = () => {
        record.audioPlays++;
        return Promise.reject(new Error('NotAllowedError: autoplay'));
      };
    };
  } else {
    // jsdom ships a stub HTMLAudioElement constructor; remove it so
    // `typeof window.Audio === 'function'` is genuinely false.
    win.Audio = undefined;
  }
  win.SpeechSynthesisUtterance = function SpeechSynthesisUtterance(text) {
    this.text = text;
  };
  win.speechSynthesis = {
    speak: utter => record.spoken.push(utter && utter.text),
    cancel: () => {},
    getVoices: () => [],
  };
  win.addEventListener('kiss-voice-post', event => {
    record.posts.push(event.detail);
  });

  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);
  return {win, record};
}

function sendHostMessage(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

// ---------------------------------------------------------------------------

async function main() {
  await test(
    'webview mode: dictated task delegates the ack to the host (voiceAck)',
    async () => {
      const {win, record} = makeWindow({
        mode: 'webview',
        ackAudioUrl: 'https://localhost/media/working-on-it.mp3',
      });
      sendHostMessage(win, {type: 'voiceSpeech', text: 'Fix the bug', speaker: 1});
      await flush();
      const acks = record.posts.filter(m => m && m.type === 'voiceAck');
      assert.strictEqual(
        acks.length,
        1,
        `expected exactly one voiceAck post, got ${JSON.stringify(record.posts)}`,
      );
    },
  );

  await test(
    'webview mode: never plays Audio and never speaks with the robotic voice',
    async () => {
      const {win, record} = makeWindow({
        mode: 'webview',
        ackAudioUrl: 'https://localhost/media/working-on-it.mp3',
      });
      sendHostMessage(win, {type: 'voiceSpeech', text: 'Run the tests', speaker: 2});
      await flush();
      assert.deepStrictEqual(
        record.audioUrls,
        [],
        'webview must not construct an Audio element for the ack',
      );
      assert.deepStrictEqual(
        record.spoken,
        [],
        `robotic Web Speech ack must never fire, spoke: ${JSON.stringify(record.spoken)}`,
      );
    },
  );

  await test(
    'browser mode: plays the synthesized clip; rejected play() stays silent',
    async () => {
      const {win, record} = makeWindow({
        mode: 'browser',
        ackAudioUrl: 'https://localhost/media/working-on-it.mp3',
      });
      sendHostMessage(win, {type: 'voiceSpeech', text: 'Deploy it', speaker: 1});
      await flush();
      assert.deepStrictEqual(
        record.audioUrls,
        ['https://localhost/media/working-on-it.mp3'],
        'browser mode must try the synthesized ack clip',
      );
      assert.strictEqual(record.audioPlays, 1, 'clip play() must be attempted');
      assert.deepStrictEqual(
        record.spoken,
        [],
        `rejected play() must degrade to silence, spoke: ${JSON.stringify(record.spoken)}`,
      );
      const acks = record.posts.filter(m => m && m.type === 'voiceAck');
      assert.deepStrictEqual(acks, [], 'browser mode must not post voiceAck');
    },
  );

  await test(
    'browser mode without an Audio API: silent, no robotic fallback, no crash',
    async () => {
      const {win, record} = makeWindow(
        {mode: 'browser', ackAudioUrl: 'https://localhost/media/working-on-it.mp3'},
        {withAudio: false},
      );
      sendHostMessage(win, {type: 'voiceSpeech', text: 'Ship it', speaker: 3});
      await flush();
      assert.deepStrictEqual(
        record.spoken,
        [],
        `robotic Web Speech ack must never fire, spoke: ${JSON.stringify(record.spoken)}`,
      );
    },
  );

  console.log(`\n${passed} passed, ${failures.length} failed`);
  if (failures.length) process.exit(1);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
