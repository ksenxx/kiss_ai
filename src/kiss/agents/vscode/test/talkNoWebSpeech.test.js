// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test that the agent ``talk`` tool NEVER falls back to the
// robotic Web Speech API (window.speechSynthesis): playback plays the
// GPT-synthesized clip (``ev.audioB64``) through an Audio element and
// nothing else.  When a talk event carries no audio, or the clip's
// ``play()`` is rejected (autoplay policy), the utterance degrades to
// SILENCE and the serialized talk queue advances immediately so the
// next queued talk clip still plays — speechSynthesis.speak must never
// be called.
//
// Runs the REAL production ``media/main.js`` in jsdom (only the
// vscode host API, a recording Web Speech API, and a recording Audio
// API are stubs, as in every webview test).  Run directly with
// ``node``:
//
//     node src/kiss/agents/vscode/test/talkNoWebSpeech.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/** Build a jsdom window running the production chat webview. */
function makeWebview() {
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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win};
}

/**
 * Install a recording Web Speech API on *win* (jsdom has none).  The
 * production code must NEVER call ``speak`` — the returned array of
 * spoken utterances is asserted empty by every test below.
 */
function installSpeech(win) {
  const spoken = [];
  win.SpeechSynthesisUtterance = function SpeechSynthesisUtterance(text) {
    this.text = text;
    this.lang = '';
  };
  win.speechSynthesis = {
    speak: u => spoken.push(u),
    getVoices: () => [],
  };
  return spoken;
}

/**
 * Install a recording Audio constructor on *win*.  Each constructed
 * clip's ``play()`` returns the next promise in *playResults* (a
 * resolved promise once the list is exhausted).  Returns the array of
 * created ``src`` strings.
 */
function installAudio(win, playResults) {
  const created = [];
  const results = (playResults || []).slice();
  win.Audio = function Audio(src) {
    created.push(src);
    const result = results.length ? results.shift() : Promise.resolve();
    this.play = () => result;
  };
  return created;
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Let pending microtasks and zero-delay timers run. */
function tick() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

const B64 = 'SUQzBAAAAAAAAA=='; // decodes to "ID3..." — an MP3 tag header

async function testTalkWithoutAudioIsSilentAndQueueAdvances() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  const created = installAudio(win, [Promise.resolve()]);

  // A live talk event WITHOUT synthesized audio: silence, no robotic
  // Web Speech fallback, and the talk queue must advance immediately.
  send(win, {type: 'talk', language: 'en-US', text: 'no clip here',
             talkId: 'nws1'});

  assert.strictEqual(spoken.length, 0,
                     'talk without audioB64 must not use Web Speech');
  assert.strictEqual(created.length, 0,
                     'talk without audioB64 creates no Audio element');

  // The queue advanced: a subsequent talk WITH a good clip plays it.
  send(win, {type: 'talk', language: 'en-US', text: 'with clip',
             talkId: 'nws2', audioB64: B64, audioMime: 'audio/mpeg'});
  await tick();

  assert.strictEqual(created.length, 1,
                     'next queued talk clip still plays');
  assert.strictEqual(created[0], 'data:audio/mpeg;base64,' + B64);
  assert.strictEqual(spoken.length, 0,
                     'Web Speech API stays silent throughout');
  console.log('PASS: talk without audio is silent and the queue advances');
}

async function testRejectedClipPlayIsSilentAndQueueAdvances() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  const created = installAudio(win, [
    Promise.reject(new Error('autoplay blocked')),
    Promise.resolve(),
  ]);

  // A talk whose clip play() rejects (autoplay policy): silence — no
  // robotic Web Speech fallback — and the queue must not stall.
  send(win, {type: 'talk', language: 'en', text: 'blocked clip',
             talkId: 'nws3', audioB64: B64});
  await tick();

  assert.strictEqual(created.length, 1, 'the rejected clip was attempted');
  assert.strictEqual(spoken.length, 0,
                     'rejected play() must not use Web Speech');

  // The queue advanced past the rejected clip: the next talk plays.
  send(win, {type: 'talk', language: 'en', text: 'next clip',
             talkId: 'nws4', audioB64: B64});
  await tick();

  assert.strictEqual(created.length, 2,
                     'next queued talk clip still plays after rejection');
  assert.strictEqual(created[1], 'data:audio/mpeg;base64,' + B64);
  assert.strictEqual(spoken.length, 0,
                     'Web Speech API stays silent throughout');
  console.log('PASS: rejected clip play() is silent and the queue advances');
}

async function main() {
  await testTalkWithoutAudioIsSilentAndQueueAdvances();
  await testRejectedClipPlayIsSilentAndQueueAdvances();
  console.log('All talkNoWebSpeech tests passed.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
