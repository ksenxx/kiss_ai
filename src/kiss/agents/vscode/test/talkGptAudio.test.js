// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for GPT-synthesized ``talk`` audio playback: when a
// backend ``{type: 'talk', ...}`` event carries ``audioB64`` (base64
// MP3 synthesized server-side by speech_synthesis.py with a GPT audio
// model), the webview must play it through an Audio element instead of
// the Web Speech API, and must fall back to the Web Speech API when
// the Audio API is unavailable or playback is rejected (autoplay
// policy).
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/talkGptAudio.test.js

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

/** Install a recording Web Speech API on *win* (jsdom has none). */
function installSpeech(win) {
  const spoken = [];
  win.SpeechSynthesisUtterance = function SpeechSynthesisUtterance(text) {
    this.text = text;
    this.lang = '';
  };
  win.speechSynthesis = {
    speak: u => spoken.push(u),
  };
  return spoken;
}

/**
 * Install a recording Audio constructor on *win*.  ``playResult``
 * controls what ``play()`` returns (e.g. a resolved or rejected
 * promise, or undefined like older browsers).
 */
function installAudio(win, playResult) {
  const created = [];
  win.Audio = function Audio(src) {
    created.push(src);
    this.play = () => playResult;
  };
  return created;
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

const B64 = 'SUQzBAAAAAAAAA=='; // decodes to "ID3..." — an MP3 tag header

function testAudioEventPlaysAudioNotSpeech() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  const created = installAudio(win, Promise.resolve());

  send(win, {type: 'talk', language: 'en-US', text: 'hi there',
             emotion: 'cheerful', talkId: 'a1', audioB64: B64,
             audioMime: 'audio/mpeg'});

  assert.strictEqual(created.length, 1, 'exactly one Audio element');
  assert.strictEqual(created[0], 'data:audio/mpeg;base64,' + B64);
  assert.strictEqual(spoken.length, 0, 'Web Speech API must stay silent');
  console.log('PASS: talk event with audioB64 plays audio, not speech');
}

function testMissingMimeDefaultsToMpeg() {
  const {win} = makeWebview();
  installSpeech(win);
  const created = installAudio(win, Promise.resolve());

  send(win, {type: 'talk', text: 'hello', talkId: 'a2', audioB64: B64});

  assert.strictEqual(created.length, 1);
  assert.strictEqual(created[0], 'data:audio/mpeg;base64,' + B64);
  console.log('PASS: missing audioMime defaults to audio/mpeg');
}

function testNoAudioApiFallsBackToSpeech() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  win.Audio = undefined; // device without the Audio API

  send(win, {type: 'talk', language: 'en', text: 'fallback please',
             talkId: 'a3', audioB64: B64});

  assert.strictEqual(spoken.length, 1, 'falls back to Web Speech API');
  assert.strictEqual(spoken[0].text, 'fallback please');
  console.log('PASS: missing Audio API falls back to Web Speech');
}

async function testRejectedPlayFallsBackToSpeech() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  installAudio(win, Promise.reject(new Error('autoplay blocked')));

  send(win, {type: 'talk', language: 'en', text: 'blocked audio',
             talkId: 'a4', audioB64: B64});

  await new Promise(resolve => setTimeout(resolve, 0));
  assert.strictEqual(spoken.length, 1, 'rejected play() falls back');
  assert.strictEqual(spoken[0].text, 'blocked audio');
  console.log('PASS: rejected play() falls back to Web Speech');
}

function testDuplicateTalkIdPlaysOnce() {
  const {win} = makeWebview();
  installSpeech(win);
  const created = installAudio(win, Promise.resolve());

  const ev = {type: 'talk', text: 'once only', talkId: 'dup1',
              audioB64: B64};
  send(win, ev);
  send(win, ev);

  assert.strictEqual(created.length, 1, 'duplicate talkId plays once');
  console.log('PASS: duplicate talkId with audio plays exactly once');
}

function testThrowingAudioConstructorFallsBack() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  win.Audio = function Audio() {
    throw new Error('data URLs forbidden');
  };

  send(win, {type: 'talk', text: 'still spoken', talkId: 'a5',
             audioB64: B64});

  assert.strictEqual(spoken.length, 1, 'constructor failure falls back');
  assert.strictEqual(spoken[0].text, 'still spoken');
  console.log('PASS: throwing Audio constructor falls back to speech');
}

async function main() {
  testAudioEventPlaysAudioNotSpeech();
  testMissingMimeDefaultsToMpeg();
  testNoAudioApiFallsBackToSpeech();
  await testRejectedPlayFallsBackToSpeech();
  testDuplicateTalkIdPlaysOnce();
  testThrowingAudioConstructorFallsBack();
  console.log('All talkGptAudio tests passed.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
