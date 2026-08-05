// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win};
}

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

function installAudio(win, playResult) {
  const created = [];
  win.Audio = function Audio(src) {
    created.push(src);
    this.play = () => playResult;
  };
  return created;
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

const B64 = 'SUQzBAAAAAAAAA==';

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

function testNoAudioApiDegradesToSilenceAndQueueAdvances() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  win.Audio = undefined;

  send(win, {type: 'talk', language: 'en', text: 'silence please',
             talkId: 'a3', audioB64: B64});

  assert.strictEqual(spoken.length, 0, 'never falls back to Web Speech');
  const created = installAudio(win, Promise.resolve());
  send(win, {type: 'talk', language: 'en', text: 'audio works now',
             talkId: 'a3b', audioB64: B64});
  assert.strictEqual(created.length, 1, 'queue advanced past silent talk');
  assert.strictEqual(spoken.length, 0, 'Web Speech API must stay silent');
  console.log('PASS: missing Audio API degrades to silence, queue advances');
}

async function testRejectedPlayDegradesToSilenceAndQueueAdvances() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  installAudio(win, Promise.reject(new Error('autoplay blocked')));

  send(win, {type: 'talk', language: 'en', text: 'blocked audio',
             talkId: 'a4', audioB64: B64});

  await new Promise(resolve => setTimeout(resolve, 0));
  assert.strictEqual(spoken.length, 0, 'never falls back to Web Speech');
  const created = installAudio(win, Promise.resolve());
  send(win, {type: 'talk', language: 'en', text: 'unblocked audio',
             talkId: 'a4b', audioB64: B64});
  assert.strictEqual(created.length, 1, 'queue advanced past blocked talk');
  assert.strictEqual(spoken.length, 0, 'Web Speech API must stay silent');
  console.log('PASS: rejected play() degrades to silence, queue advances');
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

function testThrowingAudioConstructorDegradesToSilence() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  win.Audio = function Audio() {
    throw new Error('data URLs forbidden');
  };

  send(win, {type: 'talk', text: 'never spoken', talkId: 'a5',
             audioB64: B64});

  assert.strictEqual(spoken.length, 0, 'never falls back to Web Speech');
  const created = installAudio(win, Promise.resolve());
  send(win, {type: 'talk', text: 'plays fine', talkId: 'a5b',
             audioB64: B64});
  assert.strictEqual(created.length, 1, 'queue advanced past the failure');
  console.log('PASS: throwing Audio constructor degrades to silence');
}

async function main() {
  testAudioEventPlaysAudioNotSpeech();
  testMissingMimeDefaultsToMpeg();
  testNoAudioApiDegradesToSilenceAndQueueAdvances();
  await testRejectedPlayDegradesToSilenceAndQueueAdvances();
  testDuplicateTalkIdPlaysOnce();
  testThrowingAudioConstructorDegradesToSilence();
  console.log('All talkGptAudio tests passed.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
