// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send, stubRaf} = require('./simplify2_harness');

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

const B64 = 'SUQzBAAAAAAAAA==';

function testThrowingAudioConstructorDegradesToSilence() {
  const {win} = makePage({beforeScripts: stubRaf});
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

testThrowingAudioConstructorDegradesToSilence();
console.log('All talkGptAudio tests passed.');
// The page is left with a clip "playing": its stall sampler
// (main.js TALK_STALL_CHECK_MS interval) would keep node alive.
process.exit(0);
