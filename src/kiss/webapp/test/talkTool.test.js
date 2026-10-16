// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send, stubRaf} = require('./simplify2_harness');
const {installSpeech} = require('./speech_helpers');

function installAudio(win) {
  const created = [];
  win.Audio = function Audio(src) {
    created.push(this);
    this.src = src;
    this.play = () => Promise.resolve();
  };
  return created;
}

const B64 = 'SUQzBAAAAAAAAA==';

function testAudiolessTalkIsSilentAndQueueAdvances() {
  const {win} = makePage({beforeScripts: stubRaf});
  const spoken = installSpeech(win);
  const created = installAudio(win);
  const activeTab = win._testApi.getActiveTabId();

  send(win, {type: 'talk', language: 'es', text: 'hola usuario',
             talkId: 'tt-noaudio', tabId: activeTab});

  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');
  assert.strictEqual(created.length, 0, 'no clip to play — stays silent');

  send(win, {type: 'talk', language: 'es', text: 'con audio',
             talkId: 'tt-audio', tabId: activeTab,
             audioB64: B64, audioMime: 'audio/mpeg'});

  assert.strictEqual(created.length, 1, 'queue advanced to the clip talk');
  assert.strictEqual(created[0].src, 'data:audio/mpeg;base64,' + B64);
  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');
  console.log('PASS: audio-less talk is silent and the queue advances');
}

function testTalkForOtherWindowsTabStaysSilent() {
  const {win} = makePage({beforeScripts: stubRaf});
  const spoken = installSpeech(win);
  const created = installAudio(win);

  send(win, {type: 'talk', language: 'de', text: 'hallo',
             tabId: 'some-other-windows-tab', audioB64: B64});

  assert.strictEqual(created.length, 0, "another window's copy is silent");
  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');
  console.log("PASS: talk copy for another window's tab stays silent");
}

function testTalkWithoutLanguagePlaysClip() {
  const {win} = makePage({beforeScripts: stubRaf});
  const spoken = installSpeech(win);
  const created = installAudio(win);

  send(win, {type: 'talk', text: 'plain default', talkId: 'tt-nolang',
             audioB64: B64});

  assert.strictEqual(created.length, 1, 'clip plays without a language');
  assert.strictEqual(created[0].src, 'data:audio/mpeg;base64,' + B64);
  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');
  console.log('PASS: talk without language still plays the clip');
}

function testTalkEmptyTextIsIgnored() {
  const {win} = makePage({beforeScripts: stubRaf});
  const spoken = installSpeech(win);
  const created = installAudio(win);

  send(win, {type: 'talk', language: 'en'});
  send(win, {type: 'talk', language: 'en', text: ''});
  send(win, {type: 'talk', language: 'en', text: '', audioB64: B64});

  assert.strictEqual(created.length, 0, 'empty text must not play');
  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');
  console.log('PASS: talk with empty/missing text is ignored');
}

testAudiolessTalkIsSilentAndQueueAdvances();
testTalkForOtherWindowsTabStaysSilent();
testTalkWithoutLanguagePlaysClip();
testTalkEmptyTextIsIgnored();
console.log('All talkTool tests passed.');
// The pages are left with a clip "playing": its stall sampler
// (main.js TALK_STALL_CHECK_MS interval) would keep node alive.
process.exit(0);
