// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const VOICE_JS_PATH = path.join(__dirname, '..', 'media', 'voice.js');

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.message}`);
  }
}

function makeWindow() {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body>' +
      '<button id="voice-btn" class="toggle-btn"></button>' +
      '<textarea id="task-input"></textarea>' +
      '</body></html>',
    {runScripts: 'dangerously', url: 'https://localhost/'},
  );
  const win = dom.window;
  win.__VOICE__ = {mode: 'webview'};

  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);

  return win;
}

function sendHostMessage(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

test('a second wake during an in-flight transcription still triggers', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  win.Date.now = () => 1000000;
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  assert.ok(btn.classList.contains('voice-transcribing'));
  win.Date.now = () => 1010000;
  sendHostMessage(win, {type: 'voiceWake'});
  assert.ok(
    btn.classList.contains('voice-triggered'),
    'second wake must flash red even while a transcription is in flight',
  );
  assert.ok(
    !btn.classList.contains('voice-transcribing'),
    'the new capture replaces the stale yellow flash',
  );
});

test('a late translation still lands after an interleaved wake', () => {
  const win = makeWindow();
  const inp = win.document.getElementById('task-input');
  win.Date.now = () => 1000000;
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  win.Date.now = () => 1010000;
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceSpeech', text: 'first utterance'});
  assert.strictEqual(inp.value, 'first utterance');
  sendHostMessage(win, {type: 'voiceTranscribing'});
  sendHostMessage(win, {type: 'voiceSpeech', text: 'second utterance'});
  assert.strictEqual(inp.value, 'first utterance second utterance');
});

test('interleaved silence result does not block later wakes or text', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  const inp = win.document.getElementById('task-input');
  win.Date.now = () => 1000000;
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  win.Date.now = () => 1010000;
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceSpeech', text: ''});
  assert.strictEqual(inp.value, '');
  win.Date.now = () => 1020000;
  sendHostMessage(win, {type: 'voiceTranscribing'});
  sendHostMessage(win, {type: 'voiceSpeech', text: 'still works'});
  assert.strictEqual(inp.value, 'still works');
  assert.ok(!btn.classList.contains('voice-triggered'));
  assert.ok(!btn.classList.contains('voice-transcribing'));
});

test('a late terminal event keeps the newer round\'s indicator', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  const inp = win.document.getElementById('task-input');
  win.Date.now = () => 1000000;
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  win.Date.now = () => 1010000;
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceSpeech', text: 'first'});
  assert.strictEqual(inp.value, 'first');
  assert.ok(
    btn.classList.contains('voice-triggered'),
    'late round-1 result must not clear round 2\'s red flash',
  );
  sendHostMessage(win, {type: 'voiceTranscribing'});
  assert.ok(btn.classList.contains('voice-transcribing'));
  sendHostMessage(win, {type: 'voiceSpeech', text: 'second'});
  assert.strictEqual(inp.value, 'first second');
  assert.ok(!btn.classList.contains('voice-triggered'));
  assert.ok(!btn.classList.contains('voice-transcribing'));
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
