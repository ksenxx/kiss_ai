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
  win.localStorage.setItem('kissVoiceEnabled', '1');

  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);

  return win;
}

function sendHostMessage(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

test('wake flashes the mic button red (voice-triggered)', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  sendHostMessage(win, {type: 'voiceWake'});
  assert.ok(
    btn.classList.contains('voice-triggered'),
    'wake must add the red voice-triggered class',
  );
  assert.ok(
    !btn.classList.contains('voice-transcribing'),
    'wake alone must not show the yellow transcribing flash',
  );
});

test('transcribing turns the flash yellow (voice-transcribing)', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  assert.ok(
    btn.classList.contains('voice-transcribing'),
    'voiceTranscribing must add the yellow voice-transcribing class',
  );
  assert.ok(
    !btn.classList.contains('voice-triggered'),
    'yellow must replace the red wake flash',
  );
});

test('translated speech clears the flash and inserts the text', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  sendHostMessage(win, {type: 'voiceSpeech', text: 'Fix the parser bug'});
  assert.ok(!btn.classList.contains('voice-triggered'));
  assert.ok(!btn.classList.contains('voice-transcribing'));
  assert.strictEqual(inp.value, 'Fix the parser bug');
});

test('silence clears the flash without touching the input', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  const inp = win.document.getElementById('task-input');
  inp.value = 'draft';
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceSpeech', text: ''});
  assert.ok(!btn.classList.contains('voice-triggered'));
  assert.ok(!btn.classList.contains('voice-transcribing'));
  assert.strictEqual(inp.value, 'draft');
});

test('transcribing without a prior wake still flashes yellow', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  sendHostMessage(win, {type: 'voiceTranscribing'});
  assert.ok(btn.classList.contains('voice-transcribing'));
});

test('a listener error clears the flash', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  sendHostMessage(win, {
    type: 'voiceState',
    listening: false,
    error: 'listener exited',
  });
  assert.ok(!btn.classList.contains('voice-triggered'));
  assert.ok(!btn.classList.contains('voice-transcribing'));
  assert.ok(btn.classList.contains('voice-error'));
});

test('the listener stopping clears the flash', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceState', listening: false});
  assert.ok(!btn.classList.contains('voice-triggered'));
  assert.ok(!btn.classList.contains('voice-transcribing'));
});

test('turning voice off locally clears the flash immediately', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  btn.click();
  assert.ok(!btn.classList.contains('voice-triggered'));
  assert.ok(!btn.classList.contains('voice-transcribing'));
});

test('full wake -> transcribing -> speech cycle repeats cleanly', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  const inp = win.document.getElementById('task-input');
  for (let round = 0; round < 2; round++) {
    win.Date.now = () => 1000000 + round * 10000;
    sendHostMessage(win, {type: 'voiceWake'});
    assert.ok(btn.classList.contains('voice-triggered'), `round ${round}`);
    sendHostMessage(win, {type: 'voiceTranscribing'});
    assert.ok(btn.classList.contains('voice-transcribing'), `round ${round}`);
    sendHostMessage(win, {type: 'voiceSpeech', text: `part${round}`});
    assert.ok(!btn.classList.contains('voice-triggered'), `round ${round}`);
    assert.ok(!btn.classList.contains('voice-transcribing'), `round ${round}`);
  }
  assert.strictEqual(inp.value, 'part0 part1');
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
