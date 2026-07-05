// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test (webview mode): the wake word keeps
// working while a previous transcription is still in flight.
//
// Since the listener fix, the extension host may deliver a second
// {type: 'voiceWake'} BEFORE the first utterance's {type:
// 'voiceSpeech'} arrives (translations are reported asynchronously).
// The webview must:
//
//  - accept that interleaved second wake (flash green again) instead
//    of ignoring it,
//  - still insert the first utterance's late translation, and
//  - insert both translations in arrival order.
//
// This test runs the real ``media/voice.js`` against a real jsdom
// document (no mocks for the code under test).
//
// Run directly with ``node test/voiceWakeDuringTranscription.test.js``.

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

/**
 * Build a fresh jsdom window containing the two elements voice.js
 * needs (#voice-btn, #task-input), inject the webview-mode config and
 * execute the real voice.js.
 */
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

// ---------------------------------------------------------------------------

test('a second wake during an in-flight transcription still triggers', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  // Round 1: wake + capture end; the gpt-audio call is now in flight.
  win.Date.now = () => 1000000;
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  assert.ok(btn.classList.contains('voice-transcribing'));
  // Round 2 starts BEFORE round 1's translation arrives (the listener
  // keeps detecting while the API call runs on a background thread).
  win.Date.now = () => 1010000; // past the 2s wake cooldown
  sendHostMessage(win, {type: 'voiceWake'});
  assert.ok(
    btn.classList.contains('voice-triggered'),
    'second wake must flash green even while a transcription is in flight',
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
  // Round 1's translation arrives only now.
  sendHostMessage(win, {type: 'voiceSpeech', text: 'first utterance'});
  assert.strictEqual(inp.value, 'first utterance');
  // Round 2's translation appends.
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
  // Round 1 turns out to be a failed/silent translation.
  sendHostMessage(win, {type: 'voiceSpeech', text: ''});
  assert.strictEqual(inp.value, '');
  // Round 2 completes normally.
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
  sendHostMessage(win, {type: 'voiceWake'}); // round 1
  sendHostMessage(win, {type: 'voiceTranscribing'}); // round 1 in flight
  win.Date.now = () => 1010000;
  sendHostMessage(win, {type: 'voiceWake'}); // round 2 capturing (green)
  // Round 1's late terminal result must insert its text but NOT
  // clear the green flash that belongs to round 2's active capture.
  sendHostMessage(win, {type: 'voiceSpeech', text: 'first'});
  assert.strictEqual(inp.value, 'first');
  assert.ok(
    btn.classList.contains('voice-triggered'),
    'late round-1 result must not clear round 2\'s green flash',
  );
  // Round 2 finishes normally: yellow, then terminal clears all.
  sendHostMessage(win, {type: 'voiceTranscribing'});
  assert.ok(btn.classList.contains('voice-transcribing'));
  sendHostMessage(win, {type: 'voiceSpeech', text: 'second'});
  assert.strictEqual(inp.value, 'first second');
  assert.ok(!btn.classList.contains('voice-triggered'));
  assert.ok(!btn.classList.contains('voice-transcribing'));
});

// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
