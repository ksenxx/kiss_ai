// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the speaker prefix and the auto-submit signal in
// media/voice.js (VS Code webview mode).
//
// After the "Sorcar" wake word, the extension host translates the
// speech that followed and identifies the speaker; the webview
// receives ``{type: 'voiceSpeech', text, speaker}``.  voice.js must:
//
//  1. Insert the translation prefixed with ``Speaker #N says that: ``
//     when a speaker number is present.
//  2. Insert the bare translation when no speaker was identified
//     (legacy string payloads or a failed speaker model).
//  3. Raise exactly one ``kiss-voice-submit`` window event per
//     non-empty translation, so main.js submits the task to the agent
//     in the highlighted tab (or steers a running one).
//  4. Never raise the submit event (nor touch the input) for empty,
//     blank, or non-string translations.
//  5. Reject bogus speaker values (0, negatives, fractions, strings)
//     by inserting without a prefix.
//
// Runs the real media/voice.js in a real jsdom document — no mocks
// for the code under test.  Run with:
//
//     node test/voiceSpeakerPrefixSubmit.test.js

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
 * Build a fresh jsdom window with the elements voice.js needs, inject
 * the webview-mode config, run the real voice.js and count both
 * 'input' events on the textarea and 'kiss-voice-submit' events on
 * the window.
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

  const counters = {input: 0, submit: 0};
  win.document
    .getElementById('task-input')
    .addEventListener('input', () => counters.input++);
  win.addEventListener('kiss-voice-submit', () => counters.submit++);

  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);

  return {win, counters};
}

function sendHostMessage(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

// ---------------------------------------------------------------------------

test('speaker payload inserts prefixed text and submits once', () => {
  const {win, counters} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceSpeech', text: 'Fix the bug', speaker: 1});
  assert.strictEqual(inp.value, 'Speaker #1 says that: Fix the bug');
  assert.strictEqual(counters.input, 1);
  assert.strictEqual(counters.submit, 1);
});

test('each speaker number appears in its own prefix', () => {
  const {win, counters} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceSpeech', text: 'first task', speaker: 1});
  assert.strictEqual(inp.value, 'Speaker #1 says that: first task');
  inp.value = ''; // main.js clears the input after a real submit
  sendHostMessage(win, {type: 'voiceSpeech', text: 'second task', speaker: 2});
  assert.strictEqual(inp.value, 'Speaker #2 says that: second task');
  sendHostMessage(win, {type: 'voiceSpeech', text: 'third task', speaker: 1});
  assert.strictEqual(
    inp.value,
    'Speaker #2 says that: second task Speaker #1 says that: third task',
  );
  assert.strictEqual(counters.submit, 3);
});

test('payload without a speaker inserts bare text and submits', () => {
  const {win, counters} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceSpeech', text: 'Hello everyone'});
  assert.strictEqual(inp.value, 'Hello everyone');
  assert.strictEqual(counters.submit, 1);
});

test('translation is trimmed before prefixing', () => {
  const {win} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {
    type: 'voiceSpeech',
    text: '  Run the tests \n',
    speaker: 3,
  });
  assert.strictEqual(inp.value, 'Speaker #3 says that: Run the tests');
});

test('prefixed speech appends to an existing draft with a space', () => {
  const {win} = makeWindow();
  const inp = win.document.getElementById('task-input');
  inp.value = 'draft text';
  sendHostMessage(win, {type: 'voiceSpeech', text: 'and more', speaker: 2});
  assert.strictEqual(inp.value, 'draft text Speaker #2 says that: and more');
});

test('empty translation never submits nor touches the input', () => {
  const {win, counters} = makeWindow();
  const inp = win.document.getElementById('task-input');
  inp.value = 'precious draft';
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceSpeech', text: '', speaker: 1});
  sendHostMessage(win, {type: 'voiceSpeech', text: '   ', speaker: 2});
  sendHostMessage(win, {type: 'voiceSpeech', text: 42, speaker: 1});
  sendHostMessage(win, {type: 'voiceSpeech'});
  assert.strictEqual(inp.value, 'precious draft');
  assert.strictEqual(counters.input, 0);
  assert.strictEqual(counters.submit, 0);
});

test('bogus speaker values insert without a prefix', () => {
  for (const speaker of [0, -1, 1.5, '2', null, NaN, Infinity]) {
    const {win, counters} = makeWindow();
    const inp = win.document.getElementById('task-input');
    sendHostMessage(win, {type: 'voiceSpeech', text: 'Do it', speaker});
    assert.strictEqual(
      inp.value,
      'Do it',
      `speaker=${String(speaker)} must not produce a prefix`,
    );
    assert.strictEqual(counters.submit, 1);
  }
});

test('the word sorcar never reaches the input on wake', () => {
  const {win} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceWake'});
  assert.strictEqual(inp.value, '');
});

// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
