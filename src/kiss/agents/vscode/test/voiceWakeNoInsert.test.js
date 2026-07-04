// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test: hearing the "Sorcar" wake word must NOT
// type the literal word "sorcar" (or any other text) into the task
// input textbox.
//
// Bug being reproduced: ``triggerWake()`` in media/voice.js set
// ``inp.value = 'sorcar'`` as a visible placeholder while the host
// recorded/translated the speech that followed.  Users saw the word
// "sorcar" appear in their input box — and any draft they had typed
// was destroyed by the placeholder.  The wake event must only show a
// transient visual indicator on the mic button; text appears in the
// input only when the translated speech arrives as
// ``{type: 'voiceSpeech', text}``.
//
// This test runs the real ``media/voice.js`` against a real jsdom
// document (no mocks for the code under test) and locks in:
//
//  1. A voiceWake message leaves an empty input EMPTY — the word
//     "sorcar" never appears.
//  2. A voiceWake message preserves an existing user draft untouched.
//  3. The wake event still gives visible feedback (the transient
//     'voice-triggered' class on the mic button) and focuses the input.
//  4. The full wake → translated-speech flow never shows "sorcar":
//     after wake the input is empty, then the translation is inserted.
//  5. Wake followed by silence (empty voiceSpeech) leaves the input
//     exactly as it was — nothing to clean up because nothing was
//     inserted.
//
// Run directly with ``node test/voiceWakeNoInsert.test.js``.

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
 * execute the real voice.js.  Returns the window plus a counter of
 * 'input' events and a counter of focus() calls on the textarea.
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

  const inp = win.document.getElementById('task-input');
  const inputEvents = {count: 0};
  inp.addEventListener('input', () => inputEvents.count++);
  const focusCalls = {count: 0};
  inp.addEventListener('focus', () => focusCalls.count++);

  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);

  return {win, inputEvents, focusCalls};
}

function sendHostMessage(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

// ---------------------------------------------------------------------------

test('voiceWake never types "sorcar" into an empty task input', () => {
  const {win, inputEvents} = makeWindow();
  const inp = win.document.getElementById('task-input');
  assert.strictEqual(inp.value, '');
  sendHostMessage(win, {type: 'voiceWake'});
  assert.strictEqual(
    inp.value,
    '',
    `wake inserted ${JSON.stringify(inp.value)} into the input box`,
  );
  assert.strictEqual(inputEvents.count, 0);
});

test('voiceWake preserves an existing user draft untouched', () => {
  const {win, inputEvents} = makeWindow();
  const inp = win.document.getElementById('task-input');
  inp.value = 'precious draft';
  sendHostMessage(win, {type: 'voiceWake'});
  assert.strictEqual(inp.value, 'precious draft');
  assert.strictEqual(inputEvents.count, 0);
});

test('voiceWake flashes the mic button and focuses the input', () => {
  const {win, focusCalls} = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  sendHostMessage(win, {type: 'voiceWake'});
  assert.ok(btn.classList.contains('voice-triggered'));
  assert.strictEqual(focusCalls.count, 1);
});

test('wake then translated speech: only the translation appears', () => {
  const {win, inputEvents} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceWake'});
  assert.strictEqual(inp.value, '');
  sendHostMessage(win, {type: 'voiceSpeech', text: 'Fix the parser bug'});
  assert.strictEqual(inp.value, 'Fix the parser bug');
  assert.strictEqual(inputEvents.count, 1); // only the speech insert
});

test('wake then silence leaves the input exactly as it was', () => {
  const {win, inputEvents} = makeWindow();
  const inp = win.document.getElementById('task-input');
  inp.value = 'draft';
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceSpeech', text: ''});
  assert.strictEqual(inp.value, 'draft');
  assert.strictEqual(inputEvents.count, 0);
});

// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
