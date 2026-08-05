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
  assert.strictEqual(inputEvents.count, 1);
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

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
