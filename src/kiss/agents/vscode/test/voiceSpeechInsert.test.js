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

  const inputEvents = {count: 0};
  win.document
    .getElementById('task-input')
    .addEventListener('input', () => inputEvents.count++);

  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);

  return {win, inputEvents};
}

function sendHostMessage(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

test('voiceSpeech after a wake types only the translated text', () => {
  const {win, inputEvents} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceWake'});
  assert.strictEqual(inp.value, '');
  sendHostMessage(win, {type: 'voiceSpeech', text: 'Hello everyone'});
  assert.strictEqual(inp.value, 'Hello everyone');
  assert.strictEqual(inputEvents.count, 1);
});

test('voiceSpeech types into an empty input', () => {
  const {win, inputEvents} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceSpeech', text: 'Fix the bug'});
  assert.strictEqual(inp.value, 'Fix the bug');
  assert.strictEqual(inputEvents.count, 1);
});

test('voiceSpeech appends to an existing user draft with a space', () => {
  const {win} = makeWindow();
  const inp = win.document.getElementById('task-input');
  inp.value = 'draft text';
  sendHostMessage(win, {type: 'voiceSpeech', text: 'and more'});
  assert.strictEqual(inp.value, 'draft text and more');
});

test('voiceSpeech trims surrounding whitespace from the translation', () => {
  const {win} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceSpeech', text: '  Bonjour to you \n'});
  assert.strictEqual(inp.value, 'Bonjour to you');
});

test('empty voiceSpeech after a wake leaves the input empty', () => {
  const {win, inputEvents} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceWake'});
  assert.strictEqual(inp.value, '');
  sendHostMessage(win, {type: 'voiceSpeech', text: ''});
  assert.strictEqual(inp.value, '');
  assert.strictEqual(inputEvents.count, 0);
});

test('blank voiceSpeech never touches other user text', () => {
  const {win, inputEvents} = makeWindow();
  const inp = win.document.getElementById('task-input');
  inp.value = 'precious draft';
  sendHostMessage(win, {type: 'voiceSpeech', text: '   '});
  assert.strictEqual(inp.value, 'precious draft');
  assert.strictEqual(inputEvents.count, 0);
});

test('non-string voiceSpeech payloads are treated as no speech', () => {
  const {win, inputEvents} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceSpeech', text: 42});
  assert.strictEqual(inp.value, '');
  sendHostMessage(win, {type: 'voiceSpeech'});
  assert.strictEqual(inp.value, '');
  assert.strictEqual(inputEvents.count, 0);
});

test('translated text can be inserted again after a new wake', () => {
  const {win} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceSpeech', text: 'first sentence'});
  sendHostMessage(win, {type: 'voiceSpeech', text: 'second sentence'});
  assert.strictEqual(inp.value, 'first sentence second sentence');
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
