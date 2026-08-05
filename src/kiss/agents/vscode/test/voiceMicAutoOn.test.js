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

function makeWindow(mode, storedEnabled) {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body>' +
      '<button id="voice-btn" class="toggle-btn"></button>' +
      '<textarea id="task-input"></textarea>' +
      '</body></html>',
    {runScripts: 'dangerously', url: 'https://localhost/'},
  );
  const win = dom.window;
  if (storedEnabled !== undefined) {
    win.localStorage.setItem('kissVoiceEnabled', storedEnabled);
  }
  win.__VOICE__ = {mode};
  const posted = [];
  win.addEventListener('kiss-voice-post', e => posted.push(e.detail));
  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);
  return {win, posted};
}

test('webview mode: fresh storage (first install) keeps the mic OFF', () => {
  const {win, posted} = makeWindow('webview');
  assert.strictEqual(posted.length, 0);
  const btn = win.document.getElementById('voice-btn');
  assert.ok(btn.classList.contains('voice-off'));
});

test('webview mode: explicit opt-in (stored "1") auto-enables on load', () => {
  const {win, posted} = makeWindow('webview', '1');
  assert.strictEqual(
    JSON.stringify(posted),
    JSON.stringify([{type: 'voiceToggle', enabled: true, sensitivity: 80}]),
  );
  const btn = win.document.getElementById('voice-btn');
  assert.ok(btn.classList.contains('voice-loading'));
});

test('webview mode: explicit user opt-out (stored "0") stays off', () => {
  const {win, posted} = makeWindow('webview', '0');
  assert.strictEqual(posted.length, 0);
  const btn = win.document.getElementById('voice-btn');
  assert.ok(btn.classList.contains('voice-off'));
});

test('webview mode: turning the mic on is remembered for next launch', () => {
  const first = makeWindow('webview');
  const btn = first.win.document.getElementById('voice-btn');
  btn.click();
  assert.strictEqual(
    JSON.stringify(first.posted[0]),
    JSON.stringify({type: 'voiceToggle', enabled: true, sensitivity: 80}),
  );
  assert.strictEqual(
    first.win.localStorage.getItem('kissVoiceEnabled'),
    '1',
  );
  const second = makeWindow('webview', '1');
  assert.strictEqual(
    JSON.stringify(second.posted),
    JSON.stringify([{type: 'voiceToggle', enabled: true, sensitivity: 80}]),
  );
});

test('webview mode: turning the mic off is remembered for next launch', () => {
  const first = makeWindow('webview', '1');
  const btn = first.win.document.getElementById('voice-btn');
  btn.click();
  assert.strictEqual(
    JSON.stringify(first.posted[1]),
    JSON.stringify({type: 'voiceToggle', enabled: false, sensitivity: 80}),
  );
  assert.strictEqual(
    first.win.localStorage.getItem('kissVoiceEnabled'),
    '0',
  );
});

test('browser mode: fresh storage does NOT auto-start the mic', () => {
  const {win, posted} = makeWindow('browser');
  assert.strictEqual(posted.length, 0);
  const btn = win.document.getElementById('voice-btn');
  assert.ok(btn.classList.contains('voice-off'));
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
