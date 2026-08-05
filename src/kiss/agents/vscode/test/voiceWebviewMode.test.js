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

  const posted = [];
  win.addEventListener('kiss-voice-post', event => {
    posted.push(event.detail);
  });

  const inp = win.document.getElementById('task-input');
  const inputEvents = {count: 0};
  inp.addEventListener('input', () => inputEvents.count++);
  const focusCalls = {count: 0};
  inp.addEventListener('focus', () => focusCalls.count++);

  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);

  return {win, posted, inputEvents, focusCalls};
}

function sendHostMessage(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

test('voiceWake message never types text into the task input', () => {
  const {win, inputEvents, focusCalls} = makeWindow();
  const inp = win.document.getElementById('task-input');
  const btn = win.document.getElementById('voice-btn');
  assert.strictEqual(inp.value, '');
  sendHostMessage(win, {type: 'voiceWake'});
  assert.strictEqual(inp.value, '');
  assert.strictEqual(inputEvents.count, 0);
  assert.ok(btn.classList.contains('voice-triggered'));
  assert.strictEqual(focusCalls.count, 1);
});

test('duplicate wake events within the cooldown fire only once', () => {
  const {win, focusCalls} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceWake'});
  inp.blur();
  sendHostMessage(win, {type: 'voiceWake'});
  inp.blur();
  sendHostMessage(win, {type: 'voiceWake'});
  assert.strictEqual(focusCalls.count, 1);
});

test('clicking the toggle posts voiceToggle through the bridge', () => {
  const {win, posted} = makeWindow();
  assert.strictEqual(
    JSON.stringify(posted),
    JSON.stringify([{type: 'voiceToggle', enabled: true, sensitivity: 80}]),
  );
  const btn = win.document.getElementById('voice-btn');
  btn.click();
  assert.strictEqual(
    JSON.stringify(posted[1]),
    JSON.stringify({type: 'voiceToggle', enabled: false, sensitivity: 80}),
  );
  btn.click();
  assert.strictEqual(
    JSON.stringify(posted[2]),
    JSON.stringify({type: 'voiceToggle', enabled: true, sensitivity: 80}),
  );
});

test('mic state persists in localStorage and is applied on load', () => {
  const first = makeWindow();
  first.win.document.getElementById('voice-btn').click();
  assert.strictEqual(
    first.win.localStorage.getItem('kissVoiceEnabled'),
    '0',
  );
  const second = makeWindow();
  assert.strictEqual(
    JSON.stringify(second.posted),
    JSON.stringify([{type: 'voiceToggle', enabled: true, sensitivity: 80}]),
  );
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body>' +
      '<button id="voice-btn"></button>' +
      '<textarea id="task-input"></textarea>' +
      '</body></html>',
    {runScripts: 'dangerously', url: 'https://localhost/'},
  );
  const win = dom.window;
  win.__VOICE__ = {mode: 'webview'};
  const posted = [];
  win.addEventListener('kiss-voice-post', e => posted.push(e.detail));
  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);
  assert.strictEqual(posted.length, 0);
  const dom2 = new JSDOM(
    '<!DOCTYPE html><html><body>' +
      '<button id="voice-btn"></button>' +
      '<textarea id="task-input"></textarea>' +
      '</body></html>',
    {runScripts: 'dangerously', url: 'https://localhost/'},
  );
  const win2 = dom2.window;
  win2.localStorage.setItem('kissVoiceEnabled', '0');
  win2.__VOICE__ = {mode: 'webview'};
  const posted2 = [];
  win2.addEventListener('kiss-voice-post', e => posted2.push(e.detail));
  const script2 = win2.document.createElement('script');
  script2.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win2.document.body.appendChild(script2);
  assert.strictEqual(posted2.length, 0);
});

test('voiceState messages drive the toggle UI classes', () => {
  const {win} = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  assert.ok(btn.classList.contains('voice-loading'));
  sendHostMessage(win, {type: 'voiceState', listening: true});
  assert.ok(btn.classList.contains('voice-listening'));
  assert.ok(btn.classList.contains('active'));
  sendHostMessage(win, {
    type: 'voiceState',
    listening: false,
    error: 'mic unavailable',
  });
  assert.ok(btn.classList.contains('voice-error'));
  assert.ok(btn.getAttribute('data-tooltip').includes('mic unavailable'));
  assert.strictEqual(win.localStorage.getItem('kissVoiceEnabled'), '0');
});

test('wake preserves the input when it already has other text', () => {
  const {win, focusCalls} = makeWindow();
  const inp = win.document.getElementById('task-input');
  inp.value = 'draft text';
  sendHostMessage(win, {type: 'voiceWake'});
  assert.strictEqual(inp.value, 'draft text');
  assert.strictEqual(focusCalls.count, 1);
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
