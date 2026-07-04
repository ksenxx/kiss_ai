// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for the "Sorcar" voice wake word in VS Code webview
// mode (media/voice.js).
//
// Extension webviews cannot capture the microphone, so voice.js runs in
// "webview" mode there: the extension host owns the local Vosk listener
// and forwards ``{type: 'voiceWake'}`` messages; the toggle button posts
// ``{type: 'voiceToggle', enabled}`` back through the 'kiss-voice-post'
// bridge event.  This test runs the real ``media/voice.js`` against a
// real jsdom document (no mocks for the code under test) and locks in:
//
//  1. A voiceWake message types "sorcar" into #task-input, fires the
//     'input' event (so main.js reacts), and keeps listening.
//  2. Clicking the toggle posts voiceToggle {enabled: true/false}
//     through the kiss-voice-post bridge.
//  3. voiceState messages drive the button UI (listening / error).
//  4. Rapid duplicate wake events are debounced by the cooldown.
//
// Run directly with ``node test/voiceWebviewMode.test.js``.

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
 * Build a fresh jsdom window whose document contains the two elements
 * voice.js needs (#voice-btn, #task-input), inject the webview-mode
 * config, and execute the real voice.js.  Returns the window plus the
 * list of messages posted through the 'kiss-voice-post' bridge and a
 * counter of 'input' events seen on the textarea.
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

  const posted = [];
  win.addEventListener('kiss-voice-post', event => {
    posted.push(event.detail);
  });

  const inputEvents = {count: 0};
  win.document
    .getElementById('task-input')
    .addEventListener('input', () => inputEvents.count++);

  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);

  return {win, posted, inputEvents};
}

function sendHostMessage(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

// ---------------------------------------------------------------------------

test('voiceWake message types "sorcar" into the task input', () => {
  const {win, inputEvents} = makeWindow();
  const inp = win.document.getElementById('task-input');
  assert.strictEqual(inp.value, '');
  sendHostMessage(win, {type: 'voiceWake'});
  assert.strictEqual(inp.value, 'sorcar');
  assert.strictEqual(inputEvents.count, 1);
});

test('duplicate wake events within the cooldown fire only once', () => {
  const {win, inputEvents} = makeWindow();
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceWake'});
  assert.strictEqual(inputEvents.count, 1);
});

test('clicking the toggle posts voiceToggle through the bridge', () => {
  const {win, posted} = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  btn.click();
  // Note: JSON comparison because jsdom detail objects come from
  // another realm (different Object prototype than the test's).
  assert.strictEqual(
    JSON.stringify(posted),
    JSON.stringify([{type: 'voiceToggle', enabled: true}]),
  );
  btn.click();
  assert.strictEqual(
    JSON.stringify(posted[1]),
    JSON.stringify({type: 'voiceToggle', enabled: false}),
  );
});

test('enabled state persists in localStorage and re-arms on load', () => {
  const first = makeWindow();
  first.win.document.getElementById('voice-btn').click();
  assert.strictEqual(
    first.win.localStorage.getItem('kissVoiceEnabled'),
    '1',
  );
  // A brand-new window (fresh storage) must NOT auto-enable.
  const second = makeWindow();
  assert.strictEqual(second.posted.length, 0);
  // But a window whose storage says enabled must re-request listening
  // on load — seed the flag BEFORE voice.js executes.
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body>' +
      '<button id="voice-btn"></button>' +
      '<textarea id="task-input"></textarea>' +
      '</body></html>',
    {runScripts: 'dangerously', url: 'https://localhost/'},
  );
  const win = dom.window;
  win.localStorage.setItem('kissVoiceEnabled', '1');
  win.__VOICE__ = {mode: 'webview'};
  const posted = [];
  win.addEventListener('kiss-voice-post', e => posted.push(e.detail));
  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);
  assert.strictEqual(
    JSON.stringify(posted),
    JSON.stringify([{type: 'voiceToggle', enabled: true}]),
  );
});

test('voiceState messages drive the toggle UI classes', () => {
  const {win} = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  btn.click(); // enable → 'loading' until the host confirms
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
  // After an error the stored preference must be off so the next
  // reload does not spin forever.
  assert.strictEqual(win.localStorage.getItem('kissVoiceEnabled'), '0');
});

test('wake still works while the input already has other text', () => {
  const {win} = makeWindow();
  const inp = win.document.getElementById('task-input');
  inp.value = 'draft text';
  sendHostMessage(win, {type: 'voiceWake'});
  assert.strictEqual(inp.value, 'sorcar');
});

// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
