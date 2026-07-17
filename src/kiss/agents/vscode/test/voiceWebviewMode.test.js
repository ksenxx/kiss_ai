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
//  1. A voiceWake message never types text into #task-input — it only
//     flashes the mic button, focuses the input, and keeps listening.
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
  // The mic is closed by default (fresh install); these tests exercise
  // the listening flow, so seed the explicit opt-in that auto-enables
  // the mic at load.  Default-off behavior is locked in by
  // voiceMicAutoOn.test.js and the persistence test below.
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

// ---------------------------------------------------------------------------

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
  // Blur between wakes: a non-debounced triggerWake would refocus the
  // input each time, so the focus counter reveals every extra firing.
  sendHostMessage(win, {type: 'voiceWake'});
  inp.blur();
  sendHostMessage(win, {type: 'voiceWake'});
  inp.blur();
  sendHostMessage(win, {type: 'voiceWake'});
  assert.strictEqual(focusCalls.count, 1);
});

test('clicking the toggle posts voiceToggle through the bridge', () => {
  const {win, posted} = makeWindow();
  // The seeded opt-in auto-enables the mic at load, so the first
  // click turns it OFF and the second back ON.
  // Note: JSON comparison because jsdom detail objects come from
  // another realm (different Object prototype than the test's).
  assert.strictEqual(
    JSON.stringify(posted),
    JSON.stringify([{type: 'voiceToggle', enabled: true, sensitivity: 85}]),
  );
  const btn = win.document.getElementById('voice-btn');
  btn.click();
  assert.strictEqual(
    JSON.stringify(posted[1]),
    JSON.stringify({type: 'voiceToggle', enabled: false, sensitivity: 85}),
  );
  btn.click();
  assert.strictEqual(
    JSON.stringify(posted[2]),
    JSON.stringify({type: 'voiceToggle', enabled: true, sensitivity: 85}),
  );
});

test('mic state persists in localStorage and is applied on load', () => {
  // The seeded opt-in auto-enables at load; the first click therefore
  // disables and persists the opt-out.
  const first = makeWindow();
  first.win.document.getElementById('voice-btn').click();
  assert.strictEqual(
    first.win.localStorage.getItem('kissVoiceEnabled'),
    '0',
  );
  // A window with a stored opt-in ('1') MUST auto-enable (makeWindow
  // seeds the opt-in).
  const second = makeWindow();
  assert.strictEqual(
    JSON.stringify(second.posted),
    JSON.stringify([{type: 'voiceToggle', enabled: true, sensitivity: 85}]),
  );
  // A brand-new window (fresh storage — first install) must stay OFF:
  // the mic is closed by default so Sorcar never responds to the wake
  // word until the user turns it on.  Build the dom manually so no
  // opt-in is seeded.
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
  // And a stored opt-out ('0') also stays off.
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
  // Auto-enabled at load → 'loading' until the host confirms.
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

test('wake preserves the input when it already has other text', () => {
  const {win, focusCalls} = makeWindow();
  const inp = win.document.getElementById('task-input');
  inp.value = 'draft text';
  sendHostMessage(win, {type: 'voiceWake'});
  assert.strictEqual(inp.value, 'draft text');
  assert.strictEqual(focusCalls.count, 1);
});

// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
