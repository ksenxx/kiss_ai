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

  const timers = [];
  win.setTimeout = (fn, ms) => {
    timers.push({fn, ms, cleared: false});
    return timers.length;
  };
  win.clearTimeout = id => {
    if (typeof id === 'number' && timers[id - 1]) {
      timers[id - 1].cleared = true;
    }
  };

  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);

  return {win, timers};
}

function sendHostMessage(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function firePendingTimer(timers) {
  for (let i = timers.length - 1; i >= 0; i--) {
    if (!timers[i].cleared) {
      timers[i].cleared = true;
      timers[i].fn();
      return timers[i].ms;
    }
  }
  throw new Error('no pending timer to fire');
}

test(
  'speech after a lost terminal event must not leave the yellow ' +
    'flash blinking (safety timeout self-heals the round counter)',
  () => {
    const {win, timers} = makeWindow();
    const btn = win.document.getElementById('voice-btn');
    const inp = win.document.getElementById('task-input');

    win.Date.now = () => 1000000;
    sendHostMessage(win, {type: 'voiceWake'});
    sendHostMessage(win, {type: 'voiceTranscribing'});
    assert.ok(
      btn.classList.contains('voice-transcribing'),
      'round A must show the yellow transcribing flash',
    );

    const ms = firePendingTimer(timers);
    assert.strictEqual(ms, 60000, 'yellow safety timer must be 60s');
    assert.ok(
      !btn.classList.contains('voice-transcribing'),
      'the safety timeout must clear the stale yellow flash',
    );

    win.Date.now = () => 1010000;
    sendHostMessage(win, {type: 'voiceWake'});
    sendHostMessage(win, {type: 'voiceTranscribing'});
    assert.ok(btn.classList.contains('voice-transcribing'));
    sendHostMessage(win, {type: 'voiceSpeech', text: 'hello world'});

    assert.ok(
      !btn.classList.contains('voice-transcribing'),
      'yellow flash must clear once the spoken text was delivered',
    );
    assert.ok(!btn.classList.contains('voice-triggered'));
    assert.strictEqual(inp.value, 'hello world');
  },
);

test('every later utterance also clears the flash after the self-heal', () => {
  const {win, timers} = makeWindow();
  const btn = win.document.getElementById('voice-btn');

  win.Date.now = () => 2000000;
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  firePendingTimer(timers);

  for (let round = 1; round <= 3; round++) {
    win.Date.now = () => 2000000 + round * 10000;
    sendHostMessage(win, {type: 'voiceWake'});
    sendHostMessage(win, {type: 'voiceTranscribing'});
    sendHostMessage(win, {type: 'voiceSpeech', text: `part${round}`});
    assert.ok(
      !btn.classList.contains('voice-transcribing'),
      `round ${round} must clear the yellow flash`,
    );
  }
});

test('overlapping rounds still keep the flash for the newer round', () => {
  const {win} = makeWindow();
  const btn = win.document.getElementById('voice-btn');

  win.Date.now = () => 3000000;
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  win.Date.now = () => 3010000;
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});

  sendHostMessage(win, {type: 'voiceSpeech', text: 'first'});
  assert.ok(
    btn.classList.contains('voice-transcribing'),
    'the newer in-flight round still owns the yellow flash',
  );

  sendHostMessage(win, {type: 'voiceSpeech', text: 'second'});
  assert.ok(!btn.classList.contains('voice-transcribing'));
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
