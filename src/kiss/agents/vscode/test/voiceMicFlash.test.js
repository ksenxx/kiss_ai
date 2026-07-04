// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the mic-button flash colors around the voice
// flow (webview mode):
//
//  - After the "Sorcar" wake word is heard ({type: 'voiceWake'}), the
//    mic button flashes GREEN: class 'voice-triggered' is added and
//    stays on while the host captures the speech that follows.
//  - When the host starts the gpt-audio transcription/translation call
//    ({type: 'voiceTranscribing'}), the flash turns YELLOW: class
//    'voice-transcribing' replaces 'voice-triggered'.
//  - When the translated text arrives ({type: 'voiceSpeech', text}),
//    both flash classes are cleared and the text is inserted.
//  - Silence ({type: 'voiceSpeech', text: ''}) also clears the flash
//    without touching the input.
//
// This test runs the real ``media/voice.js`` against a real jsdom
// document (no mocks for the code under test).
//
// Run directly with ``node test/voiceMicFlash.test.js``.

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

test('wake flashes the mic button green (voice-triggered)', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  sendHostMessage(win, {type: 'voiceWake'});
  assert.ok(
    btn.classList.contains('voice-triggered'),
    'wake must add the green voice-triggered class',
  );
  assert.ok(
    !btn.classList.contains('voice-transcribing'),
    'wake alone must not show the yellow transcribing flash',
  );
});

test('transcribing turns the flash yellow (voice-transcribing)', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  assert.ok(
    btn.classList.contains('voice-transcribing'),
    'voiceTranscribing must add the yellow voice-transcribing class',
  );
  assert.ok(
    !btn.classList.contains('voice-triggered'),
    'yellow must replace the green wake flash',
  );
});

test('translated speech clears the flash and inserts the text', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  sendHostMessage(win, {type: 'voiceSpeech', text: 'Fix the parser bug'});
  assert.ok(!btn.classList.contains('voice-triggered'));
  assert.ok(!btn.classList.contains('voice-transcribing'));
  assert.strictEqual(inp.value, 'Fix the parser bug');
});

test('silence clears the flash without touching the input', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  const inp = win.document.getElementById('task-input');
  inp.value = 'draft';
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceSpeech', text: ''});
  assert.ok(!btn.classList.contains('voice-triggered'));
  assert.ok(!btn.classList.contains('voice-transcribing'));
  assert.strictEqual(inp.value, 'draft');
});

test('transcribing without a prior wake still flashes yellow', () => {
  // Defensive: even if the WAKE line is lost, a TRANSCRIBING event
  // must still show the yellow indicator on its own.
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  sendHostMessage(win, {type: 'voiceTranscribing'});
  assert.ok(btn.classList.contains('voice-transcribing'));
});

test('a listener error clears the flash', () => {
  // If the Python listener dies mid-capture, the host reports the
  // error through voiceState; the stale green/yellow flash must not
  // linger until its long safety timeout.
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  sendHostMessage(win, {
    type: 'voiceState',
    listening: false,
    error: 'listener exited',
  });
  assert.ok(!btn.classList.contains('voice-triggered'));
  assert.ok(!btn.classList.contains('voice-transcribing'));
  assert.ok(btn.classList.contains('voice-error'));
});

test('the listener stopping clears the flash', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceState', listening: false});
  assert.ok(!btn.classList.contains('voice-triggered'));
  assert.ok(!btn.classList.contains('voice-transcribing'));
});

test('turning voice off locally clears the flash immediately', () => {
  // Clicking the mic toggle off must not leave a stale green/yellow
  // flash waiting for a host message that may never come.
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  btn.click(); // enable listening
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  btn.click(); // disable listening
  assert.ok(!btn.classList.contains('voice-triggered'));
  assert.ok(!btn.classList.contains('voice-transcribing'));
});

test('full wake -> transcribing -> speech cycle repeats cleanly', () => {
  const win = makeWindow();
  const btn = win.document.getElementById('voice-btn');
  const inp = win.document.getElementById('task-input');
  for (let round = 0; round < 2; round++) {
    // A new wake is debounced for 2s; jump past the cooldown.
    win.Date.now = () => 1000000 + round * 10000;
    sendHostMessage(win, {type: 'voiceWake'});
    assert.ok(btn.classList.contains('voice-triggered'), `round ${round}`);
    sendHostMessage(win, {type: 'voiceTranscribing'});
    assert.ok(btn.classList.contains('voice-transcribing'), `round ${round}`);
    sendHostMessage(win, {type: 'voiceSpeech', text: `part${round}`});
    assert.ok(!btn.classList.contains('voice-triggered'), `round ${round}`);
    assert.ok(!btn.classList.contains('voice-transcribing'), `round ${round}`);
  }
  assert.strictEqual(inp.value, 'part0 part1');
});

// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
