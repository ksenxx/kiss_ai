// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test: the mic button must not keep blinking
// YELLOW (class 'voice-transcribing') after the user's speech was
// already delivered.
//
// Root cause of the reported bug: voice.js counts outstanding voice
// rounds (one per {type: 'voiceWake'}) and clears the flash only when
// the count returns to zero.  A terminal {type: 'voiceSpeech'} event
// can be lost — VS Code drops webview.postMessage() to a hidden
// webview, and the host used to drop malformed SPEECH payload lines —
// which leaks the counter.  From then on EVERY utterance ended with
// keepFlash=true, so the yellow flash stayed blinking for its full
// 60s safety timeout even though the spoken text was already inserted.
//
// The fix makes the flash safety timeout self-heal the state machine:
// when the timer fires (nothing arrived for the whole timeout), the
// leaked round counter is reset, so the next wake -> transcribing ->
// speech cycle clears the flash immediately again.
//
// This test runs the real ``media/voice.js`` against a real jsdom
// document (no mocks for the code under test); only the window's
// setTimeout is recorded so the test can fire the 60s safety timer
// without waiting a minute.
//
// Run directly with ``node test/voiceMicFlashStuckYellow.test.js``.

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
 * needs (#voice-btn, #task-input), record every setTimeout callback
 * scheduled by the page (so the test can fire the long flash safety
 * timers deterministically), inject the webview-mode config and
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

  // Record timers instead of running them: the yellow flash uses a
  // 60s safety timeout that the test fires by hand.
  const timers = [];
  win.setTimeout = (fn, ms) => {
    timers.push({fn, ms, cleared: false});
    return timers.length; // 1-based timer id
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

/** Fire the most recently scheduled still-pending timer callback. */
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

// ---------------------------------------------------------------------------

test(
  'speech after a lost terminal event must not leave the yellow ' +
    'flash blinking (safety timeout self-heals the round counter)',
  () => {
    const {win, timers} = makeWindow();
    const btn = win.document.getElementById('voice-btn');
    const inp = win.document.getElementById('task-input');

    // Round A: wake heard, transcription started ... but the terminal
    // voiceSpeech never arrives (hidden-webview postMessage drop or a
    // malformed SPEECH payload dropped by the host).
    win.Date.now = () => 1000000;
    sendHostMessage(win, {type: 'voiceWake'});
    sendHostMessage(win, {type: 'voiceTranscribing'});
    assert.ok(
      btn.classList.contains('voice-transcribing'),
      'round A must show the yellow transcribing flash',
    );

    // Nothing arrives for the full 60s safety window; the timer fires.
    const ms = firePendingTimer(timers);
    assert.strictEqual(ms, 60000, 'yellow safety timer must be 60s');
    assert.ok(
      !btn.classList.contains('voice-transcribing'),
      'the safety timeout must clear the stale yellow flash',
    );

    // Round B (a fresh utterance, past the 2s wake cooldown): the
    // user speaks and the translated text arrives normally.
    win.Date.now = () => 1010000;
    sendHostMessage(win, {type: 'voiceWake'});
    sendHostMessage(win, {type: 'voiceTranscribing'});
    assert.ok(btn.classList.contains('voice-transcribing'));
    sendHostMessage(win, {type: 'voiceSpeech', text: 'hello world'});

    // THE BUG: the leaked round A kept outstandingRounds > 0, so the
    // just-delivered speech kept the yellow flash blinking for 60s.
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

  // Leak a round, let the safety timer heal it.
  win.Date.now = () => 2000000;
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  firePendingTimer(timers);

  // Several normal rounds afterwards: none may leave the yellow flash.
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
  // The legitimate keepFlash behavior must survive the fix: while a
  // NEWER round is transcribing, the older round's terminal must not
  // clear the newer round's yellow flash.
  const {win} = makeWindow();
  const btn = win.document.getElementById('voice-btn');

  win.Date.now = () => 3000000;
  sendHostMessage(win, {type: 'voiceWake'}); // round 1
  sendHostMessage(win, {type: 'voiceTranscribing'});
  win.Date.now = () => 3010000;
  sendHostMessage(win, {type: 'voiceWake'}); // round 2 (overlaps)
  sendHostMessage(win, {type: 'voiceTranscribing'});

  // Round 1's late terminal: round 2 is still in flight, keep yellow.
  sendHostMessage(win, {type: 'voiceSpeech', text: 'first'});
  assert.ok(
    btn.classList.contains('voice-transcribing'),
    'the newer in-flight round still owns the yellow flash',
  );

  // Round 2's terminal: nothing left in flight, clear it.
  sendHostMessage(win, {type: 'voiceSpeech', text: 'second'});
  assert.ok(!btn.classList.contains('voice-transcribing'));
});

// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
