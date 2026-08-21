// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// I-RC1: CANCELLED entries in voice.js's roundOwners map must not live
// for ever. A cancelled round's transcript can only still arrive within
// the 60s transcribe-flash window, so entries older than that are
// evicted; fresh entries keep failing closed (the late transcript is
// dropped, not typed into whatever conversation is on screen). Without
// eviction, every listener restart re-marked the surviving keys and the
// map grew without bound.

/* global require, process, console, __dirname, global, setTimeout, clearTimeout */

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const VOICE_JS_PATH = path.join(__dirname, '..', 'media', 'voice.js');

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

  // Deterministic clock, controllable from the test.
  const clock = {now: 1_000_000};
  win.Date.now = () => clock.now;

  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);
  return {win, clock};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function testFreshCancelledRoundStillFailsClosed() {
  const {win} = makeWindow();
  const inp = win.document.getElementById('task-input');

  send(win, {type: 'voiceWake', roundId: 1});
  // Listener stops: every outstanding round is cancelled.
  send(win, {type: 'voiceState', listening: false});
  // The transcript lands moments later: it must be refused.
  send(win, {type: 'voiceSpeech', roundId: 1, text: 'late words'});
  assert.strictEqual(
    inp.value,
    '',
    'a freshly cancelled round must keep failing closed',
  );
  console.log('  ok - fresh CANCELLED rounds still fail closed');
}

function testStaleCancelledRoundIsEvicted() {
  const {win, clock} = makeWindow();
  const inp = win.document.getElementById('task-input');

  send(win, {type: 'voiceWake', roundId: 1});
  send(win, {type: 'voiceState', listening: false});

  // Long after the transcribe-flash window, a new round starts (any
  // wake runs the eviction sweep) ...
  clock.now += 61_000;
  send(win, {type: 'voiceWake', roundId: 2});

  // ... and the ancient round's id no longer pins its transcript to a
  // cancelled round: it is treated like a transcript from a round this
  // webview never saw, which is allowed in.
  send(win, {type: 'voiceSpeech', roundId: 1, text: 'ancient words'});
  assert.strictEqual(
    inp.value,
    'ancient words',
    'a CANCELLED round older than 60s must have been evicted',
  );
  console.log('  ok - CANCELLED rounds older than the flash window evict');
}

function testRepeatedRestartsDoNotAccumulateCancelledDrops() {
  const {win, clock} = makeWindow();
  const inp = win.document.getElementById('task-input');

  // Many wake/restart cycles, each more than a window apart: only the
  // most recent cycle's rounds may still be CANCELLED.
  for (let round = 1; round <= 20; round++) {
    send(win, {type: 'voiceWake', roundId: round});
    send(win, {type: 'voiceState', listening: false});
    clock.now += 61_000;
  }
  // A final wake triggers the sweep for the last cancelled batch too.
  send(win, {type: 'voiceWake', roundId: 21});

  // Every old round's late transcript now behaves like an unknown
  // round (fail open), proving the entries are gone rather than
  // parked as CANCELLED for ever.
  send(win, {type: 'voiceSpeech', roundId: 7, text: 'seven'});
  assert.strictEqual(inp.value, 'seven');
  console.log('  ok - restart cycles do not accumulate CANCELLED entries');
}

try {
  testFreshCancelledRoundStillFailsClosed();
  testStaleCancelledRoundIsEvicted();
  testRepeatedRestartsDoNotAccumulateCancelledDrops();
  console.log('rr_area_hi_voice_cancelled_eviction: all tests passed');
  process.exit(0);
} catch (err) {
  console.error(err);
  process.exit(1);
}
