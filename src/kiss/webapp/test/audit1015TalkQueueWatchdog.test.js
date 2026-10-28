// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the talk queue's stall detector.
//
// A clip that neither ends nor errors (a decode that stalls after
// play() resolved fires none of ended/error/abort/pause) used to hold
// the queue busy for the life of the page, swallowing every later
// talk.  Each playing clip is now sampled every TALK_STALL_CHECK_MS;
// one whose currentTime stops moving for TALK_STALL_CHECKS samples is
// retired (paused, source dropped, handlers off) before the queue moves
// on, while a clip that keeps progressing holds the queue however long
// it lasts.

'use strict';

const assert = require('assert');
const {makePage, send, stubRaf} = require('./simplify2_harness');
const {installAudio} = require('./speech_helpers');

const B64 = 'SUQzBAAAAAAAAA==';
const CHECK_MS = 5000;
const STALL_CHECKS = 6;

/**
 * Replace the page's interval timers with a manual clock: every
 * setInterval is recorded with its delay, and a test ticks them.
 */
function installClock(win) {
  const timers = new Map();
  let nextId = 1;
  win.setInterval = (fn, ms) => {
    const id = nextId++;
    timers.set(id, {fn, ms});
    return id;
  };
  win.clearInterval = id => {
    timers.delete(id);
  };
  return {
    /** The pending intervals armed with exactly *ms*. */
    pending: ms => Array.from(timers.values()).filter(t => t.ms === ms),
    /** Run every pending interval armed with *ms* once. */
    tick: ms => {
      for (const t of Array.from(timers.values())) {
        if (t.ms === ms) t.fn();
      }
    },
  };
}

function talkEv(id, text) {
  return {type: 'talk', language: 'en-US', text, talkId: id, audioB64: B64};
}

function testHealthyLongClipIsNeverOverlapped() {
  const {win} = makePage({beforeScripts: stubRaf});
  const players = installAudio(win);
  const clock = installClock(win);

  send(win, talkEv('h1', 'a three minute clip'));
  send(win, talkEv('h2', 'must wait for it'));
  assert.strictEqual(players.length, 1, 'the second clip waits');
  assert.strictEqual(
    clock.pending(CHECK_MS).length,
    1,
    'one stall check is armed for the playing clip',
  );

  // 180 s of healthy playback: every sample sees progress.
  for (let i = 0; i < (180 * 1000) / CHECK_MS; i++) {
    players[0].advance(CHECK_MS / 1000);
    clock.tick(CHECK_MS);
  }
  assert.strictEqual(players.length, 1, 'a progressing clip keeps the queue');
  assert.strictEqual(players[0].pauseCalls, 0, 'and is never paused');
  assert.notStrictEqual(players[0].src, '', 'and keeps its source');

  players[0].fire('ended');
  assert.strictEqual(players.length, 2, 'the second clip starts after it');
  assert.strictEqual(players[1].playCalls, 1);
  win.close();
  console.log('PASS: a healthy 180 s clip is never overlapped');
}

function testStalledClipIsRetiredBeforeTheNextStarts() {
  const {win} = makePage({beforeScripts: stubRaf});
  const players = installAudio(win);
  const clock = installClock(win);

  send(win, talkEv('s1', 'stalls after ten seconds'));
  send(win, talkEv('s2', 'must still play'));
  assert.strictEqual(players.length, 1, 'the second clip waits');

  // Ten seconds of progress, then the decode stalls.
  players[0].advance(5);
  clock.tick(CHECK_MS);
  players[0].advance(5);
  clock.tick(CHECK_MS);

  for (let i = 0; i < STALL_CHECKS - 1; i++) clock.tick(CHECK_MS);
  assert.strictEqual(
    players.length,
    1,
    'one sample short of the limit the stalled clip still holds the queue',
  );
  assert.strictEqual(players[0].pauseCalls, 0);

  clock.tick(CHECK_MS);
  assert.strictEqual(players[0].pauseCalls, 1, 'the stalled clip is paused');
  assert.strictEqual(players[0].src, '', 'its source is dropped');
  assert.strictEqual(players[0].onended, null, 'its handlers are off');
  assert.strictEqual(players[0].onpause, null);
  assert.strictEqual(players.length, 2, 'then the second clip starts');
  assert.strictEqual(players[1].playCalls, 1);
  assert.strictEqual(
    clock.pending(CHECK_MS).length,
    1,
    'only the second clip is sampled now',
  );

  // Progress after a gap shorter than the limit resets the count: the
  // second clip is never retired while it keeps moving now and then.
  players[1].advance(1);
  clock.tick(CHECK_MS);
  for (let i = 0; i < STALL_CHECKS - 1; i++) clock.tick(CHECK_MS);
  players[1].advance(1);
  clock.tick(CHECK_MS);
  for (let i = 0; i < STALL_CHECKS - 1; i++) clock.tick(CHECK_MS);
  assert.strictEqual(players[1].pauseCalls, 0, 'progress resets the count');
  win.close();
  console.log('PASS: a stalled clip is retired before the next starts');
}

function testLateEndedFromRetiredClipIsIgnored() {
  const {win} = makePage({beforeScripts: stubRaf});
  const players = installAudio(win);
  const clock = installClock(win);

  send(win, talkEv('l1', 'stalls at once'));
  send(win, talkEv('l2', 'second'));
  for (let i = 0; i < STALL_CHECKS; i++) clock.tick(CHECK_MS);
  assert.strictEqual(players.length, 2, 'the stalled clip was retired');

  // The stalled clip waking up late must not advance the queue again.
  send(win, talkEv('l3', 'third'));
  players[0].fire('ended');
  players[0].fire('pause');
  players[0].fire('error');
  assert.strictEqual(players.length, 2, 'late events of clip 1 are ignored');
  players[1].advance(1);
  players[1].fire('ended');
  assert.strictEqual(players.length, 3, 'clip 2 ending starts clip 3');
  win.close();
  console.log('PASS: a late ended from a retired clip is ignored');
}

function testStallCheckIsClearedOnNormalCompletion() {
  const {win} = makePage({beforeScripts: stubRaf});
  const players = installAudio(win);
  const clock = installClock(win);

  send(win, talkEv('n1', 'ends normally'));
  assert.strictEqual(clock.pending(CHECK_MS).length, 1);
  players[0].fire('ended');
  assert.strictEqual(
    clock.pending(CHECK_MS).length,
    0,
    'the stall check is cleared when the clip ends',
  );

  // The same for the user-agent pause path.
  send(win, talkEv('n2', 'paused by the user agent'));
  assert.strictEqual(players.length, 2);
  assert.strictEqual(clock.pending(CHECK_MS).length, 1);
  players[1].fire('pause');
  assert.strictEqual(clock.pending(CHECK_MS).length, 0);
  assert.strictEqual(players[1].src, '', 'a paused clip is retired');

  // An idle queue: a later talk plays at once and is sampled again.
  send(win, talkEv('n3', 'next'));
  assert.strictEqual(players.length, 3);
  assert.strictEqual(players[2].playCalls, 1);
  assert.strictEqual(clock.pending(CHECK_MS).length, 1);
  win.close();
  console.log('PASS: the stall check is cleared on normal completion');
}

function main() {
  testHealthyLongClipIsNeverOverlapped();
  testStalledClipIsRetiredBeforeTheNextStarts();
  testLateEndedFromRetiredClipIsIgnored();
  testStallCheckIsClearedOnNormalCompletion();
  console.log('\n4 passed, 0 failed');
}

main();
