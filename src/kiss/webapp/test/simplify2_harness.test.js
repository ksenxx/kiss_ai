// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// The shared timer and animation-frame stand-ins of simplify2_harness
// must only produce schedules a browser can produce: a one-shot timer
// runs once whether its due time or the test fires it (captureTimers /
// fireCapturedTimers, recordTimers / fireTimers), and a frame cancelled
// by an earlier frame of the same batch does not run (queueRaf).  Before
// the fix a timer that had already fired on its own ran again when the
// test drained the ledger, draining a recordTimers ledger twice ran every
// callback twice, queueRaf ignored a same-frame cancel, and a captured
// timer cancelled by an earlier callback of the same drain still ran
// (audit 2026-11-08 WT7/F1: the drain emptied the ledger before running
// anything, so the page's clearTimeout found no record to mark).

'use strict';

const assert = require('assert');
const {
  newDom,
  captureTimers,
  fireCapturedTimers,
  recordTimers,
  fireTimers,
  queueRaf,
  sleep,
} = require('./simplify2_harness');

async function testNaturallyFiredTimerIsNotFiredAgain() {
  const win = newDom('outside-only');
  const timers = captureTimers(win);
  const runs = [];
  win.setTimeout((a, b) => runs.push([a, b]), 1, 'x', 'y');
  await sleep(30);
  assert.deepStrictEqual(runs, [['x', 'y']], 'the timer fired on its own');
  fireCapturedTimers(win, timers);
  assert.strictEqual(runs.length, 1, 'a timer that fired is not fired again');
  assert.strictEqual(timers.length, 0, 'the drain forgets the fired timer');
  win.close();
  console.log('  ok - captureTimers: a timer fires once, due time or drain');
}

function testTimerCancelledByAnEarlierCallbackOfTheDrainStaysQuiet() {
  const win = newDom('outside-only');
  const timers = captureTimers(win);
  const runs = [];
  win.setTimeout(() => runs.push('kept'), 60000);
  let second;
  win.setTimeout(() => {
    runs.push('first');
    win.clearTimeout(second);
    win.setTimeout(() => runs.push('armed-by-first'), 60000);
  }, 60000);
  second = win.setTimeout(() => runs.push('second'), 60000);
  // Drain the suffix only: the first timer stays armed in the ledger.
  fireCapturedTimers(win, timers, 1);
  assert.deepStrictEqual(runs, ['first'], 'a same-drain cancel is honoured');
  assert.strictEqual(
    timers.length,
    2,
    'the timer before `from` and the one armed by the drain remain',
  );
  assert.strictEqual(
    timers[1].fired,
    false,
    'a timer armed during the drain is not part of it',
  );
  fireCapturedTimers(win, timers);
  assert.deepStrictEqual(runs, ['first', 'kept', 'armed-by-first']);
  assert.strictEqual(timers.length, 0, 'a full drain empties the ledger');
  // A callback that drains the ledger itself: the outer drain finds its
  // remaining records gone and must not drop a timer armed afterwards.
  win.setTimeout(() => {
    runs.push('outer');
    fireCapturedTimers(win, timers);
    win.setTimeout(() => runs.push('after'), 60000);
  }, 60000);
  win.setTimeout(() => runs.push('inner'), 60000);
  fireCapturedTimers(win, timers);
  assert.deepStrictEqual(
    runs.slice(3),
    ['outer', 'inner'],
    'a nested drain runs the rest once',
  );
  assert.strictEqual(
    timers.length,
    1,
    'the timer armed after the nested drain stays in the ledger',
  );
  fireCapturedTimers(win, timers);
  assert.deepStrictEqual(runs.slice(5), ['after']);
  win.close();
  console.log(
    '  ok - captureTimers: a timer cancelled from its own drain stays quiet',
  );
}

function testDrainingTheRecordedTimersTwiceRunsEachOnce() {
  const win = newDom('outside-only');
  const timers = recordTimers(win, 500);
  const runs = [];
  win.setTimeout(v => runs.push(v), 500, 'first');
  const handle = win.setTimeout(v => runs.push(v), 500, 'second');
  win.clearTimeout(handle);
  fireTimers(timers);
  fireTimers(timers);
  assert.deepStrictEqual(runs, ['first'], 'each surviving timer ran once');
  win.close();
  console.log('  ok - recordTimers: draining twice runs each timer once');
}

function testFrameCancelledByAnEarlierFrameDoesNotRun() {
  const win = newDom('outside-only');
  const raf = queueRaf(win);
  const ran = [];
  let second;
  win.requestAnimationFrame(() => {
    ran.push('first');
    win.cancelAnimationFrame(second);
  });
  second = win.requestAnimationFrame(() => ran.push('second'));
  raf.run();
  assert.deepStrictEqual(ran, ['first'], 'a same-frame cancel is honoured');
  win.requestAnimationFrame(() => ran.push('third'));
  raf.run();
  assert.deepStrictEqual(ran, ['first', 'third'], 'later frames still run');
  win.close();
  console.log('  ok - queueRaf: a frame cancelled from its own batch stays quiet');
}

async function main() {
  await testNaturallyFiredTimerIsNotFiredAgain();
  testTimerCancelledByAnEarlierCallbackOfTheDrainStaysQuiet();
  testDrainingTheRecordedTimersTwiceRunsEachOnce();
  testFrameCancelledByAnEarlierFrameDoesNotRun();
  console.log('simplify2_harness: all tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
