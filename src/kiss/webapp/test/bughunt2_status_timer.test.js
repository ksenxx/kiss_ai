// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

// The page runs on a seeded clock and its 1 s header tick is recorded
// instead of scheduled, so the elapsed label is exact and no wall-clock
// stall between `send` and the assertion can move it.
function makeClockedPage() {
  const clock = {now: 1_700_000_000_000};
  const ticks = [];
  function seedClock(win) {
    win.Date.now = () => clock.now;
    const realSetInterval = win.setInterval;
    win.setInterval = function (fn, ms, ...args) {
      if (ms !== 1000) return realSetInterval(fn, ms, ...args);
      ticks.push(fn);
      return ticks.length;
    };
  }
  const {win, posted} = makePage({beforeScripts: seedClock});
  function advance(ms) {
    clock.now += ms;
    for (const tick of ticks) tick();
  }
  return {win, posted, clock, advance};
}

function testForeignTabStatusMustNotClobberActiveTimer() {
  const {win, posted, clock, advance} = makeClockedPage();

  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'page must post ready with a tabId');
  const activeId = ready.tabId;

  send(win, {
    type: 'status',
    running: true,
    tabId: activeId,
    startTs: clock.now - 5_000,
  });
  const statusText = win.document.getElementById('status-text');
  assert.strictEqual(
    statusText.textContent,
    'Running 5s',
    'active tab timer must anchor to its own startTs',
  );

  send(win, {
    type: 'status',
    running: true,
    tabId: 'foreign-window-tab',
    startTs: clock.now - 3_600_000,
  });

  advance(1_000);

  const text = statusText.textContent;
  assert.strictEqual(
    text,
    'Running 6s',
    'BUG: a tab-stamped status event for a foreign/background tab ' +
      `clobbered the active tab's timer anchor (header now "${text}")`,
  );
  win.close();
  console.log(
    '  ok - foreign-tab status does not clobber the active running timer',
  );
}

function testBackgroundLocalTabStatusKeepsOwnT0() {
  const {win, posted, clock, advance} = makeClockedPage();
  const ready = posted.find(m => m.type === 'ready');
  const activeId = ready.tabId;

  send(win, {
    type: 'openSubagentTab',
    tab_id: 'bg-sub-tab',
    description: 'sub agent work',
  });

  send(win, {
    type: 'status',
    running: true,
    tabId: activeId,
    startTs: clock.now - 5_000,
  });

  const bgStart = clock.now - 1_800_000;
  send(win, {
    type: 'status',
    running: true,
    tabId: 'bg-sub-tab',
    startTs: bgStart,
  });

  advance(1_000);

  const statusText = win.document.getElementById('status-text');
  assert.strictEqual(
    statusText.textContent,
    'Running 6s',
    'BUG: background-tab status clobbered the active timer ' +
      `(header now "${statusText.textContent}")`,
  );
  win.close();
  console.log(
    '  ok - background-tab status keeps its own t0 without touching the UI',
  );
}

try {
  testForeignTabStatusMustNotClobberActiveTimer();
  testBackgroundLocalTabStatusKeepsOwnT0();
  console.log('\n2 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
