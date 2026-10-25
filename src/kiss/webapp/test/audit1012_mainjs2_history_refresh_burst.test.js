// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Audit 2026-10-12 (mainjs-2): the daemon replays one `status` per
// chat-bound tab after every `ready` (and again after every reconnect),
// and `tasks_updated` follows each task end.  main.js answers every one
// of them with refreshHistory() -> requestHistoryFromStart(), which
// posts a fresh `getHistory` (a history.db query with chat summaries
// on the daemon) and bumps historyGeneration so that only the LAST
// reply is ever rendered.  A window with N open tabs therefore fires N
// identical history queries in one burst, N-1 of which are thrown
// away on arrival.  The burst must coalesce: one leading request and
// at most one trailing one, sent once the leading reply has landed;
// historyRefreshSingleFlight.test.js pins the exact counts.  This test
// pins the other side: the coalescing must not swallow a lone refresh.

const assert = require('assert');
const {makePage, send, ofType, sleep} = require('./simplify2_harness');

function nextTick() {
  return sleep(20);
}

async function testASingleStatusStillRefetches() {
  // The coalescing must not swallow the refresh itself: one status
  // with the panel open (and no page one in flight) still asks for
  // page one.
  const {win, posted} = makePage();
  win.document.getElementById('sidebar').classList.add('open');
  send(win, {type: 'daemonStatus', connected: true});
  await nextTick();
  const pending = ofType(posted, 'getHistory').pop();
  send(win, {type: 'history', sessions: [], offset: 0, generation: pending.generation});
  posted.length = 0;
  send(win, {
    type: 'status',
    running: true,
    tabId: win.kissActiveTabId(),
    startTs: Date.now(),
  });
  await nextTick();
  const reqs = ofType(posted, 'getHistory');
  assert.strictEqual(reqs.length, 1, 'exactly one getHistory');
  assert.strictEqual(reqs[0].offset, 0, 'page one');
  win.close();
  console.log('  ok - a single status still refetches page one');
}

testASingleStatusStillRefetches().then(
  () => {
    console.log('\n1 passed, 0 failed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
