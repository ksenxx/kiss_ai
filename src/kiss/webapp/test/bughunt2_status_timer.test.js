// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  fakeIntervals,
  readyTabId,
} = require('./simplify2_harness');

// A status for a tab this window knows (an open background subagent tab)
// must record that tab's own t0 without repainting the header, which keeps
// counting the active tab's run.  (A status for a tab this window has never
// heard of is pinned by tab_timer_per_tab.test.js.)
function testBackgroundLocalTabStatusKeepsOwnT0() {
  const clock = {now: 1_700_000_000_000};
  let intervals;
  const {win, posted} = makePage({
    beforeScripts(win) {
      win.Date.now = () => clock.now;
      intervals = fakeIntervals(win);
    },
  });
  const activeId = readyTabId(posted);

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

  clock.now += 1_000;
  intervals.tick(1_000);

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
  testBackgroundLocalTabStatusKeepsOwnT0();
  console.log('\n1 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
