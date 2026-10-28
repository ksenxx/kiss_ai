// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  clickTab,
  fakeIntervals,
  readyTabId,
} = require('./simplify2_harness');

function statusOf(win) {
  return win.document.getElementById('status-text').textContent;
}

/**
 * A page on a frozen clock whose intervals are faked: `tick(ms)` moves
 * `Date.now()` forward by `ms` and then runs the header's 1 s
 * running-timer repaint (main.js _renderTimerTick) at once, so every
 * elapsed label is exact and an assertion that must see the header
 * AFTER a repaint waits for nothing.  `clock.now` is the page's time.
 */
function makePage() {
  const clock = {now: 1_700_000_000_000};
  let intervals;
  const {win, posted} = makeChatPage({
    beforeScripts(win) {
      win.Date.now = () => clock.now;
      intervals = fakeIntervals(win);
    },
  });
  function tick(ms) {
    clock.now += ms;
    intervals.tick(ms);
  }
  return {win, activeId: readyTabId(posted), clock, tick};
}

function testForeignNewerRunDoesNotClobberActiveTimer() {
  const {win, activeId, clock, tick} = makePage();

  send(win, {
    type: 'status',
    running: true,
    tabId: activeId,
    startTs: clock.now - 65_000,
  });
  assert.strictEqual(statusOf(win), 'Running 1m 5s');

  send(win, {
    type: 'status',
    running: true,
    tabId: 'other-window-tab',
    startTs: clock.now,
  });
  tick(1000);
  assert.strictEqual(
    statusOf(win),
    'Running 1m 6s',
    'BUG: a newer run in another window clobbered the active ' +
      "tab's running timer, or the timer stopped",
  );

  send(win, {
    type: 'task_done',
    tabId: 'other-window-tab',
    startTs: clock.now - 5_000,
    endTs: clock.now,
  });
  send(win, {type: 'status', running: false, tabId: 'other-window-tab'});
  tick(1000);
  assert.strictEqual(
    statusOf(win),
    'Running 1m 7s',
    'BUG: another window finishing its run flipped the active ' +
      "tab's header, or the timer stopped",
  );
  win.close();
  console.log('  ok - newer runs in another window never touch the header');
}

function testBackgroundTaskDoneShowsDurationAfterSwitch() {
  const {win, activeId, clock, tick} = makePage();

  send(win, {
    type: 'openSubagentTab',
    tab_id: 'bg-tab',
    parent_tab_id: activeId,
    description: 'background work',
  });
  send(win, {
    type: 'status',
    running: true,
    tabId: activeId,
    startTs: clock.now - 3_000,
  });
  assert.strictEqual(statusOf(win), 'Running 3s');

  const bgEnd = clock.now;
  send(win, {
    type: 'status',
    running: true,
    tabId: 'bg-tab',
    startTs: bgEnd - 201_000,
  });
  assert.strictEqual(statusOf(win), 'Running 3s');

  send(win, {
    type: 'task_done',
    tabId: 'bg-tab',
    startTs: bgEnd - 201_000,
    endTs: bgEnd,
  });
  send(win, {type: 'status', running: false, tabId: 'bg-tab'});

  // A sub-agent's own task_done is not the user's task finishing: the
  // page stays on the parent (its tab is about to be closed by
  // subagentDone), and the user visits the finished tab themselves.
  const activeEl = win.document.querySelector('#tab-list .chat-tab.active');
  assert.strictEqual(
    activeEl && activeEl.getAttribute('data-tab-id'),
    activeId,
    'task_done in a sub-agent tab must not pull the page onto it',
  );
  clickTab(win, 'bg-tab');
  assert.strictEqual(
    statusOf(win),
    'Done (3m 21s)',
    'a tab whose task finished in the background must show ' +
      'endTs - startTs once focused',
  );

  clickTab(win, activeId);
  assert.strictEqual(
    statusOf(win),
    'Running 3s',
    "the original tab's running timer must survive the switch",
  );
  tick(1000);
  assert.strictEqual(
    statusOf(win),
    'Running 4s',
    "the original tab's timer must keep counting after the switch",
  );
  clickTab(win, 'bg-tab');
  assert.strictEqual(
    statusOf(win),
    'Done (3m 21s)',
    'the done duration must re-render when switching back',
  );
  win.close();
  console.log(
    '  ok - a background task_done shows its duration once the tab is visited',
  );
}

function testDoneLabelSurvivesTabSwitches() {
  const {win, activeId, clock} = makePage();

  send(win, {
    type: 'status',
    running: true,
    tabId: activeId,
    startTs: clock.now - 1_000,
  });
  const endTs = clock.now;
  send(win, {
    type: 'task_done',
    tabId: activeId,
    startTs: endTs - 45_000,
    endTs,
  });
  send(win, {type: 'status', running: false, tabId: activeId});
  assert.strictEqual(statusOf(win), 'Done (45s)');

  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-x',
    parent_tab_id: activeId,
    description: 'sub work',
  });
  clickTab(win, 'sub-x');
  clickTab(win, activeId);
  assert.strictEqual(
    statusOf(win),
    'Done (45s)',
    'the done duration must survive switching away and back',
  );
  win.close();
  console.log('  ok - done duration survives tab switches');
}

function testBackgroundReplayWithExtraTimestamps() {
  const {win, activeId, clock} = makePage();

  send(win, {
    type: 'openSubagentTab',
    tab_id: 'hist-tab',
    parent_tab_id: activeId,
    description: 'history task',
  });
  const end = clock.now - 60_000;
  send(win, {
    type: 'task_events',
    tabId: 'hist-tab',
    task: 'history task',
    events: [],
    extra: JSON.stringify({startTs: end - 83_000, endTs: end}),
  });
  clickTab(win, 'hist-tab');
  assert.strictEqual(
    statusOf(win),
    'Done (1m 23s)',
    'a background tab replaying a finished task must show ' +
      'endTs - startTs after switching to it',
  );
  win.close();
  console.log('  ok - background replay with extra timestamps shows Done(…)');
}

try {
  testForeignNewerRunDoesNotClobberActiveTimer();
  testBackgroundTaskDoneShowsDurationAfterSwitch();
  testDoneLabelSurvivesTabSwitches();
  testBackgroundReplayWithExtraTimestamps();
  console.log('\n4 passed, 0 failed');
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
