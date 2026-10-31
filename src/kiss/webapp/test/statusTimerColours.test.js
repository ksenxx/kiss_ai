// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The status-bar timer (#status-text) is coloured by state, not by
// alarm: the theme accent while the tab's task runs, the quiet --dim
// once it is ready or done.  A fresh tab starts dim.  The colour
// follows the running state of the tab on screen: showing a tab (a
// switch, or a hidden tab finishing and being focused) sets its
// running state, which repaints the timer, so coming back to a tab
// never shows a stale colour.  Nothing may paint the old red (running)
// or green (done).

'use strict';

const assert = require('assert');
const {makePage, send, clickTab} = require('./simplify2_harness.js');

function statusColor(win) {
  return win.document.getElementById('status-text').style.color;
}

function testFreshTabIsDimAndRunningIsAccent() {
  const {win} = makePage();
  const tabA = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tabA, startTs: Date.now()});
  assert.strictEqual(
    statusColor(win),
    'var(--accent)',
    'a running task colours the timer in the theme accent, not red',
  );

  win._testApi.endLaunch();
  win._testApi.createNewTab();
  assert.strictEqual(
    statusColor(win),
    'var(--dim)',
    'a fresh tab starts with the quiet --dim timer, not green',
  );

  clickTab(win, tabA);
  assert.strictEqual(
    statusColor(win),
    'var(--accent)',
    'switching back to the running tab shows the accent timer again',
  );
  win.close();
  console.log(
    '  ok - fresh tab is dim, running tab is accent, survives a switch',
  );
}

function testDoneTabIsDimAfterSwitchingAway() {
  const {win} = makePage();
  const tabA = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tabA, startTs: Date.now()});
  send(win, {type: 'task_done', tabId: tabA, startTs: 1000, endTs: 3000});
  assert.strictEqual(
    statusColor(win),
    'var(--dim)',
    'a finished task rests the timer in --dim, not green',
  );

  win._testApi.endLaunch();
  win._testApi.createNewTab();
  clickTab(win, tabA);
  assert.strictEqual(
    statusColor(win),
    'var(--dim)',
    'switching back to the done tab shows the --dim timer',
  );
  win.close();
  console.log('  ok - done tab stays dim after switching away and back');
}

function testHiddenTabFinishingShowsDim() {
  const {win} = makePage();
  const tabA = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tabA, startTs: Date.now()});
  win._testApi.endLaunch();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabA, tabB, 'a second tab must have been created');

  // task_done for the hidden tab A: focusFinishedTab switches to it, and
  // the tab, now idle, shows the dim timer.
  send(win, {type: 'task_done', tabId: tabA, startTs: 1000, endTs: 3000});
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    tabA,
    'the finished tab is focused',
  );
  assert.strictEqual(
    statusColor(win),
    'var(--dim)',
    'a hidden tab that finished shows the --dim timer when focused',
  );
  win.close();
  console.log('  ok - a hidden tab that finished shows the dim timer');
}

testFreshTabIsDimAndRunningIsAccent();
testDoneTabIsDimAfterSwitchingAway();
testHiddenTabFinishingShowsDim();
console.log('All statusTimerColours tests passed');
