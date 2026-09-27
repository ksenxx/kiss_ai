// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The status-bar timer (#status-text) is coloured by state, not by
// alarm: the theme accent while the tab's task runs, the quiet --dim
// once it is ready or done.  A fresh tab starts dim.  The colour is
// part of the per-tab state that saveCurrentTab / switchToTab carry
// across tab switches, and a background tab finishing (task_done for a
// hidden tab) records the dim colour into that tab's saved state, so
// coming back to it never shows a stale colour.  Nothing may paint the
// old red (running) or green (done).

'use strict';

const assert = require('assert');
const {makeWebview, send} = require('./simplify2_harness.js');

function statusColor(win) {
  return win.document.getElementById('status-text').style.color;
}

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function testFreshTabIsDimAndRunningIsAccent() {
  const {win} = makeWebview();
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
    "switching back restores the running tab's accent timer",
  );
  win.close();
  console.log('  ok - fresh tab is dim, running tab is accent, survives a switch');
}

function testDoneTabIsDimAfterSwitchingAway() {
  const {win} = makeWebview();
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
    "the done tab's saved state carries the --dim timer across a switch",
  );
  win.close();
  console.log('  ok - done tab stays dim after switching away and back');
}

function testHiddenTabFinishingRecordsDim() {
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  send(win, {type: 'status', running: true, tabId: tabA, startTs: Date.now()});
  win._testApi.endLaunch();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabA, tabB, 'a second tab must have been created');

  // task_done for the hidden tab A: setReady writes A's saved state and
  // focusFinishedTab switches to it, restoring that saved colour.
  send(win, {type: 'task_done', tabId: tabA, startTs: 1000, endTs: 3000});
  assert.strictEqual(win._testApi.getActiveTabId(), tabA, 'the finished tab is focused');
  assert.strictEqual(
    statusColor(win),
    'var(--dim)',
    "a hidden tab finishing records the --dim timer in its saved state",
  );
  win.close();
  console.log('  ok - a hidden tab finishing records the dim timer');
}

testFreshTabIsDimAndRunningIsAccent();
testDoneTabIsDimAfterSwitchingAway();
testHiddenTabFinishingRecordsDim();
console.log('All statusTimerColours tests passed');
