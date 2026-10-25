// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the "Done (Xs)" status label a
// `task_done` event leaves on its tab.
//
// The daemon stamps `startTs` / `endTs` on the event and the label is
// computed from that span.  When the span is missing the page falls
// back to the start time it recorded itself -- and that fallback used
// the module-level `t0`, i.e. the VISIBLE tab's clock, even when the
// event named a background tab, so a task that ran for a minute in a
// hidden tab could report the few seconds the tab on screen had been
// running.  The label is also built by the same helper the timer uses
// (doneLabelFor) instead of a second copy of the arithmetic.

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');

const {send} = harness;

// The page's clock is frozen at NOW so every label is exact.
const NOW = 1_700_000_000_000;

// The sourceURL is what audit0902_main_done_label.coverage.js keys on.
function makePage() {
  return harness.makePage({
    sourceUrl: 'audit0902-donelabel-main.js',
    beforeScripts(win) {
      win.Date.now = () => NOW;
    },
  });
}

// The Chats-panel pick: chat tabs have no row of their own any more.
// The pick is a click, which main.js counts as the user moving on
// since the submit (so a finishing background task no longer pulls
// them over); the gesture is dispatched before the switch.
function clickTab(win, tabId) {
  assert.ok(
    win._testApi.openTabs().some(t => t.id === tabId),
    `tab ${tabId} must be open`,
  );
  win.document.body.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  win._testApi.switchToTab(tabId);
}

function statusText(win) {
  return win.document.getElementById('status-text').textContent;
}

function startTask(win, tabId, startTs) {
  send(win, {type: 'setTaskText', text: 'work', tabId});
  send(win, {type: 'clear', chat_id: 'chat-' + tabId, tabId});
  send(win, {type: 'status', running: true, tabId, startTs});
}

function testSpanOnTheEventWins() {
  const {win} = makePage();
  const tabId = win._testApi.getActiveTabId();
  const now = NOW;
  startTask(win, tabId, now - 3000);
  send(win, {
    type: 'task_done',
    tabId,
    startTs: now - 125000,
    endTs: now,
  });
  assert.strictEqual(
    statusText(win),
    'Done (2m 5s)',
    'the label comes from the span stamped on the event',
  );
  // A span that ends before it starts is no span: the tab's own clock
  // is used instead of a clamped zero.
  startTask(win, tabId, now - 30000);
  send(win, {type: 'task_done', tabId, startTs: now, endTs: now - 1000});
  assert.strictEqual(
    statusText(win),
    'Done (30s)',
    "an inverted span falls back to the tab's clock",
  );
  win.close();
  console.log('  ok - the span on the event wins');
}

function testFallbackUsesTheFinishedTabsClock() {
  const {win} = makePage();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  const now = NOW;
  // A has been running for a minute in the background; B, on screen,
  // started five seconds ago.
  startTask(win, tabA, now - 60000);
  clickTab(win, tabB);
  startTask(win, tabB, now - 5000);
  assert.strictEqual(win._testApi.getActiveTabId(), tabB);

  // A finishes; the event carries no span.  The user clicked B since
  // submitting, so its task_done leaves them on B (focusFinishedTab
  // only undoes an agent-made switch); they go to A themselves.
  send(win, {type: 'task_done', tabId: tabA});
  assert.strictEqual(win._testApi.getActiveTabId(), tabB);
  clickTab(win, tabA);
  assert.strictEqual(
    statusText(win),
    'Done (1m 0s)',
    "a background tab's label is measured from ITS start",
  );
  // B is still running its own five-second-old task.
  clickTab(win, tabB);
  assert.strictEqual(
    statusText(win),
    'Running 5s',
    'the visible tab keeps its own clock',
  );
  win.close();
  console.log('  ok - the fallback uses the finished tab\u2019s clock');
}

function testFallbackOnTheVisibleTab() {
  const {win} = makePage();
  const tabId = win._testApi.getActiveTabId();
  startTask(win, tabId, NOW - 70000);
  send(win, {type: 'task_done', tabId});
  assert.strictEqual(
    statusText(win),
    'Done (1m 10s)',
    'the visible tab measures from its own start',
  );
  // A tab id the page does not know is ignored: the label on screen
  // stays as it is rather than throwing.
  send(win, {type: 'task_done', tabId: 'no-such-tab'});
  assert.strictEqual(
    statusText(win),
    'Done (1m 10s)',
    'an unknown tab changes nothing',
  );
  win.close();
  console.log('  ok - the fallback on the visible tab');
}

function testFallbackWithoutAnyStart() {
  const {win} = makePage();
  const tabId = win._testApi.getActiveTabId();
  // No status/startTs ever arrived for this tab.
  send(win, {type: 'clear', chat_id: 'c', tabId});
  send(win, {type: 'task_done', tabId});
  assert.strictEqual(
    statusText(win),
    'Done (0s)',
    'with no start recorded the task reads as an instant finish',
  );
  win.close();
  console.log('  ok - the fallback without any start');
}

function main() {
  testSpanOnTheEventWins();
  testFallbackUsesTheFinishedTabsClock();
  testFallbackOnTheVisibleTab();
  testFallbackWithoutAnyStart();
  console.log('all audit0902 done-label tests passed');
}

main();
