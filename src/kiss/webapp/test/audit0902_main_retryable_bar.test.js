// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the worktree action bar after a
// RETRYABLE `worktree_result` failure.
//
// The daemon (server/merge_flow.py) flags an action it could not take
// yet -- a merge or discard is already in progress on the tab -- and a
// failed "Do nothing" with `retryable: true` precisely so that "the
// page keeps the bar's buttons instead of stripping the only retry
// controls".
// main.js did keep the bar, but every button had been disabled by the
// click that sent the action and nothing ever re-armed them: the bar
// the message told the user to retry with was dead.  These tests pin
// the contract for the visible tab and for a background tab, and the
// unchanged non-retryable behaviour (the bar goes away).

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');

const {send, clickTab, plain, ofType} = harness;
const {bar, barButtons} = require('./worktree_bar_helpers');

// The sourceURL is what audit0902_main_retryable_bar.coverage.js keys on.
function makePage() {
  return harness.makePage({sourceUrl: 'audit0902-retrybar-main.js'});
}

function clickBarButton(win, text) {
  const btn = barButtons(win).find(b => b.textContent === text);
  assert.ok(btn, `the bar must have a "${text}" button`);
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  // Discard is destructive: its first click only reveals an inline
  // confirm (ui_antipattern_destructive_confirm.test.js); the confirm
  // button is what posts the action.
  const confirmBtn = win.document.querySelector('.wt-bar .wt-confirm-yes');
  if (text === 'Discard' && confirmBtn) {
    confirmBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  }
}

function assertAllEnabled(win, why) {
  const btns = barButtons(win);
  assert.strictEqual(btns.length, 3, 'the worktree bar has three buttons');
  for (const b of btns) {
    assert.ok(!b.disabled, `${why}: "${b.textContent}" must be clickable`);
  }
}

function assertAllDisabled(win, why) {
  const btns = barButtons(win);
  assert.strictEqual(btns.length, 3, 'the worktree bar has three buttons');
  for (const b of btns) {
    assert.ok(b.disabled, `${why}: "${b.textContent}" must be disarmed`);
  }
}

function worktreeActions(posted) {
  return plain(ofType(posted, 'worktreeAction'));
}

function testRetryableResultReArmsVisibleBar() {
  const {win, posted} = makePage();
  const tabId = win._testApi.getActiveTabId();

  send(win, {type: 'worktree_done', tabId});
  assert.ok(bar(win), 'worktree_done shows the bar');
  assertAllEnabled(win, 'a fresh bar');

  clickBarButton(win, 'Discard');
  assertAllDisabled(win, 'while the discard is in flight');
  assert.deepStrictEqual(worktreeActions(posted), [
    {type: 'worktreeAction', action: 'discard', tabId},
  ]);

  // The daemon could not discard yet (another action is in progress on
  // the tab) and asks the user to retry.
  send(win, {
    type: 'worktree_result',
    tabId,
    success: false,
    retryable: true,
    message:
      'A merge or discard is already in progress on this tab. ' +
      'Wait for it to finish before discarding.',
  });
  assert.ok(bar(win), 'a retryable failure keeps the bar');
  assertAllEnabled(win, 'after a retryable failure');
  assert.ok(
    win.document
      .getElementById('output')
      .textContent.includes('already in progress'),
    'the failure is reported in the transcript',
  );

  // The retry the message asked for must actually be possible.
  clickBarButton(win, 'Discard');
  assert.deepStrictEqual(
    worktreeActions(posted),
    [
      {type: 'worktreeAction', action: 'discard', tabId},
      {type: 'worktreeAction', action: 'discard', tabId},
    ],
    'the second click sends a second worktreeAction',
  );
  assertAllDisabled(win, 'while the retried discard is in flight');

  send(win, {
    type: 'worktree_result',
    tabId,
    success: true,
    message: "Discarded branch 'kiss_wt-1'.",
  });
  assert.strictEqual(bar(win), null, 'a final success dismisses the bar');
  win.close();
  console.log('  ok - retryable worktree_result re-arms the visible bar');
}

function testFailedMergeKeepsAndReArmsTheBar() {
  const {win, posted} = makePage();
  const tabId = win._testApi.getActiveTabId();

  send(win, {type: 'worktree_done', tabId});
  clickBarButton(win, 'Auto-commit and merge');
  assertAllDisabled(win, 'while the merge is in flight');

  // What the daemon sends when a pre-commit hook rejects the merge's
  // commit: the worktree is still pending.
  send(win, {
    type: 'worktree_result',
    tabId,
    success: false,
    message:
      "Cannot merge: committing 'kiss/wt-1' failed (a pre-commit hook " +
      'may have rejected the commit). Then retry the merge.',
    retryable: true,
  });
  assert.ok(bar(win), 'a failed merge keeps the bar');
  assertAllEnabled(win, 'after a failed merge');

  clickBarButton(win, 'Auto-commit and merge');
  assert.deepStrictEqual(
    worktreeActions(posted),
    [
      {type: 'worktreeAction', action: 'merge', tabId},
      {type: 'worktreeAction', action: 'merge', tabId},
    ],
    'the merge can be retried from the kept bar',
  );
  win.close();
  console.log('  ok - a failed merge keeps the bar for the retry');
}

function testRetryableResultReArmsBackgroundBar() {
  const {win, posted} = makePage();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabA, tabB);

  // Tab B owns a pending worktree; the user clicks Discard there, then
  // goes back to reading tab A while the daemon works.
  send(win, {type: 'worktree_done', tabId: tabB});
  clickBarButton(win, 'Discard');
  assertAllDisabled(win, 'while the discard is in flight');
  clickTab(win, tabA);
  assert.strictEqual(bar(win), null, 'tab A has no bar of its own');

  send(win, {
    type: 'worktree_result',
    tabId: tabB,
    success: false,
    retryable: true,
    message: 'A merge or discard is already in progress on this tab.',
  });
  assert.strictEqual(bar(win), null, 'the background result stays off tab A');

  clickTab(win, tabB);
  assert.ok(bar(win), 'tab B comes back with its bar');
  assertAllEnabled(win, 'after a retryable failure in the background');
  assert.ok(
    win.document
      .getElementById('output')
      .textContent.includes('already in progress'),
    'the failure is in tab B\u2019s transcript',
  );
  clickBarButton(win, 'Discard');
  assert.strictEqual(
    worktreeActions(posted).length,
    2,
    'the retry from the restored bar is sent',
  );
  win.close();
  console.log('  ok - retryable worktree_result re-arms a background bar');
}

function testNonRetryableFailureStillDismisses() {
  const {win} = makePage();
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'worktree_done', tabId});
  clickBarButton(win, 'Auto-commit and merge');
  send(win, {
    type: 'worktree_result',
    tabId,
    success: false,
    message: 'Merge failed: conflicts in a.txt',
  });
  assert.strictEqual(bar(win), null, 'a non-retryable failure dismisses');

  // Background twin.
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  send(win, {type: 'worktree_done', tabId: tabB});
  clickBarButton(win, 'Discard');
  clickTab(win, tabId);
  send(win, {
    type: 'worktree_result',
    tabId: tabB,
    success: false,
    message: "Partially discarded branch 'kiss_wt-3'.",
  });
  clickTab(win, tabB);
  assert.strictEqual(bar(win), null, 'the background bar is dropped too');
  win.close();
  console.log('  ok - non-retryable worktree_result still dismisses the bar');
}

function main() {
  testRetryableResultReArmsVisibleBar();
  testFailedMergeKeepsAndReArmsTheBar();
  testRetryableResultReArmsBackgroundBar();
  testNonRetryableFailureStillDismisses();
  console.log('all audit0902 retryable-bar tests passed');
}

main();
