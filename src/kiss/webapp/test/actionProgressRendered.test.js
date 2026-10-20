// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the user-facing progress event the
// daemon broadcasts during a merge flow:
//
//   * ``worktree_progress`` ("Staging changes…", "Generating commit
//     message…", "Committing…") — once not rendered by the page at
//     all, so a remote or mobile browser user clicking Merge saw
//     nothing (F08-4).
//
// It carries a plain string that any client can show, and its terminal
// sibling (``worktree_result``) must clear the live line again; so must
// the manual Git Commit button's ``autocommit_done``.

'use strict';

const assert = require('assert');
const {makePage, send, clickTab} = require('./simplify2_harness');

function progressLines(root) {
  return Array.from(root.querySelectorAll('.wt-progress')).map(
    el => el.textContent,
  );
}

function testProgressReplacedInPlaceAndClearedByAutocommitDone() {
  const {win} = makePage();
  const doc = win.document;
  const out = doc.getElementById('output');
  const tabId = win._testApi.getActiveTabId();

  send(win, {
    type: 'worktree_progress',
    message: 'Staging changes\u2026',
    tabId,
  });
  assert.deepStrictEqual(
    progressLines(out),
    ['Staging changes\u2026'],
    'worktree_progress must render its message in the transcript',
  );

  // The three messages are one flow, not three log lines: the live line
  // is replaced in place.
  send(win, {
    type: 'worktree_progress',
    message: 'Generating commit message\u2026',
    tabId,
  });
  send(win, {type: 'worktree_progress', message: 'Committing\u2026', tabId});
  assert.deepStrictEqual(
    progressLines(out),
    ['Committing\u2026'],
    'successive worktree_progress messages must replace the live line',
  );

  // A failed manual Git Commit clears the progress line and reports
  // the outcome in the transcript.
  send(win, {
    type: 'autocommit_done',
    success: false,
    manual: true,
    message: 'Nothing to commit',
    tabId,
  });
  assert.deepStrictEqual(
    progressLines(out),
    [],
    'autocommit_done must clear the live progress line',
  );
  assert.ok(
    out.textContent.includes('Nothing to commit'),
    'a failed autocommit_done must still render its own result line',
  );
  win.close();
  console.log(
    '  ok - worktree_progress replaces in place; autocommit_done clears',
  );
}

function testWorktreeProgressRendered() {
  const {win} = makePage();
  const doc = win.document;
  const out = doc.getElementById('output');
  const tabId = win._testApi.getActiveTabId();

  send(win, {
    type: 'worktree_progress',
    message: 'Generating commit message\u2026',
    tabId,
  });
  assert.deepStrictEqual(
    progressLines(out),
    ['Generating commit message\u2026'],
    'worktree_progress must render in the page too — the remote web ' +
      'client runs this same main.js and had no other feedback during a ' +
      'merge that takes tens of seconds',
  );

  send(win, {
    type: 'worktree_result',
    success: true,
    message: "Successfully merged branch 'kiss/wt-1'",
    tabId,
  });
  assert.deepStrictEqual(
    progressLines(out),
    [],
    'worktree_result must clear the live progress line',
  );
  assert.ok(
    out.textContent.includes('Successfully merged'),
    'worktree_result must still render its own result line',
  );
  win.close();
  console.log('  ok - worktree_progress renders and is cleared by the result');
}

function testProgressForBackgroundTabGoesToThatTab() {
  const {win} = makePage();
  const doc = win.document;
  const out = doc.getElementById('output');
  const tabA = win._testApi.getActiveTabId();
  win._testApi.endLaunch();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabA, tabB, 'a second tab must have been created');

  // Progress addressed to the hidden tab must not appear on screen.
  send(win, {
    type: 'worktree_progress',
    message: 'Staging changes\u2026',
    tabId: tabA,
  });
  assert.deepStrictEqual(
    progressLines(out),
    [],
    "another tab's worktree_progress must not paint the visible tab",
  );

  // ...but it must be waiting there when the user switches back.
  clickTab(win, tabA);
  assert.deepStrictEqual(
    progressLines(out),
    ['Staging changes\u2026'],
    'a hidden tab must render the progress it received while hidden',
  );

  send(win, {
    type: 'autocommit_done',
    success: true,
    message: 'chore: hidden tab commit',
    tabId: tabA,
  });
  assert.deepStrictEqual(
    progressLines(out),
    [],
    'the terminal event must clear the progress line of that tab too',
  );
  win.close();
  console.log('  ok - progress for a hidden tab is deferred, not lost');
}

function main() {
  testProgressReplacedInPlaceAndClearedByAutocommitDone();
  testWorktreeProgressRendered();
  testProgressForBackgroundTabGoesToThatTab();
  console.log('actionProgressRendered.test.js: all tests passed');
}

main();
