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
//
// The second half covers the LAST exit of the line.  ``worktree_result``
// is not guaranteed to arrive: the merge flow runs inside a swallow-all
// handler on the daemon, and the progress events are broadcast from
// INSIDE the call it wraps, so anything raised after the first "Staging
// changes…" / "Generating commit message…" -- a git binary that dies, a
// stopped task unwinding through the merge, an LLM call that throws
// while writing the commit message -- eats the terminal event and the
// transcript keeps a live progress line that nothing will ever take
// away.  It is not a stale label: it reads as if the merge were still
// running, forever.
//
// What the daemon does still guarantee is the task-end event: control
// falls out of that handler and broadcasts task_done / task_error /
// task_stopped / task_interrupted immediately afterwards.  A task that
// has ended has no operation left in flight, so that is where the line
// has to go.  In the happy path the terminal event has already removed
// the line by then, so this is a safety net that never fires -- which is
// why the tests also pin that it does not fire too eagerly (another
// tab's task ending, and the real result line, must both survive).

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

// One case per task-end event: each is a real ending of the run whose
// post-task commit/merge put the line on screen.
const TASK_END_EVENTS = [
  ['task_done', 'a task that finished'],
  ['task_error', 'a task that failed'],
  ['task_stopped', 'a task the user stopped'],
  ['task_interrupted', 'a task that was interrupted'],
];

function testEveryTaskEndClearsWorktreeProgress() {
  TASK_END_EVENTS.forEach(([type, what]) => {
    const {win} = makePage();
    const out = win.document.getElementById('output');
    const tabId = win._testApi.getActiveTabId();

    send(win, {
      type: 'worktree_progress',
      message: 'Generating commit message\u2026',
      tabId,
    });
    assert.deepStrictEqual(
      progressLines(out),
      ['Generating commit message\u2026'],
      'the progress line must be on screen before the task ends',
    );

    send(win, {type, tabId, startTs: 1, endTs: 2});
    assert.deepStrictEqual(
      progressLines(out),
      [],
      `${what} must not leave a live "Generating commit message…" line ` +
        `behind (${type} was the only terminal event the daemon sent)`,
    );
    win.close();
  });
  console.log('  ok - every task-end event clears a worktree_progress line');
}

// The daemon addresses the whole flow to one tab, so a hidden tab's
// stranded line has to be cleaned out of that tab's own transcript --
// otherwise it is simply invisible until the user switches back to it.
function testTaskEndClearsProgressOfAHiddenTab() {
  const {win} = makePage();
  const out = win.document.getElementById('output');
  const tabA = win._testApi.getActiveTabId();
  win._testApi.endLaunch();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabA, tabB, 'a second tab must have been created');

  send(win, {
    type: 'worktree_progress',
    message: 'Generating commit message\u2026',
    tabId: tabA,
  });
  send(win, {type: 'task_error', tabId: tabA, startTs: 1, endTs: 2});

  clickTab(win, tabA);
  assert.deepStrictEqual(
    progressLines(out),
    [],
    "a hidden tab's task ending must clear its stranded progress line too, " +
      'not park it in the transcript until the user comes back',
  );
  win.close();
  console.log('  ok - a hidden tab keeps no stranded progress line');
}

// The net must not catch a live operation belonging to somebody else:
// tabs run independently, and one finishing says nothing about a merge
// running on another.
function testTaskEndOfAnotherTabLeavesTheLiveLineAlone() {
  const {win} = makePage();
  const out = win.document.getElementById('output');
  const tabA = win._testApi.getActiveTabId();
  win._testApi.endLaunch();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();

  // tab B (visible) is merging...
  send(win, {
    type: 'worktree_progress',
    message: 'Generating commit message\u2026',
    tabId: tabB,
  });
  assert.deepStrictEqual(
    progressLines(out),
    ['Generating commit message\u2026'],
    "the visible tab's merge line must be on screen to begin with",
  );

  // ...while tab A's own task ends. (A finished task pulls the view onto
  // itself, so tab B's transcript is put away rather than on screen --
  // the question is whether its live line is still IN it.)
  send(win, {type: 'task_done', tabId: tabA, startTs: 1, endTs: 2});

  clickTab(win, tabB);
  assert.deepStrictEqual(
    progressLines(out),
    ['Generating commit message\u2026'],
    "another tab's task ending must not cancel this tab's live merge " +
      'line: tabs run independently and tab B is still merging',
  );
  win.close();
  console.log('  ok - a task end elsewhere leaves the live line alone');
}

// The safety net must not eat the outcome the user is waiting to read.
function testTaskEndKeepsTheRealResultLine() {
  const {win} = makePage();
  const out = win.document.getElementById('output');
  const tabId = win._testApi.getActiveTabId();

  send(win, {
    type: 'worktree_progress',
    message: 'Committing\u2026',
    tabId,
  });
  send(win, {
    type: 'worktree_result',
    success: true,
    message: 'chore: update README',
    tabId,
  });
  send(win, {type: 'task_done', tabId, startTs: 1, endTs: 2});

  assert.deepStrictEqual(
    progressLines(out),
    [],
    'the happy path still ends with no progress line',
  );
  assert.ok(
    out.textContent.includes('chore: update README'),
    'the committed-message line is the result of the flow and must ' +
      'survive the task ending',
  );
  win.close();
  console.log('  ok - the result line survives the task ending');
}

function main() {
  testProgressReplacedInPlaceAndClearedByAutocommitDone();
  testWorktreeProgressRendered();
  testProgressForBackgroundTabGoesToThatTab();
  testEveryTaskEndClearsWorktreeProgress();
  testTaskEndClearsProgressOfAHiddenTab();
  testTaskEndOfAnotherTabLeavesTheLiveLineAlone();
  testTaskEndKeepsTheRealResultLine();
  console.log('actionProgressRendered.test.js: all tests passed');
}

main();
