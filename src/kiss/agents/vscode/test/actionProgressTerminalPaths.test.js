// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the LAST exit of the commit/merge
// progress line.
//
// ``actionProgressRendered.test.js`` covers the two happy endings:
// ``autocommit_done`` clears an ``autocommit_progress`` line and
// ``worktree_result`` clears a ``worktree_progress`` line.  Neither of
// those events is guaranteed to arrive.
//
// Both flows run on the daemon's post-task path inside a swallow-all
// handler (``task_runner._run_task_inner``):
//
//     try:
//         if self._main_dirty_files(work_dir):
//             self._autocommit_changes(tab_id, work_dir=work_dir)
//     except BaseException:
//         logger.debug("Post-task autocommit error", exc_info=True)
//
// and the worktree branch a few lines below has the same shape.  The
// progress events are broadcast from INSIDE those calls, so anything
// raised after the first "Staging changes…" / "Generating commit
// message…" -- a git binary that dies, a stopped task unwinding through
// the merge, an LLM call that throws while writing the commit message
// -- eats the terminal event and the transcript keeps a live progress
// line that nothing will ever take away.  It is not a stale label: it
// reads as if the merge were still running, forever.
//
// What the daemon does still guarantee is the task-end event: control
// falls out of that handler and broadcasts task_done / task_error /
// task_stopped / task_interrupted immediately afterwards.  A task that
// has ended has no operation left in flight, so that is where the line
// has to go.
//
// In the happy path the terminal event has already removed the line by
// then, so this is a safety net that never fires -- which is why the
// tests below also pin that it does not fire too eagerly (another tab's
// task ending, and the real result line, must both survive).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function progressLines(root) {
  return Array.from(root.querySelectorAll('.wt-progress')).map(
    el => el.textContent,
  );
}

// One case per task-end event: each is a real ending of the run whose
// post-task commit/merge put the line on screen.
const TASK_END_EVENTS = [
  ['task_done', 'a task that finished'],
  ['task_error', 'a task that failed'],
  ['task_stopped', 'a task the user stopped'],
  ['task_interrupted', 'a task that was interrupted'],
];

function testEveryTaskEndClearsAutocommitProgress() {
  TASK_END_EVENTS.forEach(([type, what]) => {
    const {win} = makeWebview();
    const out = win.document.getElementById('output');
    const tabId = win._testApi.getActiveTabId();

    send(win, {
      type: 'autocommit_progress',
      message: 'Staging changes\u2026',
      tabId,
    });
    assert.deepStrictEqual(
      progressLines(out),
      ['Staging changes\u2026'],
      'the progress line must be on screen before the task ends',
    );

    // The daemon's `except BaseException` swallowed the failure, so no
    // autocommit_done is ever sent -- only the task-end event.
    send(win, {type, tabId, startTs: 1, endTs: 2});
    assert.deepStrictEqual(
      progressLines(out),
      [],
      `${what} must not leave a live "Staging changes…" line behind ` +
        `(${type} was the only terminal event the daemon sent)`,
    );
    win.close();
  });
  console.log('  ok - every task-end event clears an autocommit_progress line');
}

function testEveryTaskEndClearsWorktreeProgress() {
  TASK_END_EVENTS.forEach(([type, what]) => {
    const {win} = makeWebview();
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
  const {win} = makeWebview();
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
  const {win} = makeWebview();
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
  const {win} = makeWebview();
  const out = win.document.getElementById('output');
  const tabId = win._testApi.getActiveTabId();

  send(win, {
    type: 'autocommit_progress',
    message: 'Committing\u2026',
    tabId,
  });
  send(win, {
    type: 'autocommit_done',
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
  testEveryTaskEndClearsAutocommitProgress();
  testEveryTaskEndClearsWorktreeProgress();
  testTaskEndClearsProgressOfAHiddenTab();
  testTaskEndOfAnotherTabLeavesTheLiveLineAlone();
  testTaskEndKeepsTheRealResultLine();
  console.log('actionProgressTerminalPaths.test.js: all tests passed');
}

main();
