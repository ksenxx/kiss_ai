// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the post-task worktree bar and for the
// worktree events the agent's own tools raise MID-task:
//
//   * the worktree bar's third button, "Do nothing", must send
//     ``worktreeAction`` with ``action: 'nothing'`` (the daemon then
//     detaches from the worktree, leaving it on disk as it is);
//   * an ``autocommit_done`` (the manual Git Commit button) must never
//     dismiss a pending worktree bar, whose branch it says nothing
//     about, and ``clear`` must not retire it either;
//   * ``worktree_created`` (the agent called ``create_worktree``) and
//     ``worktree_result`` with ``action: 'merge'`` (the agent called
//     ``commit_and_merge``) arrive while the task is still running:
//     no bar is up, nothing throws, the merge outcome lands in the
//     transcript and the composer stays usable.

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

function plain(x) {
  return JSON.parse(JSON.stringify(x));
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function bar(win) {
  return win.document.querySelector('.wt-bar');
}

function barButtons(win) {
  return Array.from(win.document.querySelectorAll('.wt-bar .wt-btns .wt-btn'));
}

function clickBarButton(win, text) {
  const btn = barButtons(win).find(b => b.textContent === text);
  assert.ok(btn, `the bar must have a "${text}" button`);
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  return btn;
}

function inputHidden(win) {
  return (
    win.document.getElementById('input-container').style.display === 'none'
  );
}

function outputText(win) {
  return win.document.getElementById('output').textContent;
}

function sendWorktreeDone(win, tabId) {
  send(win, {
    type: 'worktree_done',
    branch: 'kiss/wt-1',
    worktreeDir: '/repo/.kiss-worktrees/kiss_wt-1',
    originalBranch: 'main',
    changedFiles: ['a.txt'],
    tabId,
  });
}

function testWorktreeBarHasDoNothing() {
  const {win, posted} = makeWebview();
  const tabId = win._testApi.getActiveTabId();

  sendWorktreeDone(win, tabId);
  assert.ok(bar(win), 'worktree_done must show the action bar');
  assert.deepStrictEqual(
    barButtons(win).map(b => b.textContent),
    ['Auto-commit and merge', 'Do nothing', 'Discard'],
    'the worktree bar must offer all three choices',
  );

  const btn = clickBarButton(win, 'Do nothing');
  assert.deepStrictEqual(
    plain(posted[posted.length - 1]),
    {type: 'worktreeAction', action: 'nothing', tabId},
    '"Do nothing" must send worktreeAction/nothing for the owner tab',
  );
  assert.ok(btn.disabled, 'the clicked bar must disarm while in flight');

  // The daemon detached and answered success (kept: the worktree
  // stays on disk); the bar must go and the input must come back.
  send(win, {
    type: 'worktree_result',
    success: true,
    kept: true,
    message: "Left branch 'kiss/wt-1' and its worktree as they are.",
    tabId,
  });
  assert.strictEqual(bar(win), null, 'the result must dismiss the bar');
  assert.ok(!inputHidden(win), 'the input box must be restored');
  assert.ok(
    outputText(win).includes('Left branch'),
    'the outcome must be reported in the transcript',
  );
  win.close();
  console.log('  ok - worktree bar offers Do nothing and wires it up');
}

function testAutocommitDoneAndClearLeaveWorktreeBarAlone() {
  const {win} = makeWebview();
  const tabId = win._testApi.getActiveTabId();

  sendWorktreeDone(win, tabId);
  assert.ok(bar(win), 'the worktree bar must be up');

  // A settings-drawer Git Commit on the MAIN tree finishing says
  // nothing about the still-undecided worktree branch.
  send(win, {
    type: 'autocommit_done',
    success: true,
    committed: true,
    manual: true,
    message: 'chore: main tree commit',
    tabId,
  });
  assert.ok(
    bar(win),
    'autocommit_done must NOT dismiss a pending worktree bar',
  );

  // A new task starting in the tab does not retire it either: the
  // pending branch outlives the transcript and has its own
  // daemon-side lifecycle.
  send(win, {type: 'clear', tabId});
  assert.ok(bar(win), 'clear must NOT retire a pending worktree bar');
  win.close();
  console.log('  ok - autocommit_done and clear leave a worktree bar alone');
}

function testMidTaskWorktreeEventsRenderWithoutBar() {
  const {win, posted} = makeWebview();
  const tabId = win._testApi.getActiveTabId();

  send(win, {type: 'clear', tabId});
  send(win, {type: 'status', running: true, tabId});

  // The agent called create_worktree: the daemon records the
  // worktree and tells the client, which has nothing to show yet.
  send(win, {
    type: 'worktree_created',
    branch: 'kiss/wt-7',
    worktreeDir: '/repo/.kiss-worktrees/kiss_wt-7',
    worktreeWorkDir: '/repo/.kiss-worktrees/kiss_wt-7',
    tabId,
  });
  assert.strictEqual(bar(win), null, 'worktree_created must show no bar');
  assert.ok(!inputHidden(win), 'the composer must stay usable');

  // The agent called commit_and_merge: a merge progress line, then
  // the terminal result, both while the task is still running.
  send(win, {type: 'worktree_progress', message: 'Merging…', tabId});
  assert.ok(
    win.document.querySelector('#output .ev.wt-progress'),
    'worktree_progress must render its live line',
  );
  const before = posted.length;
  send(win, {
    type: 'worktree_result',
    action: 'merge',
    branch: 'kiss/wt-7',
    success: true,
    message: "Merged branch 'kiss/wt-7' into main.",
    tabId,
  });
  assert.strictEqual(
    win.document.querySelector('#output .ev.wt-progress'),
    null,
    'the result must clear the progress line',
  );
  assert.ok(
    win.document.querySelector('#output .ev.wt-result-ok'),
    'a successful mid-task merge must render a success line',
  );
  assert.ok(
    outputText(win).includes("Merged branch 'kiss/wt-7' into main."),
    'the merge outcome must be in the transcript',
  );
  assert.strictEqual(bar(win), null, 'no bar is pending for it to dismiss');
  assert.ok(!inputHidden(win), 'the composer must stay usable');
  assert.strictEqual(
    posted.length,
    before,
    'a mid-task worktree_result must send nothing back to the host',
  );

  // A failed merge reports in the error style.
  send(win, {
    type: 'worktree_result',
    action: 'merge',
    branch: 'kiss/wt-7',
    success: false,
    message: 'Merge conflict in a.txt.',
    tabId,
  });
  assert.ok(
    win.document.querySelector('#output .ev.wt-result-err'),
    'a failed mid-task merge must render an error line',
  );
  assert.ok(
    outputText(win).includes('Merge conflict in a.txt.'),
    'the failure must be in the transcript',
  );
  win.close();
  console.log(
    '  ok - mid-task worktree_created / worktree_result render inline',
  );
}

function testMidTaskWorktreeResultOnBackgroundTab() {
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  send(win, {type: 'clear', tabId: tabA});
  send(win, {type: 'status', running: true, tabId: tabA});

  // Open and switch to a second tab so tab A's events are deferred
  // into its hidden fragment.
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabA, tabB, 'a second tab must be active');

  send(win, {
    type: 'worktree_created',
    branch: 'kiss/wt-8',
    worktreeDir: '/repo/.kiss-worktrees/kiss_wt-8',
    tabId: tabA,
  });
  send(win, {
    type: 'worktree_result',
    action: 'merge',
    branch: 'kiss/wt-8',
    success: true,
    message: "Merged branch 'kiss/wt-8' into main.",
    tabId: tabA,
  });
  assert.strictEqual(bar(win), null, 'the visible tab must show no bar');
  assert.ok(
    !outputText(win).includes('kiss/wt-8'),
    "tab A's merge outcome must not paint the visible tab",
  );

  win._testApi.switchToTab(tabA);
  assert.ok(
    outputText(win).includes("Merged branch 'kiss/wt-8' into main."),
    "switching back must show tab A's merge outcome",
  );
  assert.strictEqual(bar(win), null, 'tab A holds no pending bar');
  win.close();
  console.log('  ok - a hidden tab keeps its mid-task merge outcome');
}

function main() {
  testWorktreeBarHasDoNothing();
  testAutocommitDoneAndClearLeaveWorktreeBarAlone();
  testMidTaskWorktreeEventsRenderWithoutBar();
  testMidTaskWorktreeResultOnBackgroundTab();
  console.log('worktreeBarAndMidTaskResult.test.js: all tests passed');
}

main();
