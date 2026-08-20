// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the two post-task action bars:
//
//   * the worktree bar's third button, "Do nothing", which must send
//     ``worktreeAction`` with ``action: 'nothing'`` (the daemon then
//     detaches from the worktree, leaving it on disk as it is);
//   * the non-worktree manual-commit bar (``main_tree_done``), whose
//     Auto commit / Discard / Do nothing buttons must send
//     ``autocommitAction`` / ``mainTreeAction`` and whose terminal
//     events (``autocommit_done`` / ``main_tree_result``) must
//     dismiss it again — without an ``autocommit_done`` ever
//     dismissing a WORKTREE bar, whose branch it says nothing about.

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

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function bar(win) {
  return win.document.querySelector('.wt-bar');
}

function barButtons(win) {
  return Array.from(win.document.querySelectorAll('.wt-bar .wt-btn'));
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

function testWorktreeBarHasDoNothing() {
  const {win, posted} = makeWebview();
  const tabId = win._testApi.getActiveTabId();

  send(win, {
    type: 'worktree_done',
    branch: 'kiss/wt-1',
    worktreeDir: '/repo/.kiss-worktrees/kiss_wt-1',
    originalBranch: 'main',
    changedFiles: ['a.txt'],
    tabId,
  });
  assert.ok(bar(win), 'worktree_done must show the action bar');
  assert.deepStrictEqual(
    barButtons(win).map(b => b.textContent),
    ['Auto-commit and merge', 'Discard', 'Do nothing'],
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
    win.document.getElementById('output').textContent.includes('Left branch'),
    'the outcome must be reported in the transcript',
  );
  win.close();
  console.log('  ok - worktree bar offers Do nothing and wires it up');
}

function testMainTreeBarButtonsAndWiring() {
  const {win, posted} = makeWebview();
  const tabId = win._testApi.getActiveTabId();

  send(win, {
    type: 'main_tree_done',
    workDir: '/repo',
    changedFiles: ['a.txt'],
    tabId,
  });
  assert.ok(bar(win), 'main_tree_done must show the action bar');
  assert.ok(inputHidden(win), 'the bar must replace the input box');
  assert.deepStrictEqual(
    barButtons(win).map(b => b.textContent),
    ['Auto commit', 'Discard', 'Do nothing'],
    'the main-tree bar must offer all three choices',
  );

  clickBarButton(win, 'Auto commit');
  assert.deepStrictEqual(
    plain(posted[posted.length - 1]),
    {type: 'autocommitAction', tabId, workDir: '/repo'},
    '"Auto commit" must reuse the manual Git Commit command with the ' +
      "run's workDir",
  );

  // A terminal event for ANOTHER repository sharing the tab (e.g. a
  // settings-drawer Git Commit issued from a second window) must
  // leave the bar's controls in place: its own tree is still dirty.
  send(win, {
    type: 'autocommit_done',
    success: true,
    committed: true,
    manual: true,
    message: 'chore: unrelated repo',
    workDir: '/other-repo',
    tabId,
  });
  assert.ok(
    bar(win),
    "an autocommit_done for a different workDir must not dismiss the bar",
  );

  // Its own terminal event (matching workDir; a successful manual
  // commit is reported by toast, hence no transcript line) must
  // dismiss the bar.
  send(win, {
    type: 'autocommit_done',
    success: true,
    committed: true,
    manual: true,
    message: 'chore: committed',
    workDir: '/repo',
    tabId,
  });
  assert.strictEqual(
    bar(win),
    null,
    'autocommit_done must dismiss the main-tree bar',
  );
  assert.ok(!inputHidden(win), 'the input box must be restored');

  // Discard: a fresh bar, then mainTreeAction/discard.
  send(win, {
    type: 'main_tree_done',
    workDir: '/repo',
    changedFiles: ['a.txt'],
    tabId,
  });
  clickBarButton(win, 'Discard');
  assert.deepStrictEqual(
    plain(posted[posted.length - 1]),
    {type: 'mainTreeAction', action: 'discard', tabId, workDir: '/repo'},
    '"Discard" must send mainTreeAction/discard',
  );
  send(win, {
    type: 'main_tree_result',
    success: true,
    message: 'Discarded 1 uncommitted file(s) in repo.',
    tabId,
  });
  assert.strictEqual(bar(win), null, 'main_tree_result must dismiss the bar');
  assert.ok(
    win.document
      .getElementById('output')
      .textContent.includes('Discarded 1 uncommitted file(s)'),
    'the discard outcome must be reported in the transcript',
  );

  // Do nothing: a fresh bar, then mainTreeAction/nothing.
  send(win, {
    type: 'main_tree_done',
    workDir: '/repo',
    changedFiles: ['a.txt'],
    tabId,
  });
  clickBarButton(win, 'Do nothing');
  assert.deepStrictEqual(
    plain(posted[posted.length - 1]),
    {type: 'mainTreeAction', action: 'nothing', tabId, workDir: '/repo'},
    '"Do nothing" must send mainTreeAction/nothing',
  );
  send(win, {
    type: 'main_tree_result',
    success: true,
    message: 'Left the changes in the working tree, uncommitted.',
    tabId,
  });
  assert.strictEqual(bar(win), null, 'the result must dismiss the bar');
  assert.ok(!inputHidden(win), 'the input box must be restored');
  win.close();
  console.log('  ok - main-tree bar buttons send the right commands');
}

function testAutocommitDoneLeavesWorktreeBarAlone() {
  const {win} = makeWebview();
  const tabId = win._testApi.getActiveTabId();

  send(win, {
    type: 'worktree_done',
    branch: 'kiss/wt-1',
    worktreeDir: '/repo/.kiss-worktrees/kiss_wt-1',
    originalBranch: 'main',
    changedFiles: ['a.txt'],
    tabId,
  });
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
  win.close();
  console.log('  ok - autocommit_done leaves a pending worktree bar alone');
}

function testBackgroundTabMainTreeBarIsDeferred() {
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.endLaunch();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabA, tabB, 'a second tab must have been created');

  send(win, {
    type: 'main_tree_done',
    workDir: '/repo',
    changedFiles: ['a.txt'],
    tabId: tabA,
  });
  assert.strictEqual(
    bar(win),
    null,
    "another tab's main_tree_done must not paint the visible tab",
  );

  clickTab(win, tabA);
  assert.ok(
    bar(win),
    'the hidden tab must show the bar it received when switched to',
  );
  assert.deepStrictEqual(
    barButtons(win).map(b => b.textContent),
    ['Auto commit', 'Discard', 'Do nothing'],
    'the deferred bar must be the main-tree bar',
  );

  // A result addressed to the (now visible) owner clears it.
  send(win, {
    type: 'main_tree_result',
    success: true,
    message: 'Left the changes in the working tree, uncommitted.',
    tabId: tabA,
  });
  assert.strictEqual(bar(win), null, 'the result must dismiss the bar');

  // And a background result clears a still-hidden tab's bar too.
  send(win, {
    type: 'main_tree_done',
    workDir: '/repo',
    changedFiles: ['a.txt'],
    tabId: tabB,
  });
  assert.strictEqual(bar(win), null, 'tab B is hidden; no visible bar');
  send(win, {
    type: 'main_tree_result',
    success: true,
    message: 'Left the changes in the working tree, uncommitted.',
    tabId: tabB,
  });
  clickTab(win, tabB);
  assert.strictEqual(
    bar(win),
    null,
    "a background result must clear the hidden tab's deferred bar",
  );
  win.close();
  console.log('  ok - a hidden tab defers and clears its main-tree bar');
}

function testClearRetiresStaleMainTreeBarButNotWorktreeBar() {
  const {win} = makeWebview();
  const tabId = win._testApi.getActiveTabId();

  // A new task starting in the tab (its `clear` event) invalidates
  // the previous task's main-tree bar — acting on the old workDir
  // from under the new transcript would be wrong.
  send(win, {
    type: 'main_tree_done',
    workDir: '/repo',
    changedFiles: ['a.txt'],
    tabId,
  });
  assert.ok(bar(win), 'the main-tree bar must be up');
  send(win, {type: 'clear', tabId});
  assert.strictEqual(
    bar(win),
    null,
    'clear must retire a stale main-tree bar',
  );

  // A worktree bar survives the same event: its pending branch
  // outlives the transcript and has its own daemon-side lifecycle.
  send(win, {
    type: 'worktree_done',
    branch: 'kiss/wt-1',
    worktreeDir: '/repo/.kiss-worktrees/kiss_wt-1',
    originalBranch: 'main',
    changedFiles: ['a.txt'],
    tabId,
  });
  assert.ok(bar(win), 'the worktree bar must be up');
  send(win, {type: 'clear', tabId});
  assert.ok(bar(win), 'clear must NOT retire a pending worktree bar');
  win.close();
  console.log('  ok - clear retires stale main-tree bars only');
}

function main() {
  testWorktreeBarHasDoNothing();
  testMainTreeBarButtonsAndWiring();
  testAutocommitDoneLeavesWorktreeBarAlone();
  testBackgroundTabMainTreeBarIsDeferred();
  testClearRetiresStaleMainTreeBarButNotWorktreeBar();
  console.log('mainTreeBarAndDoNothing.test.js: all tests passed');
}

main();
