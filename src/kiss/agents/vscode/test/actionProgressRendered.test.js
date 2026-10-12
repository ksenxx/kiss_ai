// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the user-facing progress event the
// daemon broadcasts during a merge flow:
//
//   * ``worktree_progress`` ("Staging changes…", "Generating commit
//     message…", "Committing…") — once rendered only by the VS Code
//     host's native progress toast, so a remote or mobile browser user
//     clicking Merge saw nothing at all (F08-4).
//
// It carries a plain string that any client can show, and its terminal
// sibling (``worktree_result``) must clear the live line again; so must
// the manual Git Commit button's ``autocommit_done``.

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

// The Chats-panel pick: chat tabs have no row of their own any more, so a
// chat is brought on screen through the test API (sub-agent and content
// tabs on the group strip are still clicked).
function clickTab(win, tabId) {
  assert.ok(
    win._testApi.openTabs().some(t => t.id === tabId),
    `tab ${tabId} must be open`,
  );
  const el = win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  if (el) el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  else win._testApi.switchToTab(tabId);
}

function progressLines(root) {
  return Array.from(root.querySelectorAll('.wt-progress')).map(
    el => el.textContent,
  );
}

function testProgressReplacedInPlaceAndClearedByAutocommitDone() {
  const {win} = makeWebview();
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
  const {win} = makeWebview();
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
    'worktree_progress must render in the webview too — the remote web ' +
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
  const {win} = makeWebview();
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
