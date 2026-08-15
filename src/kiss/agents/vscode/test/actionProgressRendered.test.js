// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the two user-facing progress events the
// daemon broadcasts during the commit/merge flows:
//
//   * ``autocommit_progress`` ("Staging changes…", "Generating commit
//     message…", "Committing…") — emitted three times per non-worktree
//     task and, until this test, rendered by NO client (F08-3).
//   * ``worktree_progress``   ("Generating commit message…") — rendered
//     only by the VS Code host's native progress toast, so a remote or
//     mobile browser user clicking Merge saw nothing at all (F08-4).
//
// Both carry a plain string that any client can show, and both have a
// terminal sibling (``autocommit_done`` / ``worktree_result``) that must
// clear the live line again.

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

function testAutocommitProgressRendered() {
  const {win} = makeWebview();
  const doc = win.document;
  const out = doc.getElementById('output');
  const tabId = win._testApi.getActiveTabId();

  send(win, {
    type: 'autocommit_progress',
    message: 'Staging changes\u2026',
    tabId,
  });
  assert.deepStrictEqual(
    progressLines(out),
    ['Staging changes\u2026'],
    'autocommit_progress must render its message in the transcript ' +
      '(the daemon broadcasts it three times per task and no client ' +
      'showed any of them)',
  );

  // The three messages are one flow, not three log lines: the live line
  // is replaced in place.
  send(win, {
    type: 'autocommit_progress',
    message: 'Generating commit message\u2026',
    tabId,
  });
  send(win, {type: 'autocommit_progress', message: 'Committing\u2026', tabId});
  assert.deepStrictEqual(
    progressLines(out),
    ['Committing\u2026'],
    'successive autocommit_progress messages must replace the live line',
  );

  // The terminal event clears the progress line and reports the outcome.
  send(win, {
    type: 'autocommit_done',
    success: true,
    message: 'chore: update README',
    tabId,
  });
  assert.deepStrictEqual(
    progressLines(out),
    [],
    'autocommit_done must clear the live progress line',
  );
  assert.ok(
    out.textContent.includes('chore: update README'),
    'autocommit_done must still render its own result line',
  );
  win.close();
  console.log('  ok - autocommit_progress renders, replaces and clears');
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
    type: 'autocommit_progress',
    message: 'Staging changes\u2026',
    tabId: tabA,
  });
  assert.deepStrictEqual(
    progressLines(out),
    [],
    "another tab's autocommit_progress must not paint the visible tab",
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
  testAutocommitProgressRendered();
  testWorktreeProgressRendered();
  testProgressForBackgroundTabGoesToThatTab();
  console.log('actionProgressRendered.test.js: all tests passed');
}

main();
