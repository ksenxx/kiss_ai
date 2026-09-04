// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the worktree action bar after a
// RETRYABLE `worktree_result` failure.
//
// The daemon (server/merge_flow.py) flags a deferred discard -- a
// sub-agent is still writing into the worktree -- and a failed "Do
// nothing" with `retryable: true` precisely so that "the webview keeps
// the bar's buttons instead of stripping the only retry controls".
// main.js did keep the bar, but every button had been disabled by the
// click that sent the action and nothing ever re-armed them: the bar
// the message told the user to retry with was dead.  These tests pin
// the contract for the visible tab and for a background tab, and the
// unchanged non-retryable behaviour (the bar goes away).

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
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=audit0902-retrybar-main.js',
  );

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
  return posted.filter(m => m.type === 'worktreeAction').map(plain);
}

function testRetryableResultReArmsVisibleBar() {
  const {win, posted} = makeWebview();
  const tabId = win._testApi.getActiveTabId();

  send(win, {type: 'worktree_done', tabId});
  assert.ok(bar(win), 'worktree_done shows the bar');
  assertAllEnabled(win, 'a fresh bar');

  clickBarButton(win, 'Discard');
  assertAllDisabled(win, 'while the discard is in flight');
  assert.deepStrictEqual(worktreeActions(posted), [
    {type: 'worktreeAction', action: 'discard', tabId},
  ]);

  // The daemon deferred the discard (a sub-agent is still writing) and
  // asks the user to retry.
  send(win, {
    type: 'worktree_result',
    tabId,
    success: false,
    retryable: true,
    message:
      "Discard deferred: a sub-agent of branch 'kiss_wt-1' is still " +
      'running. Retry once it finishes.',
  });
  assert.ok(bar(win), 'a retryable failure keeps the bar');
  assertAllEnabled(win, 'after a retryable failure');
  assert.ok(
    win.document
      .getElementById('output')
      .textContent.includes('Discard deferred'),
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

function testRetryableResultReArmsBackgroundBar() {
  const {win, posted} = makeWebview();
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
    message: "Discard deferred: a sub-agent of branch 'kiss_wt-2' is running.",
  });
  assert.strictEqual(bar(win), null, 'the background result stays off tab A');

  clickTab(win, tabB);
  assert.ok(bar(win), 'tab B comes back with its bar');
  assertAllEnabled(win, 'after a retryable failure in the background');
  assert.ok(
    win.document
      .getElementById('output')
      .textContent.includes('Discard deferred'),
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
  const {win} = makeWebview();
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
  testRetryableResultReArmsBackgroundBar();
  testNonRetryableFailureStillDismisses();
  console.log('all audit0902 retryable-bar tests passed');
}

main();
