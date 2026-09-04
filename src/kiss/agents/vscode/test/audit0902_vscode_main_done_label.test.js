// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the "Done (Xs)" status label a
// `task_done` event leaves on its tab.
//
// The daemon stamps `startTs` / `endTs` on the event and the label is
// computed from that span.  When the span is missing the webview falls
// back to the start time it recorded itself -- and that fallback used
// the module-level `t0`, i.e. the VISIBLE tab's clock, even when the
// event named a background tab, so a task that ran for a minute in a
// hidden tab could report the few seconds the tab on screen had been
// running.  The label is also built by the same helper the timer uses
// (doneLabelFor) instead of a second copy of the arithmetic.

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
      '\n//# sourceURL=audit0902-donelabel-main.js',
  );

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

function statusText(win) {
  return win.document.getElementById('status-text').textContent;
}

function startTask(win, tabId, startTs) {
  send(win, {type: 'setTaskText', text: 'work', tabId});
  send(win, {type: 'clear', chat_id: 'chat-' + tabId, tabId});
  send(win, {type: 'status', running: true, tabId, startTs});
}

function testSpanOnTheEventWins() {
  const {win} = makeWebview();
  const tabId = win._testApi.getActiveTabId();
  const now = Date.now();
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
  assert.ok(
    /^Done \(30s\)$/.test(statusText(win)) ||
      /^Done \(31s\)$/.test(statusText(win)),
    `an inverted span falls back to the tab's clock, got "${statusText(win)}"`,
  );
  win.close();
  console.log('  ok - the span on the event wins');
}

function testFallbackUsesTheFinishedTabsClock() {
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  const now = Date.now();
  // A has been running for a minute in the background; B, on screen,
  // started five seconds ago.
  startTask(win, tabA, now - 60000);
  clickTab(win, tabB);
  startTask(win, tabB, now - 5000);
  assert.strictEqual(win._testApi.getActiveTabId(), tabB);

  // A finishes; the event carries no span.  Its task_done also brings
  // the user to A (a finished task may switch tabs).
  send(win, {type: 'task_done', tabId: tabA});
  assert.strictEqual(win._testApi.getActiveTabId(), tabA);
  const label = statusText(win);
  assert.ok(
    /^Done \(1m 0s\)$/.test(label) || /^Done \(1m 1s\)$/.test(label),
    `a background tab's label is measured from ITS start, got "${label}"`,
  );
  // B is still running its own five-second-old task.
  clickTab(win, tabB);
  assert.ok(
    /^Running \d+s$/.test(statusText(win)),
    'the visible tab keeps its own clock',
  );
  win.close();
  console.log('  ok - the fallback uses the finished tab\u2019s clock');
}

function testFallbackOnTheVisibleTab() {
  const {win} = makeWebview();
  const tabId = win._testApi.getActiveTabId();
  const now = Date.now();
  startTask(win, tabId, now - 70000);
  send(win, {type: 'task_done', tabId});
  const label = statusText(win);
  assert.ok(
    /^Done \(1m 1?0?s\)$/.test(label) || /^Done \(1m 1[01]s\)$/.test(label),
    `the visible tab measures from its own start, got "${label}"`,
  );
  // A tab id the webview does not know: nothing to measure from, so
  // the label reads as an instant finish rather than throwing.
  send(win, {type: 'task_done', tabId: 'no-such-tab'});
  assert.strictEqual(statusText(win), label, 'an unknown tab changes nothing');
  win.close();
  console.log('  ok - the fallback on the visible tab');
}

function testFallbackWithoutAnyStart() {
  const {win} = makeWebview();
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
