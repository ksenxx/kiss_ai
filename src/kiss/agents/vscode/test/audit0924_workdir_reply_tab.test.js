// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) regression tests for a late "Working directory"
// reply from the VS Code host landing in a panel that was not waiting
// for it.
//
// History: this file used to check that a reply was routed to the TAB
// that asked (openWorkDir / pickWorkDir carried a tabId, the host echoed
// it, and only that tab's panel reacted).  The working directory is one
// global value now: requests and replies name no tab, every tab shows
// the same directory, and `submit` carries none.  The tab-correlation
// tests (and the host test that `workDirError` echoes a tabId) are gone
// with the behaviour; `workDirPanelHost.test.js` asserts the replies
// carry no tabId.
//
// What remains, and is tested here: the webview reacts to a reply only
// while the open panel is waiting on a request it sent
// (workDirRequestPending).  Closing the panel abandons the request, so
// a late `workDirPicked` for it adopts the directory (it IS the global
// value now) but leaves a since-reopened panel, and whatever was typed
// into it, alone; a late `workDirError` is not shown in that panel.  A
// reply that arrives while the panel is waiting (also after a tab
// switch with the panel kept open) closes it or shows the error.

/* global require, __dirname, console, process */

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
  send(win, {type: 'configData', config: {work_dir: '/work/ws'}});
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function byId(win, id) {
  const node = win.document.getElementById(id);
  assert.ok(node, `#${id} must exist`);
  return node;
}

function typeInto(win, id, value) {
  const node = byId(win, id);
  node.value = value;
  node.dispatchEvent(new win.Event('input', {bubbles: true}));
}

function pressEnter(win, id) {
  byId(win, id).dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
}

function msgs(posted, type) {
  return posted.filter(m => m && m.type === type);
}

function panelOpen(win) {
  return byId(win, 'workdir-panel').classList.contains('open');
}

function errorText(win) {
  const el = byId(win, 'workdir-error');
  return el.hidden ? '' : el.textContent;
}

/** The panel's "Current:" line ('' while it is hidden). */
function currentLine(win) {
  const el = byId(win, 'workdir-current');
  return el.hidden ? '' : el.textContent;
}

function openPanelViaMenu(win) {
  click(win, byId(win, 'more-btn'));
  click(win, byId(win, 'workdir-btn'));
  assert.ok(panelOpen(win), 'the menu item opens the panel');
}

function switchToTab(win, tabId) {
  click(win, win.document.querySelector(`.chat-tab[data-tab-id="${tabId}"]`));
  assert.strictEqual(win._testApi.getActiveTabId(), tabId);
}

function submitPrompt(win, posted, text) {
  const before = msgs(posted, 'submit').length;
  byId(win, 'task-input').value = text;
  click(win, byId(win, 'send-btn'));
  const sent = msgs(posted, 'submit');
  assert.strictEqual(sent.length, before + 1, 'one submit is posted');
  return sent[sent.length - 1];
}

/**
 * The panel asks for a folder (through *ask*), the user closes it,
 * creates a second tab and reopens the panel there with a path typed
 * into it.  Returns the two tab ids.
 */
function askThenCloseAndReopen(win, posted, ask) {
  const tabA = win._testApi.getActiveTabId();
  openPanelViaMenu(win);
  ask(win, posted);
  click(win, byId(win, 'workdir-panel-close'));
  assert.ok(!panelOpen(win), 'the user closed the panel');

  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabB, tabA);
  openPanelViaMenu(win);
  typeInto(win, 'workdir-input', '/typed/later');
  assert.strictEqual(byId(win, 'workdir-input').value, '/typed/later');
  return {tabA, tabB};
}

function askViaPickButton(win, posted) {
  click(win, byId(win, 'workdir-pick-btn'));
  const picks = msgs(posted, 'pickWorkDir');
  assert.strictEqual(picks.length, 1);
  assert.strictEqual(picks[0].tabId, undefined, 'the request names no tab');
}

function askViaTypedPath(win, posted) {
  typeInto(win, 'workdir-input', '/asked/first');
  pressEnter(win, 'workdir-input');
  const opens = msgs(posted, 'openWorkDir');
  assert.strictEqual(opens.length, 1);
  assert.strictEqual(opens[0].path, '/asked/first');
  assert.strictEqual(opens[0].tabId, undefined, 'the request names no tab');
}

// ---------------------------------------------------------------------------

function testLatePickLeavesTheReopenedPanelOpen() {
  const {win, posted} = makeWebview();
  const {tabA, tabB} = askThenCloseAndReopen(win, posted, askViaPickButton);

  send(win, {type: 'workDirPicked', path: '/picked/late'});
  assert.ok(panelOpen(win), 'a reply to an abandoned request keeps the panel');
  assert.strictEqual(
    byId(win, 'workdir-input').value,
    '/typed/later',
    'the typed path is intact',
  );
  assert.strictEqual(errorText(win), '');
  // The folder is the global directory all the same: the panel's
  // "Current:" line follows it, in this tab and in the one that asked.
  assert.strictEqual(currentLine(win), 'Current: /picked/late');

  click(win, byId(win, 'workdir-panel-close'));
  let sub = submitPrompt(win, posted, 'from b');
  assert.strictEqual(sub.tabId, tabB);
  assert.strictEqual(sub.workDir, undefined, 'no tab carries a directory');
  switchToTab(win, tabA);
  openPanelViaMenu(win);
  assert.strictEqual(currentLine(win), 'Current: /picked/late');
  click(win, byId(win, 'workdir-panel-close'));
  sub = submitPrompt(win, posted, 'from a');
  assert.strictEqual(sub.tabId, tabA);
  assert.strictEqual(sub.workDir, undefined);
  assert.strictEqual(msgs(posted, 'setWorkDir').length, 0, 'the host did it');
}

function testLateErrorStaysOutOfTheReopenedPanel() {
  const {win, posted} = makeWebview();
  askThenCloseAndReopen(win, posted, askViaTypedPath);

  send(win, {type: 'workDirError', text: 'Not a directory: /asked/first'});
  assert.ok(panelOpen(win));
  assert.strictEqual(errorText(win), '', 'the stale error is not shown');
  assert.strictEqual(byId(win, 'workdir-input').value, '/typed/later');
  assert.strictEqual(currentLine(win), 'Current: /work/ws');
}

function testReplyForTheWaitingPanelCloses() {
  const {win, posted} = makeWebview();
  openPanelViaMenu(win);
  askViaTypedPath(win, posted);
  send(win, {type: 'workDirPicked', path: '/asked/first'});
  assert.ok(!panelOpen(win), 'the waiting panel closes on its answer');
  const sub = submitPrompt(win, posted, 'go');
  assert.strictEqual(sub.workDir, undefined);
  openPanelViaMenu(win);
  assert.strictEqual(currentLine(win), 'Current: /asked/first');
}

function testErrorForTheWaitingPanelIsShown() {
  const {win, posted} = makeWebview();
  openPanelViaMenu(win);
  askViaTypedPath(win, posted);
  send(win, {type: 'workDirError', text: 'Not a directory: x'});
  assert.ok(panelOpen(win), 'an error leaves the panel on screen');
  assert.strictEqual(errorText(win), 'Not a directory: x');
  assert.strictEqual(currentLine(win), 'Current: /work/ws');

  // The panel is still waiting: a second try that works closes it.
  typeInto(win, 'workdir-input', '/asked/again');
  pressEnter(win, 'workdir-input');
  assert.strictEqual(errorText(win), '', 'a new request clears the error');
  send(win, {type: 'workDirPicked', path: '/asked/again'});
  assert.ok(!panelOpen(win));
}

function testPanelKeptOpenAcrossATabSwitchClosesOnTheReply() {
  // The panel stays open across a tab switch and keeps waiting: the
  // host's answer closes it whichever tab is active by then.
  const {win, posted} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  openPanelViaMenu(win);
  askViaPickButton(win, posted);
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabB, tabA);
  assert.ok(panelOpen(win));
  typeInto(win, 'workdir-input', '/for/b');
  pressEnter(win, 'workdir-input');
  const opens = msgs(posted, 'openWorkDir');
  assert.strictEqual(opens.length, 1);
  assert.strictEqual(opens[0].tabId, undefined);

  send(win, {type: 'workDirPicked', path: '/picked/by-dialog'});
  assert.ok(!panelOpen(win), 'the first answer closes the waiting panel');
  // The second answer finds no waiting panel: it only moves the
  // directory on.
  send(win, {type: 'workDirPicked', path: '/for/b'});
  assert.ok(!panelOpen(win));
  openPanelViaMenu(win);
  assert.strictEqual(currentLine(win), 'Current: /for/b');
}

function testReopenedPanelWithItsOwnRequestClosesOnAnyReply() {
  // The reopened panel asked again, so it is waiting: the webview
  // cannot tell the earlier request's late answer from its own (replies
  // name no request), and either answer is the global directory now.
  const {win, posted} = makeWebview();
  askThenCloseAndReopen(win, posted, askViaPickButton);
  typeInto(win, 'workdir-input', '/second/ask');
  pressEnter(win, 'workdir-input');
  assert.strictEqual(msgs(posted, 'openWorkDir').length, 1);
  send(win, {type: 'workDirPicked', path: '/picked/late'});
  assert.ok(!panelOpen(win), 'the waiting panel closes');
  openPanelViaMenu(win);
  assert.strictEqual(currentLine(win), 'Current: /picked/late');
}

// ---------------------------------------------------------------------------

const tests = [
  [
    'a late workDirPicked leaves the reopened panel and its typed path',
    testLatePickLeavesTheReopenedPanelOpen,
  ],
  [
    'a late workDirError is not shown in the reopened panel',
    testLateErrorStaysOutOfTheReopenedPanel,
  ],
  [
    'a reply for the waiting panel closes it',
    testReplyForTheWaitingPanelCloses,
  ],
  [
    'an error for the waiting panel is shown',
    testErrorForTheWaitingPanelIsShown,
  ],
  [
    'a panel kept open across a tab switch closes on the reply',
    testPanelKeptOpenAcrossATabSwitchClosesOnTheReply,
  ],
  [
    'a reopened panel that asked again closes on any reply',
    testReopenedPanelWithItsOwnRequestClosesOnAnyReply,
  ],
];

let failed = 0;
for (const [name, fn] of tests) {
  try {
    fn();
    console.log('ok - ' + name);
  } catch (e) {
    failed += 1;
    console.log('not ok - ' + name);
    console.log(e && e.stack ? e.stack : String(e));
  }
}
if (failed) {
  console.log(`${failed} of ${tests.length} tests failed`);
  process.exit(1);
}
console.log(`all ${tests.length} audit0924_workdir_reply_tab tests passed`);
