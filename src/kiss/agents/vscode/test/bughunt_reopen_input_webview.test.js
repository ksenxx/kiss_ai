// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test: the REAL ``media/main.js`` webview, driven inside
// jsdom, must keep accepting user input after the VS Code sidebar view
// that STARTED a running task is closed and re-opened.
//
// Closing and re-opening the secondary-sidebar view tears the webview
// down and rebuilds it from scratch — the fresh webview restores its
// tabs from ``vscode.getState()`` (persisted across the dispose) and
// posts ``ready`` with ``restoredTabs`` so the extension can
// ``resumeSession`` each one.  The daemon then re-broadcasts
// ``status running:true`` followed by the ``task_events`` replay.
//
// REQUIREMENT (from the task): a tab where a task STARTED must, after a
// close+reopen, behave EXACTLY like a tab that LOADS the task — i.e.
//   * while the task is still running, a typed message is sent as an
//     ``appendUserMessage`` (injected into the live agent), and
//   * after the task finishes, a typed message is sent as a ``submit``
//     (starts a new run).
// If the re-opened webview never re-learns that the task is running
// (its ``isRunning`` stays false), a mid-run message is wrongly sent as
// a ``submit`` — which the extension drops because the tab is still in
// ``_runningTabs`` — so the user's input is silently ignored.
//
// This drives the production ``chat.html`` + ``panelCopy.js`` +
// ``main.js`` (no mocks of project code) and carries the persisted
// ``vscode`` state across the two webview instances exactly as VS Code
// does across a view dispose/reopen.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/bughunt_reopen_input_webview.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

// Persisted webview state, shared across webview instances — this is
// the ``vscode.getState()`` / ``vscode.setState()`` blob that VS Code
// keeps alive while the view is closed and hands back on reopen.
let persistedState;

/**
 * Build a jsdom window running the production chat webview, sharing the
 * module-level ``persistedState`` so a second instance restores exactly
 * what the first one persisted (mirrors VS Code's view dispose/reopen).
 */
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
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => persistedState,
      setState: s => {
        persistedState = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Type *text* into the task input and click the send button. */
function typeAndSend(win, text) {
  const inp = win.document.getElementById('task-input');
  const sendBtn = win.document.getElementById('send-btn');
  inp.value = text;
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  sendBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

async function runTests() {
  // ---- First webview: start a task in its only tab ------------------
  const wv1 = makeWebview();
  const ready1 = wv1.posted.find(m => m.type === 'ready');
  assert.ok(ready1 && ready1.tabId, 'webview must post ready with a tabId');
  const TAB = ready1.tabId;

  // User submits a task.
  typeAndSend(wv1.win, 'do a long task');
  const submit = wv1.posted.find(m => m.type === 'submit');
  assert.ok(submit && submit.tabId === TAB, 'first send must be a submit');

  // Extension/daemon echo a ``clear`` (assigns the backend chat id —
  // required for the tab to be restorable) then ``status running:true``.
  send(wv1.win, {type: 'clear', chat_id: 'chat-1', tabId: TAB});
  send(wv1.win, {
    type: 'status',
    running: true,
    tabId: TAB,
    startTs: Date.now(),
  });

  // Sanity: while running, a typed message goes out as appendUserMessage.
  typeAndSend(wv1.win, 'mid-run note before close');
  assert.ok(
    wv1.posted.some(m => m.type === 'appendUserMessage'),
    'sanity: the launching tab sends appendUserMessage while running',
  );
  assert.ok(persistedState, 'tab state must have been persisted');
  wv1.win.close();

  // ---- Close + reopen: a fresh webview restores from getState -------
  const wv2 = makeWebview();
  const ready2 = wv2.posted.find(m => m.type === 'ready');
  assert.ok(ready2, 're-opened webview must post ready');
  // Compare via JSON: ``restoredTabs`` are objects from the jsdom realm
  // (different prototypes than Node literals), so deepStrictEqual would
  // spuriously fail on the prototype check.
  assert.strictEqual(
    JSON.stringify(ready2.restoredTabs),
    JSON.stringify([{tabId: TAB, chatId: 'chat-1'}]),
    'BUG: re-opened webview must restore the running tab so the ' +
      'extension can resumeSession it (got ' +
      JSON.stringify(ready2.restoredTabs) +
      ')',
  );

  // The daemon's _replay_session re-broadcasts running status BEFORE the
  // task_events replay (server.py ordering), then the event stream.
  send(wv2.win, {
    type: 'status',
    running: true,
    tabId: TAB,
    startTs: Date.now(),
  });
  send(wv2.win, {
    type: 'task_events',
    events: [],
    task: 'do a long task',
    tabId: TAB,
    chat_id: 'chat-1',
  });

  // (1) DURING the task: a typed message MUST be an appendUserMessage.
  wv2.posted.length = 0;
  typeAndSend(wv2.win, 'please also update the docs');
  const duringMsgs = wv2.posted.filter(
    m => m.type === 'appendUserMessage' || m.type === 'submit',
  );
  assert.deepStrictEqual(
    duringMsgs.map(m => m.type),
    ['appendUserMessage'],
    'BUG: after reopen, a mid-run message must be sent as ' +
      'appendUserMessage (was: ' +
      JSON.stringify(duringMsgs.map(m => m.type)) +
      '). The re-opened tab never re-learned the task is running.',
  );

  // (2) AFTER the task finishes: a typed message MUST be a submit.
  send(wv2.win, {type: 'status', running: false, tabId: TAB});
  wv2.posted.length = 0;
  typeAndSend(wv2.win, 'now do a follow-up task');
  const afterMsgs = wv2.posted.filter(
    m => m.type === 'appendUserMessage' || m.type === 'submit',
  );
  assert.deepStrictEqual(
    afterMsgs.map(m => m.type),
    ['submit'],
    'after the task finishes, a typed message must be a submit (was: ' +
      JSON.stringify(afterMsgs.map(m => m.type)) +
      ')',
  );

  wv2.win.close();
}

runTests().then(
  () => {
    console.log('\nAll tests passed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    process.exit(1);
  },
);
