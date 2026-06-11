// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Bug-hunt integration test for the webview 'status' event handler in
// ``media/main.js``.
//
// Bug locked in:
//
//   ``handleEvent``'s ``case 'status'`` anchors the GLOBAL running
//   timer (``t0 = ev.startTs``) and clears the global ``endTs``
//   UNCONDITIONALLY — even when the status event is tab-stamped for a
//   background tab or for a tab owned by ANOTHER VS Code window (the
//   kiss-web daemon broadcasts tab-stamped events to every connected
//   client).  The very next block in the same handler correctly gates
//   ALL UI updates on ``ev.tabId === undefined || ev.tabId ===
//   activeTabId`` — the timer anchor must obey the same rule.  With
//   the bug, a broadcast ``status`` for a foreign tab whose agent
//   started an hour ago flips the ACTIVE tab's header from
//   "Running 6s" to "Running 60m 1s".
//
// This test drives the real ``media/main.js`` (plus the real
// ``media/chat.html`` markup and ``media/panelCopy.js``) inside jsdom
// — no mocks of project code — exactly like ``panelCopy.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/bughunt2_status_timer.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview: the real
 * ``chat.html`` body (placeholders blanked), ``panelCopy.js`` and
 * ``main.js`` evaluated in the window, and a recording
 * ``acquireVsCodeApi`` stub (the only host API the webview has).
 */
function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  // Blank every template placeholder and drop <script> tags — the
  // scripts under test are evaluated manually below so the test
  // controls ordering and globals.
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  // jsdom lacks these layout APIs that main.js calls; harmless no-ops.
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  // Evaluate the real webview scripts in load order.
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function testForeignTabStatusMustNotClobberActiveTimer() {
  const {win, posted} = makeWebview();

  // main.js init() announces the active tab id in its 'ready' message.
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const activeId = ready.tabId;

  // 1. The active tab's agent started 5 s ago.
  send(win, {
    type: 'status',
    running: true,
    tabId: activeId,
    startTs: Date.now() - 5_000,
  });
  const statusText = win.document.getElementById('status-text');
  assert.match(
    statusText.textContent,
    /^Running [4-7]s$/,
    'active tab timer must anchor to its own startTs',
  );

  // 2. A broadcast status for a tab of ANOTHER window (unknown tab id)
  //    whose agent started an hour ago.  Per the handler's own routing
  //    comment this event must not touch this window's UI state.
  send(win, {
    type: 'status',
    running: true,
    tabId: 'foreign-window-tab',
    startTs: Date.now() - 3_600_000,
  });

  // 3. Let the 1 s timer tick re-render the header from the global t0.
  await sleep(1_300);

  const text = statusText.textContent;
  assert.match(
    text,
    /^Running [4-9]s$/,
    'BUG: a tab-stamped status event for a foreign/background tab ' +
      `clobbered the active tab's timer anchor (header now "${text}")`,
  );
  win.close();
  console.log(
    '  ok - foreign-tab status does not clobber the active running timer',
  );
}

async function testBackgroundLocalTabStatusKeepsOwnT0() {
  const {win, posted} = makeWebview();
  const ready = posted.find(m => m.type === 'ready');
  const activeId = ready.tabId;

  // Materialise a LOCAL background tab the way the backend does it.
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'bg-sub-tab',
    description: 'sub agent work',
  });

  // Active tab running, started 5 s ago.
  send(win, {
    type: 'status',
    running: true,
    tabId: activeId,
    startTs: Date.now() - 5_000,
  });

  // Background tab's agent started 30 minutes ago.
  const bgStart = Date.now() - 1_800_000;
  send(win, {
    type: 'status',
    running: true,
    tabId: 'bg-sub-tab',
    startTs: bgStart,
  });

  await sleep(1_300);

  const statusText = win.document.getElementById('status-text');
  assert.match(
    statusText.textContent,
    /^Running [4-9]s$/,
    'BUG: background-tab status clobbered the active timer ' +
      `(header now "${statusText.textContent}")`,
  );
  win.close();
  console.log(
    '  ok - background-tab status keeps its own t0 without touching the UI',
  );
}

async function runTests() {
  await testForeignTabStatusMustNotClobberActiveTimer();
  await testBackgroundLocalTabStatusKeepsOwnT0();
}

runTests().then(
  () => {
    console.log('\n2 passed, 0 failed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    process.exit(1);
  },
);
