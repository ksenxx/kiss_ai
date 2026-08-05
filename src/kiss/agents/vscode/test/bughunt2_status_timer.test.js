// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function testForeignTabStatusMustNotClobberActiveTimer() {
  const {win, posted} = makeWebview();

  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const activeId = ready.tabId;

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

  send(win, {
    type: 'status',
    running: true,
    tabId: 'foreign-window-tab',
    startTs: Date.now() - 3_600_000,
  });

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

  send(win, {
    type: 'openSubagentTab',
    tab_id: 'bg-sub-tab',
    description: 'sub agent work',
  });

  send(win, {
    type: 'status',
    running: true,
    tabId: activeId,
    startTs: Date.now() - 5_000,
  });

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
