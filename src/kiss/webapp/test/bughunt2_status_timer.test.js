// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send, sleep} = require('./simplify2_harness');

async function testForeignTabStatusMustNotClobberActiveTimer() {
  const {win, posted} = makePage();

  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'page must post ready with a tabId');
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
  const {win, posted} = makePage();
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
