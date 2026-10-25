// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The daemon broadcasts ONE `autocommit_done` per Git Commit, stamped
// with the owning tab id (src/kiss/tests/server/test_single_owner_ui_broadcast.py:
// no per-viewer fan-out, no `mirrorOf`).  The page owning that tab
// appends the outcome as a result panel and shows no blocking action
// bar for it.

'use strict';

const assert = require('assert');
const {makePage, send, stubRaf} = require('./simplify2_harness');

function testAutocommitDoneAppendsResultPanel() {
  const {win} = makePage({beforeScripts: stubRaf});
  send(win, {
    type: 'autocommit_done',
    tabId: win._testApi.getActiveTabId(),
    success: true,
    committed: true,
    message: 'Committed: chore',
  });
  assert.strictEqual(
    win.document.querySelector('#input-area .wt-bar'),
    null,
    'a finished commit must not show a blocking action bar',
  );
  const results = Array.from(win.document.querySelectorAll('.wt-result-ok'));
  assert.ok(
    results.some(el => (el.textContent || '').includes('Committed: chore')),
    'the commit outcome must be appended as a result panel',
  );
  win.close();
  console.log('  ok - autocommit_done appends its result panel');
}

try {
  testAutocommitDoneAppendsResultPanel();
  console.log('1 passed, 0 failed');
} catch (err) {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
}
