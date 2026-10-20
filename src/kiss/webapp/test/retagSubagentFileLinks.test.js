// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// End-to-end jsdom tests: when the daemon re-announces an open sub-agent
// under a new tab id the client renames the tab (retagSubagentTab).  The
// file-link bookkeeping of the transcript -- the `data-path-tab` stamp
// on every path span, the in-flight check keys and the deferred tail
// sweep's tab id -- must follow the rename, or the sub-agent's own
// commit_and_merge never re-verifies its links, a reconnect re-asks
// under a tab id the daemon has closed, closing the tab leaks the
// spans, and the frame's collapse work is dropped.

/* global require, console, process */

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');
const {clickClose} = require('./subagent_tab_helpers');

const REL = 'reports/out.html';

/**
 * Boot a page with a running parent task and a running sub-agent tab
 * `sa-1` for task `t1`.
 */
function bootSubagent() {
  const {win, posted} = makePage();
  const parentId = posted.find(m => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  send(win, {type: 'status', running: true, tabId: parentId, startTs: 1});
  announce(win, parentId, 'sa-1');
  send(win, {type: 'status', running: true, tabId: 'sa-1', startTs: 1});
  return {win, posted, parentId};
}

function announce(win, parentId, tabId) {
  send(win, {
    type: 'openSubagentTab',
    tab_id: tabId,
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: 't1',
    isSubagentTab: true,
    isDone: false,
  });
}

function checkPathsFor(posted, tabId) {
  return posted.filter(m => m.type === 'checkPaths' && m.tabId === tabId);
}

/** Stream a result naming REL into sa-1 and retag the tab onto sa-2. */
function streamAndRetag(win, posted, parentId) {
  send(win, {
    type: 'result',
    tabId: 'sa-1',
    taskId: 't1',
    success: true,
    summary: '<p>Report at <code>' + REL + '</code></p>',
    total_tokens: 1,
    cost: 'N/A',
  });
  const first = checkPathsFor(posted, 'sa-1');
  assert.strictEqual(first.length, 1, 'setup: one check under sa-1');
  assert.ok(first[0].paths.includes(REL), 'setup: the check names REL');
  announce(win, parentId, 'sa-2');
  assert.ok(
    posted.some(m => m.type === 'closeTab' && m.tabId === 'sa-1'),
    'setup: the retag must release the old id',
  );
  assert.ok(
    win._testApi.openTabs().some(t => t.id === 'sa-2') &&
      !win._testApi.openTabs().some(t => t.id === 'sa-1'),
    'setup: the tab must now be sa-2',
  );
}

function testRetagReissuesTheOutstandingCheck() {
  const {win, posted, parentId} = bootSubagent();
  streamAndRetag(win, posted, parentId);
  // The check on the wire under sa-1 is answered under sa-1: the daemon
  // echoes the id the request named.  Renaming the client's key cannot
  // rename that request, so the retag must ask again under sa-2.
  const reissued = checkPathsFor(posted, 'sa-2');
  assert.strictEqual(reissued.length, 1, 'the retag re-asks under sa-2');
  assert.deepStrictEqual(Array.from(reissued[0].paths), [REL]);
  const counts = win._testApi.pendingFileLinkCounts();
  assert.strictEqual(counts.spans, 1, 'one span in flight');
  assert.strictEqual(counts.checks, 1, 'one key (the sa-2 one) in flight');
  const old = checkPathsFor(posted, 'sa-1')[0];
  send(win, {
    type: 'pathsExist',
    tabId: 'sa-1',
    workDir: old.workDir,
    results: {[REL]: true},
  });
  win._testApi.switchToTab('sa-2');
  const O = win.document.getElementById('output');
  assert.ok(
    O.querySelector('[data-path-candidate]'),
    'the retired id resolves nothing',
  );
  assert.strictEqual(
    win._testApi.pendingFileLinkCounts().checks,
    1,
    'the sa-2 key is still in flight',
  );
  send(win, {
    type: 'pathsExist',
    tabId: 'sa-2',
    workDir: reissued[0].workDir,
    results: {[REL]: true},
  });
  const link = O.querySelector('[data-path]');
  assert.ok(link, 'the sa-2 reply promotes the link');
  assert.strictEqual(link.getAttribute('data-path'), REL);
  const after = win._testApi.pendingFileLinkCounts();
  assert.strictEqual(
    after.spans + after.checks,
    0,
    'nothing stays pending, got ' + JSON.stringify(after),
  );
  win.close();
  console.log('  ok - a retag re-asks about the outstanding links');
}

function testWorktreeResultRechecksUnderNewId() {
  const {win, posted, parentId} = bootSubagent();
  streamAndRetag(win, posted, parentId);
  const before = checkPathsFor(posted, 'sa-2').length;
  send(win, {type: 'worktree_result', tabId: 'sa-2', success: true});
  const rechecks = checkPathsFor(posted, 'sa-2');
  assert.strictEqual(
    rechecks.length,
    before + 1,
    'worktree_result for the retagged tab must re-check its links',
  );
  assert.deepStrictEqual(Array.from(rechecks[before].paths), [REL]);
  assert.strictEqual(
    checkPathsFor(posted, 'sa-1').length,
    1,
    'no check may still name the retired id',
  );
  win.close();
  console.log('  ok - worktree_result re-checks the links under the new id');
}

function testReconnectReissuesUnderNewId() {
  const {win, posted, parentId} = bootSubagent();
  streamAndRetag(win, posted, parentId);
  const before = checkPathsFor(posted, 'sa-2').length;
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  assert.strictEqual(
    checkPathsFor(posted, 'sa-1').length,
    1,
    'the reissue must not ask under the retired id',
  );
  const reissued = checkPathsFor(posted, 'sa-2');
  assert.strictEqual(reissued.length, before + 1, 'the reissue must name sa-2');
  assert.deepStrictEqual(Array.from(reissued[before].paths), [REL]);
  win.close();
  console.log('  ok - a reconnect re-asks under the new id');
}

function testCloseForgetsTheSpans() {
  const {win, posted, parentId} = bootSubagent();
  streamAndRetag(win, posted, parentId);
  assert.strictEqual(win._testApi.pendingFileLinkCounts().spans, 1);
  clickClose(win, 'sa-2');
  const counts = win._testApi.pendingFileLinkCounts();
  assert.strictEqual(
    counts.spans + counts.checks,
    0,
    'closing the retagged tab must release its spans and keys, got ' +
      JSON.stringify(counts),
  );
  win.close();
  console.log('  ok - closing the retagged tab releases its bookkeeping');
}

function nextFrames(win, n) {
  return new Promise(resolve => {
    let left = n;
    function step() {
      if (--left <= 0) return resolve();
      win.requestAnimationFrame(step);
    }
    win.requestAnimationFrame(step);
  });
}

async function testPendingSweepFollowsActiveTabRetag() {
  const {win, parentId} = bootSubagent();
  win._testApi.switchToTab('sa-1');
  assert.strictEqual(win._testApi.getActiveTabId(), 'sa-1');
  const O = win.document.getElementById('output');
  const stamp = ev => Object.assign({tabId: 'sa-1', taskId: 't1'}, ev);
  send(win, stamp({type: 'tool_call', name: 'Bash', command: 'make zero'}));
  send(win, stamp({type: 'tool_call', name: 'Bash', command: 'make one'}));
  send(win, stamp({type: 'system_output', text: 'one\n'}));
  send(win, stamp({type: 'tool_call', name: 'Bash', command: 'make two'}));
  const panels = O.querySelectorAll('.ev.tc');
  assert.strictEqual(panels.length, 3, 'setup: three tool panels');
  const panel1 = panels[0];
  assert.ok(panel1.classList.contains('collapsed'), 'setup: folded');
  panel1.classList.remove('collapsed');
  // A buffered chunk defers the older-panel collapse to the frame...
  send(win, stamp({type: 'system_output', text: 'two\n'}));
  assert.ok(!panel1.classList.contains('collapsed'), 'setup: deferred');
  // ...and the rename of the tab on screen lands before that frame.
  announce(win, parentId, 'sa-2');
  assert.strictEqual(win._testApi.getActiveTabId(), 'sa-2');
  await nextFrames(win, 3);
  assert.ok(
    panel1.classList.contains('collapsed'),
    'the sweep deferred for the retagged tab must still run',
  );
  win.close();
  console.log('  ok - a pending tail sweep follows the active tab retag');
}

async function main() {
  testRetagReissuesTheOutstandingCheck();
  testWorktreeResultRechecksUnderNewId();
  testReconnectReissuesUnderNewId();
  testCloseForgetsTheSpans();
  await testPendingSweepFollowsActiveTabRetag();
  console.log('retagSubagentFileLinks: all passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
