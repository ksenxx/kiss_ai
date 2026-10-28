// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) test for a hidden tab's `task_events` replay
// (media/main.js): the replay REPLACES the tab's transcript, so the
// file-link bookkeeping of the discarded one (candidate path spans
// waiting for the daemon's answer, in-flight checkPaths keys) must be
// released with it, exactly as the visible tab's replay does through
// clearOutput() -> forgetPendingFileLinks().  Spans of a detached
// fragment are unreachable DOM that every later reply would still walk.
//
// The release is transactional: a replay that throws half way hands the
// tab its old transcript back, and that transcript's spans and in-flight
// checks come back with it (the daemon's later answer must still make
// its links clickable), while whatever the failed replay registered for
// the discarded new fragment goes.

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');
const {
  makeWorkDirPage,
  pendingCounts,
  checkPathCommands,
  findLinks,
} = require('./file_link_helpers');

const {send} = harness;
const WORK_DIR = '/tmp/kiss-link-bg-replay';

function sendToolResult(win, tabId, content) {
  send(win, {type: 'tool_result', content, tabId, workDir: WORK_DIR});
}

/** A page with two chat tabs: A on screen, B hidden. */
function bootTwoTabs() {
  const {win, posted} = makeWorkDirPage(WORK_DIR);
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  win._testApi.switchToTab(tabA);
  assert.strictEqual(win._testApi.getActiveTabId(), tabA, 'A is on screen');
  return {win, posted, tabA, tabB};
}

function testFailedReplayRestoresTheOldTranscriptAndItsLinks() {
  const {win, posted, tabA, tabB} = bootTwoTabs();
  sendToolResult(win, tabB, 'touched ./src/old.py');
  assert.deepStrictEqual(pendingCounts(win), {spans: 1, checks: 1});

  // A malformed transcript: the first replayed event is fine and
  // registers its own path, the second one blows the replay up.  The
  // throw is reported on the window (jsdom's listener error path).
  const errors = [];
  win.addEventListener('error', e => {
    errors.push(e.error || e.message);
    e.preventDefault();
  });
  posted.length = 0;
  send(win, {
    type: 'task_events',
    tabId: tabB,
    task: 'broken replay',
    task_id: 8,
    events: [
      {
        type: 'tool_result',
        content: 'wrote ./src/replayed.py',
        workDir: WORK_DIR,
      },
      null,
    ],
  });
  assert.strictEqual(errors.length, 1, 'the broken replay threw');
  assert.deepStrictEqual(
    checkPathCommands(posted).map(c => c.paths),
    [['./src/replayed.py']],
    'the failed replay had asked about its own path before it died',
  );
  assert.strictEqual(win._testApi.getActiveTabId(), tabA, 'A stays on screen');
  assert.deepStrictEqual(
    pendingCounts(win),
    {spans: 1, checks: 1},
    'the restored transcript keeps its span and in-flight check; the ' +
      "failed replay's registrations are gone",
  );

  // The daemon's answer to the OLD transcript's check lands, and the
  // restored transcript's link becomes clickable.
  send(win, {
    type: 'pathsExist',
    tabId: tabB,
    workDir: WORK_DIR,
    results: {'./src/old.py': true},
  });
  assert.deepStrictEqual(pendingCounts(win), {spans: 0, checks: 0});
  win._testApi.switchToTab(tabB);
  assert.strictEqual(
    findLinks(win, './src/old.py').length,
    1,
    'BUG: the old transcript came back without its pending link',
  );
  assert.strictEqual(
    findLinks(win, './src/replayed.py').length,
    0,
    'nothing of the failed replay is shown',
  );

  // A reply meant for the discarded replay resolves nothing and leaves
  // no bookkeeping behind.
  send(win, {
    type: 'pathsExist',
    tabId: tabB,
    workDir: WORK_DIR,
    results: {'./src/replayed.py': true},
  });
  assert.deepStrictEqual(pendingCounts(win), {spans: 0, checks: 0});
  win.close();
}

function testHiddenReplayReleasesTheOldTranscriptsLinks() {
  const {win, posted, tabA, tabB} = bootTwoTabs();

  // Ten panels in hidden B, none of them ever answered.
  for (let i = 0; i < 10; i += 1) {
    sendToolResult(win, tabB, `touched ./src/gen${i}.py`);
  }
  const before = pendingCounts(win);
  assert.strictEqual(before.spans, 10, 'ten spans of B must be waiting');
  assert.strictEqual(before.checks, 10, 'ten checks of B must be in flight');

  // The daemon replays B's transcript (a reattach): two panels, one path.
  posted.length = 0;
  send(win, {
    type: 'task_events',
    tabId: tabB,
    task: 'replayed task',
    task_id: 7,
    events: [
      {
        type: 'tool_result',
        content: 'wrote ./src/replayed.py',
        workDir: WORK_DIR,
      },
      {type: 'text_delta', text: 'done'},
      {type: 'text_end'},
    ],
  });
  assert.strictEqual(win._testApi.getActiveTabId(), tabA, 'A stays on screen');
  const after = pendingCounts(win);
  assert.strictEqual(
    after.spans,
    1,
    'the replaced transcript released its spans; only the replayed one waits',
  );
  assert.strictEqual(
    after.checks,
    1,
    'the replaced transcript released its in-flight checks; only the replayed one is pending',
  );
  assert.deepStrictEqual(
    checkPathCommands(posted).map(c => c.paths),
    [['./src/replayed.py']],
    'the replayed panel asked about its own path',
  );

  // The daemon's answer settles the replayed check; nothing else waits.
  send(win, {
    type: 'pathsExist',
    tabId: tabB,
    workDir: WORK_DIR,
    results: {'./src/replayed.py': true},
  });
  assert.deepStrictEqual(
    pendingCounts(win),
    {spans: 0, checks: 0},
    'the replayed check is answered and nothing of the old transcript lingers',
  );
  win.close();
}

testHiddenReplayReleasesTheOldTranscriptsLinks();
testFailedReplayRestoresTheOldTranscriptAndItsLinks();
console.log('bgReplayFileLinks_sweep1026.test.js: all tests passed');
