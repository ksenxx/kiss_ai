// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the lifetime of the file-link bookkeeping.
//
// A candidate file path is registered in two module-level sets while its
// `checkPaths` -> `pathsExist` round trip is in flight:
//   _pendingFileLinkSpans  the DOM nodes waiting to be promoted
//   _pendingPathChecks     the "tabId\0workDir\0path" triples in flight
//
// Entries used to leave those sets ONLY when a matching reply arrived, so
//   * a closed or cleared transcript kept its detached spans alive for
//     the rest of the session, and every later reply had to walk them;
//   * a reply that could never arrive (the daemon died between request
//     and reply) permanently suppressed re-checks of that triple, so
//     those paths stayed grey forever even though the file existed.

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');
const {
  makeWorkDirPage,
  pendingCounts,
  checkPathCommands,
} = require('./file_link_helpers');

const {send, clickTab} = harness;
const WORK_DIR = '/tmp/kiss-link-cleanup';

function isOpen(win, tabId) {
  return win._testApi.openTabs().some(t => t.id === tabId);
}

// Without a row of chat tabs there is no close button for a chat; an
// idle chat is closed by leaving it. Pressing "+" moves to a fresh chat and
// retires the one on screen (no task running, no question, no draft);
// "+" on a chat that is still new only focuses the composer, so the
// daemon's task-text echo makes the chat "used" first.
function leaveAndRetire(win, tabId) {
  assert.strictEqual(win._testApi.getActiveTabId(), tabId);
  send(win, {type: 'setTaskText', text: 'first task', tabId});
  win.document.getElementById('new-chat-btn').click();
  assert.notStrictEqual(win._testApi.getActiveTabId(), tabId);
  assert.ok(!isOpen(win, tabId), `tab ${tabId} must have been retired`);
}

// A tool_result whose body mentions a file path: the ordinary way a
// candidate span is born.
function sendToolResult(win, tabId, content) {
  send(win, {
    type: 'tool_result',
    content,
    tabId,
    workDir: WORK_DIR,
  });
}

// A dropped connection is the one case where the reply can never come.
// Until it was handled, the in-flight key survived the outage and every
// later panel mentioning that path sent nothing at all.
function testLostReplyDoesNotWedgeThePathForEver() {
  const {win, posted} = makeWorkDirPage(WORK_DIR);
  const tabId = win._testApi.getActiveTabId();

  sendToolResult(win, tabId, 'wrote ./src/foo.py');
  assert.deepStrictEqual(
    checkPathCommands(posted).map(c => c.paths),
    [['./src/foo.py']],
    'the first panel must ask whether the path exists',
  );

  // While the check is in flight a second panel mentions the same path.
  // No second request is needed -- the one reply resolves both spans --
  // so this must stay deduped.
  posted.length = 0;
  sendToolResult(win, tabId, 'read ./src/foo.py again');
  assert.deepStrictEqual(
    checkPathCommands(posted),
    [],
    'a check already in flight must not be sent twice',
  );

  // The daemon dies before answering, then comes back: the reconnect
  // re-asks about the path (fileLinkReissueOnReconnect pins the reissue).
  send(win, {type: 'daemonStatus', connected: false});
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: true});
  assert.strictEqual(checkPathCommands(posted).length, 1, 'precondition: reissued');

  // That reissue is itself in flight now, so a later panel mentioning
  // the same path is deduped against it rather than asking a third time.
  posted.length = 0;
  sendToolResult(win, tabId, 'and ./src/foo.py once more');
  assert.deepStrictEqual(
    checkPathCommands(posted),
    [],
    'the reissued check is in flight, so a later panel must not ask again',
  );

  // ...and the eventual reply still resolves every span of that path,
  // including the ones rendered before the outage.
  send(win, {
    type: 'pathsExist',
    tabId,
    workDir: WORK_DIR,
    results: {'./src/foo.py': true},
  });
  assert.strictEqual(
    win.document.querySelectorAll('#output [data-path="./src/foo.py"]').length,
    3,
    'every span of that path must become a link',
  );
  win.close();
  console.log('  ok - a lost reply does not wedge the path for ever');
}

function testClosingATabReleasesItsPendingSpans() {
  const {win} = makeWorkDirPage(WORK_DIR);
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();

  // Twenty panels in the tab, none of them ever answered.
  for (let i = 0; i < 20; i += 1) {
    sendToolResult(win, tabB, `touched ./src/gen${i}.py`);
  }
  const before = pendingCounts(win);
  assert.strictEqual(before.spans, 20, 'twenty spans must be waiting');
  assert.strictEqual(before.checks, 20, 'twenty checks must be in flight');

  // "+" moves the user to a fresh chat (the transcript of B detaches
  // into its fragment) and closes the idle chat B behind them.
  leaveAndRetire(win, tabB);
  const after = pendingCounts(win);
  assert.strictEqual(
    after.spans,
    0,
    'a closed tab must release its detached spans — they are unreachable ' +
      'DOM that every later reply would still have to walk',
  );
  assert.strictEqual(
    after.checks,
    0,
    'a closed tab must release its in-flight check keys',
  );
  win.close();
  console.log('  ok - closing a tab releases its pending file links');
}

function testClearingATranscriptReleasesItsPendingSpans() {
  const {win} = makeWorkDirPage(WORK_DIR);
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  clickTab(win, tabA);

  sendToolResult(win, tabA, 'visible ./src/a.py');
  sendToolResult(win, tabB, 'hidden ./src/b.py');
  assert.strictEqual(
    pendingCounts(win).spans,
    2,
    'both tabs must have a span waiting',
  );

  // A new task starts in the hidden tab: its transcript is discarded.
  send(win, {type: 'clear', tabId: tabB});
  assert.deepStrictEqual(
    pendingCounts(win),
    {spans: 1, checks: 1},
    "clearing a background tab must release exactly that tab's spans",
  );

  // ...and now in the visible one.
  send(win, {type: 'clear', tabId: tabA});
  assert.deepStrictEqual(
    pendingCounts(win),
    {spans: 0, checks: 0},
    'clearing the visible transcript must release its spans too',
  );
  win.close();
  console.log('  ok - clearing a transcript releases its pending file links');
}

function testShowWelcomeReleasesPendingSpans() {
  const {win} = makeWorkDirPage(WORK_DIR);
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  clickTab(win, tabA);

  sendToolResult(win, tabA, 'visible ./src/a.py');
  sendToolResult(win, tabB, 'hidden ./src/b.py');

  send(win, {type: 'showWelcome', tabId: tabB});
  assert.deepStrictEqual(
    pendingCounts(win),
    {spans: 1, checks: 1},
    'resetting a background tab to the welcome screen must release its ' +
      'spans',
  );

  send(win, {type: 'showWelcome', tabId: tabA});
  assert.deepStrictEqual(
    pendingCounts(win),
    {spans: 0, checks: 0},
    'resetting the visible tab to the welcome screen must release its ' +
      'spans',
  );
  win.close();
  console.log('  ok - showWelcome releases pending file links');
}

// An ordinary answered round trip emptying both sets is the tail of
// fileLinkReissueOnReconnect.testReconnectReissuesTheLostCheck.

function main() {
  testLostReplyDoesNotWedgeThePathForEver();
  testClosingATabReleasesItsPendingSpans();
  testClearingATranscriptReleasesItsPendingSpans();
  testShowWelcomeReleasesPendingSpans();
  console.log('fileLinkPendingCleanup.test.js: all tests passed');
}

main();
