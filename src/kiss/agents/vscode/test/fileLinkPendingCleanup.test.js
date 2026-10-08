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
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const WORK_DIR = '/tmp/kiss-link-cleanup';

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

  send(win, {type: 'configData', config: {work_dir: WORK_DIR}, apiKeys: {}});
  win._testApi.endLaunch();
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function isOpen(win, tabId) {
  return win._testApi.openTabs().some(t => t.id === tabId);
}

// There is no row of chat tabs: the user picks a chat in the Chats
// panel, which is what switchToTab stands in for.
function clickTab(win, tabId) {
  assert.ok(isOpen(win, tabId), `tab ${tabId} must be open`);
  win._testApi.switchToTab(tabId);
}

// Without a row of chat tabs there is no close button for a chat; an
// idle chat is closed by leaving it. Pressing "+" opens a fresh chat and
// retires the one on screen (no task running, no question, no draft).
function leaveAndRetire(win, tabId) {
  assert.strictEqual(win._testApi.getActiveTabId(), tabId);
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

// The probe's object is built in the JSDOM realm; copy it into this one
// so deepStrictEqual compares values rather than prototypes.
function counts(win) {
  const c = win._testApi.pendingFileLinkCounts();
  return {spans: c.spans, checks: c.checks};
}

function checkPathCommands(posted) {
  return posted
    .filter(m => m && m.type === 'checkPaths')
    .map(m => ({tabId: m.tabId, paths: Array.from(m.paths || [])}));
}

// A dropped connection is the one case where the reply can never come.
// Until it was handled, the in-flight key survived the outage and every
// later panel mentioning that path sent nothing at all.
function testLostReplyDoesNotWedgeThePathForEver() {
  const {win, posted} = makeWebview();
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

  // The daemon dies before answering, then comes back. The key of a
  // check nothing can answer must not survive the outage, and the spans
  // it was for must be asked about again -- see
  // fileLinkReissueOnReconnect.test.js for the reissue itself.
  send(win, {type: 'daemonStatus', connected: false});
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: true});
  assert.deepStrictEqual(
    checkPathCommands(posted).map(c => c.paths),
    [['./src/foo.py']],
    'a reply that can never arrive must not suppress the check for the ' +
      'rest of the session — the path would stay inert for ever',
  );

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
  const {win} = makeWebview();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();

  // Twenty panels in the tab, none of them ever answered.
  for (let i = 0; i < 20; i += 1) {
    sendToolResult(win, tabB, `touched ./src/gen${i}.py`);
  }
  const before = counts(win);
  assert.strictEqual(before.spans, 20, 'twenty spans must be waiting');
  assert.strictEqual(before.checks, 20, 'twenty checks must be in flight');

  // "+" moves the user to a fresh chat (the transcript of B detaches
  // into its fragment) and closes the idle chat B behind them.
  leaveAndRetire(win, tabB);
  const after = counts(win);
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
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  clickTab(win, tabA);

  sendToolResult(win, tabA, 'visible ./src/a.py');
  sendToolResult(win, tabB, 'hidden ./src/b.py');
  assert.strictEqual(
    counts(win).spans,
    2,
    'both tabs must have a span waiting',
  );

  // A new task starts in the hidden tab: its transcript is discarded.
  send(win, {type: 'clear', tabId: tabB});
  assert.deepStrictEqual(
    counts(win),
    {spans: 1, checks: 1},
    "clearing a background tab must release exactly that tab's spans",
  );

  // ...and now in the visible one.
  send(win, {type: 'clear', tabId: tabA});
  assert.deepStrictEqual(
    counts(win),
    {spans: 0, checks: 0},
    'clearing the visible transcript must release its spans too',
  );
  win.close();
  console.log('  ok - clearing a transcript releases its pending file links');
}

function testShowWelcomeReleasesPendingSpans() {
  const {win} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  clickTab(win, tabA);

  sendToolResult(win, tabA, 'visible ./src/a.py');
  sendToolResult(win, tabB, 'hidden ./src/b.py');

  send(win, {type: 'showWelcome', tabId: tabB});
  assert.deepStrictEqual(
    counts(win),
    {spans: 1, checks: 1},
    'resetting a background tab to the welcome screen must release its ' +
      'spans',
  );

  send(win, {type: 'showWelcome', tabId: tabA});
  assert.deepStrictEqual(
    counts(win),
    {spans: 0, checks: 0},
    'resetting the visible tab to the welcome screen must release its ' +
      'spans',
  );
  win.close();
  console.log('  ok - showWelcome releases pending file links');
}

// Control: an ordinary answered round trip must still empty both sets,
// and must not be disturbed by the new cleanup.
function testAnsweredCheckStillEmptiesBothSets() {
  const {win} = makeWebview();
  const tabId = win._testApi.getActiveTabId();
  sendToolResult(win, tabId, 'wrote ./src/ok.py and ./src/gone.py');
  assert.strictEqual(counts(win).spans, 2);

  send(win, {
    type: 'pathsExist',
    tabId,
    workDir: WORK_DIR,
    results: {'./src/ok.py': true, './src/gone.py': false},
  });
  assert.deepStrictEqual(
    counts(win),
    {spans: 0, checks: 0},
    'an answered check must leave nothing behind',
  );
  assert.ok(
    win.document.querySelector('#output [data-path="./src/ok.py"]'),
    'an existing file must become a link',
  );
  assert.ok(
    win.document.querySelector('#output [data-path-missing]'),
    'a missing file must be greyed out',
  );
  win.close();
  console.log('  ok - an answered round trip still empties both sets');
}

function main() {
  testLostReplyDoesNotWedgeThePathForEver();
  testClosingATabReleasesItsPendingSpans();
  testClearingATranscriptReleasesItsPendingSpans();
  testShowWelcomeReleasesPendingSpans();
  testAnsweredCheckStillEmptiesBothSets();
  console.log('fileLinkPendingCleanup.test.js: all tests passed');
}

main();
