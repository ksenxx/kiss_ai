// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) test for the file-link checks that were on the wire
// when the daemon went away.
//
// A path found in a transcript starts as an inert [data-path-candidate]
// span and only becomes clickable once the host answers the `checkPaths`
// it triggered. When the socket dies mid-round-trip that answer never
// comes, so the webview drops the in-flight key -- otherwise the dedup
// set would suppress every later check of the same path.
//
// Dropping the key is only half of it. The SPANS are deliberately kept:
// they are still on screen, still grey, still unclickable. Nothing ever
// re-asked about them, so unless a LATER panel happened to mention the
// very same path again, those links stayed dead for the rest of the
// session -- and a finished task renders no further panels, which is
// precisely when the user goes to click them.
//
// The reconnect must reissue the checks for the spans it kept.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const WORK_DIR = '/tmp/kiss-link-reissue';

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

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function sendToolResult(win, tabId, content, workDir) {
  send(win, {
    type: 'tool_result',
    content,
    tabId,
    workDir: workDir === undefined ? WORK_DIR : workDir,
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
    .map(m => ({
      tabId: m.tabId,
      workDir: m.workDir,
      paths: Array.from(m.paths || []).sort(),
    }));
}

function outage(win) {
  send(win, {type: 'daemonStatus', connected: false});
}

function reconnect(win) {
  send(win, {type: 'daemonStatus', connected: true});
}

// The task finished, so no further panel will ever mention the path
// again: the reconnect is the last chance those links have.
function testReconnectReissuesTheLostCheck() {
  const {win, posted} = makeWebview();
  const tabId = win._testApi.getActiveTabId();

  sendToolResult(win, tabId, 'wrote ./src/foo.py and ./src/bar.py');
  assert.deepStrictEqual(
    checkPathCommands(posted).map(c => c.paths),
    [['./src/bar.py', './src/foo.py']],
    'the panel must ask whether its paths exist',
  );

  // The socket dies with the reply still owed, and the run is over.
  outage(win);
  posted.length = 0;
  reconnect(win);

  assert.deepStrictEqual(
    checkPathCommands(posted),
    [
      {
        tabId,
        workDir: WORK_DIR,
        paths: ['./src/bar.py', './src/foo.py'],
      },
    ],
    'reconnecting must re-ask about the spans whose reply was lost; ' +
      'nothing else will ever ask again and the links stay dead',
  );

  send(win, {
    type: 'pathsExist',
    tabId,
    workDir: WORK_DIR,
    results: {'./src/foo.py': true, './src/bar.py': false},
  });
  assert.ok(
    win.document.querySelector('#output [data-path="./src/foo.py"]'),
    'the reissued answer must promote the existing file to a link',
  );
  assert.ok(
    win.document.querySelector('#output [data-path-missing]'),
    'and grey out the missing one',
  );
  assert.deepStrictEqual(
    counts(win),
    {spans: 0, checks: 0},
    'the reissued round trip must leave nothing behind',
  );
  win.close();
  console.log('  ok - a reconnect reissues the checks whose reply was lost');
}

// Spans belong to a tab as well as to the workDir they were checked
// under, and a reply only resolves spans stamped with the same pair.
// The reissue must keep both stamps or it resolves nothing.
function testReissueIsGroupedByTabAndWorkDir() {
  const {win, posted} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  clickTab(win, tabA);

  sendToolResult(win, tabA, 'visible ./src/a.py');
  sendToolResult(win, tabB, 'hidden ./src/b.py');

  outage(win);
  posted.length = 0;
  reconnect(win);

  const sent = checkPathCommands(posted);
  assert.strictEqual(
    sent.length,
    2,
    'two tabs waiting on two different paths must produce two requests, ' +
      `got ${JSON.stringify(sent)}`,
  );
  assert.deepStrictEqual(
    sent.slice().sort((x, y) => (x.paths[0] < y.paths[0] ? -1 : 1)),
    [
      {tabId: tabA, workDir: WORK_DIR, paths: ['./src/a.py']},
      {tabId: tabB, workDir: WORK_DIR, paths: ['./src/b.py']},
    ],
    'each tab must be re-asked under its own id and the workDir its ' +
      'spans were checked in, or the reply cannot match them',
  );

  // The reply carries the asking tab, and resolves only that tab.
  send(win, {
    type: 'pathsExist',
    tabId: tabB,
    workDir: WORK_DIR,
    results: {'./src/b.py': true},
  });
  assert.deepStrictEqual(
    counts(win),
    {spans: 1, checks: 1},
    "answering one tab must not clear the other tab's outstanding span",
  );
  clickTab(win, tabB);
  assert.ok(
    win.document.querySelector('#output [data-path="./src/b.py"]'),
    "a background tab's reissued check must resolve its own spans",
  );
  win.close();
  console.log('  ok - the reissue keeps each span\u2019s tab and workDir');
}

// Nothing outstanding must produce no traffic: every open window
// reconnects at once after a daemon restart, and a needless burst from
// each of them is exactly what the backoff work was about.
function testReconnectWithNothingOutstandingIsSilent() {
  const {win, posted} = makeWebview();
  const tabId = win._testApi.getActiveTabId();

  sendToolResult(win, tabId, 'wrote ./src/answered.py');
  send(win, {
    type: 'pathsExist',
    tabId,
    workDir: WORK_DIR,
    results: {'./src/answered.py': true},
  });
  assert.deepStrictEqual(counts(win), {spans: 0, checks: 0});

  outage(win);
  posted.length = 0;
  reconnect(win);
  assert.deepStrictEqual(
    checkPathCommands(posted),
    [],
    'a reconnect with nothing outstanding must not re-ask anything',
  );
  win.close();
  console.log('  ok - a reconnect with nothing outstanding is silent');
}

// A tab closed during the outage took its spans with it; the reconnect
// must not resurrect them.
function testClosedTabIsNotReissued() {
  const {win, posted} = makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  clickTab(win, tabA);

  sendToolResult(win, tabA, 'kept ./src/kept.py');
  sendToolResult(win, tabB, 'doomed ./src/doomed.py');

  outage(win);
  const closeBtn = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabB)}] .chat-tab-close`,
  );
  assert.ok(closeBtn, 'tab B must have a close button');
  closeBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  posted.length = 0;
  reconnect(win);
  assert.deepStrictEqual(
    checkPathCommands(posted),
    [{tabId: tabA, workDir: WORK_DIR, paths: ['./src/kept.py']}],
    'a tab closed during the outage must not be re-asked about',
  );
  win.close();
  console.log('  ok - a tab closed during the outage is not reissued');
}

function main() {
  testReconnectReissuesTheLostCheck();
  testReissueIsGroupedByTabAndWorkDir();
  testReconnectWithNothingOutstandingIsSilent();
  testClosedTabIsNotReissued();
  console.log('fileLinkReissueOnReconnect.test.js: all tests passed');
}

main();
