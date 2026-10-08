// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E test (webview harness): a filesystem-root work dir must never be
// adopted by a tab or stamped on a tab-scoped command.
//
// A pre-guard daemon (or the task history it persisted) can hand the
// webview `work_dir: '/'` — via configData or a task_events replay —
// which used to root drag-and-drop resolution and every daemon-bound
// command at the whole disk.  `workDirForTab()` and the two replay
// adoption sites ignore roots (isRootDir in media/main.js).
//
// workDirForTab() resolves the GLOBAL working directory (configData
// work_dir) first and a tab's own folder (learned from a replay) only
// while there is no usable global one, so the replay guard is observed
// with an empty global value, and each case also checks that a real
// global value wins over whatever the replay recorded.

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

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function drop(win, uriList) {
  const container = win.document.getElementById('input-container');
  assert.ok(container, '#input-container must exist');
  const ev = new win.Event('drop', {bubbles: true, cancelable: true});
  ev.dataTransfer = {
    getData: type => (type === 'text/uri-list' ? uriList : ''),
    files: [],
  };
  container.dispatchEvent(ev);
}

function lastMsg(posted, type) {
  for (let i = posted.length - 1; i >= 0; i -= 1) {
    if (posted[i] && posted[i].type === type) return posted[i];
  }
  return null;
}

// The work dir the active tab stamps on a drop's resolveDroppedPaths.
function droppedWorkDir(win, posted, uriList) {
  const before = posted.length;
  drop(win, uriList);
  const cmd = lastMsg(posted.slice(before), 'resolveDroppedPaths');
  assert.ok(cmd, 'drop must send resolveDroppedPaths');
  return cmd.workDir;
}

function setConfigWorkDir(win, workDir) {
  send(win, {type: 'configData', config: {work_dir: workDir}});
}

function replay(win, workDir, tabFields) {
  send(win, {
    type: 'task_events',
    ...(tabFields || {}),
    task: 'replayed task',
    events: [],
    extra: JSON.stringify({work_dir: workDir}),
  });
}

// Chat tabs have no row of their own: the Chats panel's pick is
// switchToTab.
function activateTab(win, tabId) {
  assert.ok(
    win._testApi.openTabs().some(t => t.id === tabId),
    `the ${tabId} tab must be open`,
  );
  win._testApi.switchToTab(tabId);
  assert.strictEqual(win._testApi.getActiveTabId(), tabId);
}

const BG_TABS = [
  {tabId: 'tab-a', chatId: 'chat-a', title: 'a', workDir: ''},
  {tabId: 'tab-bg', chatId: 'chat-bg', title: 'bg', workDir: ''},
];
const BG_FIELDS = {tabId: 'tab-bg', chat_id: 'chat-bg'};

// A root global work dir (a pre-guard daemon whose fallback had
// degenerated to '/') must not be stamped on tab-scoped commands ...
function testRootConfigWorkDirNotStamped() {
  const {win, posted} = makeWebview();
  setConfigWorkDir(win, '/');
  assert.strictEqual(
    droppedWorkDir(win, posted, 'file:///x/y/src/a.ts\n'),
    '',
    'a root global work dir must be treated as no work dir',
  );
  win.close();
  console.log('ok - root config work dir is not stamped on commands');
}

// ... and, being treated as none, it leaves the tab's own folder as the
// fallback instead of hiding it.
function testRootConfigWorkDirFallsThroughToTabFolder() {
  const {win, posted} = makeWebview();
  setConfigWorkDir(win, '/');
  replay(win, '/proj/repo');
  assert.strictEqual(
    droppedWorkDir(win, posted, 'file:///proj/repo/src/a.ts\n'),
    '/proj/repo',
    'a root global work dir must fall through to the tab folder',
  );
  win.close();
  console.log('ok - root config work dir falls through to the tab folder');
}

// A replayed task that truthfully recorded work_dir '/' (run before
// the daemon guard existed) must not poison the live tab: the folder
// the tab already learned survives (workDirForTab() would hide a stored
// root on its own, so a surviving real folder is what proves the replay
// never stored it), and a real global one is untouched by the replay.
function testReplayRootWorkDirNotAdopted() {
  const {win, posted} = makeWebview();
  setConfigWorkDir(win, '');
  replay(win, '/proj/repo');
  replay(win, '/');
  assert.strictEqual(
    droppedWorkDir(win, posted, 'file:///proj/repo/src/a.ts\n'),
    '/proj/repo',
    'a replayed root work_dir must not displace the tab folder',
  );
  setConfigWorkDir(win, '/x/y');
  assert.strictEqual(
    droppedWorkDir(win, posted, 'file:///x/y/src/a.ts\n'),
    '/x/y',
    'the global work dir is stamped once the daemon reports one',
  );
  win.close();
  console.log('ok - replayed root work_dir is not adopted by the tab');
}

// Control: a real replayed work_dir is still learned as the tab's own
// folder (the fallback while no global directory exists), and the
// global directory wins over it as soon as the daemon reports one.
function testReplayRealWorkDirIsTheFallbackOnly() {
  const {win, posted} = makeWebview();
  setConfigWorkDir(win, '');
  replay(win, '/proj/repo');
  assert.strictEqual(
    droppedWorkDir(win, posted, 'file:///proj/repo/src/a.ts\n'),
    '/proj/repo',
    'a real replayed work_dir is the fallback while no global one exists',
  );
  setConfigWorkDir(win, '/x/y');
  assert.strictEqual(
    droppedWorkDir(win, posted, 'file:///x/y/src/a.ts\n'),
    '/x/y',
    'the global work dir wins over the replayed tab folder',
  );
  win.close();
  console.log('ok - real replayed work_dir is only the fallback');
}

// The same replay poisoning through the BACKGROUND-tab branch of
// `task_events` (a hidden tab's transcript restored behind the active
// one): the root must not stick to the hidden tab either.
function testBgReplayRootWorkDirNotAdopted() {
  const {win, posted} = makeWebview();
  setConfigWorkDir(win, '');
  send(win, {type: 'tabs_state', tabs: BG_TABS});
  replay(win, '/proj/bg', BG_FIELDS);
  replay(win, '/', BG_FIELDS);
  activateTab(win, 'tab-bg');
  assert.strictEqual(
    droppedWorkDir(win, posted, 'file:///proj/bg/src/a.ts\n'),
    '/proj/bg',
    'a bg-replayed root work_dir must not displace the hidden tab folder',
  );
  setConfigWorkDir(win, '/x/y');
  assert.strictEqual(
    droppedWorkDir(win, posted, 'file:///x/y/src/a.ts\n'),
    '/x/y',
    'the global work dir is stamped once the daemon reports one',
  );
  win.close();
  console.log('ok - bg-replayed root work_dir is not adopted by the tab');
}

// Control for the background branch: a real replayed work_dir is still
// learned as the hidden tab's folder, and the global directory wins.
function testBgReplayRealWorkDirIsTheFallbackOnly() {
  const {win, posted} = makeWebview();
  setConfigWorkDir(win, '');
  send(win, {type: 'tabs_state', tabs: BG_TABS});
  replay(win, '/proj/bg', BG_FIELDS);
  activateTab(win, 'tab-bg');
  assert.strictEqual(
    droppedWorkDir(win, posted, 'file:///proj/bg/src/a.ts\n'),
    '/proj/bg',
    'a real bg-replayed work_dir is the fallback while no global one exists',
  );
  setConfigWorkDir(win, '/x/y');
  assert.strictEqual(
    droppedWorkDir(win, posted, 'file:///x/y/src/a.ts\n'),
    '/x/y',
    'the global work dir wins over the bg-replayed tab folder',
  );
  win.close();
  console.log('ok - real bg-replayed work_dir is only the fallback');
}

testRootConfigWorkDirNotStamped();
testRootConfigWorkDirFallsThroughToTabFolder();
testReplayRootWorkDirNotAdopted();
testReplayRealWorkDirIsTheFallbackOnly();
testBgReplayRootWorkDirNotAdopted();
testBgReplayRealWorkDirIsTheFallbackOnly();
console.log('all rootWorkDirTabGuard tests passed');
