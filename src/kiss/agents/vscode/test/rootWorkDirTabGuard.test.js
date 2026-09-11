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
// adoption sites now ignore roots (isRootDir in media/main.js).

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

// A root config work dir (a pre-guard daemon whose fallback had
// degenerated to '/') must not be stamped on tab-scoped commands.
function testRootConfigWorkDirNotStamped() {
  const {win, posted} = makeWebview();
  send(win, {type: 'configData', config: {work_dir: '/'}});
  drop(win, 'file:///x/y/src/a.ts\n');
  const cmd = lastMsg(posted, 'resolveDroppedPaths');
  assert.ok(cmd, 'drop must send resolveDroppedPaths');
  assert.strictEqual(
    cmd.workDir,
    '',
    'a root config work dir must be treated as no work dir',
  );
  win.close();
  console.log('ok - root config work dir is not stamped on commands');
}

// A replayed task that truthfully recorded work_dir '/' (run before
// the daemon guard existed) must not re-poison the live tab.
function testReplayRootWorkDirNotAdopted() {
  const {win, posted} = makeWebview();
  send(win, {type: 'configData', config: {work_dir: '/x/y'}});
  send(win, {
    type: 'task_events',
    task: 'old poisoned task',
    events: [],
    extra: JSON.stringify({work_dir: '/'}),
  });
  drop(win, 'file:///x/y/src/a.ts\n');
  const cmd = lastMsg(posted, 'resolveDroppedPaths');
  assert.ok(cmd, 'drop must send resolveDroppedPaths');
  assert.strictEqual(
    cmd.workDir,
    '/x/y',
    'a replayed root work_dir must not displace the real fallback',
  );
  win.close();
  console.log('ok - replayed root work_dir is not adopted by the tab');
}

// Control: a real replayed work_dir must still repin the tab (the
// normal heal path for tabs that predate per-tab pinning).
function testReplayRealWorkDirStillAdopted() {
  const {win, posted} = makeWebview();
  send(win, {type: 'configData', config: {work_dir: '/x/y'}});
  send(win, {
    type: 'task_events',
    task: 'healthy task',
    events: [],
    extra: JSON.stringify({work_dir: '/proj/repo'}),
  });
  drop(win, 'file:///proj/repo/src/a.ts\n');
  const cmd = lastMsg(posted, 'resolveDroppedPaths');
  assert.ok(cmd, 'drop must send resolveDroppedPaths');
  assert.strictEqual(
    cmd.workDir,
    '/proj/repo',
    'a real replayed work_dir must still repin the tab',
  );
  win.close();
  console.log('ok - real replayed work_dir still repins the tab');
}

// The same replay poisoning through the BACKGROUND-tab branch of
// `task_events` (a hidden tab's transcript restored behind the active
// one): the root must not stick to the hidden tab either.
function testBgReplayRootWorkDirNotAdopted() {
  const {win, posted} = makeWebview();
  send(win, {type: 'configData', config: {work_dir: '/x/y'}});
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 'tab-a', chatId: 'chat-a', title: 'a', workDir: ''},
      {tabId: 'tab-bg', chatId: 'chat-bg', title: 'bg', workDir: ''},
    ],
  });
  send(win, {
    type: 'task_events',
    tabId: 'tab-bg',
    chat_id: 'chat-bg',
    task: 'old poisoned bg task',
    events: [],
    extra: JSON.stringify({work_dir: '/'}),
  });
  const bgEl = win.document.querySelector(
    '.chat-tab[data-tab-id="tab-bg"]',
  );
  assert.ok(bgEl, 'the background tab must be rendered');
  bgEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  drop(win, 'file:///x/y/src/a.ts\n');
  const cmd = lastMsg(posted, 'resolveDroppedPaths');
  assert.ok(cmd, 'drop must send resolveDroppedPaths');
  assert.strictEqual(
    cmd.workDir,
    '/x/y',
    'a bg-replayed root work_dir must not displace the real fallback',
  );
  win.close();
  console.log('ok - bg-replayed root work_dir is not adopted by the tab');
}

// Control for the background branch: a real replayed work_dir still
// repins the hidden tab.
function testBgReplayRealWorkDirStillAdopted() {
  const {win, posted} = makeWebview();
  send(win, {type: 'configData', config: {work_dir: '/x/y'}});
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 'tab-a', chatId: 'chat-a', title: 'a', workDir: ''},
      {tabId: 'tab-bg', chatId: 'chat-bg', title: 'bg', workDir: ''},
    ],
  });
  send(win, {
    type: 'task_events',
    tabId: 'tab-bg',
    chat_id: 'chat-bg',
    task: 'healthy bg task',
    events: [],
    extra: JSON.stringify({work_dir: '/proj/bg'}),
  });
  const bgEl = win.document.querySelector(
    '.chat-tab[data-tab-id="tab-bg"]',
  );
  assert.ok(bgEl, 'the background tab must be rendered');
  bgEl.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  drop(win, 'file:///proj/bg/src/a.ts\n');
  const cmd = lastMsg(posted, 'resolveDroppedPaths');
  assert.ok(cmd, 'drop must send resolveDroppedPaths');
  assert.strictEqual(
    cmd.workDir,
    '/proj/bg',
    'a real bg-replayed work_dir must still repin the tab',
  );
  win.close();
  console.log('ok - real bg-replayed work_dir still repins the tab');
}

testRootConfigWorkDirNotStamped();
testReplayRootWorkDirNotAdopted();
testReplayRealWorkDirStillAdopted();
testBgReplayRootWorkDirNotAdopted();
testBgReplayRealWorkDirStillAdopted();
console.log('all rootWorkDirTabGuard tests passed');
