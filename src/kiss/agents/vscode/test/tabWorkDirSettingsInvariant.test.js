// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// E2E (JSDOM) tests for the INVARIANT behind workDirForTab():
//
//   The global working directory (configData.config.work_dir, the one
//   value every task on every surface runs in) wins for every tab; a
//   tab's own workDir (the folder its last task ran in, learned from a
//   task_events replay or the registry) is only the fallback while the
//   daemon has reported no global directory, or reported a filesystem
//   root, which is never a working directory.
//
// This file used to assert the opposite ("a tab keeps the directory it
// was bound under when the configured work_dir changes"); that
// per-tab contract no longer exists.

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

function lastMsg(posted, type) {
  for (let i = posted.length - 1; i >= 0; i -= 1) {
    if (posted[i] && posted[i].type === type) return posted[i];
  }
  return null;
}

function initialTabId(posted) {
  const ready = posted.find(m => m && m.type === 'ready');
  return ready ? ready.tabId : '';
}

function sendConfig(win, workDir) {
  send(win, {type: 'configData', config: {work_dir: workDir}, apiKeys: {}});
}

// The directory the active tab's daemon-bound commands carry: typing an
// @-mention posts getFiles stamped with workDirForTab(activeTabId).
function mentionWorkDir(win, posted) {
  const before = posted.length;
  const inp = win.document.getElementById('task-input');
  inp.value = '@readm';
  inp.dispatchEvent(new win.window.Event('input', {bubbles: true}));
  const msg = lastMsg(posted.slice(before), 'getFiles');
  assert.ok(msg, 'typing an @-mention must post a getFiles command');
  inp.value = '';
  inp.dispatchEvent(new win.window.Event('input', {bubbles: true}));
  return msg.workDir;
}

// Bind the initial tab to a real chat the way the daemon does (a "clear"
// or a task_events replay), list it in the registry as a tab with no
// folder of its own, then move the global directory.
function bindTab(win, tabId, bindKind, extra) {
  if (bindKind === 'clear') {
    send(win, {type: 'clear', tabId: tabId, chat_id: 'chat-real-task'});
  } else if (bindKind === 'task_events') {
    send(win, {
      type: 'task_events',
      tabId: tabId,
      chat_id: 'chat-real-task',
      task_id: 42,
      task: 'Real persisted task',
      events: [],
      extra: JSON.stringify(extra || {startTs: 1_700_000_000_000}),
    });
  } else {
    throw new Error('unknown bindKind: ' + bindKind);
  }
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: tabId, chatId: 'chat-real-task', title: 'b', workDir: ''}],
  });
}

function workDirOnMentionAfterGlobalChange(initialWd, newWd, bindKind) {
  const {win, posted} = makeWebview();
  const tabId = initialTabId(posted);
  assert.ok(tabId, 'main.js must announce the initial tab id');
  sendConfig(win, initialWd);
  bindTab(win, tabId, bindKind);
  assert.strictEqual(
    mentionWorkDir(win, posted),
    initialWd,
    'before the change the tab runs in the global directory',
  );
  sendConfig(win, newWd);
  const wd = mentionWorkDir(win, posted);
  win.close();
  return wd;
}

function testGlobalWinsAfterChange_ClearBind() {
  const wd = workDirOnMentionAfterGlobalChange(
    '/path/initial',
    '/path/new',
    'clear',
  );
  assert.strictEqual(
    wd,
    '/path/new',
    'INVARIANT: when the global work_dir changes, a tab bound via a ' +
      '"clear" event MUST route its commands to the NEW global directory ' +
      '-- observed workDir = ' +
      JSON.stringify(wd),
  );
  console.log('  ok - clear-bound tab follows the global work_dir change');
}

function testGlobalWinsAfterChange_TaskEventsBind() {
  const wd = workDirOnMentionAfterGlobalChange(
    '/path/initial',
    '/path/new',
    'task_events',
  );
  assert.strictEqual(
    wd,
    '/path/new',
    'INVARIANT: when the global work_dir changes, a tab bound via a ' +
      '"task_events" replay (whose persisted "extra" carries no work_dir) ' +
      'MUST route its commands to the NEW global directory -- observed ' +
      'workDir = ' +
      JSON.stringify(wd),
  );
  console.log(
    '  ok - task_events-bound tab follows the global work_dir change',
  );
}

// A replayed task that recorded its own work_dir gives the tab a folder
// of its own.  The global directory still wins over it.
function testGlobalWinsOverTaskRecordedWorkDir() {
  const {win, posted} = makeWebview();
  const tabId = initialTabId(posted);
  sendConfig(win, '/path/initial');
  bindTab(win, tabId, 'task_events', {work_dir: '/path/task-recorded'});
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '/path/initial',
    'the tab-recorded folder never displaces the global directory',
  );
  sendConfig(win, '/path/new');
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '/path/new',
    'and the tab follows every later change of the global directory',
  );
  send(win, {type: 'workDirChanged', workDir: '/path/newer'});
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '/path/newer',
    "including the daemon's workDirChanged broadcast of a pick made elsewhere",
  );
  win.close();
  console.log(
    '  ok - the global directory wins over a task-recorded tab folder',
  );
}

// The tab's own folder is used only until the daemon reports a global
// directory.
function testTabFolderIsOnlyTheFallback() {
  const {win, posted} = makeWebview();
  const tabId = initialTabId(posted);
  sendConfig(win, '');
  bindTab(win, tabId, 'task_events', {work_dir: '/path/task-recorded'});
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '/path/task-recorded',
    'with no global directory yet, the folder the task ran in is the fallback',
  );
  sendConfig(win, '/path/global');
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '/path/global',
    'once the daemon reports the global directory, it wins',
  );
  win.close();
  console.log(
    '  ok - a tab folder is only the fallback before a global one exists',
  );
}

// Root-dir guard: a filesystem root is never a working directory, on
// either side of the precedence.
function testRootDirsAreSkippedOnBothSides() {
  const {win, posted} = makeWebview();
  const tabId = initialTabId(posted);
  // A root global value (a pre-guard daemon) falls through to the tab's
  // own folder ...
  sendConfig(win, '/');
  bindTab(win, tabId, 'task_events', {work_dir: '/path/task-recorded'});
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '/path/task-recorded',
    'a root global work_dir must be treated as none, leaving the tab folder',
  );
  // ... and a root tab folder (an old poisoned task row) is not adopted,
  // so with no usable directory at all the command carries none.
  send(win, {
    type: 'task_events',
    tabId: tabId,
    chat_id: 'chat-real-task',
    task_id: 43,
    task: 'old poisoned task',
    events: [],
    extra: JSON.stringify({work_dir: '/'}),
  });
  assert.strictEqual(
    mentionWorkDir(win, posted),
    '/path/task-recorded',
    'a replayed root work_dir must not displace the tab folder',
  );
  win.close();
  console.log(
    '  ok - filesystem roots are skipped as global and as tab folder',
  );
}

function main() {
  testGlobalWinsAfterChange_ClearBind();
  testGlobalWinsAfterChange_TaskEventsBind();
  testGlobalWinsOverTaskRecordedWorkDir();
  testTabFolderIsOnlyTheFallback();
  testRootDirsAreSkippedOnBothSides();
  console.log('tabWorkDirSettingsInvariant.test.js: all assertions passed.');
}

main();
