// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// JSDOM end-to-end tests for the page side of "a task starts: hide
// the editor window, show the right sidebar" (media/main.js
// showChatForRun, status handler):
//
//   * remote webapp, desktop split layout: the daemon's `status
//     running:true` for the chat tab on screen folds the content pane
//     away (body.content-pane-folded, the content tabs kept) and the
//     task-info panel docks beside the chat again (no body.meta-hidden);
//   * only a task that just STARTED counts: a repeated running status,
//     a tab catching up on a run already going (the daemon's `attached`
//     status: a reload's replay, a history open), a start in a
//     background tab and a sub-agent's start leave the layout alone;
//     the next task folds the pane again;
//   * the phone layout has no pane to fold and nothing is stored.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const FOLD_KEY = 'kiss-content-pane-folded';

/**
 * A fresh remote page. `desktop` is the (min-width: 900px) media query
 * of the remote page.
 */
function makePage(opts) {
  const {desktop = true} = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace('{{BODY_CLASS_ATTR}}', ' class="remote-chat"');
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
  win.Element.prototype.setPointerCapture = function () {};
  win.Element.prototype.releasePointerCapture = function () {};
  const posted = [];
  let state;
  win.acquireTransport = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.matchMedia = function (query) {
    return {
      matches: query === '(min-width: 900px)' ? desktop : false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=runstart-main.js',
  );
  OPEN_WINDOWS.push(win);
  return {win, posted};
}

const OPEN_WINDOWS = [];

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function hasClass(win, cls) {
  return win.document.body.classList.contains(cls);
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/** Register chat tabs; `task_events` marks each one's status as known. */
function openChats(win, ids) {
  send(win, {
    type: 'tabs_state',
    tabs: ids.map(id => ({
      tabId: id,
      chatId: 'chat-' + id,
      title: id,
      workDir: '/ws',
    })),
  });
  for (const id of ids)
    send(win, {type: 'task_events', tabId: id, task: 'the task', events: []});
}

/** Open a file in the content pane for chat tab `tabId`. */
function openFile(win, tabId) {
  send(win, {
    type: 'fileContent',
    tabId,
    path: '/ws/notes.md',
    name: 'notes.md',
    content: '# Notes',
    version: 'v1',
  });
  const fileTab = win._testApi.openTabs().find(t => t.isContentTab);
  assert.ok(fileTab, 'the file opened as a content tab');
  return fileTab;
}

function start(win, tabId) {
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
}

function end(win, tabId) {
  send(win, {type: 'status', running: false, tabId});
}

function assertPaneShown(win, why) {
  assert.ok(hasClass(win, 'content-pane-open'), why + ': the pane is open');
  assert.ok(!hasClass(win, 'content-pane-folded'), why + ': not folded');
  assert.ok(hasClass(win, 'meta-hidden'), why + ': the panel slid off');
}

function assertPaneFolded(win, why) {
  assert.ok(!hasClass(win, 'content-pane-open'), why + ': the pane is gone');
  assert.ok(hasClass(win, 'content-pane-folded'), why + ': folded');
  assert.ok(!hasClass(win, 'meta-hidden'), why + ': the panel is docked');
}

const TESTS = [];
function test(name, fn) {
  TESTS.push({name, fn});
}

test('a task starting in the chat on screen folds the pane and docks the panel', () => {
  const {win} = makePage();
  openChats(win, ['a1']);
  const fileTab = openFile(win, 'a1');
  assertPaneShown(win, 'file open');
  assert.strictEqual(win._testApi.getActiveTabId(), 'a1');

  start(win, 'a1');
  assertPaneFolded(win, 'task started');
  assert.ok(
    win._testApi.openTabs().some(t => t.id === fileTab.id),
    'the content tab stays open behind the fold',
  );
  assert.strictEqual(
    win.localStorage.getItem(FOLD_KEY),
    '1',
    'the fold is stored',
  );

  // The user brings the pane back while the task runs; the daemon's
  // repeated running status (a viewer attaching, a reconnect) is not
  // a new start and leaves it shown.
  click(win, win.document.getElementById('content-pane-show'));
  assertPaneShown(win, 'unfolded by the user');
  start(win, 'a1');
  assertPaneShown(win, 'repeated running status');

  // The task ends; the next task folds the pane again.
  end(win, 'a1');
  assertPaneShown(win, 'task ended');
  start(win, 'a1');
  assertPaneFolded(win, 'next task started');
});

test('a start in a background tab or a sub-agent leaves the layout alone', () => {
  const {win} = makePage();
  openChats(win, ['a1', 'a2']);
  win._testApi.switchToTab('a1');
  openFile(win, 'a1');
  assertPaneShown(win, 'file open');

  start(win, 'a2');
  assertPaneShown(win, 'background tab started');
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    'a1',
    'the user stays on a1',
  );

  // A sub-agent of a1 starting: the user's chat is still on screen,
  // but a child task is not the user's task starting.
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub1',
    parent_tab_id: 'a1',
    description: 'child',
    task_id: 'child-task',
    isSubagentTab: true,
    isDone: false,
  });
  win._testApi.switchToTab('sub1');
  assert.strictEqual(win._testApi.getActiveTabId(), 'sub1');
  start(win, 'sub1');
  assertPaneShown(win, 'sub-agent started');

  // Back on a1, its own start folds the pane.
  win._testApi.switchToTab('a1');
  start(win, 'a1');
  assertPaneFolded(win, 'a1 started');
});

test('a tab catching up on a task already running does not fold', () => {
  const {win} = makePage();
  // A registry tab replayed after the page loaded: the daemon marks
  // the running status of a run already going `attached`
  // (server.py _broadcast_viewer_running).
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'a1', chatId: 'chat-a1', title: 'a1', workDir: '/ws'}],
  });
  openFile(win, 'a1');
  assertPaneShown(win, 'file open');
  send(win, {
    type: 'status',
    running: true,
    tabId: 'a1',
    startTs: Date.now() - 600000,
    attached: true,
  });
  assertPaneShown(win, 'replayed running status');
  send(win, {type: 'task_events', tabId: 'a1', task: 'the task', events: []});
  assertPaneShown(win, 'transcript replayed');
  // The task ends and a fresh one starts.
  end(win, 'a1');
  start(win, 'a1');
  assertPaneFolded(win, 'fresh start after the replay');
});

test('the phone layout has no pane to fold', () => {
  const {win} = makePage({desktop: false});
  openChats(win, ['a1']);
  openFile(win, 'a1');
  win._testApi.switchToTab('a1');
  start(win, 'a1');
  assert.ok(!hasClass(win, 'content-pane-folded'), 'nothing folded');
  assert.strictEqual(
    win.localStorage.getItem(FOLD_KEY),
    null,
    'nothing stored',
  );
});

let failed = 0;
for (const {name, fn} of TESTS) {
  try {
    fn();
    console.log('  ok - ' + name);
  } catch (err) {
    failed++;
    console.error('  FAIL - ' + name);
    console.error(err && err.stack ? err.stack : err);
  }
  // The pages' timers would keep the process alive.
  for (const win of OPEN_WINDOWS.splice(0)) win.close();
}
if (failed) {
  console.error('\n' + failed + ' test(s) failed');
  process.exit(1);
}
console.log('\nAll tests passed');
