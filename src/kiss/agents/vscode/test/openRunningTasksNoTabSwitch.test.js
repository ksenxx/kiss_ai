// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end test for the remote-webapp reconnect path. The server pushes an
// `openRunningTasks` event to every client that says `ready` -- including the
// `ready` a dropped WebSocket sends when it comes back. Once the user is
// working in the page, that reconnect must not drag them onto a task that is
// still running: restored tasks get reachable background tabs and nothing
// else. (Landing on the newest running task is a launch-time affair, and
// lives in launchTabSwitch.test.js.) Every test here therefore ends the
// launch first, the way a real tap or keystroke would.

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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

  const sent = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => {
        sent.push(msg);
      },
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  win._sentMessages = sent;
  return win;
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function tabIds(win) {
  return Array.from(win.document.querySelectorAll('.chat-tab')).map(el =>
    el.getAttribute('data-tab-id'),
  );
}

function tabTitle(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  const label = el.querySelector('.chat-tab-label');
  return (label || el).textContent.trim();
}

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function resumeCalls(win) {
  return win._sentMessages.filter(m => m && m.type === 'resumeSession');
}

// A launched window with a live backend, in which the user has since touched
// the page: exactly the state a mid-session snapshot arrives in.
// `daemonStatus connected` is what both clients send once the backend is
// reachable -- without it the window would still be sitting behind the "server
// is starting" overlay and no gesture could mean anything.
function endLaunch(win) {
  send(win, {type: 'daemonStatus', connected: true});
  win._testApi.endLaunch();
}

// A reconnect that restores one running task must leave the user where they
// were, and must resume that task into its own (background) tab.
function testRestoredRunningTaskDoesNotStealFocus() {
  const win = makeWebview();
  const api = win._testApi;
  endLaunch(win);
  const userTab = api.getActiveTabId();
  const before = tabIds(win);

  send(win, {
    type: 'openRunningTasks',
    tasks: [{chatId: 'chat-1', taskId: 'task-1', title: 'background work'}],
  });

  assert.strictEqual(
    api.getActiveTabId(),
    userTab,
    'restoring a still-running task must not switch tabs',
  );

  const after = tabIds(win);
  assert.strictEqual(
    after.length,
    before.length + 1,
    'the restored task must get its own tab',
  );
  const restoredId = after.find(id => before.indexOf(id) < 0);
  assert.ok(restoredId, 'a new tab id must appear in the tab bar');
  assert.strictEqual(
    tabTitle(win, restoredId),
    'background work',
    'the restored tab must be titled after its task',
  );

  const calls = resumeCalls(win);
  assert.strictEqual(calls.length, 1, 'exactly one session must be resumed');
  assert.strictEqual(calls[0].id, 'chat-1');
  assert.strictEqual(calls[0].taskId, 'task-1');
  assert.strictEqual(
    calls[0].tabId,
    restoredId,
    'the session must resume into its own tab, not the tab the user is on',
  );

  clickTab(win, restoredId);
  assert.strictEqual(
    api.getActiveTabId(),
    restoredId,
    'the restored tab must still be reachable by clicking it',
  );

  win.close();
  console.log('  ok - a restored running task never steals focus');
}

// Several restored tasks must each get their own tab, none of them active.
function testManyRestoredTasksEachGetTheirOwnTab() {
  const win = makeWebview();
  const api = win._testApi;
  endLaunch(win);
  const userTab = api.getActiveTabId();

  send(win, {
    type: 'openRunningTasks',
    tasks: [
      {chatId: 'chat-a', taskId: 'task-a', title: 'alpha'},
      {chatId: 'chat-b', taskId: 'task-b', title: 'beta'},
      {chatId: 'chat-c', taskId: 'task-c', title: ''},
    ],
  });

  assert.strictEqual(
    api.getActiveTabId(),
    userTab,
    'no restored task may become the active tab',
  );

  const calls = resumeCalls(win);
  assert.strictEqual(calls.length, 3, 'every task must be resumed');
  const usedTabIds = calls.map(c => c.tabId);
  assert.strictEqual(
    new Set(usedTabIds).size,
    3,
    'each restored task needs a distinct tab',
  );
  assert.ok(
    usedTabIds.every(id => id !== userTab),
    "no task may resume into the user's tab",
  );
  assert.strictEqual(
    tabTitle(win, calls[2].tabId),
    'new chat',
    'a task without a title falls back to the default tab title',
  );

  win.close();
  console.log('  ok - every restored task gets its own background tab');
}

// A task already shown in a tab must not be duplicated, and must not be
// switched to either.
function testAlreadyOpenTaskIsNeitherDuplicatedNorFocused() {
  const win = makeWebview();
  const api = win._testApi;
  endLaunch(win);

  send(win, {
    type: 'openRunningTasks',
    tasks: [{chatId: 'chat-1', taskId: 'task-1', title: 'first pass'}],
  });
  const restoredId = resumeCalls(win)[0].tabId;
  const userTab = api.getActiveTabId();
  const tabsAfterFirst = tabIds(win);

  send(win, {
    type: 'openRunningTasks',
    tasks: [{chatId: 'chat-1', taskId: 'task-1', title: 'first pass'}],
  });

  assert.deepStrictEqual(
    tabIds(win),
    tabsAfterFirst,
    'a task that already has a tab must not get a second one',
  );
  assert.strictEqual(
    resumeCalls(win).length,
    1,
    'an already-open session must not be resumed twice',
  );
  assert.strictEqual(
    api.getActiveTabId(),
    userTab,
    'an already-open running task must not be switched to',
  );
  assert.notStrictEqual(
    api.getActiveTabId(),
    restoredId,
    'the restored tab must stay in the background',
  );

  win.close();
  console.log('  ok - an already-open running task is left alone');
}

// Junk entries must be ignored without disturbing the tab bar.
function testMalformedTasksAreIgnored() {
  const win = makeWebview();
  const api = win._testApi;
  endLaunch(win);
  const userTab = api.getActiveTabId();
  const before = tabIds(win);

  send(win, {type: 'openRunningTasks', tasks: [null, {}, {taskId: 'x'}]});
  send(win, {type: 'openRunningTasks', tasks: 'not-an-array'});
  send(win, {type: 'openRunningTasks'});

  assert.deepStrictEqual(tabIds(win), before, 'no tab may be created');
  assert.strictEqual(resumeCalls(win).length, 0, 'nothing may be resumed');
  assert.strictEqual(api.getActiveTabId(), userTab, 'the active tab is kept');

  win.close();
  console.log('  ok - malformed openRunningTasks payloads are ignored');
}

function runTests() {
  testRestoredRunningTaskDoesNotStealFocus();
  testManyRestoredTasksEachGetTheirOwnTab();
  testAlreadyOpenTaskIsNeitherDuplicatedNorFocused();
  testMalformedTasksAreIgnored();
}

try {
  runTests();
  console.log('\n4 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
