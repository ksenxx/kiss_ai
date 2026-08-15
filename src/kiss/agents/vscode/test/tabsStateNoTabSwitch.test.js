// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end test for the mid-session reconnect path. The server answers
// every `ready` -- including the one a dropped WebSocket sends when it comes
// back -- with the shared-registry `tabs_state` snapshot, followed by a
// replayed `status {running:true}` for every task still going. Once the user
// is working in the page, that resync must not drag them onto a task that is
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

function snapshotEntry(tabId, title, chatId) {
  return {tabId: tabId, chatId: chatId || '', title: title || '', workDir: ''};
}

// The registry resync a reconnect triggers: the canonical snapshot plus a
// replayed running `status` for each still-live task.
function resync(win, entries, runningTabIds) {
  send(win, {type: 'tabs_state', tabs: entries});
  for (const tabId of runningTabIds || []) {
    send(win, {type: 'status', running: true, tabId: tabId});
  }
}

// A launched window with a live backend, in which the user has since touched
// the page: exactly the state a mid-session resync arrives in.
// `daemonStatus connected` is what both clients send once the backend is
// reachable -- without it the window would still be sitting behind the "server
// is starting" overlay and no gesture could mean anything. The user works in
// one registered tab of their own.
function endLaunch(win) {
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {
    type: 'tabs_state',
    tabs: [snapshotEntry('tab-user', 'my chat', 'chat-user')],
  });
  win._testApi.endLaunch();
  assert.strictEqual(win._testApi.getActiveTabId(), 'tab-user');
}

// A reconnect that restores one running task must leave the user where they
// were, and must open that task in its own (background) tab.
function testRestoredRunningTaskDoesNotStealFocus() {
  const win = makeWebview();
  const api = win._testApi;
  endLaunch(win);
  const userTab = api.getActiveTabId();
  const before = tabIds(win);

  resync(
    win,
    [
      snapshotEntry('tab-user', 'my chat', 'chat-user'),
      snapshotEntry('tab-bg', 'background work', 'chat-1'),
    ],
    ['tab-bg'],
  );

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
  assert.ok(after.includes('tab-bg'), 'the restored tab is in the tab bar');
  assert.strictEqual(
    tabTitle(win, 'tab-bg'),
    'background work',
    'the restored tab must be titled after its task',
  );
  assert.strictEqual(
    win._sentMessages.filter(m => m && m.type === 'resumeSession').length,
    0,
    'the daemon replays registry tabs itself; the client resumes nothing',
  );

  clickTab(win, 'tab-bg');
  assert.strictEqual(
    api.getActiveTabId(),
    'tab-bg',
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

  resync(
    win,
    [
      snapshotEntry('tab-user', 'my chat', 'chat-user'),
      snapshotEntry('tab-a', 'alpha', 'chat-a'),
      snapshotEntry('tab-b', 'beta', 'chat-b'),
      snapshotEntry('tab-c', '', 'chat-c'),
    ],
    ['tab-a', 'tab-b', 'tab-c'],
  );

  assert.strictEqual(
    api.getActiveTabId(),
    userTab,
    'no restored task may become the active tab',
  );

  const ids = tabIds(win);
  for (const id of ['tab-a', 'tab-b', 'tab-c']) {
    assert.ok(ids.includes(id), `restored task ${id} needs its own tab`);
    assert.notStrictEqual(id, userTab, "and not the user's tab");
  }
  assert.strictEqual(
    tabTitle(win, 'tab-c'),
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

  const entries = [
    snapshotEntry('tab-user', 'my chat', 'chat-user'),
    snapshotEntry('tab-1', 'first pass', 'chat-1'),
  ];
  resync(win, entries, ['tab-1']);
  const userTab = api.getActiveTabId();
  const tabsAfterFirst = tabIds(win);

  resync(win, entries, ['tab-1']);

  assert.deepStrictEqual(
    tabIds(win),
    tabsAfterFirst,
    'a task that already has a tab must not get a second one',
  );
  assert.strictEqual(
    api.getActiveTabId(),
    userTab,
    'an already-open running task must not be switched to',
  );
  assert.notStrictEqual(
    api.getActiveTabId(),
    'tab-1',
    'the restored tab must stay in the background',
  );

  win.close();
  console.log('  ok - an already-open running task is left alone');
}

// Junk snapshots must not disturb the tab bar, and junk running statuses
// must not create or focus anything.
function testMalformedPayloadsAreIgnored() {
  const win = makeWebview();
  const api = win._testApi;
  endLaunch(win);
  const userTab = api.getActiveTabId();
  const before = tabIds(win);

  send(win, {type: 'tabs_state', tabs: 'not-an-array'});
  send(win, {type: 'tabs_state'});
  send(win, {
    type: 'tabs_state',
    tabs: [null, {}, {taskId: 'x'}, snapshotEntry('tab-user', 'my chat',
                                                  'chat-user')],
  });
  send(win, {type: 'status', running: true, tabId: 'tab-ghost'});

  assert.deepStrictEqual(tabIds(win), before, 'no tab may appear or vanish');
  assert.strictEqual(api.getActiveTabId(), userTab, 'the active tab is kept');

  win.close();
  console.log('  ok - malformed tabs_state payloads are ignored');
}

function runTests() {
  testRestoredRunningTaskDoesNotStealFocus();
  testManyRestoredTasksEachGetTheirOwnTab();
  testAlreadyOpenTaskIsNeitherDuplicatedNorFocused();
  testMalformedPayloadsAreIgnored();
}

try {
  runTests();
  console.log('\n4 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
