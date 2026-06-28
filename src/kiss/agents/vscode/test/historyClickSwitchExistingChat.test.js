// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for History clicks and chat-tab identity.
//
// Invariant: one webview/client must not have two tabs displaying the same
// backend chat id.  When the user clicks a History row whose chat id is already
// open in another tab, the webview must switch to that tab instead of opening a
// duplicate tab and issuing another resumeSession for the same chat.
//
// This drives the production media/chat.html + panelCopy.js + main.js in JSDOM.
// Run directly with:
//
//     node src/kiss/agents/vscode/test/historyClickSwitchExistingChat.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(initialState) {
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
  let state = initialState;
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function chatTabs(win) {
  return Array.from(win.document.querySelectorAll('#tab-list .chat-tab'));
}

function activeTab(win) {
  return win.document.querySelector('#tab-list .chat-tab.active');
}

function activeTabLabel(win) {
  const tab = activeTab(win);
  const label = tab && tab.querySelector('.chat-tab-label');
  return label ? label.textContent : '';
}

function historyRows(win) {
  return Array.from(win.document.querySelectorAll('#history-list .sidebar-item'));
}

function disableWorkspaceFilter(win) {
  send(win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
  const ws = win.document.getElementById('hf-workspace');
  if (ws && ws.checked) {
    ws.checked = false;
    ws.dispatchEvent(new win.Event('change', {bubbles: true}));
  }
}

function countMessages(posted, type) {
  return posted.filter(msg => msg && msg.type === type).length;
}

function testHistoryClickSwitchesToExistingChatTab() {
  const {win, posted} = makeWebview();
  disableWorkspaceFilter(win);

  const ready = posted.find(msg => msg && msg.type === 'ready');
  assert.ok(ready && ready.tabId, 'main.js must announce the initial tab id');
  const firstTabId = ready.tabId;

  // Load a real persisted chat into the first tab.  This is the same event the
  // daemon sends when a task is replayed or running task state is attached.
  send(win, {
    type: 'task_events',
    tabId: firstTabId,
    chat_id: 'chat-existing',
    task_id: 101,
    task: 'Existing task opened already',
    events: [],
    extra: JSON.stringify({startTs: 1_700_000_000_000, endTs: 1_700_000_001_000}),
  });

  assert.strictEqual(chatTabs(win).length, 1, 'sanity: one chat tab initially');
  assert.strictEqual(
    activeTabLabel(win),
    'Existing task opened already',
    'sanity: the first tab displays the existing chat',
  );

  // Open a second empty tab so the existing chat is no longer active.
  win.document.querySelector('.chat-tab-add').click();
  assert.strictEqual(chatTabs(win).length, 2, 'sanity: plus opens one new tab');
  assert.strictEqual(activeTabLabel(win), 'new chat', 'sanity: new tab is active');

  const resumeBefore = countMessages(posted, 'resumeSession');

  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      {
        id: 'chat-existing',
        task_id: 101,
        title: 'Existing task opened already',
        preview: 'Existing task opened already',
        has_events: true,
        failed: false,
        is_running: false,
        tokens: 0,
        cost: 0,
        steps: 0,
        is_favorite: false,
        timestamp: 1_700_000_000,
        work_dir: '',
        startTs: 1_700_000_000_000,
        endTs: 1_700_000_001_000,
      },
    ],
  });

  const rows = historyRows(win);
  assert.strictEqual(rows.length, 1, 'history row must render');
  rows[0].click();

  assert.strictEqual(
    chatTabs(win).length,
    2,
    'clicking a history row for an already-open chat must not create a duplicate tab',
  );
  assert.strictEqual(
    activeTabLabel(win),
    'Existing task opened already',
    'history click must switch focus back to the already-open chat tab',
  );
  assert.strictEqual(
    countMessages(posted, 'resumeSession'),
    resumeBefore,
    'switching to an already-open chat must not issue another resumeSession',
  );

  win.close();
  console.log('  ok - history click switches to existing tab with same chat id');
}

function testRestoreDropsDuplicatePersistedChatIds() {
  const {win, posted} = makeWebview({
    activeTabIndex: 0,
    chatId: 'frontend-a',
    tabs: [
      {title: 'A', chatId: 'frontend-a', backendChatId: 'chat-dup'},
      {title: 'B duplicate', chatId: 'frontend-b', backendChatId: 'chat-dup'},
      {title: 'C', chatId: 'frontend-c', backendChatId: 'chat-other'},
    ],
  });

  assert.strictEqual(
    chatTabs(win).length,
    2,
    'startup restore must drop duplicate persisted tabs with the same backend chat id',
  );
  const ready = posted.find(msg => msg && msg.type === 'ready');
  assert.strictEqual(
    JSON.stringify(ready.restoredTabs),
    JSON.stringify([
      {tabId: 'frontend-a', chatId: 'chat-dup'},
      {tabId: 'frontend-c', chatId: 'chat-other'},
    ]),
    'ready must resume each backend chat id at most once',
  );

  win.close();
  console.log('  ok - startup restore drops duplicate persisted backend chat ids');
}

function main() {
  testHistoryClickSwitchesToExistingChatTab();
  testRestoreDropsDuplicatePersistedChatIds();
  console.log('historyClickSwitchExistingChat.test.js: all assertions passed.');
}

main();
