// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  sendEmptyWorkDir,
  readyTabId,
  ofType,
} = require('./simplify2_harness');
const {historyRows, makeRow} = require('./history_helpers');

function makePage(initialState) {
  return makeChatPage({initialState});
}

// The open chats (there is no row of chat tabs any more: the chat on
// screen is the one picked in the Chats panel, the rest stay hidden).
function chatTabs(win) {
  return win._testApi.openTabs().filter(t => !t.isContentTab);
}

function activeTabLabel(win) {
  const id = win._testApi.getActiveTabId();
  const tab = chatTabs(win).find(t => t.id === id);
  return tab ? tab.title : '';
}

function countMessages(posted, type) {
  return ofType(posted, type).length;
}

function testHistoryClickSwitchesToExistingChatTab() {
  const {win, posted} = makePage();
  sendEmptyWorkDir(win);

  const firstTabId = readyTabId(posted);

  send(win, {
    type: 'task_events',
    tabId: firstTabId,
    chat_id: 'chat-existing',
    task_id: 101,
    task: 'Existing task opened already',
    events: [],
    extra: JSON.stringify({startTs: 1_700_000_000_000}),
  });
  // The task is still running: leaving it for "+" must keep its tab
  // open (an idle chat left behind would be retired instead).
  send(win, {type: 'status', running: true, tabId: firstTabId});
  assert.ok(chatTabs(win)[0].isRunning, 'sanity: the first chat is running');

  assert.strictEqual(chatTabs(win).length, 1, 'sanity: one chat tab initially');
  assert.strictEqual(
    activeTabLabel(win),
    'Existing task opened already',
    'sanity: the first tab displays the existing chat',
  );

  win.document.querySelector('#new-chat-btn').click();
  assert.strictEqual(chatTabs(win).length, 2, 'sanity: plus opens one new tab');
  assert.strictEqual(
    activeTabLabel(win),
    'new chat',
    'sanity: new tab is active',
  );
  assert.notStrictEqual(win._testApi.getActiveTabId(), firstTabId);

  const resumeBefore = countMessages(posted, 'resumeSession');

  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      makeRow({
        id: 'chat-existing',
        task_id: 101,
        title: 'Existing task opened already',
        has_events: true,
        is_running: true,
        endTs: 0,
      }),
    ],
  });

  const rows = historyRows(win);
  assert.strictEqual(rows.length, 1, 'history row must render');
  rows[0].click();

  assert.strictEqual(
    chatTabs(win).filter(t => t.title === 'Existing task opened already')
      .length,
    1,
    'clicking a history row for an already-open chat must not create a duplicate tab',
  );
  assert.strictEqual(
    win._testApi.getActiveTabId(),
    firstTabId,
    'history click must switch focus back to the already-open chat tab',
  );
  assert.strictEqual(
    activeTabLabel(win),
    'Existing task opened already',
    'the already-open chat is the one on screen',
  );
  // The empty "new chat" left behind is retired (retire-on-leave), so
  // the running chat is the only one open.
  assert.strictEqual(
    chatTabs(win).length,
    1,
    'leaving the empty new chat for the running one retires it',
  );
  assert.strictEqual(
    countMessages(posted, 'resumeSession'),
    resumeBefore,
    'switching to an already-open chat must not issue another resumeSession',
  );

  win.close();
  console.log(
    '  ok - history click switches to existing tab with same chat id',
  );
}

function testRestoreIgnoresStaleLocalTabs() {
  const {win, posted} = makePage({
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
    1,
    'a stale local tab set is not restored: the daemon registry is the ' +
      'only source of tabs and answers with tabs_state',
  );
  const ready = posted.find(msg => msg && msg.type === 'ready');
  assert.strictEqual(
    JSON.stringify(ready.restoredTabs),
    '[]',
    'ready carries no tabs from local storage',
  );

  win.close();
  console.log('  ok - startup ignores a stale local tab set');
}

function main() {
  testHistoryClickSwitchesToExistingChatTab();
  testRestoreIgnoresStaleLocalTabs();
  console.log('historyClickSwitchExistingChat.test.js: all assertions passed.');
}

main();
