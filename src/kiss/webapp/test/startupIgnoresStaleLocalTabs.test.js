// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Startup ignores a stale local tab set.  (A history click on an
// already-open chat switching to its tab is pinned by
// historyClickNoInputCopy.test.js::testSwitchBranchLeavesInputEmpty.)

'use strict';

const assert = require('assert');
const {makePage} = require('./simplify2_harness');

// The open chats (there is no row of chat tabs any more: the chat on
// screen is the one picked in the Chats panel, the rest stay hidden).
function chatTabs(win) {
  return win._testApi.openTabs().filter(t => !t.isContentTab);
}

function testRestoreIgnoresStaleLocalTabs() {
  const {win, posted} = makePage({
    initialState: {
      activeTabIndex: 0,
      chatId: 'frontend-a',
      tabs: [
        {title: 'A', chatId: 'frontend-a', backendChatId: 'chat-dup'},
        {title: 'B duplicate', chatId: 'frontend-b', backendChatId: 'chat-dup'},
        {title: 'C', chatId: 'frontend-c', backendChatId: 'chat-other'},
      ],
    },
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
  testRestoreIgnoresStaleLocalTabs();
  console.log('startupIgnoresStaleLocalTabs.test.js: all assertions passed.');
}

main();
