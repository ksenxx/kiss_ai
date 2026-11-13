// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end test: "+" from a BUSY chat (a task is running, so the chat
// stays open) MOVES the composer draft into the new chat.  The busy
// chat keeps no copy, neither on screen when the user switches back nor
// in the persisted tab state a reload would restore.
//
// Before the fix createNewTab() ran saveCurrentTab() before carrying
// the text, so the draft landed in both chats whenever the chat left
// behind could not be retired.
//
// "+" over a file (stacked surface: the file takes the chat's place and
// hides the text box) leaves the chat the file belongs to, and moves
// that chat's draft: not whatever the hidden text box still holds.

'use strict';

const assert = require('assert');
const {
  makePage: makeHarnessPage,
  send,
  click,
  sleep,
  stubMonaco,
  stubRaf,
  tabIds: chatIds,
} = require('./simplify2_harness');

function makePage() {
  return makeHarnessPage({
    bodyClass: 'remote-chat',
    extraScripts: ['contentContextMenu.js'],
    beforeScripts: win => {
      stubRaf(win);
      stubMonaco(win);
    },
  });
}

/** Open `path` as a file of chat `tabId` and return its tab's id. */
function openFile(win, tabId, path) {
  send(win, {
    type: 'fileContent',
    tabId: tabId,
    path: path,
    name: path.split('/').pop(),
    content: 'text',
    version: 'v1',
  });
  return win._testApi.openTabs().find(t => t.contentPath === path).id;
}

function persistedDrafts(win, getState) {
  win.dispatchEvent(new win.Event('pagehide'));
  return JSON.parse(JSON.stringify(getState().inputDrafts));
}

function clickPlus(win) {
  click(win, win.document.getElementById('new-chat-btn'));
}

function composer(win) {
  return win.document.getElementById('task-input');
}

function testPlusFromBusyChatMovesTheDraft() {
  const {win, getState} = makePage();
  const boot = win._testApi.getActiveTabId();
  send(win, {type: 'setTaskText', text: 'long task'});
  send(win, {type: 'status', running: true, tabId: boot});
  composer(win).value = 'carry me';

  clickPlus(win);

  const fresh = win._testApi.getActiveTabId();
  assert.notStrictEqual(fresh, boot, 'a new chat opens');
  assert.strictEqual(composer(win).value, 'carry me', 'the draft moved');
  assert.ok(chatIds(win).includes(boot), 'the busy chat stays open');

  win._testApi.switchToTab(boot);
  assert.strictEqual(
    composer(win).value,
    '',
    'the busy chat keeps no copy of the draft',
  );
  const drafts = (getState() || {}).inputDrafts || {};
  assert.ok(!drafts[boot], 'no copy is persisted for the busy chat');

  win._testApi.switchToTab(fresh);
  assert.strictEqual(composer(win).value, 'carry me', 'the new chat has it');
  win.close();
  console.log('  ok - + from a busy chat moves the draft, leaving no copy');
}

async function testPlusOverAFileLeavesTheChatItBelongsTo() {
  const api = win => win._testApi;

  // No new chat is open: "+" opens one.  The text box under the file
  // still holds chat A's draft, but the file on screen is chat B's.
  const one = makePage();
  send(one.win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 'chat-A', title: 'chat A', chatId: 'cA'},
      {tabId: 'chat-B', title: 'chat B', chatId: 'cB'},
    ],
  });
  api(one.win).switchToTab('chat-B');
  composer(one.win).value = 'draft of B';
  api(one.win).switchToTab('chat-A');
  composer(one.win).value = 'draft of A';
  openFile(one.win, 'chat-A', '/w/a.txt');
  api(one.win).switchToTab(openFile(one.win, 'chat-B', '/w/b.txt'));
  assert.ok(
    one.win.document.body.classList.contains('content-tab-open'),
    'precondition: a file is on screen and hides the text box',
  );
  clickPlus(one.win);
  const opened = api(one.win).getActiveTabId();
  assert.ok(!['chat-A', 'chat-B'].includes(opened), 'a new chat opens');
  assert.strictEqual(composer(one.win).value, 'draft of B', "chat B's draft moved");
  assert.deepStrictEqual(
    persistedDrafts(one.win, one.getState),
    {'chat-A': 'draft of A', [opened]: 'draft of B'},
    'chat B keeps no copy and chat A keeps its own',
  );
  await sleep(0);
  one.win.close();

  // A new chat is already open: "+" brings it forward with the draft.
  const two = makePage();
  send(two.win, {
    type: 'tabs_state',
    tabs: [
      {tabId: 'chat-B', title: 'chat B', chatId: 'cB'},
      {tabId: 'chat-F', title: 'new chat', chatId: ''},
    ],
  });
  api(two.win).switchToTab('chat-B');
  composer(two.win).value = 'draft of B';
  openFile(two.win, 'chat-B', '/w/b.txt');
  clickPlus(two.win);
  assert.strictEqual(api(two.win).getActiveTabId(), 'chat-F');
  assert.strictEqual(composer(two.win).value, 'draft of B');
  assert.deepStrictEqual(persistedDrafts(two.win, two.getState), {
    'chat-F': 'draft of B',
  });
  await sleep(0);
  two.win.close();

  // The file belongs to a chat that is itself new: "+" shows that chat.
  const three = makePage();
  const boot = api(three.win).getActiveTabId();
  openFile(three.win, boot, '/w/c.txt');
  assert.notStrictEqual(api(three.win).getActiveTabId(), boot);
  clickPlus(three.win);
  assert.strictEqual(api(three.win).getActiveTabId(), boot);
  assert.deepStrictEqual(chatIds(three.win).length, 2, 'no chat was opened');
  await sleep(0);
  three.win.close();
  console.log('  ok - + over a file leaves the chat the file belongs to');
}

async function main() {
  testPlusFromBusyChatMovesTheDraft();
  await testPlusOverAFileLeavesTheChatItBelongsTo();
  console.log('\n2 passed, 0 failed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
