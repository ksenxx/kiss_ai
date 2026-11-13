// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for which tabs a close keeps and which tab
// comes on screen after it (media/main.js):
// * "Close Inactive" keeps an idle chat whose sub-agent still runs (a
//   user's close takes the descendants along, the running one too);
// * "Close Others" from a file in the content pane closes the chats in
//   the chat pane without restoring and fetching each one in between;
// * closing a file on the stacked (phone) layout lands on the chat that
//   opened it, not on whichever chat sits next to it in the tab list;
// * widening the window while a file whose chat is gone is on screen
//   puts another open chat in the chat pane instead of a new chat;
// * closing a sub-agent's tab that sits behind its own sub-agent's on
//   the strip lands on their chat instead of a new chat.
'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');
const h = require('./ui_antipattern_harness');
const {tabEl, deferChatTranscripts} = require('./subagent_tab_helpers');

const {test, report} = h.makeRunner();

/** A remote page on the desktop (split) or phone (stacked) layout;
 *  `fire(matches)` crosses the desktop breakpoint. */
function page(desktop) {
  let fire = null;
  const {win, posted} = harness.makePage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js', 'contentContextMenu.js'],
    beforeScripts(w) {
      harness.stubRaf(w);
      fire = harness.fakeMatchMedia(w, desktop);
      harness.stubMonaco(w);
    },
  });
  const tabA = win._testApi.getActiveTabId();
  win._testApi.endLaunch();
  return {win, posted, tabA, fire};
}

/** Register chats A (the boot tab), B and C; B and C have deferred
 *  transcripts, fetched the first time they come on screen. */
function threeChats(win, tabA) {
  h.send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: tabA, chatId: '', title: 'A', workDir: ''},
      {tabId: 'tab-b', chatId: 'c-b', title: 'B', workDir: ''},
      {tabId: 'tab-c', chatId: 'c-c', title: 'C', workDir: ''},
    ],
  });
  deferChatTranscripts(win);
}

/** Open /ws/a.py from chat *owner*; returns the file tab's id. */
function openFile(win, owner) {
  h.send(win, {
    type: 'fileContent',
    tabId: owner,
    path: '/ws/a.py',
    name: 'a.py',
    content: 'print(1)\n',
    version: 'v1',
  });
  const file = win._testApi.openTabs().find(t => t.isContentTab);
  assert.ok(file, 'the file opened in a tab');
  return file.id;
}

function closeWithX(win, tabId) {
  const el = tabEl(win, tabId);
  assert.ok(el, `tab ${tabId} is on the strip`);
  h.click(win, el.querySelector('.chat-tab-close'));
}

/** The daemon announces sub-agent tab *tabId* below tab *parentId*. */
function announceSubagent(win, tabId, parentId, extra) {
  h.send(win, {
    type: 'openSubagentTab',
    tab_id: tabId,
    parent_tab_id: parentId,
    description: tabId,
    task_id: 'task-' + tabId,
    ...extra,
  });
}

async function main() {
  await test('Close Inactive keeps an idle chat whose sub-agent runs', () => {
    const {win, posted, tabA} = page(true);
    threeChats(win, tabA);
    h.send(win, {
      type: 'openSubagentTab',
      tab_id: 'sub-a',
      parent_tab_id: tabA,
      description: 'a sub-agent',
      task_id: 'sub-task-a',
      taskIndex: 0,
      isSubagentTab: true,
    });
    h.send(win, {type: 'status', running: false, tabId: tabA});
    const subA = win._testApi.openTabs().find(t => t.id === 'sub-a');
    assert.ok(subA.isRunning, 'precondition: the sub-agent runs');
    h.pickTabMenuItem(win, tabEl(win, tabA), 'Close Inactive');
    assert.deepStrictEqual(
      harness.tabIds(win),
      [tabA, 'sub-a'],
      'B and C close, A stays',
    );
    const closed = h.ofType(posted, 'closeTab').map(m => m.tabId);
    assert.deepStrictEqual([...closed].sort(), ['tab-b', 'tab-c']);
    win.close();
  });

  await test('Close Others from a file in the content pane fetches no chat', () => {
    const {win, posted, tabA} = page(true);
    threeChats(win, tabA);
    const file = openFile(win, tabA);
    const fileTab = win.document.querySelector(
      '#content-tab-list .chat-tab[data-tab-id="' + file + '"]',
    );
    assert.ok(fileTab, 'the file is on the content pane strip');
    h.pickTabMenuItem(win, fileTab, 'Close Others');
    const tabs = win._testApi.openTabs();
    assert.strictEqual(tabs.length, 2, 'the file and one fresh chat');
    assert.strictEqual(tabs[0].id, file);
    assert.ok(!tabs[1].isContentTab && tabs[1].id !== tabA);
    assert.deepStrictEqual(
      [...h.scopedFetches(posted, ['tab-b', 'tab-c'])],
      [],
      'no chat closed by the same action was restored and fetched',
    );
    win.close();
  });

  await test('closing a file on the phone layout lands on its own chat', () => {
    const {win, tabA} = page(false);
    h.send(win, {
      type: 'tabs_state',
      tabs: [
        {tabId: tabA, chatId: '', title: 'A', workDir: ''},
        {tabId: 'tab-b', chatId: 'c-b', title: 'B', workDir: ''},
      ],
    });
    const file = openFile(win, tabA);
    assert.strictEqual(win._testApi.getActiveTabId(), file, 'the file shows');
    closeWithX(win, file);
    assert.strictEqual(
      win._testApi.getActiveTabId(),
      tabA,
      'the chat that opened the file comes back',
    );
    win.close();
  });

  await test('widening onto a file whose chat closed shows an open chat', () => {
    const {win, posted, tabA, fire} = page(false);
    h.send(win, {
      type: 'tabs_state',
      tabs: [
        {tabId: tabA, chatId: '', title: 'A', workDir: ''},
        {tabId: 'tab-b', chatId: 'c-b', title: 'B', workDir: ''},
      ],
    });
    const file = openFile(win, tabA);
    win._testApi.switchToTab(tabA);
    closeWithX(win, tabA);
    win._testApi.switchToTab(file);
    assert.strictEqual(win._testApi.getActiveTabId(), file, 'the file shows');
    const before = h.ofType(posted, 'newChat').length;
    fire(true);
    assert.strictEqual(
      h.ofType(posted, 'newChat').length,
      before,
      'no new chat is opened while chat B is there',
    );
    assert.strictEqual(win._testApi.getActiveTabId(), 'tab-b');
    assert.deepStrictEqual(harness.tabIds(win), ['tab-b', file]);
    win.close();
  });

  await test("closing a sub-agent behind its own sub-agent's tab lands on their chat", () => {
    const {win, posted, tabA} = page(true);
    // sub-c dispatched sub-g and finished first: its tab closed, sub-g's
    // stayed.  sub-c's history was then reopened (behind sub-g on the
    // strip) and the daemon announced the running sub-g below it again.
    announceSubagent(win, 'sub-c', tabA);
    announceSubagent(win, 'sub-g', 'sub-c');
    h.send(win, {type: 'subagentDone', tab_id: 'sub-c'});
    announceSubagent(win, 'sub-c', tabA, {isDone: true, reopened: true});
    announceSubagent(win, 'sub-g', 'sub-c');
    assert.deepStrictEqual(
      harness.tabIds(win),
      [tabA, 'sub-g', 'sub-c'],
      'precondition: the reopened sub-agent sits behind its own sub-agent',
    );
    win._testApi.switchToTab('sub-c');
    const before = h.ofType(posted, 'newChat').length;
    closeWithX(win, 'sub-c');
    assert.deepStrictEqual(
      harness.tabIds(win),
      [tabA],
      'the user closed the sub-agent and, with it, its own sub-agent',
    );
    assert.strictEqual(
      win._testApi.getActiveTabId(),
      tabA,
      'their chat, still open, comes on screen',
    );
    assert.strictEqual(
      h.ofType(posted, 'newChat').length,
      before,
      'no new chat is opened',
    );
    win.close();
  });

  report('tabCloseActions_sweep1030');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
