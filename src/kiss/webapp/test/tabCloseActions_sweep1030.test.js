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
//   puts another open chat in the chat pane instead of a new chat.
'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');
const h = require('./ui_antipattern_harness');
const {tabEl} = require('./subagent_tab_helpers');

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
  for (const id of ['tab-b', 'tab-c']) {
    h.send(win, {type: 'status', running: false, tabId: id});
    h.send(win, {
      type: 'replay_deferred',
      tabId: id,
      chatId: id === 'tab-b' ? 'c-b' : 'c-c',
      taskId: '',
    });
  }
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

/** The open tabs' ids, as an array of this realm (deepStrictEqual
 *  compares prototypes). */
function openIds(win) {
  return win._testApi.openTabs().map(t => String(t.id));
}

function scopedFetches(posted, ids) {
  return h
    .ofType(posted, 'resumeSession')
    .filter(m => m.scoped === true && ids.includes(m.tabId));
}

function pickTabMenuItem(win, tabNode, label) {
  h.rightClick(win, tabNode);
  const item = h
    .all(win, '#tab-context-menu .tab-ctx-item')
    .find(el => el.textContent === label);
  assert.ok(item, `the "${label}" item exists`);
  h.click(win, item);
}

function closeWithX(win, tabId) {
  const el = tabEl(win, tabId);
  assert.ok(el, `tab ${tabId} is on the strip`);
  h.click(win, el.querySelector('.chat-tab-close'));
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
    pickTabMenuItem(win, tabEl(win, tabA), 'Close Inactive');
    assert.deepStrictEqual(
      [...openIds(win)],
      [tabA, 'sub-a'],
      'B and C close, A stays',
    );
    const closed = h.ofType(posted, 'closeTab').map(m => m.tabId);
    assert.deepStrictEqual([...closed].sort(), ['tab-b', 'tab-c']);
    win.close();
  });

  await test('Close Inactive closes an idle chat whose sub-agent stopped', () => {
    const {win, tabA} = page(true);
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
    h.send(win, {type: 'status', running: true, tabId: 'tab-c'});
    h.send(win, {type: 'status', running: false, tabId: 'sub-a'});
    pickTabMenuItem(win, tabEl(win, tabA), 'Close Inactive');
    assert.deepStrictEqual(
      [...openIds(win)],
      ['tab-c'],
      'only the running chat stays',
    );
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
    pickTabMenuItem(win, fileTab, 'Close Others');
    const tabs = win._testApi.openTabs();
    assert.strictEqual(tabs.length, 2, 'the file and one fresh chat');
    assert.strictEqual(tabs[0].id, file);
    assert.ok(!tabs[1].isContentTab && tabs[1].id !== tabA);
    assert.deepStrictEqual(
      [...scopedFetches(posted, ['tab-b', 'tab-c'])],
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
    assert.deepStrictEqual([...openIds(win)], ['tab-b', file]);
    win.close();
  });

  report('tabCloseActions_sweep1030');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
