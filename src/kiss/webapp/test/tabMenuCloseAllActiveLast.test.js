// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for the batch entries of a content tab's
// context menu in media/main.js ("Close Others" and "Close All"; the
// menu's third entry is "Close"):
// * they close content tabs (files, reports, browser and terminal
//   tabs) only: a chat's or a sub-agent's tab is not the user's to
//   close, so it stays, idle or running, and a chat whose transcript
//   the daemon deferred is not fetched;
// * the content tab on screen is closed LAST, so no tab that is about
//   to be closed anyway is brought on screen in between.
'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');
const h = require('./ui_antipattern_harness');
const {deferChatTranscripts} = require('./subagent_tab_helpers');

const {test, report} = h.makeRunner();

// The two registry chats whose transcripts the daemon deferred.
const DEFERRED = ['tab-b', 'tab-c'];

// The boot chat (on screen, with a running sub-agent), two registry
// chats whose transcripts the daemon deferred (tab-c runs), and three
// files the boot chat opened.  `chats` lists every tab that is not a
// file, in tab order.
function pageWithFiles(opts) {
  const {win, posted} = h.makePage(opts);
  const tabA = win._testApi.getActiveTabId();
  win._testApi.endLaunch();
  h.send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: tabA, chatId: '', title: 'new chat', workDir: ''},
      {tabId: 'tab-b', chatId: 'c-b', title: 'B', workDir: ''},
      {tabId: 'tab-c', chatId: 'c-c', title: 'C', workDir: ''},
    ],
  });
  deferChatTranscripts(win);
  h.send(win, {type: 'status', running: true, tabId: 'tab-c'});
  h.send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-a',
    parent_tab_id: tabA,
    description: 'a sub-agent',
    task_id: 'sub-task-a',
    taskIndex: 0,
  });
  const chats = harness.tabIds(win);
  const files = ['a.py', 'b.py', 'c.py'].map(name => {
    const path = '/ws/' + name;
    h.send(win, {
      type: 'fileContent',
      tabId: tabA,
      path,
      name,
      content: 'print(1)\n',
      version: 'v1',
    });
    return win._testApi.openTabs().find(t => t.contentPath === path).id;
  });
  assert.strictEqual(
    h.scopedFetches(posted, DEFERRED).length,
    0,
    'precondition: the deferred chats were not fetched',
  );
  return {win, posted, tabA, chats, files};
}

/** The strip entry of content tab *id* (on either tab row). */
function fileTabEl(win, id) {
  const el = Array.from(h.contentTabStrips(win)).find(
    t => t.dataset.tabId === id,
  );
  assert.ok(el, `file tab ${id} is on a tab row`);
  return el;
}

/**
 * How many content views were brought on screen while *act* ran (the
 * view the action found on screen is not one).
 */
function viewsShownDuring(win, act) {
  const watch = new win.MutationObserver(() => {});
  watch.observe(h.byId(win, 'content-tab-area'), {
    attributes: true,
    attributeFilter: ['style'],
    attributeOldValue: true,
    subtree: true,
  });
  act();
  const records = watch
    .takeRecords()
    .filter(r => r.target.classList.contains('content-tab-view'));
  watch.disconnect();
  const hidden = style => /display:\s*none/.test(style || '');
  const shown = new Set();
  records.forEach((rec, i) => {
    const next = records.slice(i + 1).find(r => r.target === rec.target);
    const after = next ? next.oldValue : rec.target.getAttribute('style');
    if (hidden(rec.oldValue) && !hidden(after)) shown.add(rec.target);
  });
  return shown.size;
}

/** No chat or sub-agent tab was closed, nor a deferred chat fetched. */
function assertChatsUntouched(posted) {
  assert.deepStrictEqual(h.ofType(posted, 'closeTab'), [], 'no chat closed');
  assert.deepStrictEqual(
    h.scopedFetches(posted, DEFERRED),
    [],
    'no deferred chat was brought on screen',
  );
}

async function main() {
  await test('Close All closes every content tab and no chat', () => {
    const {win, posted, chats, files} = pageWithFiles();
    h.rightClick(win, fileTabEl(win, files[0]));
    assert.deepStrictEqual(
      h.all(win, '#tab-context-menu .tab-ctx-item').map(el => el.textContent),
      ['Close', 'Close Others', 'Close All'],
      'no entry does what another one does',
    );
    h.pickTabMenuItem(win, fileTabEl(win, files[0]), 'Close All');
    assert.deepStrictEqual(
      harness.tabIds(win),
      chats,
      'the chats and the sub-agent stay, idle or running',
    );
    assertChatsUntouched(posted);
    win.close();
  });

  await test('Close Others keeps the clicked file and every chat', () => {
    const {win, posted, chats, files} = pageWithFiles();
    h.pickTabMenuItem(win, fileTabEl(win, files[1]), 'Close Others');
    assert.deepStrictEqual(harness.tabIds(win), [...chats, files[1]]);
    assertChatsUntouched(posted);
    win.close();
  });

  // The stacked (phone) layout shows a file in place of the chat: the
  // file on screen is the first of the three here.
  function fileScreen() {
    const page = pageWithFiles({narrow: true});
    page.win._testApi.switchToTab(page.files[0]);
    assert.strictEqual(
      page.win._testApi.getActiveTabId(),
      page.files[0],
      'precondition: the first file is on screen',
    );
    return page;
  }

  await test('Close All closes the file on screen last', () => {
    const {win, posted, tabA, chats, files} = fileScreen();
    const shown = viewsShownDuring(win, () =>
      h.pickTabMenuItem(win, fileTabEl(win, files[0]), 'Close All'),
    );
    assert.strictEqual(shown, 0, 'no file about to close came on screen');
    assert.deepStrictEqual(harness.tabIds(win), chats);
    assert.strictEqual(
      win._testApi.getActiveTabId(),
      tabA,
      'the chat that opened the files comes back',
    );
    assertChatsUntouched(posted);
    win.close();
  });

  await test('Close Others from a file off screen brings that file on screen', () => {
    const {win, posted, chats, files} = fileScreen();
    const shown = viewsShownDuring(win, () =>
      h.pickTabMenuItem(win, fileTabEl(win, files[2]), 'Close Others'),
    );
    assert.strictEqual(shown, 1, 'only the kept file came on screen');
    assert.deepStrictEqual(harness.tabIds(win), [...chats, files[2]]);
    assert.strictEqual(win._testApi.getActiveTabId(), files[2]);
    assertChatsUntouched(posted);
    win.close();
  });

  report('tabMenuCloseAllActiveLast');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
