// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for the history list's ACTIVE-TASK
// highlight in media/main.js: the row of the task the visible tab
// shows carries `.history-active-task`, its chat panel carries
// `.history-active-chat`, and the row (or, while the panel is folded,
// the chat's header) is scrolled into view.  No panel ever unfolds on
// its own: folding is the user's alone.  The task the page shows is
// read from the visible tab itself: its task_settings event and its
// chat binding (tabs_state), re-read on every tab switch.
//
// Covered behavior:
//  - the visible tab's task highlights the row of that very task and
//    its chat panel, leaves every panel folded, and scrolls the chat's
//    header into view once;
//  - a tab bound to a chat that names no task yet falls back to the
//    chat's newest row;
//  - a task shown BEFORE the history lands is applied (and scrolled
//    to) when the rows arrive;
//  - the identical-refresh fast path and a changed-data rebuild both
//    keep the highlight; the rebuild scrolls again (the emptied list
//    lands at the top);
//  - a named task without a loaded row highlights no row (another
//    task must not pass for it) but still highlights its chat;
//  - a panel the user unfolded stays unfolded across moves away and
//    back and across rebuilds (the row is the scroll target); a panel
//    the user folded stays folded (the header stands in);
//  - a running task does not unfold its chat's panel;
//  - a row the filters hide is marked but its scroll waits for the
//    filter change that shows it;
//  - a fresh tab (no chat, no task) clears both highlights;
//  - the page paints its in-page list from its own tabs (task_settings,
//    tab switches) and never posts `activeTask`.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  fakeMatchMedia,
} = require('./simplify2_harness');

function makePage() {
  // jsdom has no scrollIntoView: record every call and its target.
  const scrolled = [];
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js'],
    beforeScripts(win) {
      win.Element.prototype.scrollIntoView = function () {
        scrolled.push(this);
      };
      stubRaf(win);
      fakeMatchMedia(win, false);
    },
  });
  return {win, posted, scrolled};
}

function byType(posted, type) {
  return posted.filter(m => m && m.type === type);
}

function lastMessage(posted, type) {
  const msgs = byType(posted, type);
  return msgs.length ? msgs[msgs.length - 1] : null;
}

function sendEmptyWorkDir(win) {
  send(win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
}

/** A booted page with the Chats drawer open (which asks for history). */
function boot() {
  const made = makePage();
  sendEmptyWorkDir(made.win);
  made.win.document.getElementById('menu-btn').click();
  return made;
}

/**
 * The visible tab shows task *taskId* of chat *chatId*: the daemon's
 * task_settings event for the tab, as a replay or a live task sends it.
 */
function showTask(win, chatId, taskId) {
  send(win, {
    type: 'task_settings',
    tabId: win._testApi.getActiveTabId(),
    taskId: String(taskId),
    settings: {model: 'm', chat_id: chatId, task_id: taskId, start_ts: 1},
  });
}

/**
 * Bring a tab bound to *chatId* that names no task yet on screen: the
 * daemon's tabs_state binds a second tab to the chat, and the Chats
 * panel's pick switches to it.
 */
function showChatWithoutTask(win, chatId) {
  const tabId = 'bound-' + chatId;
  const tabs = Array.from(win._testApi.openTabs(), t => ({
    tabId: t.id,
    chatId: t.backendChatId || '',
    title: t.title || '',
    workDir: '',
  }));
  tabs.push({tabId, chatId, title: chatId, workDir: ''});
  send(win, {type: 'tabs_state', tabs});
  win._testApi.switchToTab(tabId);
  assert.strictEqual(win._testApi.getActiveTabId(), tabId);
}

function session(overrides) {
  return Object.assign(
    {
      id: 'chat-1',
      task_id: 1,
      title: 'Refactor the payment gateway retries',
      preview: 'Refactor the payment gateway retries',
      has_events: true,
      failed: false,
      is_running: false,
      tokens: 0,
      cost: 0,
      steps: 0,
      is_favorite: false,
      timestamp: 1_700_000_000,
      work_dir: '',
    },
    overrides || {},
  );
}

// Two chats, three tasks, newest first as the daemon sends them.
function threeSessions() {
  return [
    session({
      id: 'chat-2',
      task_id: 3,
      title: 'chat two',
      preview: 'chat two',
      timestamp: 1_700_000_300,
    }),
    session({
      id: 'chat-1',
      task_id: 2,
      title: 'chat one, second',
      preview: 'chat one, second',
      timestamp: 1_700_000_200,
    }),
    session({
      id: 'chat-1',
      task_id: 1,
      title: 'chat one, first',
      preview: 'chat one, first',
      timestamp: 1_700_000_100,
    }),
  ];
}

// renderHistory drops replies whose generation is stale, so always
// answer the generation the page last asked for.
function sendHistory(win, posted, sessions) {
  const req = lastMessage(posted, 'getHistory');
  assert.ok(req, 'the page must have requested history');
  send(win, {
    type: 'history',
    offset: 0,
    generation: req.generation,
    sessions,
  });
}

function rows(win) {
  return Array.from(
    win.document.querySelectorAll('#history-list .running-item'),
  );
}

// Rows carry no ids of their own: they sit under their chat's group in
// the daemon's order (newest first), the same order as threeSessions().
function rowFor(win, chatId, taskId) {
  const ownTasks = threeSessions()
    .filter(s => s.id === chatId)
    .map(s => s.task_id);
  const idx = ownTasks.indexOf(taskId);
  const row = group(win, chatId).querySelectorAll('.running-item')[idx];
  assert.ok(idx >= 0 && row, `row for ${chatId}/${taskId} must exist`);
  return row;
}

function highlighted(win) {
  return rows(win).filter(r => r.classList.contains('history-active-task'));
}

function group(win, chatId) {
  const g = win.document.querySelector(
    `#history-list .history-chat-group[data-chat-id="${chatId}"]`,
  );
  assert.ok(g, `group for ${chatId} must exist`);
  return g;
}

function collapsed(win, chatId) {
  return group(win, chatId).classList.contains('collapsed');
}

function activeChat(win, chatId) {
  return group(win, chatId).classList.contains('history-active-chat');
}

function header(win, chatId) {
  return group(win, chatId).querySelector(':scope > .history-chat-header');
}

function chevron(win, chatId) {
  return header(win, chatId).querySelector('.history-chat-collapse');
}

function clickChevron(win, chatId) {
  chevron(win, chatId).dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
}

function testVisibleTaskHighlightsAndScrolls() {
  const {win, posted, scrolled} = boot();
  sendHistory(win, posted, threeSessions());
  assert.strictEqual(highlighted(win).length, 0, 'nothing highlighted yet');
  assert.ok(
    !activeChat(win, 'chat-1') && !activeChat(win, 'chat-2'),
    'no chat highlighted yet',
  );
  assert.ok(
    collapsed(win, 'chat-1') && collapsed(win, 'chat-2'),
    'idle chats start collapsed',
  );
  scrolled.length = 0;

  showTask(win, 'chat-1', 2);
  const row = rowFor(win, 'chat-1', 2);
  assert.deepStrictEqual(
    highlighted(win),
    [row],
    'only the shown task is highlighted',
  );
  assert.ok(activeChat(win, 'chat-1'), 'the shown chat is highlighted');
  assert.ok(!activeChat(win, 'chat-2'), 'the other chat is not');
  assert.ok(
    collapsed(win, 'chat-1') && collapsed(win, 'chat-2'),
    'the shown chat does not unfold on its own',
  );
  assert.deepStrictEqual(
    scrolled,
    [header(win, 'chat-1')],
    'the folded chat\u2019s header is scrolled into view once',
  );

  // The same settings again change nothing and scroll nothing.
  showTask(win, 'chat-1', 2);
  assert.strictEqual(
    scrolled.length,
    1,
    'a repeated repaint does not scroll again',
  );

  // A named task without a loaded row: no other row passes for it,
  // but its chat is highlighted and the previous chat is not.
  showTask(win, 'chat-2', 999);
  assert.strictEqual(
    highlighted(win).length,
    0,
    'no stand-in for a named task',
  );
  assert.ok(activeChat(win, 'chat-2'), 'the shown chat is highlighted');
  assert.ok(!activeChat(win, 'chat-1'), 'the previous chat no longer');
  assert.ok(
    collapsed(win, 'chat-1') && collapsed(win, 'chat-2'),
    'both chats stay collapsed',
  );
  assert.strictEqual(scrolled.length, 1, 'nothing to scroll to');

  // A chat that names no task yet: its newest row stands in.
  showChatWithoutTask(win, 'chat-2');
  const newest = rowFor(win, 'chat-2', 3);
  assert.deepStrictEqual(highlighted(win), [newest]);
  assert.deepStrictEqual(scrolled.slice(1), [header(win, 'chat-2')]);

  // A fresh tab (no chat, no task) clears both highlights.
  win._testApi.createNewTab();
  assert.strictEqual(highlighted(win).length, 0);
  assert.ok(!activeChat(win, 'chat-1') && !activeChat(win, 'chat-2'));
  assert.ok(collapsed(win, 'chat-2'), 'still collapsed');
  assert.strictEqual(scrolled.length, 2, 'nothing to scroll to');
  win.close();
  console.log(
    'PASS the visible task highlights its row and chat and scrolls once',
  );
}

function testTaskBeforeHistoryAppliesOnRender() {
  const {win, posted, scrolled} = boot();
  showTask(win, 'chat-1', 1);
  assert.strictEqual(rows(win).length, 0, 'no rows yet');
  scrolled.length = 0;
  sendHistory(win, posted, threeSessions());
  const row = rowFor(win, 'chat-1', 1);
  assert.deepStrictEqual(highlighted(win), [row]);
  assert.ok(activeChat(win, 'chat-1') && !activeChat(win, 'chat-2'));
  assert.ok(collapsed(win, 'chat-1'), 'the panel is still folded');
  assert.deepStrictEqual(
    scrolled,
    [header(win, 'chat-1')],
    'the pending scroll lands with the rows, on the folded chat\u2019s header',
  );
  win.close();
  console.log('PASS a task shown ahead of the history is applied on render');
}

function testRefreshAndRebuildKeepHighlight() {
  const {win, posted, scrolled} = boot();
  sendHistory(win, posted, threeSessions());
  showTask(win, 'chat-1', 1);
  // The user unfolds the chat: the row itself becomes the scroll
  // target from here on.
  clickChevron(win, 'chat-1');
  assert.ok(!collapsed(win, 'chat-1'));
  const before = rowFor(win, 'chat-1', 1);
  scrolled.length = 0;

  // Identical data: the fast path keeps the very same row elements.
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, threeSessions());
  assert.strictEqual(
    rowFor(win, 'chat-1', 1),
    before,
    'fast path kept the row',
  );
  assert.deepStrictEqual(highlighted(win), [before]);
  assert.ok(activeChat(win, 'chat-1'), 'the chat highlight is kept');
  assert.strictEqual(
    scrolled.length,
    0,
    'an identical refresh does not scroll',
  );

  // Changed data: the list is rebuilt (scroll offset lost), so the
  // fresh row is highlighted and scrolled to again.
  const changed = threeSessions();
  changed[0].tokens = 4242;
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, changed);
  const after = rowFor(win, 'chat-1', 1);
  assert.notStrictEqual(after, before, 'the row was rebuilt');
  assert.deepStrictEqual(highlighted(win), [after]);
  assert.ok(activeChat(win, 'chat-1') && !activeChat(win, 'chat-2'));
  assert.ok(!collapsed(win, 'chat-1'), 'the user\u2019s unfold survives');
  assert.deepStrictEqual(
    scrolled,
    [after],
    'a rebuild scrolls the fresh row into view',
  );
  win.close();
  console.log('PASS refreshes and rebuilds keep the highlight');
}

function testFoldIsTheUsersAlone() {
  const {win, posted, scrolled} = boot();
  sendHistory(win, posted, threeSessions());
  showTask(win, 'chat-1', 2);
  assert.ok(collapsed(win, 'chat-1'), 'folded while on screen');

  // The user unfolds the chat they look at.
  clickChevron(win, 'chat-1');
  assert.ok(!collapsed(win, 'chat-1'), 'the user unfolded the chat');

  // Moving to another chat and back leaves the unfold alone; the
  // row itself is the scroll target.
  showTask(win, 'chat-2', 3);
  assert.ok(!collapsed(win, 'chat-1'), 'moving away keeps the unfold');
  assert.ok(collapsed(win, 'chat-2'), 'the chat moved to stays folded');
  assert.ok(activeChat(win, 'chat-2') && !activeChat(win, 'chat-1'));
  scrolled.length = 0;
  showTask(win, 'chat-1', 1);
  assert.ok(!collapsed(win, 'chat-1'), 'moving back keeps the unfold');
  const row = rowFor(win, 'chat-1', 1);
  assert.deepStrictEqual(highlighted(win), [row]);
  assert.ok(activeChat(win, 'chat-1') && !activeChat(win, 'chat-2'));
  assert.deepStrictEqual(scrolled, [row], 'the row itself is scrolled to');

  // The user folds it again while looking at it: a rebuild keeps the
  // fold, and the rebuild's scroll lands on the header, the
  // highlighted row being hidden inside.
  clickChevron(win, 'chat-1');
  assert.ok(collapsed(win, 'chat-1'), 'the user folded the chat');
  scrolled.length = 0;
  const changed = threeSessions();
  changed[2].cost = 1.5;
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, changed);
  assert.ok(collapsed(win, 'chat-1'), 'the fold survives a rebuild');
  assert.deepStrictEqual(highlighted(win), [rowFor(win, 'chat-1', 1)]);
  assert.ok(activeChat(win, 'chat-1'));
  assert.deepStrictEqual(
    scrolled,
    [header(win, 'chat-1')],
    'the header stands in for the hidden row',
  );

  // Moving away and back no longer unfolds it either.
  showTask(win, 'chat-2', 3);
  showTask(win, 'chat-1', 2);
  assert.ok(collapsed(win, 'chat-1'), 'moving back keeps the fold');
  assert.deepStrictEqual(highlighted(win), [rowFor(win, 'chat-1', 2)]);
  win.close();
  console.log('PASS folding and unfolding a chat is the user\u2019s alone');
}

function testRunningTaskDoesNotUnfold() {
  const {win, posted} = boot();
  showTask(win, 'chat-1', 2);
  const running = threeSessions();
  running[1].is_running = true;
  sendHistory(win, posted, running);
  assert.strictEqual(
    group(win, 'chat-1').dataset.hasRunning,
    '1',
    'the chat is marked running for the Running section',
  );
  assert.ok(collapsed(win, 'chat-1'), 'a task starting does not unfold');
  assert.ok(activeChat(win, 'chat-1'), 'the chat is highlighted');
  assert.deepStrictEqual(highlighted(win), [rowFor(win, 'chat-1', 2)]);

  // The user unfolds it; the task finishing (a rebuild) keeps that.
  clickChevron(win, 'chat-1');
  assert.ok(!collapsed(win, 'chat-1'));
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, threeSessions());
  assert.ok(!collapsed(win, 'chat-1'), 'the user\u2019s unfold stands');
  assert.ok(collapsed(win, 'chat-2'), 'the other chat is folded');
  win.close();
  console.log('PASS a running task leaves the fold to the user');
}

function testFilteredRowWaitsForTheFilter() {
  const {win, posted, scrolled} = boot();
  sendHistory(win, posted, threeSessions());
  // The user unfolds the chat so the row (not the header) is the
  // scroll target.
  clickChevron(win, 'chat-1');
  const completedBox = win.document.getElementById('hf-completed');
  completedBox.checked = false;
  completedBox.dispatchEvent(new win.Event('change', {bubbles: true}));
  const row = rowFor(win, 'chat-1', 1);
  assert.strictEqual(
    row.style.display,
    'none',
    'the filter hides completed tasks',
  );

  scrolled.length = 0;
  showTask(win, 'chat-1', 1);
  assert.deepStrictEqual(highlighted(win), [row], 'the hidden row is marked');
  assert.strictEqual(scrolled.length, 0, 'a hidden row is not scrolled to');

  // Showing completed tasks again lands the pending scroll.
  completedBox.checked = true;
  completedBox.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.strictEqual(row.style.display, '');
  assert.deepStrictEqual(scrolled, [row], 'the scroll waited for the filter');
  win.close();
  console.log(
    'PASS a filtered-out row is scrolled to once the filter shows it',
  );
}

function testPagePaintsFromOwnTabs() {
  const {win, posted, scrolled} = makePage();
  sendEmptyWorkDir(win);
  const mine = byType(posted, 'ready')[0].tabId;
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: mine, chatId: 'chat-1', title: 'one', workDir: ''},
      {tabId: 'other-1', chatId: 'chat-2', title: 'two', workDir: ''},
    ],
  });
  win.document.getElementById('menu-btn').click();
  sendHistory(win, posted, threeSessions());
  // The visible tab is bound to chat-1 but names no task yet: the
  // chat's newest row stands in.
  assert.deepStrictEqual(highlighted(win), [rowFor(win, 'chat-1', 2)]);

  send(win, {
    type: 'task_settings',
    tabId: mine,
    taskId: '1',
    settings: {model: 'm', chat_id: 'chat-1', task_id: 1, start_ts: 1},
  });
  const own = rowFor(win, 'chat-1', 1);
  assert.deepStrictEqual(
    highlighted(win),
    [own],
    'the tab\u2019s own task takes over',
  );
  assert.ok(
    scrolled.includes(header(win, 'chat-1')),
    'the folded chat\u2019s header is scrolled into view',
  );
  assert.ok(activeChat(win, 'chat-1') && !activeChat(win, 'chat-2'));
  assert.ok(collapsed(win, 'chat-1') && collapsed(win, 'chat-2'));

  // Switching to the other chat (the Chats panel's pick) follows it.
  win._testApi.switchToTab('other-1');
  assert.deepStrictEqual(highlighted(win), [rowFor(win, 'chat-2', 3)]);
  assert.ok(activeChat(win, 'chat-2') && !activeChat(win, 'chat-1'));
  assert.ok(collapsed(win, 'chat-1') && collapsed(win, 'chat-2'));

  assert.strictEqual(
    byType(posted, 'activeTask').length,
    0,
    'the page never posts activeTask (its postMessage is the daemon)',
  );
  win.close();
  console.log('PASS the page paints its in-page list from its own tabs');
}

function main() {
  testVisibleTaskHighlightsAndScrolls();
  testTaskBeforeHistoryAppliesOnRender();
  testRefreshAndRebuildKeepHighlight();
  testFoldIsTheUsersAlone();
  testRunningTaskDoesNotUnfold();
  testFilteredRowWaitsForTheFilter();
  testPagePaintsFromOwnTabs();
  console.log('historyActiveTaskHighlight: all tests passed');
}

try {
  main();
} catch (err) {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
}
