// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end jsdom tests: when a task is deleted from the task history
// panel, any open tab (on any client) whose chat webview shows the
// deleted task's chat MUST remove the task and its chat from the tab.
//
// The backend broadcasts
//   {type: 'taskDeleted', taskId, chatId, chatHasMoreTasks}
// to EVERY connected client (see WebPrinter.broadcast in
// web_server.py — 'taskDeleted' is a GLOBAL system broadcast, never a
// per-task fan-out).  These tests exercise the real ``media/main.js``
// against the real ``media/chat.html`` in jsdom (the same harness as
// ``adjacentTaskScroll.test.js``) and lock the required webview
// behaviour:
//
//   1. A tab whose current (header) task IS the deleted task is
//      closed entirely.
//   2. ``chatHasMoreTasks: false`` closes the tab even when the tab's
//      current task differs from the deleted one (the whole chat is
//      gone from the DB).
//   3. An INACTIVE (background) tab showing the deleted task's chat
//      has the task's ``.adjacent-task`` block pruned from its saved
//      ``outputFragment`` and is closed when its current task was the
//      deleted one.
//   4. A tab bound to a DIFFERENT chat is untouched.
//   5. Full live lifecycle: submit ('clear' with chat_id) → streamed
//      events adopt the real taskId → task_done → taskDeleted for
//      that task closes the tab (and, being the last tab, a fresh
//      empty tab replaces it).
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/taskDeletedRemovesChat.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom webview running the real chat.html + panelCopy.js +
 * main.js, with a stubbed acquireVsCodeApi that records every message
 * posted to the extension host.
 */
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
  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: (msg) => posted.push(msg),
      getState: () => state,
      setState: (s) => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

/** Deliver a message-event from the extension host to the webview. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Tab ids currently in the webview's tab bar (chat tabs only). */
function tabIds(win) {
  return Array.from(
    win.document.querySelectorAll('.chat-tab:not(.chat-tab-add)'),
  )
    .map((el) => el.dataset.tabId)
    .filter(Boolean);
}

/** Load a persisted chat (history replay) into tab `tabId`. */
function loadChat(win, tabId, chatId, taskId, title) {
  send(win, {
    type: 'task_events',
    tabId,
    chat_id: chatId,
    task_id: taskId,
    task: title,
    events: [
      {type: 'task_start', task: title},
      {type: 'system_output', text: 'hello from ' + taskId + '\n'},
    ],
  });
}

// ---------------------------------------------------------------------------
// 1. Deleting the tab's current (header) task closes the tab, removing
//    the task and its chat from the webview.
// ---------------------------------------------------------------------------
function testDeleteCurrentTaskClosesTab() {
  const {win, posted} = makeWebview();
  const tab1 = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  loadChat(win, tab1, 'chat-A', '42', 'Task forty-two');
  // Second tab showing another chat so tab1's closure is observable
  // (closing the LAST tab auto-creates a fresh one).
  win._demoApi.createNewTab();
  const tab2 = win._demoApi.getActiveTabId();
  assert.ok(tab2 && tab2 !== tab1, 'second tab must exist and be active');
  loadChat(win, tab2, 'chat-B', '77', 'Task seventy-seven');

  assert.ok(tabIds(win).includes(tab1), 'precondition: tab1 open');
  send(win, {
    type: 'taskDeleted',
    chatId: 'chat-A',
    taskId: '42',
    chatHasMoreTasks: true,
  });
  assert.ok(
    !tabIds(win).includes(tab1),
    'tab whose current task was deleted must be closed',
  );
  assert.ok(
    tabIds(win).includes(tab2),
    'the unrelated tab (different chat) must stay open',
  );
  // The backend was told the tab is gone.
  assert.ok(
    posted.some((m) => m.type === 'closeTab' && m.tabId === tab1),
    'closeTab must be posted to the extension host for the closed tab',
  );
  win.close();
  console.log('PASS deleting the current task closes its tab');
}

// ---------------------------------------------------------------------------
// 2. chatHasMoreTasks:false closes the tab even when its current task
//    differs from the deleted one — the whole chat is gone.
// ---------------------------------------------------------------------------
function testChatEmptyClosesTabWithDifferentCurrentTask() {
  const {win, posted} = makeWebview();
  const tab1 = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  loadChat(win, tab1, 'chat-A', '42', 'Task forty-two');
  win._demoApi.createNewTab();
  const tab2 = win._demoApi.getActiveTabId();
  loadChat(win, tab2, 'chat-B', '77', 'Task seventy-seven');

  // Delete a DIFFERENT task ('41') of chat-A, and the chat is now
  // empty in the DB (e.g. '42' was deleted moments earlier by another
  // client) — the tab showing chat-A must close.
  send(win, {
    type: 'taskDeleted',
    chatId: 'chat-A',
    taskId: '41',
    chatHasMoreTasks: false,
  });
  assert.ok(
    !tabIds(win).includes(tab1),
    'tab must close when its chat has no tasks left in the DB',
  );
  assert.ok(tabIds(win).includes(tab2), 'unrelated tab must survive');
  win.close();
  console.log('PASS chatHasMoreTasks:false closes the chat tab');
}

// ---------------------------------------------------------------------------
// 3. INACTIVE tab: the deleted task's .adjacent-task block is pruned
//    from the tab's saved outputFragment; the tab closes only when the
//    deleted task is its current task.
// ---------------------------------------------------------------------------
function testInactiveTabFragmentPruned() {
  const {win, posted} = makeWebview();
  const tab1 = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  loadChat(win, tab1, 'chat-A', '42', 'Task forty-two');
  const O = win.document.getElementById('output');
  // Render the previous task '41' of chat-A above '42' (adjacent
  // overscroll reply) so an .adjacent-task[data-task-id="41"] block
  // exists in tab1's DOM.
  send(win, {
    type: 'adjacent_task_events',
    tabId: tab1,
    direction: 'prev',
    task: 'Task forty-one',
    task_id: '41',
    events: [
      {type: 'task_start', task: 'Task forty-one'},
      {type: 'system_output', text: 'older\n'},
    ],
  });
  assert.ok(
    O.querySelector('.adjacent-task[data-task-id="41"]'),
    'precondition: adjacent task 41 rendered in tab1',
  );
  // Switch away: tab1 becomes INACTIVE, its DOM moves into
  // tab.outputFragment.
  win._demoApi.createNewTab();
  const tab2 = win._demoApi.getActiveTabId();
  loadChat(win, tab2, 'chat-B', '77', 'Task seventy-seven');
  assert.strictEqual(win._demoApi.getActiveTabId(), tab2);

  // Another client deletes task 41 (NOT tab1's current task '42').
  send(win, {
    type: 'taskDeleted',
    chatId: 'chat-A',
    taskId: '41',
    chatHasMoreTasks: true,
  });
  assert.ok(
    tabIds(win).includes(tab1),
    'tab1 must stay open (its current task 42 still exists)',
  );
  // Re-activate tab1 (click its tab-bar element) and verify block 41
  // is gone but task 42 remains.
  const tab1El = win.document.querySelector(
    '.chat-tab[data-tab-id="' + tab1 + '"]',
  );
  assert.ok(tab1El, 'tab1 element must exist in the tab bar');
  tab1El.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
  assert.strictEqual(win._demoApi.getActiveTabId(), tab1);
  assert.strictEqual(
    O.querySelector('.adjacent-task[data-task-id="41"]'),
    null,
    "the deleted task's block must be pruned from the inactive tab's " +
      'outputFragment',
  );
  assert.ok(
    O.textContent.includes('hello from 42'),
    "the surviving task's content must remain in the tab",
  );
  win.close();
  console.log("PASS inactive tab's fragment pruned of the deleted task");
}

// ---------------------------------------------------------------------------
// 4. A tab bound to a DIFFERENT chat is untouched even when its current
//    task id NUMERICALLY matches the deleted id (ids are per-DB-global,
//    but the chat filter must gate first).
// ---------------------------------------------------------------------------
function testUnrelatedChatUntouched() {
  const {win, posted} = makeWebview();
  const tab1 = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  loadChat(win, tab1, 'chat-B', '42', 'Unrelated chat task');
  send(win, {
    type: 'taskDeleted',
    chatId: 'chat-A',
    taskId: '42',
    chatHasMoreTasks: false,
  });
  assert.ok(
    tabIds(win).includes(tab1),
    'a tab bound to a different chat must not be closed',
  );
  const O = win.document.getElementById('output');
  assert.ok(
    O.textContent.includes('hello from 42'),
    "the unrelated tab's content must be untouched",
  );
  win.close();
  console.log('PASS unrelated chat tab untouched');
}

// ---------------------------------------------------------------------------
// 5. Full live lifecycle: submit → 'clear' binds chat_id → streamed
//    events adopt the real taskId → task_done → taskDeleted closes the
//    tab; as the LAST tab, a fresh empty tab replaces it (no stale
//    chat content left anywhere).
// ---------------------------------------------------------------------------
function testLiveTaskLifecycleThenDelete() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  const O = win.document.getElementById('output');
  // Live submit flow exactly as broadcast by the backend
  // (commands.py _cmd_run → 'clear' with chat_id, then the stream).
  send(win, {type: 'setTaskText', text: 'My live task', tabId});
  send(win, {type: 'clear', chat_id: 'chat-live', tabId});
  send(win, {type: 'status', running: true, tabId});
  send(win, {type: 'system_prompt', text: 'sys', tabId, taskId: '123'});
  send(win, {type: 'prompt', text: 'My live task', tabId, taskId: '123'});
  send(win, {type: 'system_output', text: 'working\n', tabId, taskId: '123'});
  send(win, {type: 'taskExecuted', tabId, taskId: '123'});
  send(win, {type: 'task_done', tabId, taskId: '123'});
  send(win, {type: 'status', running: false, tabId});
  assert.ok(
    O.textContent.includes('working'),
    'precondition: live task output rendered',
  );

  // The user deletes the just-finished task from the History panel
  // (it was the only task of its chat).
  send(win, {
    type: 'taskDeleted',
    chatId: 'chat-live',
    taskId: '123',
    chatHasMoreTasks: false,
  });
  assert.ok(
    !tabIds(win).includes(tabId),
    'the live tab showing the deleted task must be closed',
  );
  // Closing the last tab creates a fresh empty tab — the deleted chat
  // content must not survive anywhere in the webview.
  assert.ok(
    !O.textContent.includes('working'),
    'no stale chat content of the deleted task may remain',
  );
  assert.ok(
    posted.some((m) => m.type === 'closeTab' && m.tabId === tabId),
    'closeTab must be posted to the extension host',
  );
  win.close();
  console.log('PASS live lifecycle then delete closes the tab');
}

testDeleteCurrentTaskClosesTab();
testChatEmptyClosesTabWithDifferentCurrentTask();
testInactiveTabFragmentPruned();
testUnrelatedChatUntouched();
testLiveTaskLifecycleThenDelete();
console.log('All taskDeletedRemovesChat tests passed');
