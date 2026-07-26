// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function tabIds(win) {
  return Array.from(
    win.document.querySelectorAll('.chat-tab:not(.chat-tab-add)'),
  )
    .map((el) => el.dataset.tabId)
    .filter(Boolean);
}

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

function testDeleteCurrentTaskClosesTab() {
  const {win, posted} = makeWebview();
  const tab1 = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  loadChat(win, tab1, 'chat-A', '42', 'Task forty-two');
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
  assert.ok(
    posted.some((m) => m.type === 'closeTab' && m.tabId === tab1),
    'closeTab must be posted to the extension host for the closed tab',
  );
  win.close();
  console.log('PASS deleting the current task closes its tab');
}

function testChatEmptyClosesTabWithDifferentCurrentTask() {
  const {win, posted} = makeWebview();
  const tab1 = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  loadChat(win, tab1, 'chat-A', '42', 'Task forty-two');
  win._demoApi.createNewTab();
  const tab2 = win._demoApi.getActiveTabId();
  loadChat(win, tab2, 'chat-B', '77', 'Task seventy-seven');

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

function testInactiveTabFragmentPruned() {
  const {win, posted} = makeWebview();
  const tab1 = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  loadChat(win, tab1, 'chat-A', '42', 'Task forty-two');
  const O = win.document.getElementById('output');
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
  win._demoApi.createNewTab();
  const tab2 = win._demoApi.getActiveTabId();
  loadChat(win, tab2, 'chat-B', '77', 'Task seventy-seven');
  assert.strictEqual(win._demoApi.getActiveTabId(), tab2);

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

function testLiveTaskLifecycleThenDelete() {
  const {win, posted} = makeWebview();
  const tabId = posted.find((m) => m.type === 'ready').tabId;
  win._demoApi.hideWelcome();
  const O = win.document.getElementById('output');
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
