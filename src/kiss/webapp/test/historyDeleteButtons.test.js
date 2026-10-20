// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (jsdom) tests for the delete buttons of the CHATS panel:
// every chat header and every finished task row carries a trash button
// that, after the inline "Delete / Cancel" confirmation, asks the
// daemon to delete the chat (``deleteChat``) or the task
// (``deleteTask``) from history.db.  A running task has no delete
// button; Cancel sends nothing.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  click,
  ofType,
} = require('./simplify2_harness');

function makePage() {
  const {win, posted} = makeChatPage({bodyClass: 'remote-chat'});
  send(win, {type: 'daemonStatus', connected: true});
  return {win, posted};
}

function makeSession(overrides) {
  return Object.assign(
    {
      id: 'chat-1',
      task_id: 'task-1',
      title: 'refactor the parser',
      preview: 'refactor the parser',
      has_events: true,
      timestamp: 1700000000,
      work_dir: '/home/user/proj',
      model: 'test-model',
      is_favorite: false,
    },
    overrides || {},
  );
}

function loadHistory(win, sessions) {
  send(win, {type: 'history', offset: 0, generation: 0, sessions});
}

function testTaskRowDelete() {
  const {win, posted} = makePage();
  loadHistory(win, [makeSession()]);
  const row = win.document.querySelector('#history-list .running-item');
  assert.ok(row, 'the task row is rendered');
  const del = row.querySelector('.sidebar-item-actions > .sidebar-item-delete');
  assert.ok(del, 'a finished task row has a delete button');
  assert.strictEqual(del.getAttribute('aria-label'), 'Delete task');
  const confirm = row.querySelector(
    '.sidebar-item-actions > .sidebar-item-confirm',
  );
  assert.strictEqual(
    confirm.style.display,
    'none',
    'the confirmation is hidden',
  );

  click(win, del);
  assert.strictEqual(
    confirm.style.display,
    '',
    'the click shows Delete / Cancel',
  );
  assert.strictEqual(
    ofType(posted, 'deleteTask').length,
    0,
    'nothing sent yet',
  );

  click(win, confirm.querySelector('.sidebar-confirm-no'));
  assert.strictEqual(
    confirm.style.display,
    'none',
    'Cancel hides the confirmation',
  );
  assert.strictEqual(
    ofType(posted, 'deleteTask').length,
    0,
    'Cancel sends nothing',
  );

  click(win, del);
  click(win, confirm.querySelector('.sidebar-confirm-yes'));
  const sent = ofType(posted, 'deleteTask');
  assert.strictEqual(sent.length, 1, 'Delete asks the daemon once');
  assert.strictEqual(sent[0].taskId, 'task-1');
  assert.ok(row.isConnected, 'the row stays until the daemon answers');
  assert.strictEqual(
    ofType(posted, 'resumeSession').length,
    0,
    'none of the clicks opened the task',
  );
  // The daemon deleted the rows and nudged every page: the refetch
  // comes back without the task and the row goes.
  loadHistory(win, []);
  assert.ok(!row.isConnected, 'the refetched history drops the row');
  win.close();
  console.log('  ok - a task row deletes its task after confirmation');
}

function testRunningRowHasNoDelete() {
  const {win} = makePage();
  loadHistory(win, [makeSession({is_running: true})]);
  const row = win.document.querySelector('#history-list .running-item');
  assert.ok(row);
  assert.strictEqual(
    row.querySelector('.sidebar-item-delete'),
    null,
    'a running task cannot be deleted',
  );
  win.close();
  console.log('  ok - a running task row has no delete button');
}

function testChatHeaderDelete() {
  const {win, posted} = makePage();
  loadHistory(win, [
    makeSession(),
    makeSession({task_id: 'task-2', timestamp: 1700000100}),
  ]);
  const group = win.document.querySelector('#history-list .history-chat-group');
  assert.ok(group, 'the chat group is rendered');
  const header = group.querySelector('.history-chat-header');
  const del = header.querySelector(
    '.history-chat-actions > .sidebar-item-delete',
  );
  assert.ok(del, 'the chat header has a delete button');
  assert.strictEqual(del.getAttribute('aria-label'), 'Delete chat');
  const confirm = header.querySelector(
    '.history-chat-actions > .sidebar-item-confirm',
  );

  click(win, del);
  assert.strictEqual(confirm.style.display, '');
  // A click on the confirmation strip itself does not open the chat.
  click(win, confirm);
  assert.strictEqual(ofType(posted, 'resumeSession').length, 0);
  click(win, confirm.querySelector('.sidebar-confirm-yes'));
  const sent = ofType(posted, 'deleteChat');
  assert.strictEqual(sent.length, 1, 'Delete asks the daemon once');
  assert.strictEqual(sent[0].chatId, 'chat-1');
  assert.ok(group.isConnected, 'the group stays until the daemon answers');
  assert.strictEqual(ofType(posted, 'deleteTask').length, 0);
  assert.strictEqual(ofType(posted, 'resumeSession').length, 0);
  loadHistory(win, []);
  assert.ok(!group.isConnected, 'the refetched history drops the group');
  win.close();
  console.log('  ok - a chat header deletes its chat after confirmation');
}

function testChatWithRunningTaskStaysUntilRefresh() {
  const {win, posted} = makePage();
  loadHistory(win, [
    makeSession(),
    makeSession({task_id: 'task-2', timestamp: 1700000100, is_running: true}),
  ]);
  const group = win.document.querySelector('#history-list .history-chat-group');
  const header = group.querySelector('.history-chat-header');
  click(win, header.querySelector('.sidebar-item-delete'));
  click(win, header.querySelector('.sidebar-confirm-yes'));
  assert.strictEqual(ofType(posted, 'deleteChat').length, 1);
  assert.ok(
    group.isConnected,
    'a chat with a running task stays: the daemon refuses with a notice',
  );
  send(win, {
    type: 'notice',
    text: 'Cannot delete the chat: its task is still running. Stop it first.',
  });
  assert.ok(group.isConnected, 'the notice does not remove the group');
  assert.ok(
    win.document.querySelector('#history-list .history-chat-group'),
    'the chat is still listed',
  );
  win.close();
  console.log(
    '  ok - a chat with a running task is not removed optimistically',
  );
}

function testRemoteChatHeaderHasNoDelete() {
  const {win} = makePage();
  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    machineLabels: {'10.0.0.9': 'box'},
    sessions: [makeSession({machine: '10.0.0.9'})],
  });
  const header = win.document.querySelector(
    '#history-list .history-chat-header',
  );
  assert.ok(header, 'the remote chat is listed');
  const del = header.querySelector(
    '.history-chat-actions > .sidebar-item-delete',
  );
  assert.ok(
    !del || del.style.display === 'none',
    'a chat that lives on another machine has no delete button',
  );
  win.close();
  console.log('  ok - a remote-only chat header has no delete button');
}

testTaskRowDelete();
testRunningRowHasNoDelete();
testChatHeaderDelete();
testChatWithRunningTaskStaysUntilRefresh();
testRemoteChatHeaderHasNoDelete();
console.log('historyDeleteButtons.test.js passed');
