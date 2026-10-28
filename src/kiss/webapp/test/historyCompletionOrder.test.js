// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The CHATS panel sorts by completion time: chats by the completion of
// their latest-finished task, flat-view rows by their own completion.
// A row without an endTs (running or legacy) sorts by its launch
// timestamp.  Runs with: node test/historyCompletionOrder.test.js

const assert = require('assert');
const {makePage: makeChatPage} = require('./simplify2_harness');
const {
  sendHistory,
  makeRow,
  localIsoDate: localIso,
} = require('./history_helpers');

function makePage(legacy) {
  const {win, posted} = makeChatPage({
    beforeScripts(win) {
      if (legacy) win.localStorage.setItem('kissSorcar.historyLegacyView', '1');
    },
  });
  win.document.getElementById('sidebar').classList.add('open');
  return {win, posted};
}

const DAY = 86_400;
const T0 = 1_700_000_000;

function row(chat, taskId, launched, endSec, running) {
  return makeRow({
    id: chat,
    task_id: taskId,
    title: 'task ' + taskId,
    timestamp: launched,
    startTs: launched * 1000,
    endTs: endSec ? endSec * 1000 : 0,
    is_running: !!running,
  });
}

// Chat A's only task launched first but finished last; chat B's last
// task launched last but finished early; chat C's task has no endTs.
const SESSIONS = [
  row('B', 'b2', T0 + 3 * DAY, T0 + 3 * DAY + 60),
  row('C', 'c1', T0 + 4 * DAY, 0),
  row('B', 'b1', T0, T0 + 60),
  row('A', 'a1', T0 + DAY, T0 + 5 * DAY),
  row('D', 'd1', T0 - DAY, 0, true),
];

function testChatsSortByLastCompletion() {
  const {win, posted} = makePage(false);
  sendHistory(win, posted, SESSIONS);
  const items = Array.from(
    win.document.querySelectorAll(
      '#history-list > .history-chat-group, #history-list > .history-day-sep',
    ),
  ).map(el => el.dataset.chatId || 'sep:' + el.dataset.day);
  const chats = items.filter(x => !x.startsWith('sep:'));
  assert.deepStrictEqual(
    chats,
    ['D', 'A', 'C', 'B'],
    'running, then by completion',
  );
  // A sits under the day it completed, not the day it launched.
  const aSep = items[items.indexOf('A') - 1];
  const dayA = new Date((T0 + 5 * DAY) * 1000);
  assert.strictEqual(
    aSep,
    'sep:' +
      dayA.getFullYear() +
      '-' +
      (dayA.getMonth() + 1) +
      '-' +
      dayA.getDate(),
  );
  const bRows = Array.from(
    win.document.querySelectorAll(
      '.history-chat-group[data-chat-id="B"] .sidebar-item-text',
    ),
  ).map(el => el.textContent);
  assert.deepStrictEqual(bRows, ['task b2', 'task b1']);
  win.close();
}

function testFlatRowsSortByCompletion() {
  const {win, posted} = makePage(true);
  sendHistory(win, posted, SESSIONS);
  const rows = Array.from(
    win.document.querySelectorAll(
      '#history-list > .sidebar-item .sidebar-item-text',
    ),
  ).map(el => el.textContent);
  assert.deepStrictEqual(rows, [
    'task d1',
    'task a1',
    'task c1',
    'task b2',
    'task b1',
  ]);
  win.close();
}

function visibleFlatRows(win) {
  return Array.from(
    win.document.querySelectorAll('#history-list > .sidebar-item'),
  )
    .filter(el => el.style.display !== 'none')
    .map(el => el.querySelector('.sidebar-item-text').textContent);
}

function setDate(win, id, sec) {
  const input = win.document.getElementById(id);
  input.value = sec === null ? '' : localIso(sec);
  input.dispatchEvent(new win.Event('change', {bubbles: true}));
}

function testDateFilterUsesCompletion() {
  const {win, posted} = makePage(true);
  sendHistory(win, posted, SESSIONS);
  // a1 launched on day 1 but finished on day 5: From=day 5 keeps it.
  // c1 (no endTs) counts at its day-4 launch; d1 (running) at day -1.
  setDate(win, 'hf-from', T0 + 5 * DAY);
  assert.deepStrictEqual(visibleFlatRows(win), ['task a1']);
  // To=day 1 drops a1 (finished day 5) and keeps the day-0 tasks.
  setDate(win, 'hf-from', null);
  setDate(win, 'hf-to', T0 + DAY);
  assert.deepStrictEqual(visibleFlatRows(win), ['task d1', 'task b1']);
  win.close();
}

function testAgoLabelUsesCompletion() {
  const {win, posted} = makePage(true);
  const now = Math.floor(Date.now() / 1000);
  const HOUR = 3600;
  sendHistory(win, posted, [
    row('R', 'running', now - 3 * HOUR, 0, true),
    row('F', 'finished', now - 3 * DAY, now - 2 * HOUR),
    // A corrupt completion time falls back to the launch.
    Object.assign(row('X', 'corrupt', now - 5 * HOUR, 0), {endTs: 9e15}),
  ]);
  const labels = {};
  win.document.querySelectorAll('#history-list > .sidebar-item').forEach(el => {
    const ago = el.querySelector('.sidebar-item-launched');
    labels[el.querySelector('.sidebar-item-text').textContent] = [
      ago.textContent,
      ago.title.split(' ')[0],
    ];
  });
  assert.deepStrictEqual(labels, {
    'task running': ['3 hours ago', 'Launched'],
    'task finished': ['2 hours ago', 'Finished'],
    'task corrupt': ['5 hours ago', 'Launched'],
  });
  // The date filter agrees: the corrupt row is dated by its launch.
  setDate(win, 'hf-from', now - 5 * HOUR);
  assert.ok(visibleFlatRows(win).includes('task corrupt'));
  win.close();
}

testChatsSortByLastCompletion();
testFlatRowsSortByCompletion();
testDateFilterUsesCompletion();
testAgoLabelUsesCompletion();
console.log('historyCompletionOrder: all tests passed');
