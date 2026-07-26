// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Validates that the history-list and frequent-tasks-list delete widgets
// behave identically (same tooltip/aria-label, same confirm/cancel flow).
// The two lists historically built the widget with duplicated code that
// drifted: the history delete button lacked the tooltip and aria-label
// that the frequent-task delete button carried.

'use strict';

const assert = require('assert');
const {makeWebview, send} = require('./simplify2_harness.js');

function historyRow(win) {
  return win.document.querySelector('#history-list .sidebar-item');
}

function frequentRow(win) {
  return win.document.querySelector('#frequent-list .sidebar-item');
}

function loadLists(win) {
  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      {
        id: 'chat-1',
        task_id: 'task-1',
        preview: 'refactor the parser',
        title: 'refactor the parser',
        has_events: true,
        ts: Date.now() / 1000,
      },
    ],
  });
  send(win, {type: 'frequentTasks', tasks: [{task: 'run the tests', count: 4}]});
}

function checkDeleteWidget(row, expectedAria) {
  const delBtn = row.querySelector('.sidebar-item-delete');
  assert.ok(delBtn, 'delete button rendered');
  assert.strictEqual(
    delBtn.dataset.tooltip,
    'Delete',
    'delete button must have the Delete tooltip in BOTH lists',
  );
  assert.strictEqual(
    delBtn.getAttribute('aria-label'),
    expectedAria,
    'delete button must carry an aria-label in BOTH lists',
  );
  const confirmWrap = row.querySelector('.sidebar-item-confirm');
  const confirmBtn = row.querySelector('.sidebar-confirm-yes');
  const cancelBtn = row.querySelector('.sidebar-confirm-no');
  assert.ok(confirmWrap && confirmBtn && cancelBtn, 'confirm widget rendered');
  assert.strictEqual(confirmWrap.style.display, 'none');
  assert.strictEqual(confirmBtn.dataset.tooltip, 'Confirm delete');
  assert.strictEqual(cancelBtn.dataset.tooltip, 'Cancel');

  delBtn.click();
  assert.strictEqual(confirmWrap.style.display, '', 'confirm shown on delete');
  assert.strictEqual(delBtn.style.display, 'none', 'delete hidden on confirm');

  cancelBtn.click();
  assert.strictEqual(confirmWrap.style.display, 'none', 'cancel hides confirm');
  assert.strictEqual(delBtn.style.display, '', 'cancel restores delete button');

  delBtn.click();
  confirmBtn.click();
  return {delBtn, confirmWrap};
}

function main() {
  const {win, posted} = makeWebview();
  loadLists(win);

  const hRow = historyRow(win);
  assert.ok(hRow, 'history row rendered');
  checkDeleteWidget(hRow, 'Delete task');
  const delMsg = posted.find(m => m.type === 'deleteTask');
  assert.ok(delMsg, 'confirm posts deleteTask');
  assert.strictEqual(delMsg.taskId, 'task-1');
  assert.strictEqual(historyRow(win), null, 'history row removed on confirm');
  assert.ok(
    !posted.some(m => m.type === 'resumeSession'),
    'delete/confirm/cancel clicks must not bubble into the history row ' +
      'click handler (no resumeSession may be posted)',
  );

  const fRow = frequentRow(win);
  assert.ok(fRow, 'frequent row rendered');
  const taskInput = win.document.getElementById('task-input');
  assert.ok(taskInput, 'task input present');
  taskInput.value = 'my precious draft';
  const cnt = fRow.querySelector('.frequent-item-count');
  assert.ok(cnt, 'frequent count rendered');
  const fDel = fRow.querySelector('.sidebar-item-delete');
  fDel.click();
  assert.strictEqual(cnt.style.display, 'none', 'count hidden while confirming');
  fRow.querySelector('.sidebar-confirm-no').click();
  assert.strictEqual(cnt.style.display, '', 'count restored on cancel');
  checkDeleteWidget(fRow, 'Delete frequent task');
  const fMsg = posted.find(m => m.type === 'deleteFrequentTask');
  assert.ok(fMsg, 'confirm posts deleteFrequentTask');
  assert.strictEqual(fMsg.task, 'run the tests');
  assert.strictEqual(frequentRow(win), null, 'frequent row removed on confirm');
  assert.strictEqual(
    taskInput.value,
    'my precious draft',
    'delete/confirm/cancel clicks must not bubble into the frequent row ' +
      'click handler (task input must keep the user draft)',
  );

  console.log('  ok - history and frequent delete widgets behave identically');
}

main();
