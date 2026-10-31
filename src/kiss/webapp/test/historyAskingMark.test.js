// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// History panel: a running task that is blocked on an ask_user_question
// (`awaiting_answer` on its history row) shows a pulsing "?" in the
// spinner's place; once the user has answered, the next history refresh
// (the daemon's `tasks_updated` nudge) puts the spinner back.

'use strict';

const assert = require('assert');
const {
  makePage: buildPage,
  send,
  stubRaf,
  sendEmptyWorkDir,
  loadMainCss,
  settle: tick,
} = require('./simplify2_harness');
const {
  makeRow,
  rowByTitle,
  answerHistory,
  openChatsPanel,
} = require('./history_helpers');

function makePage() {
  return buildPage({
    beforeScripts: win => {
      stubRaf(win);
      loadMainCss(win);
    },
  });
}

function statusMark(row) {
  return row.querySelector('.sidebar-item-running');
}

function assertSpinner(row, where) {
  const mark = statusMark(row);
  assert.ok(mark, 'a running row carries a status mark ' + where);
  assert.ok(mark.classList.contains('status-spinner'), 'spinner ' + where);
  assert.ok(!mark.classList.contains('sidebar-item-asking'), 'no "?" ' + where);
  assert.strictEqual(mark.textContent, '', 'the spinner has no glyph ' + where);
}

function assertAskingMark(win, row, where) {
  const mark = statusMark(row);
  assert.ok(mark, 'a waiting row carries a status mark ' + where);
  assert.strictEqual(
    row.firstElementChild,
    mark,
    "in the spinner's place " + where,
  );
  assert.ok(
    mark.classList.contains('sidebar-item-asking'),
    '"?" class ' + where,
  );
  assert.ok(!mark.classList.contains('status-spinner'), 'no spinner ' + where);
  assert.strictEqual(mark.textContent, '?', 'the glyph is "?" ' + where);
  assert.strictEqual(mark.dataset.tooltip, 'Waiting for your answer');
  assert.strictEqual(
    mark.getAttribute('aria-label'),
    'Waiting for your answer',
  );
  const cs = win.getComputedStyle(mark);
  assert.strictEqual(
    cs.color,
    'var(--warning)',
    'warning colour; got ' + cs.color,
  );
  const anim =
    (cs.getPropertyValue('animation-name') || '') +
    (cs.getPropertyValue('animation') || '');
  assert.ok(anim.indexOf('running-pulse') >= 0, 'the "?" pulses; got ' + anim);
  assert.ok(anim.indexOf('status-spin') < 0, 'and does not spin');
}

function testAskingRowShowsQuestionMark() {
  const {win, posted} = makePage();
  openChatsPanel(win);
  sendEmptyWorkDir(win);
  answerHistory(win, posted, [
    makeRow({task_id: 1, title: 'running task', is_running: true}),
    makeRow({
      task_id: 2,
      title: 'asking task',
      is_running: true,
      awaiting_answer: true,
    }),
    makeRow({
      task_id: 3,
      title: 'finished task',
      awaiting_answer: true,
      endTs: 1,
    }),
  ]);
  assertSpinner(rowByTitle(win, 'running task'), 'on a plain running row');
  const asking = rowByTitle(win, 'asking task');
  assertAskingMark(win, asking, 'on a waiting row');
  assert.strictEqual(asking.dataset.category, 'running', 'still a running row');
  assert.strictEqual(
    statusMark(rowByTitle(win, 'finished task')),
    null,
    'a stale awaiting_answer on a finished row draws nothing',
  );
  win.close();
  console.log(
    '  ok - a task waiting for an answer shows "?" instead of the spinner',
  );
}

async function testMarkFlipsWithTasksUpdated() {
  const {win, posted} = makePage();
  openChatsPanel(win);
  sendEmptyWorkDir(win);
  answerHistory(win, posted, [
    makeRow({task_id: 5, title: 'live task', is_running: true}),
  ]);
  assertSpinner(rowByTitle(win, 'live task'), 'before the question');

  // The daemon's tasks_updated after the question opened: the client
  // refetches, and the row repaints with the "?".
  posted.length = 0;
  send(win, {type: 'tasks_updated'});
  answerHistory(win, posted, [
    makeRow({
      task_id: 5,
      title: 'live task',
      is_running: true,
      awaiting_answer: true,
    }),
  ]);
  assertAskingMark(
    win,
    rowByTitle(win, 'live task'),
    'while the question is open',
  );

  // Answered: the next nudge brings the spinner back.  (A refresh in the
  // same tick as the previous one folds into its end-of-tick refetch,
  // so let that tick end first.)
  await tick();
  posted.length = 0;
  send(win, {type: 'tasks_updated'});
  answerHistory(win, posted, [
    makeRow({task_id: 5, title: 'live task', is_running: true}),
  ]);
  assertSpinner(rowByTitle(win, 'live task'), 'after the answer');
  win.close();
  console.log(
    '  ok - the "?" comes and goes with the daemon\'s tasks_updated nudges',
  );
}

async function main() {
  testAskingRowShowsQuestionMark();
  await testMarkFlipsWithTasksUpdated();
  console.log('\n2 passed, 0 failed');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
