// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

const TASK_ID = 'stopped-task-1';

function stoppedTaskEventsWithoutResult() {
  return [
    {type: 'prompt', text: 'review the repo', ts: 1000, taskId: TASK_ID},
    {type: 'text_delta', delta: 'Looking at the repo…', ts: 1500, taskId: TASK_ID},
    {
      type: 'tool_call',
      name: 'run_parallel',
      extras: {tasks: '["review A", "review B"]'},
      ts: 2000,
      taskId: TASK_ID,
    },
    {type: 'task_stopped', ts: 3000},
    {type: 'followup_suggestion', text: 'Resume the stopped task.', ts: 3001},
  ];
}

function stoppedTaskEventsWithResult() {
  const evs = stoppedTaskEventsWithoutResult();
  evs.splice(3, 0, {
    type: 'result',
    text: 'Task stopped by user',
    success: false,
    total_tokens: 0,
    cost: '$0.0000',
    step_count: 0,
    ts: 2500,
    taskId: TASK_ID,
  });
  return evs;
}

function resultPanels(win) {
  return Array.from(win.document.querySelectorAll('.ev.rc'));
}

function testSymptomNoResultRowMeansNoPanel() {
  const {win} = makePage();
  send(win, {
    type: 'task_events',
    task: 'review the repo',
    task_id: TASK_ID,
    events: stoppedTaskEventsWithoutResult(),
  });
  assert.strictEqual(
    resultPanels(win).length,
    0,
    'a stream with no persisted result row must render no Result panel',
  );
  console.log('  ok - pre-fix stream (no result row) reproduces the symptom');
}

function testHistoryReplayShowsStoppedResultPanel() {
  const {win} = makePage();
  send(win, {
    type: 'task_events',
    task: 'review the repo',
    task_id: TASK_ID,
    events: stoppedTaskEventsWithResult(),
  });
  const panels = resultPanels(win);
  assert.strictEqual(panels.length, 1, 'history replay must render 1 Result panel');
  const txt = panels[0].textContent;
  assert.ok(txt.includes('Task stopped by user'), 'panel shows the stop summary');
  assert.ok(txt.includes('Status: FAILED'), 'panel shows the FAILED status');
  console.log('  ok - history replay renders the stopped-task Result panel');
}

function testLiveStopShowsResultPanel() {
  const {win} = makePage();
  for (const ev of stoppedTaskEventsWithResult()) {
    send(win, Object.assign({}, ev));
  }
  const panels = resultPanels(win);
  assert.strictEqual(panels.length, 1, 'live stop must render 1 Result panel');
  assert.ok(panels[0].textContent.includes('Task stopped by user'));
  console.log('  ok - live stop renders the stopped-task Result panel');
}

function testAdjacentTaskReplayShowsStoppedResultPanel() {
  const {win} = makePage();
  send(win, {
    type: 'task_events',
    task: 'current task',
    task_id: 'current-1',
    events: [
      {type: 'prompt', text: 'current task', ts: 5000, taskId: 'current-1'},
      {
        type: 'result',
        summary: 'done',
        success: true,
        total_tokens: 1,
        cost: '$0.01',
        ts: 5001,
        taskId: 'current-1',
      },
    ],
  });
  send(win, {
    type: 'adjacent_task_events',
    direction: 'prev',
    task: 'review the repo',
    task_id: TASK_ID,
    events: stoppedTaskEventsWithResult(),
  });
  const adjacent = win.document.querySelector('.adjacent-task');
  assert.ok(adjacent, 'adjacent task block rendered');
  const panels = adjacent.querySelectorAll('.ev.rc');
  assert.strictEqual(
    panels.length,
    1,
    'adjacent-task replay must render the stopped-task Result panel',
  );
  assert.ok(panels[0].textContent.includes('Task stopped by user'));
  console.log('  ok - adjacent-task replay renders the stopped-task Result panel');
}

function main() {
  testSymptomNoResultRowMeansNoPanel();
  testHistoryReplayShowsStoppedResultPanel();
  testLiveStopShowsResultPanel();
  testAdjacentTaskReplayShowsStoppedResultPanel();
  console.log('stoppedTaskResultPanelReplay.test.js: all assertions passed.');
}

try {
  main();
  process.exit(0);
} catch (err) {
  console.error(err);
  process.exit(1);
}
