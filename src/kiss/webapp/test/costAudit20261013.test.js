'use strict';
// Cost audit 2026-10-13: what the status row's Cost shows after a late
// or neighbouring usage event.
//
//   * A finished task's late side-channel charge arrives as a usage_info
//     stamped with that task's id.  A HIDDEN tab that meanwhile runs a
//     newer task must drop it exactly like the visible tab does (the
//     background path had no such check, so the newer task's Cost was
//     repainted with the old task's total).
//   * The previous/next-task replay renders a neighbour through the live
//     status row.  A neighbour whose transcript carries no usage event
//     must not have the live task's numbers written into its dataset
//     (the row was not cleared before the replay, so it did).

const assert = require('assert');
const harness = require('./simplify2_harness');

const {send} = harness;

function makePage() {
  const page = harness.makePage();
  page.win._testApi.endLaunch();
  page.win._testApi.hideWelcome();
  return page;
}

function budgetText(win) {
  return win.document.getElementById('status-budget').textContent;
}

const TS = 1767225600000;

function usage(taskId, cost, tokens) {
  return {
    type: 'usage_info',
    text: '',
    taskId,
    total_tokens: tokens,
    cost,
    cost_usd: Number(cost.slice(1)),
    total_steps: 1,
    ts: TS,
  };
}

function testHiddenTabDropsAnotherTasksLateUsage() {
  const {win} = makePage();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  win._testApi.switchToTab(tabA);
  assert.strictEqual(win._testApi.getActiveTabId(), tabA);

  // B (hidden) shows the finished task old-9, then starts new-1 there.
  send(win, {
    type: 'task_events',
    tabId: tabB,
    task_id: 'old-9',
    task: 'old task',
    events: [{type: 'prompt', text: 'old task', ts: TS}, usage('old-9', '$0.5000', 100)],
  });
  send(win, {
    type: 'task_events',
    tabId: tabB,
    task_id: 'new-1',
    task: 'new task',
    events: [{type: 'prompt', text: 'new task', ts: TS}],
  });
  send(win, {type: 'status', running: true, tabId: tabB, startTs: TS});
  send(win, {...usage('new-1', '$0.0100', 10), tabId: tabB});

  // old-9's late side-channel charge (task_update.charge_side_channel_usage)
  // reaches every tab; B must not take it for new-1's total.
  send(win, {...usage('old-9', '$0.7500', 150), tabId: tabB});
  // The same event on the visible tab (which shows no task) is harmless.
  send(win, {...usage('old-9', '$0.7500', 150), tabId: tabA});

  win._testApi.switchToTab(tabB);
  assert.strictEqual(
    budgetText(win),
    'Cost: $0.01',
    'a hidden tab running new-1 must keep new-1\u2019s cost, not old-9\u2019s late total',
  );
  // new-1's own next total is still applied.
  win._testApi.switchToTab(tabA);
  send(win, {...usage('new-1', '$0.0200', 20), tabId: tabB});
  win._testApi.switchToTab(tabB);
  assert.strictEqual(budgetText(win), 'Cost: $0.02');
  win.close();
  console.log('  ok - a hidden tab drops another task\u2019s late usage_info');
}

function testNeighbourWithoutUsageDoesNotInheritTheLiveCost() {
  const {win} = makePage();
  const tabId = win._testApi.getActiveTabId();
  send(win, {
    type: 'task_events',
    tabId,
    task_id: '42',
    task: 'current',
    events: [{type: 'prompt', text: 'current', ts: TS}, usage('42', '$0.4200', 400)],
  });
  assert.strictEqual(budgetText(win), 'Cost: $0.42');

  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'older',
    task_id: '41',
    events: [{type: 'prompt', text: 'older', ts: TS}],
  });
  const older = win.document.querySelector('.adjacent-task[data-task-id="41"]');
  assert.ok(older, 'the neighbour must render');
  assert.strictEqual(
    older.dataset.metricBudget,
    '',
    'a neighbour without usage events must not carry the live task\u2019s cost',
  );
  assert.strictEqual(older.dataset.metricTokens, '');

  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'next',
    task: 'newer',
    task_id: '43',
    events: [{type: 'prompt', text: 'newer', ts: TS}, usage('43', '$0.0300', 30)],
  });
  const newer = win.document.querySelector('.adjacent-task[data-task-id="43"]');
  assert.ok(newer, 'the next neighbour must render');
  assert.strictEqual(newer.dataset.metricBudget, 'Cost: $0.03');
  win.close();
  console.log('  ok - a neighbour without usage events shows no borrowed cost');
}

testHiddenTabDropsAnotherTasksLateUsage();
testNeighbourWithoutUsageDoesNotInheritTheLiveCost();
