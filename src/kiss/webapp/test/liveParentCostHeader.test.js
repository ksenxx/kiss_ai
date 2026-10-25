// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send, stubRaf, clickTab} = require('./simplify2_harness');

function headerText(win, id) {
  return win.document.getElementById(id).textContent;
}

function testParentHeaderTracksLiveAggregateUsage() {
  const {win} = makePage({beforeScripts: stubRaf});
  const api = win._testApi;
  assert.ok(api, '_testApi must be exposed by main.js');
  const parentTab = api.getActiveTabId();

  send(win, {
    type: 'usage_info',
    text: 'Steps: 2/100, Tokens: 1,000/400,000, Budget: $0.1000/$10.00, ',
    total_tokens: 1000,
    cost: '$0.1000',
    total_steps: 2,
    taskId: 'parent-task',
  });
  assert.strictEqual(headerText(win, 'status-tokens'), 'Tokens: 1.00K');
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.10');
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 2');

  api.createNewTab();
  const subTab = api.getActiveTabId();
  assert.ok(subTab !== parentTab);
  clickTab(win, parentTab);

  send(win, {
    type: 'usage_info',
    text: 'Steps: 5/100, Tokens: 30,000/400,000, Budget: $0.3000/$5.00, ',
    total_tokens: 30000,
    cost: '$0.3000',
    total_steps: 5,
    taskId: 'sub-task',
    tabId: subTab,
  });
  assert.strictEqual(
    headerText(win, 'status-budget'),
    'Cost: $0.10',
    'a sub-agent usage_info must never clobber the parent header',
  );

  send(win, {
    type: 'usage_info',
    text: 'Tokens: 31,000, Budget: $0.4000 (live, incl. parallel sub-agents), ',
    total_tokens: 31000,
    cost: '$0.4000',
    total_steps: 7,
    taskId: 'parent-task',
  });
  assert.strictEqual(headerText(win, 'status-tokens'), 'Tokens: 31.0K');
  assert.strictEqual(
    headerText(win, 'status-budget'),
    'Cost: $0.40',
    'the parent header must reflect agent + all sub-agents cost',
  );
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 7');

  clickTab(win, subTab);
  assert.strictEqual(headerText(win, 'status-tokens'), 'Tokens: 30.0K');
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.30');
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 5');
  clickTab(win, parentTab);
  assert.strictEqual(headerText(win, 'status-tokens'), 'Tokens: 31.0K');
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.40');
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 7');

  win.close();
  console.log('  ok - parent header tracks live aggregate (agent + subs)');
}

function testMisroutedParentUsageForOtherTaskDropped() {
  const {win} = makePage({beforeScripts: stubRaf});
  send(win, {type: 'task_events', task: '', events: [], task_id: 'task-A'});
  send(win, {
    type: 'usage_info',
    text: '',
    total_tokens: 111,
    cost: '$0.1110',
    total_steps: 1,
    taskId: 'task-A',
  });
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.11');
  send(win, {
    type: 'usage_info',
    text: '',
    total_tokens: 999,
    cost: '$9.9990',
    total_steps: 9,
    taskId: 'task-B',
  });
  assert.strictEqual(
    headerText(win, 'status-budget'),
    'Cost: $0.11',
    "another task's usage_info must not update this tab's header",
  );
  win.close();
  console.log('  ok - misrouted usage_info for another task is dropped');
}

// The number formats themselves (text fallback, `N/A` cost, result
// events) are metricFormatting.testHeaderCostTwoDecimalsOnAllPaths; this
// pins what a frame WITHOUT a field leaves on the header.
function testMissingFieldsKeepThePreviousNumbers() {
  const {win} = makePage({beforeScripts: stubRaf});
  send(win, {
    type: 'usage_info',
    text: 'Steps: 3/100, Tokens: 1,234/400,000, Budget: $0.5000/$10.00, ',
    taskId: 'parent-task',
  });
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 3');

  send(win, {type: 'usage_info', text: 'no metrics here', taskId: 'parent-task'});
  assert.strictEqual(headerText(win, 'status-tokens'), 'Tokens: 1.23K');
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.50');
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 3');

  send(win, {
    type: 'usage_info',
    text: '',
    total_tokens: 2500,
    cost: '$0.6000',
    taskId: 'parent-task',
  });
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 3');
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.60');

  // A result that ran nothing keeps the numbers already on the header.
  send(win, {
    type: 'result',
    text: 'success: true\nsummary: noop',
    summary: 'noop',
    success: true,
    total_tokens: 0,
    cost: 'N/A',
    step_count: 0,
    taskId: 'parent-task',
  });
  assert.strictEqual(headerText(win, 'status-tokens'), 'Tokens: 2.50K');
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.60');
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 3');

  win.close();
  console.log('  ok - frames without a field keep the previous header numbers');
}

function runTests() {
  testParentHeaderTracksLiveAggregateUsage();
  testMisroutedParentUsageForOtherTaskDropped();
  testMissingFieldsKeepThePreviousNumbers();
  console.log('liveParentCostHeader.test.js: all tests passed');
}

runTests();
