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
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
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

function headerText(win, id) {
  return win.document.getElementById(id).textContent;
}

function switchToTab(win, api, tabId) {
  const tabEl = win.document.querySelector(
    '.chat-tab[data-tab-id="' + tabId + '"]',
  );
  assert.ok(tabEl, 'tab element must exist for ' + tabId);
  tabEl.click();
  assert.strictEqual(api.getActiveTabId(), tabId);
}

function testParentHeaderTracksLiveAggregateUsage() {
  const {win} = makeWebview();
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
  assert.strictEqual(headerText(win, 'status-tokens'), 'Tokens: 1,000');
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.1000');
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 2');

  api.createNewTab();
  const subTab = api.getActiveTabId();
  assert.ok(subTab !== parentTab);
  switchToTab(win, api, parentTab);

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
    'Cost: $0.1000',
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
  assert.strictEqual(headerText(win, 'status-tokens'), 'Tokens: 31,000');
  assert.strictEqual(
    headerText(win, 'status-budget'),
    'Cost: $0.4000',
    'the parent header must reflect agent + all sub-agents cost',
  );
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 7');

  switchToTab(win, api, subTab);
  assert.strictEqual(headerText(win, 'status-tokens'), 'Tokens: 30,000');
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.3000');
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 5');
  switchToTab(win, api, parentTab);
  assert.strictEqual(headerText(win, 'status-tokens'), 'Tokens: 31,000');
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.4000');
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 7');

  win.close();
  console.log('  ok - parent header tracks live aggregate (agent + subs)');
}

function testMisroutedParentUsageForOtherTaskDropped() {
  const {win} = makeWebview();
  send(win, {type: 'task_events', task: '', events: [], task_id: 'task-A'});
  send(win, {
    type: 'usage_info',
    text: '',
    total_tokens: 111,
    cost: '$0.1110',
    total_steps: 1,
    taskId: 'task-A',
  });
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.1110');
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
    'Cost: $0.1110',
    "another task's usage_info must not update this tab's header",
  );
  win.close();
  console.log('  ok - misrouted usage_info for another task is dropped');
}

function testUsageInfoFallbackAndNABranches() {
  const {win} = makeWebview();

  send(win, {
    type: 'usage_info',
    text: 'Steps: 3/100, Tokens: 1,234/400,000, Budget: $0.5000/$10.00, ',
    taskId: 'parent-task',
  });
  assert.strictEqual(headerText(win, 'status-tokens'), 'Tokens: 1,234');
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.5000');
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 3');

  send(win, {
    type: 'usage_info',
    text: 'no metrics here',
    taskId: 'parent-task',
  });
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.5000');

  send(win, {
    type: 'usage_info',
    text: '',
    total_tokens: 2000,
    cost: 'N/A',
    total_steps: 4,
    taskId: 'parent-task',
  });
  assert.strictEqual(headerText(win, 'status-tokens'), 'Tokens: 2,000');
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.5000');
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 4');

  send(win, {
    type: 'usage_info',
    text: '',
    total_tokens: 2500,
    cost: '$0.6000',
    taskId: 'parent-task',
  });
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 4');
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.6000');

  win.close();
  console.log('  ok - usage_info fallback / N/A / missing-field branches');
}

function testResultEventHeaderBranches() {
  const {win} = makeWebview();

  send(win, {
    type: 'result',
    text: 'success: true\nsummary: done',
    summary: 'done',
    success: true,
    total_tokens: 42000,
    cost: '$0.7000',
    step_count: 12,
    taskId: 'parent-task',
  });
  assert.strictEqual(headerText(win, 'status-tokens'), 'Tokens: 42,000');
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.7000');
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 12');

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
  assert.strictEqual(headerText(win, 'status-tokens'), 'Tokens: 42,000');
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.7000');
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 12');

  win.close();
  console.log('  ok - result event header update / preserve branches');
}

function runTests() {
  testParentHeaderTracksLiveAggregateUsage();
  testMisroutedParentUsageForOtherTaskDropped();
  testUsageInfoFallbackAndNABranches();
  testResultEventHeaderBranches();
  console.log('liveParentCostHeader.test.js: all tests passed');
}

runTests();
