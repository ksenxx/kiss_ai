// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end reproduction of the "stale parent cost header during
// run_parallel" issue, plus full branch coverage of the webview's
// cost/tokens header update paths.
//
// Requirement locked in:
//
//   The cost and tokens shown at the top of the chat webview must
//   always reflect the cost so far of running the agent AND all of
//   its sub-agents at every turn.
//
// The webview can only display what the backend streams on the PARENT
// task: sub-agent ``usage_info`` events are stamped with the
// sub-agent's tab id and are routed into that tab's background state —
// they must NEVER clobber the parent header (each header shows its own
// task).  So while ``run_parallel`` blocks the parent's turn, the
// parent header goes stale unless the backend emits live aggregate
// ``usage_info`` events on the parent task.  The backend fix
// (``_LiveUsageMonitor`` in ``sorcar_agent.py``) does exactly that;
// this test drives the production webview with the event sequences of
// both the broken world (no live parent events → header stays stale)
// and the fixed world (live parent events → header tracks the
// aggregate), and pins the routing rules that make the fix correct.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/liveParentCostHeader.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview: the real
 * ``chat.html`` body (placeholders blanked), ``panelCopy.js`` and
 * ``main.js`` evaluated in the window, and a recording
 * ``acquireVsCodeApi`` stub (the only host API the webview has).
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
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
  const api = win._demoApi;
  assert.ok(api, '_demoApi must be exposed by main.js');
  const parentTab = api.getActiveTabId();

  // Parent turn 1: the agent's own per-turn usage_info.
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

  // A sub-agent tab opens (run_parallel fan-out); the user keeps
  // looking at the parent tab.
  api.createNewTab();
  const subTab = api.getActiveTabId();
  assert.ok(subTab !== parentTab);
  switchToTab(win, api, parentTab);

  // The sub-agent burns budget and streams its OWN usage_info,
  // stamped with the sub-agent's tab id.  It must land in the
  // sub-agent tab's background state, NOT the parent header.
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

  // BROKEN WORLD: without live parent-task events the header above is
  // all the parent ever shows until the whole fan-out completes — it
  // reads $0.1000 while $0.4000 has actually been spent.  FIXED WORLD:
  // the backend's _LiveUsageMonitor emits the aggregate (parent
  // session + all sub-agents) on the PARENT task while sub-agents run:
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

  // Per-tab isolation both ways: the sub-agent tab shows ITS OWN
  // usage when activated, and the parent aggregate is restored when
  // the user switches back.
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
  // A usage_info stamped for a DIFFERENT task than the one the active
  // tab is showing must be dropped (misroute guard), so a sibling
  // task's live aggregate can never corrupt this tab's header.
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

  // Structured fields missing → regex fallback on the text.
  send(win, {
    type: 'usage_info',
    text: 'Steps: 3/100, Tokens: 1,234/400,000, Budget: $0.5000/$10.00, ',
    taskId: 'parent-task',
  });
  assert.strictEqual(headerText(win, 'status-tokens'), 'Tokens: 1,234');
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.5000');
  assert.strictEqual(headerText(win, 'status-steps'), 'Steps: 3');

  // Fallback text with no parsable metrics → header unchanged.
  send(win, {
    type: 'usage_info',
    text: 'no metrics here',
    taskId: 'parent-task',
  });
  assert.strictEqual(headerText(win, 'status-budget'), 'Cost: $0.5000');

  // Structured cost 'N/A' → tokens/steps update, budget preserved.
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

  // total_steps missing → steps preserved.
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

  // A final result carries the parent's cumulative usage (offsets
  // applied backend-side, so it already includes sub-agent spend).
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

  // Zero tokens / 'N/A' cost / zero steps → header preserved.
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
