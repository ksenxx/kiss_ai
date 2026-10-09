// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// October 2026 cost audit: the header, result panel, history rows and
// Spend panel formatted every cost with two decimals, so a task that
// spent $0.0049 read "Cost: $0.00" — indistinguishable from a free
// run.  A positive amount below half a cent now shows four decimals,
// and below $0.00005 the lower bound "<$0.0001"; larger amounts and an
// exact zero keep the two-decimal form.

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
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return win;
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function headerCost(win) {
  return win.document.getElementById('status-budget').textContent;
}

function testSubCentSpendIsNotShownAsFree() {
  const win = makeWebview();
  const cases = [
    ['$0.0049', 'Cost: $0.0049'],
    ['$0.0001', 'Cost: $0.0001'],
    [0.00004, 'Cost: <$0.0001'],
    ['$0.0050', 'Cost: $0.01'],
    ['$0.1000', 'Cost: $0.10'],
    ['$0.0000', 'Cost: $0.00'],
  ];
  let step = 0;
  for (const [cost, expected] of cases) {
    step += 1;
    send(win, {
      type: 'usage_info',
      text: 'Steps: ' + step,
      total_tokens: 100 * step,
      cost,
      total_steps: step,
      taskId: 'cheap-task',
    });
    assert.strictEqual(headerCost(win), expected, 'usage_info cost ' + cost);
  }
  send(win, {
    type: 'result',
    text: 'done',
    total_tokens: 1000,
    cost: '$0.0049',
    step_count: 9,
    taskId: 'cheap-task',
  });
  assert.strictEqual(headerCost(win), 'Cost: $0.0049', 'result event cost');

  // Daemon-shaped events: the display string is rounded to "$0.0000" but
  // the exact figure travels in cost_usd and wins.
  send(win, {
    type: 'usage_info',
    text: 'Steps: 10',
    total_tokens: 2000,
    cost: '$0.0000',
    cost_usd: 0.000049,
    total_steps: 10,
    taskId: 'cheap-task',
  });
  assert.strictEqual(
    headerCost(win),
    'Cost: <$0.0001',
    'usage_info prefers cost_usd',
  );
  send(win, {
    type: 'result',
    text: 'done',
    total_tokens: 2000,
    cost: '$0.0000',
    cost_usd: 0.0012,
    step_count: 10,
    taskId: 'cheap-task',
  });
  assert.strictEqual(
    headerCost(win),
    'Cost: $0.0012',
    'result prefers cost_usd',
  );
  const panels = win.document.querySelectorAll('.rs-cost');
  const panelCost = panels[panels.length - 1];
  assert.ok(panelCost, 'the result panel renders a cost');
  assert.strictEqual(
    panelCost.textContent,
    '$0.0012',
    'result panel prefers cost_usd',
  );
  send(win, {
    type: 'usage_info',
    text: '',
    total_tokens: 2100,
    cost: '$0.0000',
    cost_usd: 0.0013,
    total_steps: 10,
    taskId: 'cheap-task',
  });
  assert.strictEqual(
    panelCost.textContent,
    '$0.0013',
    'late fold refreshes the panel',
  );
  win.close();
  console.log('  ok - sub-cent spend shows four decimals, never $0.00');
}

// A result whose exact cost is the number 0 is a free run, not an
// unknown one: the panel must say "$0.00", not "N/A".  And an event
// whose display string is "N/A" but carries a finite cost_usd must
// still update the header and the panel, while one with neither leaves
// the last known figure in place.
function testZeroAndNaCostsResolveFromCostUsd() {
  const win = makeWebview();
  send(win, {
    type: 'result',
    text: 'free',
    total_tokens: 10,
    cost: '$0.0000',
    cost_usd: 0,
    step_count: 1,
    taskId: 'free-task',
  });
  let panels = win.document.querySelectorAll('.rs-cost');
  assert.strictEqual(
    panels[panels.length - 1].textContent,
    '$0.00',
    'zero cost_usd is $0.00',
  );
  assert.strictEqual(
    headerCost(win),
    'Cost: $0.00',
    'zero cost_usd in the header',
  );

  send(win, {
    type: 'usage_info',
    text: '',
    total_tokens: 20,
    cost: 'N/A',
    cost_usd: 0.0012,
    total_steps: 1,
    taskId: 'free-task',
  });
  assert.strictEqual(
    headerCost(win),
    'Cost: $0.0012',
    'N/A string yields to cost_usd',
  );
  assert.strictEqual(
    panels[panels.length - 1].textContent,
    '$0.0012',
    'late fold despite N/A',
  );

  send(win, {
    type: 'usage_info',
    text: '',
    total_tokens: 30,
    cost: 'N/A',
    total_steps: 2,
    taskId: 'free-task',
  });
  assert.strictEqual(
    headerCost(win),
    'Cost: $0.0012',
    'a cost-less event keeps the header',
  );
  assert.strictEqual(
    panels[panels.length - 1].textContent,
    '$0.0012',
    'and the panel',
  );

  send(win, {
    type: 'result',
    text: 'unknown',
    total_tokens: 40,
    cost: 'N/A',
    step_count: 2,
    taskId: 'free-task',
  });
  panels = win.document.querySelectorAll('.rs-cost');
  assert.strictEqual(
    panels[panels.length - 1].textContent,
    'N/A',
    'no cost at all renders N/A',
  );
  assert.strictEqual(
    headerCost(win),
    'Cost: $0.0012',
    'and does not clobber the header',
  );
  win.close();
  console.log('  ok - zero and N/A costs resolve from cost_usd');
}

function runTests() {
  testSubCentSpendIsNotShownAsFree();
  testZeroAndNaCostsResolveFromCostUsd();
  console.log('subCentCostDisplay.test.js: all tests passed');
}

runTests();
