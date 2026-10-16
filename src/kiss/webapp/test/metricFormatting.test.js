// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Regression tests for the metric display formats:
//  - the header cost always shows exactly two digits after the decimal
//    point ("Cost: $0.41"), on every path that writes it (structured
//    usage_info, result events, and the text-fallback parser);
//  - token counts render as exactly three significant digits with a
//    K/M/B/T suffix (thousands/millions/billions/trillions), plain
//    below one thousand, in the header and the history rows (the result
//    panel's header carries no metrics at all);
//  - model-list items wrap the name in a .model-item-name span (the
//    CSS start-truncation hook that keeps the list from scrolling
//    horizontally on narrow screens) with U+200E bidi guards.

'use strict';

const assert = require('assert');
const {makePage, send, stubRaf} = require('./simplify2_harness');

function headerTokens(win) {
  return win.document.getElementById('status-tokens').textContent;
}

function headerBudget(win) {
  return win.document.getElementById('status-budget').textContent;
}

// Every K/M/B/T bucket, the sub-1000 passthrough, and the two rounding
// edges: 999,499 stays 999K while 999,500 rounds up a unit to 1.00M.
const TOKEN_CASES = [
  [999, '999'],
  [1000, '1.00K'],
  [12345, '12.3K'],
  [123456, '123K'],
  [999499, '999K'],
  [999500, '1.00M'],
  [1234567, '1.23M'],
  [1234567890, '1.23B'],
  [2500000000000, '2.50T'],
  [999600000000000, '1000T'],
];

function testHeaderTokensCompactOnStructuredUsageInfo() {
  const {win} = makePage({beforeScripts: stubRaf});
  for (const [tokens, expected] of TOKEN_CASES) {
    send(win, {
      type: 'usage_info',
      total_tokens: tokens,
      cost: '$0.4123',
      total_steps: 1,
    });
    assert.strictEqual(
      headerTokens(win),
      'Tokens: ' + expected,
      tokens + ' tokens',
    );
    assert.strictEqual(headerBudget(win), 'Cost: $0.41');
  }
  win.close();
  console.log('  ok - header tokens are compact for structured usage_info');
}

function testHeaderCostTwoDecimalsOnAllPaths() {
  const {win} = makePage({beforeScripts: stubRaf});

  // Structured usage_info with a numeric cost.
  send(win, {type: 'usage_info', total_tokens: 500, cost: 0.056});
  assert.strictEqual(headerBudget(win), 'Cost: $0.06');
  assert.strictEqual(headerTokens(win), 'Tokens: 500');

  // N/A cost leaves the previous cost in place.
  send(win, {type: 'usage_info', total_tokens: 600, cost: 'N/A'});
  assert.strictEqual(headerBudget(win), 'Cost: $0.06');

  // Text fallback (no structured fields): the step-counter footer.
  send(win, {
    type: 'usage_info',
    text: 'Steps: 3/100, Tokens: 12,345/400,000, Budget: $0.4123/$9.00, ',
  });
  assert.strictEqual(headerTokens(win), 'Tokens: 12.3K');
  assert.strictEqual(headerBudget(win), 'Cost: $0.41');

  // A result event refreshes both counters.
  send(win, {
    type: 'result',
    success: true,
    text: 'done',
    total_tokens: 1234567,
    cost: '$1.2999',
  });
  assert.strictEqual(headerTokens(win), 'Tokens: 1.23M');
  assert.strictEqual(headerBudget(win), 'Cost: $1.30');

  win.close();
  console.log('  ok - header cost shows two decimals on every write path');
}

function testResultPanelHeaderCarriesNoMetrics() {
  // The run's spend lives in the status bar only; the Result panel's
  // header is just its title.
  const {win} = makePage({beforeScripts: stubRaf});
  send(win, {
    type: 'result',
    success: true,
    text: 'all done',
    total_tokens: 45678,
    cost: '$0.4123',
  });
  const header = win.document.querySelector('.rc .rc-h');
  assert.ok(header, 'the result panel must render its header');
  assert.strictEqual(header.textContent.trim(), 'Result');
  assert.strictEqual(headerTokens(win), 'Tokens: 45.7K');
  assert.strictEqual(headerBudget(win), 'Cost: $0.41');
  win.close();
  console.log('  ok - the result panel header shows no tokens or cost');
}

function testHistoryRowsUseCompactFormats() {
  const {win} = makePage({beforeScripts: stubRaf});
  send(win, {
    type: 'history',
    offset: 0,
    sessions: [
      {
        id: 'h1',
        task_id: 7,
        title: 'big run',
        preview: 'big run',
        timestamp: 1700000000,
        has_events: false,
        failed: false,
        is_running: false,
        tokens: 1234567,
        cost: 0.1234,
        steps: 42,
        is_favorite: false,
      },
    ],
  });
  const metrics = win.document.querySelector('.running-item-metrics');
  assert.ok(metrics, 'the history row must render its metrics span');
  assert.ok(
    metrics.textContent.includes('1.23M tok • $0.12'),
    'history metrics must be compact; got: ' + metrics.textContent,
  );
  win.close();
  console.log('  ok - history rows use the compact formats');
}

function testModelItemsWrapNameForStartTruncation() {
  const {win} = makePage({beforeScripts: stubRaf});
  const long = 'claude-extremely-long-model-name-20261114-thinking-64k';
  send(win, {
    type: 'models',
    models: [
      {name: long, inp: 5, out: 25, uses: 1, vendor: 'anthropic'},
      {name: 'gpt-5.6-sol', inp: 2, out: 8, uses: 0, vendor: 'openai'},
    ],
    selected: 'gpt-5.6-sol',
  });
  win.document
    .getElementById('model-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const names = Array.from(
    win.document.querySelectorAll('#model-list .model-item .model-item-name'),
  );
  assert.strictEqual(
    names.length,
    2,
    'every model item must wrap its name in .model-item-name',
  );
  for (const el of names) {
    const raw = el.textContent;
    assert.ok(
      raw.startsWith('\u200e') && raw.endsWith('\u200e'),
      'the name must carry U+200E bidi guards; got: ' + JSON.stringify(raw),
    );
  }
  const stripped = names.map(el => el.textContent.replace(/\u200e/g, ''));
  assert.deepStrictEqual(stripped, [long, 'gpt-5.6-sol']);
  win.close();
  console.log('  ok - model items wrap the name for start-truncation');
}

testHeaderTokensCompactOnStructuredUsageInfo();
testHeaderCostTwoDecimalsOnAllPaths();
testResultPanelHeaderCarriesNoMetrics();
testHistoryRowsUseCompactFormats();
testModelItemsWrapNameForStartTruncation();
console.log('metricFormatting.test.js: all tests passed');
