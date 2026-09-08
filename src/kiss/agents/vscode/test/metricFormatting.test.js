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
//    below one thousand, in the header, the result panel, and the
//    history rows;
//  - model-list items wrap the name in a .model-item-name span (the
//    CSS start-truncation hook that keeps the list from scrolling
//    horizontally on narrow screens) with U+200E bidi guards.

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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

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
  const {win} = makeWebview();
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
  const {win} = makeWebview();

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

function testResultPanelUsesCompactFormats() {
  const {win} = makeWebview();
  send(win, {
    type: 'result',
    success: true,
    text: 'all done',
    total_tokens: 45678,
    cost: '$0.4123',
  });
  const rs = win.document.querySelector('.rc .rs');
  assert.ok(rs, 'the result panel must render its metrics strip');
  const text = rs.textContent;
  assert.ok(
    text.includes('Tokens 45.7K'),
    'result panel tokens must be compact; got: ' + text,
  );
  assert.ok(
    text.includes('Cost $0.41'),
    'result panel cost must show two decimals; got: ' + text,
  );
  win.close();
  console.log('  ok - the result panel uses the compact formats');
}

function testResultPanelKeepsNAForMissingCost() {
  const {win} = makeWebview();
  send(win, {type: 'result', success: true, text: 'done', total_tokens: 10});
  const rs = win.document.querySelector('.rc .rs');
  assert.ok(rs, 'the result panel must render its metrics strip');
  assert.ok(
    rs.textContent.includes('Cost N/A'),
    'a missing cost must stay N/A; got: ' + rs.textContent,
  );
  win.close();
  console.log('  ok - a missing result cost stays N/A');
}

function testHistoryRowsUseCompactFormats() {
  const {win} = makeWebview();
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
  const {win} = makeWebview();
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
testResultPanelUsesCompactFormats();
testResultPanelKeepsNAForMissingCost();
testHistoryRowsUseCompactFormats();
testModelItemsWrapNameForStartTruncation();
console.log('metricFormatting.test.js: all tests passed');
