// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// 2026-11-22 cost audit: the Spend panel's abbreviated dollar amounts
// (the all-time line and the cost-by-model rows) were rounded twice.
// fmtCostAbbrev handed the amount to fmtTokens, which rounds to a whole
// number first (right for a token count) and then to three significant
// digits: $1,234.50 became 1235, then "$1.24K" (it is $1.23K), and
// $999,499.75 became 999500, then "$1.00M" (it is $999K).  The amount is
// now rounded once, with its cents; token counts keep their rounding.

'use strict';

const assert = require('assert');
const {makePage, send, stubRaf} = require('./simplify2_harness');

/** The all-time line and the one model row a spendReport of `cost` draws. */
function drawn(win, cost, tokens) {
  send(win, {
    type: 'spendReport',
    total: {cost, tokens, tasks: 1},
    days: [],
    totalByModel: [{model: 'priced-model', cost, tokens, tasks: 1}],
    daysByModel: {},
  });
  return {
    total: win.document.querySelector('.spend-total').textContent,
    row: win.document.querySelector('.spend-model-value').textContent,
  };
}

function testSpendAmountsAreRoundedOnce() {
  const {win} = makePage({beforeScripts: stubRaf});
  const cases = [
    // [dollars, text] — the first four read one unit too high before.
    [1234.5, '$1.23K'],
    [1234.99, '$1.23K'],
    [12549.5, '$12.5K'],
    [999499.75, '$999K'],
    // On either side of those ties, and the forms that never changed.
    [1234.49, '$1.23K'],
    [1235.01, '$1.24K'],
    [12549.49, '$12.5K'],
    [12550.01, '$12.6K'],
    [999500, '$1.00M'],
    [1000, '$1.00K'],
    [2040, '$2.04K'],
    [98940.46, '$98.9K'],
    [999.99, '$999.99'],
    [4.2, '$4.20'],
    [0, '$0.00'],
  ];
  for (const [cost, text] of cases) {
    const got = drawn(win, cost, 10);
    assert.strictEqual(
      got.total,
      'All time \u00b7 ' + text + ' \u00b7 10 tok \u00b7 1 task',
      'all-time line of $' + cost,
    );
    assert.strictEqual(got.row, text, 'model row of $' + cost);
  }
  win.close();
  console.log('  ok - a Spend amount is rounded once, with its cents');
}

function testTokenCountsKeepTheirRounding() {
  const {win} = makePage({beforeScripts: stubRaf});
  // A token count is a whole number: 999,950 reads 1.00M, 212,500 213K,
  // and a fractional count (never sent, but the formatter takes it)
  // still rounds to a whole number first.
  const cases = [
    [999, '999'],
    [1000, '1.00K'],
    [212500, '213K'],
    [999950, '1.00M'],
    [1234.5, '1.24K'],
  ];
  for (const [tokens, text] of cases) {
    assert.strictEqual(
      drawn(win, 1, tokens).total,
      'All time \u00b7 $1.00 \u00b7 ' + text + ' tok \u00b7 1 task',
      'token count ' + tokens,
    );
  }
  win.close();
  console.log('  ok - token counts are formatted as before');
}

function runTests() {
  testSpendAmountsAreRoundedOnce();
  testTokenCountsKeepTheirRounding();
  console.log('spendAbbrevRounding.test.js: all tests passed');
}

runTests();
