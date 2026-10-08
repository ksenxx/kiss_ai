// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The Result panel shows the run's final spend, not the pre-fold figure.
//
// ``SorcarAgent.run`` emits the executor's ``result`` first and only then
// folds spend that was banked outside the executor (the pre-run
// classifier, a sub-task reclaimed at the end), republishing the run
// total as a trailing ``usage_info``.  The header already followed that
// event; the Result panel baked its Tokens/Cost into static HTML and kept
// the stale figure, so the two surfaces disagreed on one task's cost.
// The trailing panel now takes the later total, while an earlier
// session's panel (anything appended after it) is left alone.

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

function panels(win) {
  return Array.from(win.document.querySelectorAll('#output > .rc'));
}

function metrics(panel) {
  return {
    tokens: panel.querySelector('.rs-tokens').textContent,
    cost: panel.querySelector('.rs-cost').textContent,
  };
}

function testTrailingResultPanelTakesTheFoldedTotal() {
  const win = makeWebview();
  send(win, {
    type: 'result',
    success: true,
    is_continue: false,
    summary: 'done',
    total_tokens: 1000,
    cost: '$1.0000',
  });
  assert.strictEqual(panels(win).length, 1);
  assert.deepStrictEqual(metrics(panels(win)[0]), {tokens: '1.00K', cost: '$1.00'});

  // The classifier fold republishes the run total after the result.
  send(win, {
    type: 'usage_info',
    text: 'Steps: 3, Total tokens: 1,200, Budget: $1.0200, ',
    total_tokens: 1200,
    cost: '$1.0200',
    total_steps: 3,
  });
  assert.deepStrictEqual(metrics(panels(win)[0]), {tokens: '1.20K', cost: '$1.02'});
  assert.strictEqual(
    win.document.getElementById('status-budget').textContent,
    'Cost: $1.02',
  );
  // "N/A" never overwrites a known cost; tokens still follow.
  send(win, {type: 'usage_info', total_tokens: 1300, cost: 'N/A', total_steps: 3});
  assert.deepStrictEqual(metrics(panels(win)[0]), {tokens: '1.30K', cost: '$1.02'});
  win.close();
}

function testEarlierSessionPanelIsLeftAlone() {
  const win = makeWebview();
  send(win, {
    type: 'result',
    success: false,
    is_continue: true,
    summary: 'session 1 paused',
    total_tokens: 500,
    cost: '$0.5000',
  });
  // The continuation session appends new content after the panel ...
  send(win, {type: 'prompt', text: 'continuing the work'});
  // ... so its growing totals must not rewrite session 1's panel.
  send(win, {
    type: 'usage_info',
    text: 'Steps: 9, Total tokens: 9,000, Budget: $9.0000, ',
    total_tokens: 9000,
    cost: '$9.0000',
    total_steps: 9,
  });
  const first = panels(win)[0];
  assert.deepStrictEqual(metrics(first), {tokens: '500', cost: '$0.50'});
  win.close();
}

testTrailingResultPanelTakesTheFoldedTotal();
testEarlierSessionPanelIsLeftAlone();
console.log('resultPanelLateUsageFold.test.js passed');
