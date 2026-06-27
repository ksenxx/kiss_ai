// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end chat-webview regression test for multi-session Result
// ordering.  Drives the real chat.html + panelCopy.js + main.js in
// jsdom and dispatches the same task_events payload the backend sends.
//
// Requirement: for a multi-session agent result, the "Previous
// Sessions" panel must render before the terminal "Result" panel.

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

  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage() {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return win;
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function text(el) {
  return (el && el.textContent) || '';
}

function assertSummaryOrdering(summary, expectedFinalText) {
  const win = makeWebview();

  send(win, {
    type: 'task_events',
    task: 'multi-session task',
    events: [
      {
        type: 'result',
        success: false,
        is_continue: true,
        summary: 'session 1: prepared files',
        total_tokens: 10,
        cost: '$0.01',
      },
      {
        type: 'result',
        success: true,
        is_continue: false,
        summary: 'session 2: finished the fix',
        total_tokens: 20,
        cost: '$0.02',
      },
      {
        type: 'result',
        success: true,
        is_continue: false,
        summary,
        total_tokens: 30,
        cost: '$0.03',
      },
    ],
  });

  const panels = Array.from(win.document.querySelectorAll('#output > .rc'));
  assert.ok(panels.length >= 2, 'expected at least Previous Sessions + Result panels');
  const headings = panels.map(panel => text(panel.querySelector('.rc-h h3')));
  assert.deepStrictEqual(
    headings,
    ['Previous Sessions', 'Result'],
    'BUG: stale per-session Result panels must be removed when the merged multi-session Result arrives; got ' +
      headings.join(' -> '),
  );
  const previousIdx = headings.lastIndexOf('Previous Sessions');
  const resultIdx = headings.lastIndexOf('Result');
  assert.ok(previousIdx >= 0, 'BUG: Previous Sessions panel did not render');
  assert.ok(resultIdx >= 0, 'terminal Result panel did not render');
  assert.ok(
    previousIdx < resultIdx,
    'BUG: Previous Sessions panel must appear before terminal Result panel; got ' +
      headings.join(' -> '),
  );

  assert.ok(
    text(panels[previousIdx]).includes('session 1: prepared files'),
    'Previous Sessions panel must contain prior session text',
  );
  assert.ok(
    !text(panels[previousIdx]).includes('session 2: finished the fix'),
    'Previous Sessions panel must not contain the final session text',
  );
  assert.ok(
    text(panels[resultIdx]).includes(expectedFinalText),
    'terminal Result panel must contain final text',
  );
  assert.ok(
    !text(panels[resultIdx]).includes('Previous Session'),
    'terminal Result panel must not duplicate previous-session headings',
  );

  win.close();
}

function testPreviousSessionsRenderBeforeTerminalResult() {
  const finalSessionSummary =
    '### Previous Session 1\n' +
    'session 1: prepared files\n\n' +
    '---\n\n' +
    '### Final Session\n' +
    'session 2: finished the fix';
  assertSummaryOrdering(finalSessionSummary, 'session 2: finished the fix');
  console.log('  ok - Previous Sessions renders before terminal Result');
}

function testExhaustionSummaryWithoutFinalSessionHeader() {
  const exhaustionSummary =
    '### Previous Session 1\n' +
    'session 1: prepared files\n\n' +
    '---\n\n' +
    'Task failed after 2 sub-sessions';
  assertSummaryOrdering(exhaustionSummary, 'Task failed after 2 sub-sessions');
  console.log('  ok - Previous Sessions renders before exhaustion Result');
}

function main() {
  testPreviousSessionsRenderBeforeTerminalResult();
  testExhaustionSummaryWithoutFinalSessionHeader();
  console.log('multiSessionResultOrder.test.js: all assertions passed.');
}

main();
