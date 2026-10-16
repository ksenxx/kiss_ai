// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');

const {send} = harness;

function makePage() {
  return harness.makePage().win;
}

function text(el) {
  return (el && el.textContent) || '';
}

function resultEvents(summary) {
  return [
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
  ];
}

function sendReplayedTaskEvents(win, summary) {
  send(win, {
    type: 'task_events',
    task: 'multi-session task',
    events: resultEvents(summary),
  });
}

function sendLiveResultEvents(win, summary) {
  resultEvents(summary).forEach(event => send(win, event));
}

function assertRenderedSummaryOrdering(win, expectedFinalText) {
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
}

function assertSummaryOrdering(summary, expectedFinalText, sendEvents) {
  const win = makePage();
  sendEvents(win, summary);
  assertRenderedSummaryOrdering(win, expectedFinalText);
  win.close();
}

function testPreviousSessionsRenderBeforeTerminalResult() {
  const finalSessionSummary =
    '### Previous Session 1\n' +
    'session 1: prepared files\n\n' +
    '---\n\n' +
    '### Final Session\n' +
    'session 2: finished the fix';
  assertSummaryOrdering(
    finalSessionSummary,
    'session 2: finished the fix',
    sendReplayedTaskEvents,
  );
  console.log('  ok - Previous Sessions renders before replayed terminal Result');
}

function testLivePreviousSessionsRenderBeforeTerminalResult() {
  const finalSessionSummary =
    '### Previous Session 1\n' +
    'session 1: prepared files\n\n' +
    '---\n\n' +
    '### Final Session\n' +
    'session 2: finished the fix';
  assertSummaryOrdering(
    finalSessionSummary,
    'session 2: finished the fix',
    sendLiveResultEvents,
  );
  console.log('  ok - Previous Sessions renders before live terminal Result');
}

function testExhaustionSummaryWithoutFinalSessionHeader() {
  const exhaustionSummary =
    '### Previous Session 1\n' +
    'session 1: prepared files\n\n' +
    '---\n\n' +
    'Task failed after 2 sub-sessions';
  assertSummaryOrdering(
    exhaustionSummary,
    'Task failed after 2 sub-sessions',
    sendReplayedTaskEvents,
  );
  console.log('  ok - Previous Sessions renders before exhaustion Result');
}

function main() {
  testPreviousSessionsRenderBeforeTerminalResult();
  testLivePreviousSessionsRenderBeforeTerminalResult();
  testExhaustionSummaryWithoutFinalSessionHeader();
  console.log('multiSessionResultOrder.test.js: all assertions passed.');
}

main();
