// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A multi-session task ends with one merged Result (the "### Previous
// Session N" / "### Final Session" summary).  The page renders only
// its final session, as the Result panel (the earlier sessions' notes
// are not shown), and the stale
// per-session Result panels of the earlier `is_continue` results are
// removed when the merged one arrives (replayed or live).  The split
// of a single merged summary is pinned by resultPanelHtml.test.js.

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

function assertOnlyFinalSessionRendered(win, expectedFinalText) {
  const panels = Array.from(win.document.querySelectorAll('#output > .rc'));
  const headings = panels.map(panel => text(panel.querySelector('.rc-h h3')));
  assert.deepStrictEqual(
    headings,
    ['Result'],
    'BUG: stale per-session Result panels must be removed and no Previous Sessions panel shown when the merged multi-session Result arrives; got ' +
      headings.join(' -> '),
  );
  const [result] = panels;
  assert.ok(
    !text(result).includes('session 1: prepared files'),
    'the earlier sessions must not be shown',
  );
  assert.ok(
    text(result).includes(expectedFinalText),
    'terminal Result panel must contain final text',
  );
  assert.ok(
    !text(result).includes('Previous Session'),
    'terminal Result panel must not duplicate previous-session headings',
  );
}

function assertFinalSessionAlone(summary, expectedFinalText, sendEvents) {
  const win = makePage();
  sendEvents(win, summary);
  assertOnlyFinalSessionRendered(win, expectedFinalText);
  win.close();
}

const FINAL_SESSION_SUMMARY =
  '### Previous Session 1\n' +
  'session 1: prepared files\n\n' +
  '---\n\n' +
  '### Final Session\n' +
  'session 2: finished the fix';

function testReplayedMergedResultShowsFinalSessionAlone() {
  assertFinalSessionAlone(
    FINAL_SESSION_SUMMARY,
    'session 2: finished the fix',
    sendReplayedTaskEvents,
  );
  console.log('  ok - replayed merged Result shows the final session alone');
}

function testLiveMergedResultShowsFinalSessionAlone() {
  assertFinalSessionAlone(
    FINAL_SESSION_SUMMARY,
    'session 2: finished the fix',
    sendLiveResultEvents,
  );
  console.log('  ok - live merged Result shows the final session alone');
}

function testExhaustionSummaryWithoutFinalSessionHeader() {
  const exhaustionSummary =
    '### Previous Session 1\n' +
    'session 1: prepared files\n\n' +
    '---\n\n' +
    'Task failed after 2 sub-sessions';
  assertFinalSessionAlone(
    exhaustionSummary,
    'Task failed after 2 sub-sessions',
    sendReplayedTaskEvents,
  );
  console.log('  ok - exhaustion Result shows the banner alone');
}

function main() {
  testReplayedMergedResultShowsFinalSessionAlone();
  testLiveMergedResultShowsFinalSessionAlone();
  testExhaustionSummaryWithoutFinalSessionHeader();
  console.log('multiSessionResultOrder.test.js: all assertions passed.');
}

main();
