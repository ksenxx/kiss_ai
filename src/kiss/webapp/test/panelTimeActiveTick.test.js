// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A thinking_start with no thought tokens yet shows the Thoughts panel
// at once, with a live .panel-time footer (the panel's last child) that
// advances once a second, and stops ticking when the task stops.  The
// page runs on a frozen clock and its intervals on a virtual one
// (fakeIntervals), so the elapsed labels are exact and the test does not
// wait on wall-clock time.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  fakeIntervals,
  parsePanelTimeMs,
} = require('./simplify2_harness');

const clock = {now: 1_700_000_000_000};
let intervals;

function seedClock(win) {
  win.Date.now = () => clock.now;
  intervals = fakeIntervals(win);
}

function advance(ms) {
  clock.now += ms;
  intervals.tick(ms);
}

function footers(panel) {
  return panel.querySelectorAll(':scope > .panel-time');
}

function panelTimeMs(panel) {
  const footer = panel.querySelector(':scope > .panel-time');
  return footer ? parsePanelTimeMs(footer.textContent) : NaN;
}

function runTests() {
  const {win, posted} = makePage({beforeScripts: seedClock});
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'page must post ready with a tabId');
  const TAB = ready.tabId;

  send(win, {type: 'clear', chat_id: 'chat-active-tick', tabId: TAB});
  send(win, {type: 'status', running: true, tabId: TAB, startTs: clock.now});
  const before = new Set(intervals.intervals);
  send(win, {type: 'thinking_start', tabId: TAB});
  const panelTick = intervals.intervals.find(iv => !before.has(iv));
  assert.ok(
    panelTick && panelTick.ms === 1000,
    'the active panel starts the shared 1 s tick',
  );

  const output = win.document.getElementById('output');
  const panels = output.querySelectorAll('.llm-panel');
  assert.strictEqual(
    panels.length,
    1,
    'BUG: thinking_start with no thought tokens must immediately show ' +
      'the Thoughts panel so the user can see that the model is thinking.',
  );
  const panel = panels[0];
  const thinkingContent = panel.querySelector('.think');
  assert.ok(
    thinkingContent,
    'expected the tokenless Thoughts panel to contain an empty ' +
      'thinking text block',
  );
  assert.strictEqual(
    thinkingContent.textContent,
    '',
    'test setup must reproduce the tokenless-thinking state before ' +
      'checking the live time footer',
  );
  assert.strictEqual(
    footers(panel).length,
    1,
    'BUG: a tokenless active .llm-panel must show a .panel-time ' +
      'footer immediately when thinking_start arrives.',
  );

  advance(1000);
  assert.strictEqual(
    panelTimeMs(panel),
    1000,
    'BUG: the live footer of a tokenless active panel must read 1s ' +
      'after one tick; the page is not ticking the active footer.',
  );
  advance(1000);
  assert.strictEqual(
    panelTimeMs(panel),
    2000,
    'BUG: the live .panel-time footer must keep advancing each second',
  );
  assert.strictEqual(
    footers(panel).length,
    1,
    'expected the live .panel-time footer to remain a single ' +
      'direct child after the second tick',
  );

  const last = panel.lastElementChild;
  assert.ok(
    last && last.classList.contains('panel-time'),
    'BUG: live .panel-time must be the LAST child of its panel; last ' +
      'child is <' +
      (last ? last.tagName.toLowerCase() : 'null') +
      '>',
  );
  assert.strictEqual(
    thinkingContent.textContent,
    '',
    'the Thoughts panel must still have zero thought tokens while its ' +
      'live time footer is ticking',
  );

  // The task stops: the last active panel pauses, the shared 1 s tick
  // ends and the footer keeps the time it showed.
  send(win, {type: 'status', running: false, tabId: TAB});
  assert.ok(
    !intervals.intervals.includes(panelTick),
    'the shared panel tick must stop with the last active panel',
  );
  advance(1000);
  assert.strictEqual(panelTimeMs(panel), 2000, 'the paused footer is frozen');

  win.close();
}

try {
  runTests();
  console.log('\nAll tests passed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
