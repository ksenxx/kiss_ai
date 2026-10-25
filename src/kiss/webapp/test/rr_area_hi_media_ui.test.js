// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Page regressions for the area-I redundancy fixes:
// - I-R1: every tab-title write clips through clipTabTitle (30 chars +
//   ellipsis), for the active tab (setTaskText) and background tabs
//   (task_events) alike.
// - I-R4: the followup bar is labelled "Suggested next" (its click
//   behaviour, live and replayed, is pinned by
//   followupSuggestionClickCopies.test.js).
// - I-RC2: replacing the input history cache resets a live ArrowUp/
//   ArrowDown cycling position instead of resuming at a stale index
//   into the new list.

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

// Every open tab's title, rendered or not: chat tabs have no row of
// their own any more, so a background chat's title is only in the
// tab records.
function allTabTitles(win) {
  return win._testApi.openTabs().map(t => t.title);
}

function key(win, el, k) {
  el.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: k, bubbles: true, cancelable: true}),
  );
}

const LONG = 'x'.repeat(45);
const CLIPPED = 'x'.repeat(30) + '\u2026';

function testActiveTabTitleClipped() {
  const {win, posted} = makePage();
  const tabId = posted.find(m => m.type === 'ready').tabId;
  send(win, {type: 'setTaskText', tabId, text: LONG});
  assert.ok(
    allTabTitles(win).includes(CLIPPED),
    `active tab title not clipped: ${JSON.stringify(allTabTitles(win))}`,
  );
  console.log('  ok - setTaskText clips the active tab title');
}

function testBackgroundTabTitleClipped() {
  const {win, posted} = makePage();
  const tab1 = posted.find(m => m.type === 'ready').tabId;
  win._testApi.createNewTab(); // becomes the active tab
  send(win, {type: 'task_events', tabId: tab1, task: LONG, events: []});
  assert.ok(
    allTabTitles(win).includes(CLIPPED),
    `background tab title not clipped: ${JSON.stringify(allTabTitles(win))}`,
  );
  console.log('  ok - task_events clips a background tab title');
}

function testFollowupBarLabel() {
  const {win, posted} = makePage();
  const tabId = posted.find(m => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  send(win, {type: 'followup_suggestion', tabId, text: 'Do X next'});
  const bar = win.document.querySelector('#output .followup-bar');
  assert.ok(bar, 'live followup bar not rendered');
  assert.strictEqual(
    bar.querySelector('.fu-label').textContent,
    'Suggested next',
  );
  console.log('  ok - followup bar is labelled "Suggested next"');
}

function testInputHistoryReplaceResetsCycling() {
  const {win} = makePage();
  const inp = win.document.getElementById('task-input');

  send(win, {type: 'inputHistory', tasks: ['old-0', 'old-1', 'old-2']});
  key(win, inp, 'ArrowUp');
  assert.strictEqual(inp.value, 'old-0', 'first ArrowUp recalls newest');
  key(win, inp, 'ArrowUp');
  assert.strictEqual(inp.value, 'old-1', 'second ArrowUp goes further back');

  // The daemon pushes a fresh history while the user is mid-cycle.
  send(win, {type: 'inputHistory', tasks: ['new-0']});

  // A stale index would now step through the NEW list from position 1.
  key(win, inp, 'ArrowDown');
  assert.strictEqual(
    inp.value,
    'old-1',
    'ArrowDown after a history replace must not jump into the new list',
  );

  // Cycling restarts cleanly from the top of the new list.
  inp.value = '';
  key(win, inp, 'ArrowUp');
  assert.strictEqual(
    inp.value,
    'new-0',
    'ArrowUp after a history replace starts at the new newest entry',
  );
  console.log('  ok - inputHistory replace resets the cycling position');
}

function main() {
  testActiveTabTitleClipped();
  testBackgroundTabTitleClipped();
  testFollowupBarLabel();
  testInputHistoryReplaceResetsCycling();
  console.log('rr_area_hi_media_ui: all tests passed');
  process.exit(0);
}

try {
  main();
} catch (err) {
  console.error(err);
  process.exit(1);
}
