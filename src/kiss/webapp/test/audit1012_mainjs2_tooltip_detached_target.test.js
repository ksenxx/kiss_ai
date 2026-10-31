// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Audit 2026-10-12 (mainjs-2): the [data-tooltip] hover timer in
// media/main.js (showTooltipFor, armed 400 ms after mouseover) keeps a
// reference to the hovered element and shows its tooltip even when the
// element has left the document meanwhile.  History rows carry their
// preview as data-tooltip and the list is rebuilt on every `history`
// reply (one per status / tasks_updated), so hovering a row while a
// task runs replaces it under the pointer: the timer then measures a
// detached node (a 0x0 rect at the origin) and parks the tooltip in the
// top-left corner of the window, where it stays -- the removed element
// never fires the mouseout that would hide it.

const assert = require('assert');
const harness = require('./simplify2_harness');

// The 400 ms hover timer is recorded and fired by hand, never waited for.
function prepare(win) {
  win.hoverTimers = harness.recordTimers(win, 400);
}

function makePage() {
  return harness.makePage({beforeScripts: prepare}).win;
}

function testRemovedTargetShowsNoTooltip() {
  const win = makePage();
  const row = win.document.createElement('div');
  row.className = 'sidebar-item';
  row.dataset.tooltip = 'the task preview';
  win.document.getElementById('history-list').appendChild(row);
  row.dispatchEvent(new win.MouseEvent('mouseover', {bubbles: true}));
  // The history re-render replaces the row before the 400 ms timer.
  row.remove();
  // A removed element fires no mouseout: the hover delay is still armed.
  assert.strictEqual(
    win.hoverTimers.filter(t => !t.cleared).length,
    1,
    'the hover armed one live 400 ms delay',
  );
  harness.fireTimers(win.hoverTimers);
  const tip = win.document.getElementById('custom-tooltip');
  assert.ok(
    !tip.classList.contains('visible'),
    'no tooltip for an element that left the document (got "' +
      tip.textContent + '" at ' + tip.style.left + ',' + tip.style.top + ')',
  );
  win.close();
  console.log('  ok - a removed hover target shows no tooltip');
}

// An OPEN tooltip whose anchor is replaced by a re-render must close
// too: a removed element fires neither focusout nor mouseout, so a
// keyboard user who tabbed onto a Schedule row sees the row's prompt
// stay on screen after the 30 s panel poll rebuilt the list.
const cronJob = {
  id: 'j1',
  name: 'Digest',
  schedule: 'every 5m',
  kind: 'prompt',
  what: 'summarize the inbox',
  enabled: true,
  running: false,
  nextRunAt: 0,
  lastRunAt: 0,
  lastStatus: '',
};

async function testOpenTooltipClosesWhenItsAnchorIsReRendered() {
  const win = makePage();
  harness.send(win, {type: 'cronJobs', jobs: [cronJob]});
  const list = win.document.getElementById('meta-schedule-list');
  const row = list.querySelector('.sched-row');
  assert.ok(row && row.dataset.tooltip, 'the schedule row carries a tooltip');
  row.focus();
  const tip = win.document.getElementById('custom-tooltip');
  assert.ok(tip.classList.contains('visible'), 'focus opens the tooltip');
  // The panel poll answers; renderCronJobs rebuilds every row.
  harness.send(win, {type: 'cronJobs', jobs: [cronJob]});
  assert.ok(!row.isConnected, 'the focused row was replaced');
  // The anchor watch is a MutationObserver: one turn for it to deliver.
  await harness.sleep(0);
  assert.ok(
    !tip.classList.contains('visible'),
    'the tooltip closes with the row it described',
  );
  // A fresh row still gets its tooltip afterwards.
  list.querySelector('.sched-row').focus();
  assert.ok(tip.classList.contains('visible'), 'the new row\'s tooltip opens');
  win.close();
  console.log('  ok - an open tooltip closes when its anchor is re-rendered');
}

async function runTests() {
  testRemovedTargetShowsNoTooltip();
  await testOpenTooltipClosesWhenItsAnchorIsReRendered();
}

runTests().then(
  () => {
    console.log('\n2 passed, 0 failed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
