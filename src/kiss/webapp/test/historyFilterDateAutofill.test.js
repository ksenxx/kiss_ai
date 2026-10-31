// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');
const {
  makeRow,
  localIsoDate: localIso,
  visibleTitles,
} = require('./history_helpers');

/** A finished row whose task ran at the unix time `ts`. */
function session(id, ts) {
  return makeRow({
    task_id: id,
    title: 'task ' + id,
    timestamp: ts,
    startTs: ts * 1000,
    endTs: ts * 1000 + 1000,
  });
}

const TS_FIRST = Date.UTC(2024, 0, 15, 12, 0, 0) / 1000;
const TS_MID = Date.UTC(2024, 5, 10, 12, 0, 0) / 1000;
const TS_LAST = Date.UTC(2024, 11, 24, 12, 0, 0) / 1000;
const SESSIONS = [
  session(1, TS_FIRST),
  session(2, TS_MID),
  session(3, TS_LAST),
];

function testAutofillFromDateRange() {
  const {win} = makePage();
  const doc = win.document;

  send(win, {
    type: 'history',
    sessions: SESSIONS,
    offset: 0,
    dateRange: {min: TS_FIRST, max: TS_LAST},
  });

  assert.strictEqual(
    doc.getElementById('hf-from').value,
    localIso(TS_FIRST),
    'From must be auto-filled with the FIRST task date',
  );
  assert.strictEqual(
    doc.getElementById('hf-to').value,
    localIso(TS_LAST),
    'To must be auto-filled with the LAST task date',
  );

  assert.deepStrictEqual(
    visibleTitles(win),
    ['task 1', 'task 2', 'task 3'],
    'auto-filled [first, last] range must keep every row visible',
  );

  win.close();
  console.log('  ok - dates auto-filled from history dateRange');
}

function testAutofillUpdatesWhileUntouched() {
  const {win} = makePage();
  const doc = win.document;

  send(win, {
    type: 'history',
    sessions: SESSIONS.slice(0, 2),
    offset: 0,
    dateRange: {min: TS_FIRST, max: TS_MID},
  });
  assert.strictEqual(doc.getElementById('hf-to').value, localIso(TS_MID));

  send(win, {
    type: 'history',
    sessions: SESSIONS,
    offset: 0,
    dateRange: {min: TS_FIRST, max: TS_LAST},
  });
  assert.strictEqual(
    doc.getElementById('hf-to').value,
    localIso(TS_LAST),
    'untouched inputs must track the refreshed dateRange',
  );

  win.close();
  console.log('  ok - refreshes update the auto-fill while untouched');
}

function testUserValueWins() {
  const {win} = makePage();
  const doc = win.document;

  send(win, {
    type: 'history',
    sessions: SESSIONS,
    offset: 0,
    dateRange: {min: TS_FIRST, max: TS_LAST},
  });

  const hfFrom = doc.getElementById('hf-from');
  hfFrom.value = localIso(TS_MID);
  hfFrom.dispatchEvent(new win.Event('change', {bubbles: true}));

  assert.deepStrictEqual(
    visibleTitles(win),
    ['task 2', 'task 3'],
    'narrowed From date must hide the older task',
  );

  send(win, {
    type: 'history',
    sessions: SESSIONS,
    offset: 0,
    dateRange: {min: TS_FIRST, max: TS_LAST},
  });
  assert.strictEqual(
    hfFrom.value,
    localIso(TS_MID),
    'a user-set date must NOT be overwritten by a later history event',
  );
  assert.deepStrictEqual(
    visibleTitles(win),
    ['task 2', 'task 3'],
    'filter stays applied',
  );

  win.close();
  console.log('  ok - user-set dates survive history refreshes');
}

function testClearButton() {
  const {win} = makePage();
  const doc = win.document;

  const clearBtn = doc.getElementById('hf-date-clear');
  assert.ok(clearBtn, '#hf-date-clear button must exist');
  assert.strictEqual(
    clearBtn.style.display,
    'none',
    'clear button hidden while both dates are empty',
  );

  send(win, {
    type: 'history',
    sessions: SESSIONS,
    offset: 0,
    dateRange: {min: TS_FIRST, max: TS_LAST},
  });
  assert.notStrictEqual(
    clearBtn.style.display,
    'none',
    'clear button visible once dates are set',
  );

  clearBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(doc.getElementById('hf-from').value, '');
  assert.strictEqual(doc.getElementById('hf-to').value, '');
  assert.deepStrictEqual(
    visibleTitles(win),
    ['task 1', 'task 2', 'task 3'],
    'all rows visible after clear',
  );
  assert.strictEqual(
    clearBtn.style.display,
    'none',
    'clear button hides again after clearing',
  );

  send(win, {
    type: 'history',
    sessions: SESSIONS,
    offset: 0,
    dateRange: {min: TS_FIRST, max: TS_LAST},
  });
  assert.strictEqual(
    doc.getElementById('hf-from').value,
    '',
    'cleared dates must stay cleared across refreshes',
  );

  win.close();
  console.log('  ok - clear button empties, hides, and pins the range');
}

function testMissingDateRangeLeavesInputsEmpty() {
  const {win} = makePage();
  const doc = win.document;

  send(win, {type: 'history', sessions: SESSIONS, offset: 0});
  assert.strictEqual(doc.getElementById('hf-from').value, '');
  assert.strictEqual(doc.getElementById('hf-to').value, '');

  win.close();
  console.log('  ok - a missing dateRange leaves the inputs empty');
}

function testEmptyDatabaseClearsUntouchedAutofill() {
  const {win} = makePage();
  const doc = win.document;

  send(win, {
    type: 'history',
    sessions: SESSIONS,
    offset: 0,
    dateRange: {min: TS_FIRST, max: TS_LAST},
  });
  assert.notStrictEqual(doc.getElementById('hf-from').value, '');

  send(win, {
    type: 'history',
    sessions: [],
    offset: 0,
    dateRange: {min: null, max: null},
  });
  assert.strictEqual(
    doc.getElementById('hf-from').value,
    '',
    'an empty database must clear a stale auto-filled From date',
  );
  assert.strictEqual(
    doc.getElementById('hf-to').value,
    '',
    'an empty database must clear a stale auto-filled To date',
  );
  assert.strictEqual(
    doc.getElementById('hf-date-clear').style.display,
    'none',
    'clear button hides when null bounds clear the auto-fill',
  );

  win.close();
  console.log('  ok - empty database clears an untouched auto-fill');
}

function testEpochDateIsValid() {
  const {win} = makePage();
  const doc = win.document;

  send(win, {
    type: 'history',
    sessions: [session(0, 0)],
    offset: 0,
    dateRange: {min: 0, max: 0},
  });
  assert.strictEqual(doc.getElementById('hf-from').value, localIso(0));
  assert.strictEqual(doc.getElementById('hf-to').value, localIso(0));

  win.close();
  console.log('  ok - epoch-zero timestamps remain valid task dates');
}

function main() {
  testAutofillFromDateRange();
  testAutofillUpdatesWhileUntouched();
  testUserValueWins();
  testClearButton();
  testMissingDateRangeLeavesInputsEmpty();
  testEmptyDatabaseClearsUntouchedAutofill();
  testEpochDateIsValid();
  console.log('historyFilterDateAutofill.test.js: all assertions passed.');
}

main();
