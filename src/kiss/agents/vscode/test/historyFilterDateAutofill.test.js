// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the History filter bar's From/To date
// AUTO-FILL: the backend's ``history`` event now carries a
// ``dateRange`` payload with the FIRST and LAST task timestamps
// (epoch seconds) from ~/.kiss/sorcar.db, and the webview fills the
// empty ``#hf-from`` / ``#hf-to`` date inputs from it.
//
// Requirements driven by this test:
//
//   1. On a ``history`` event with
//      ``dateRange: {min: <sec>, max: <sec>}``, the empty From/To
//      inputs are filled with the LOCAL calendar dates of those
//      timestamps (``YYYY-MM-DD``, same format the inputs produce).
//
//   2. The auto-filled [min, max] range is inclusive: every history
//      row stays visible after the fill.
//
//   3. A later ``history`` event with a wider range updates the
//      auto-filled values (the user has not touched the inputs).
//
//   4. Once the USER sets a date (a ``change`` event on the input),
//      subsequent ``history`` events must NOT overwrite it.
//
//   5. A ``#hf-date-clear`` button clears both dates, un-hides all
//      rows, is hidden while both inputs are empty, and marks the
//      range as user-managed so refreshes do not re-fill it.
//
//   6. A missing / null ``dateRange`` leaves the inputs empty.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/historyFilterDateAutofill.test.js

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
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Local-timezone YYYY-MM-DD of an epoch-seconds timestamp. */
function localIso(sec) {
  const d = new Date(sec * 1000);
  const p = n => (n < 10 ? '0' + n : '' + n);
  return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate());
}

function visibleCount(win) {
  const list = win.document.getElementById('history-list');
  let n = 0;
  list.querySelectorAll('.sidebar-item').forEach(r => {
    if (r.style.display !== 'none') n++;
  });
  return n;
}

function session(id, ts) {
  return {
    id: 'chat' + id,
    task_id: id,
    title: 'task ' + id,
    timestamp: ts,
    preview: 'task ' + id,
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: '',
    startTs: ts * 1000,
    endTs: ts * 1000 + 1000,
  };
}

// Fixed timestamps: three tasks over three distinct days.
const TS_FIRST = Date.UTC(2024, 0, 15, 12, 0, 0) / 1000;
const TS_MID = Date.UTC(2024, 5, 10, 12, 0, 0) / 1000;
const TS_LAST = Date.UTC(2024, 11, 24, 12, 0, 0) / 1000;
const SESSIONS = [
  session(1, TS_FIRST),
  session(2, TS_MID),
  session(3, TS_LAST),
];

function testAutofillFromDateRange() {
  const {win} = makeWebview();
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

  // The auto-filled range is inclusive — nothing gets hidden.
  assert.strictEqual(
    visibleCount(win),
    3,
    'auto-filled [first, last] range must keep every row visible',
  );

  win.close();
  console.log('  ok - dates auto-filled from history dateRange');
}

function testAutofillUpdatesWhileUntouched() {
  const {win} = makeWebview();
  const doc = win.document;

  send(win, {
    type: 'history',
    sessions: SESSIONS.slice(0, 2),
    offset: 0,
    dateRange: {min: TS_FIRST, max: TS_MID},
  });
  assert.strictEqual(doc.getElementById('hf-to').value, localIso(TS_MID));

  // A newer task arrives; the backend now reports a wider range.
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
  const {win} = makeWebview();
  const doc = win.document;

  send(win, {
    type: 'history',
    sessions: SESSIONS,
    offset: 0,
    dateRange: {min: TS_FIRST, max: TS_LAST},
  });

  // The user narrows the From date (same path the custom picker
  // takes: set .value then dispatch a change event).
  const hfFrom = doc.getElementById('hf-from');
  hfFrom.value = localIso(TS_MID);
  hfFrom.dispatchEvent(new win.Event('change', {bubbles: true}));

  assert.strictEqual(
    visibleCount(win),
    2,
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
  assert.strictEqual(visibleCount(win), 2, 'filter stays applied');

  win.close();
  console.log('  ok - user-set dates survive history refreshes');
}

function testClearButton() {
  const {win} = makeWebview();
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
  assert.strictEqual(visibleCount(win), 3, 'all rows visible after clear');
  assert.strictEqual(
    clearBtn.style.display,
    'none',
    'clear button hides again after clearing',
  );

  // A refresh must NOT re-fill the range the user just cleared.
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
  const {win} = makeWebview();
  const doc = win.document;

  send(win, {type: 'history', sessions: SESSIONS, offset: 0});
  assert.strictEqual(doc.getElementById('hf-from').value, '');
  assert.strictEqual(doc.getElementById('hf-to').value, '');

  send(win, {
    type: 'history',
    sessions: [],
    offset: 0,
    dateRange: {min: null, max: null},
  });
  assert.strictEqual(doc.getElementById('hf-from').value, '');
  assert.strictEqual(doc.getElementById('hf-to').value, '');

  win.close();
  console.log('  ok - missing/null dateRange leaves the inputs empty');
}

function testEmptyDatabaseClearsUntouchedAutofill() {
  const {win} = makeWebview();
  const doc = win.document;

  send(win, {
    type: 'history',
    sessions: SESSIONS,
    offset: 0,
    dateRange: {min: TS_FIRST, max: TS_LAST},
  });
  assert.notStrictEqual(doc.getElementById('hf-from').value, '');

  // All tasks are deleted while the dates remain program-managed.
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
  const {win} = makeWebview();
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
