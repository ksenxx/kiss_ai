// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// The inline Delete / Cancel confirmation (main.js makeSidebarDeleteConfirm)
// posts a delete ONCE and opens nothing: the confirm click spends its
// Delete button but keeps the box in place, so the second click of a
// double click (or Enter held on the focused button) lands on a disabled
// button rather than on the row under it, while the row waits for the
// daemon's history refetch.  The confirmation's own box swallows clicks
// (a click in the gap between its buttons must not open the task row
// under it), and its buttons are type="button" so they never submit a
// surrounding form.

'use strict';

const assert = require('assert');
const {send, click, ofType} = require('./simplify2_harness');
const {
  loadHistory,
  makeSession,
  makeConnectedRemotePage: makePage,
} = require('./history_helpers');

/**
 * Load a one-row history and return that row with its delete controls:
 * the trash icon `del` and the confirmation box with its Delete (`yes`)
 * and Cancel (`no`) buttons.
 */
function loadRow(win) {
  loadHistory(win, [makeSession()]);
  const row = win.document.querySelector('#history-list .running-item');
  const confirm = row.querySelector(
    '.sidebar-item-actions > .sidebar-item-confirm',
  );
  return {
    row,
    del: row.querySelector('.sidebar-item-actions > .sidebar-item-delete'),
    confirm,
    yes: confirm.querySelector('.sidebar-confirm-yes'),
    no: confirm.querySelector('.sidebar-confirm-no'),
  };
}

/**
 * What a browser's hit test finds at the point where `el` was: `el`
 * itself while it is on screen, else the nearest displayed ancestor
 * (jsdom has no layout, so `elementFromPoint` cannot answer this).
 */
function hitTarget(el) {
  let node = el;
  while (node) {
    let shown = true;
    for (let a = node; a && a.style; a = a.parentElement) {
      if (a.style.display === 'none') shown = false;
    }
    if (shown) return node;
    node = node.parentElement;
  }
  return null;
}

/**
 * A browser's click at a point: the event is dispatched on what the
 * hit test finds there, and never on a disabled form control.
 */
function clickAt(win, el) {
  const target = hitTarget(el);
  if (!target.disabled) click(win, target);
  return target;
}

// Chromium, desktop flat history: the two clicks of a double click on
// the confirmation's Delete land on the same point.  Folding the box
// away on the first click put the row's action strip under the second
// one, which bubbled to the row and opened (openTab + resumeSession)
// the very task being deleted.
function testDoubleClickDeletesOnceAndOpensNothing() {
  const {win, posted} = makePage();
  const {row, del, confirm, yes, no} = loadRow(win);
  click(win, del);
  assert.strictEqual(clickAt(win, yes), yes);
  assert.strictEqual(confirm.style.display, '', 'the box stays in place');
  assert.strictEqual(yes.disabled, true, 'Delete is spent');
  assert.strictEqual(no.disabled, false, 'Cancel still works');
  assert.strictEqual(win.document.activeElement, no, 'focus moved to Cancel');
  // The second click of the double click, at the same point.
  assert.strictEqual(clickAt(win, yes), yes, 'still the spent Delete');
  // jsdom dispatches a synthetic click even on a disabled button (a
  // browser never does); the handler itself must refuse it.
  click(win, yes);
  assert.strictEqual(
    ofType(posted, 'deleteTask').length,
    1,
    'a double click posts the delete once',
  );
  assert.strictEqual(ofType(posted, 'openTab').length, 0, 'no tab opened');
  assert.strictEqual(
    ofType(posted, 'resumeSession').length,
    0,
    'the deleted task was not opened',
  );
  // The daemon's refetch after the delete drops the row.
  loadHistory(win, []);
  assert.strictEqual(row.isConnected, false, 'the row is gone');
  for (const b of confirm.querySelectorAll('button')) {
    assert.strictEqual(b.type, 'button', 'never a submit button');
  }
  assert.strictEqual(del.type, 'button');
  win.close();
  console.log(
    '  ok - a same-point double click posts one delete, opens nothing',
  );
}

// A refused delete (the daemon only sends a notice, no refetch) leaves
// the spent box up: Cancel folds it, the trash icon asks afresh.
function testRefusedDeleteCancelsAndAsksAgain() {
  const {win, posted} = makePage();
  const {del, confirm, yes, no} = loadRow(win);
  click(win, del);
  clickAt(win, yes);
  send(win, {
    type: 'notice',
    text: 'Cannot delete the task: its task is still running. Stop it first.',
  });
  assert.strictEqual(yes.disabled, true);
  click(win, no);
  assert.strictEqual(confirm.style.display, 'none', 'Cancel folds the box');
  assert.strictEqual(del.style.display, '', 'the trash icon is back');
  assert.strictEqual(win.document.activeElement, del);
  click(win, del);
  assert.strictEqual(confirm.style.display, '', 'asked again');
  assert.strictEqual(yes.disabled, false, 'with a live Delete');
  clickAt(win, yes);
  assert.strictEqual(
    ofType(posted, 'deleteTask').length,
    2,
    'a second, deliberate confirmation posts again',
  );
  assert.strictEqual(ofType(posted, 'resumeSession').length, 0);
  win.close();
  console.log('  ok - a refused delete can be cancelled and asked again');
}

function testConfirmBoxSwallowsClicks() {
  const {win, posted} = makePage();
  const {del, confirm} = loadRow(win);
  click(win, del);
  // A click on the box itself (the gap between Delete and Cancel).
  click(win, confirm);
  assert.strictEqual(
    ofType(posted, 'resumeSession').length,
    0,
    'the click did not open the task under the confirmation',
  );
  assert.strictEqual(confirm.style.display, '', 'the confirm stays up');
  win.close();
  console.log('  ok - the confirmation box swallows clicks');
}

try {
  testDoubleClickDeletesOnceAndOpensNothing();
  testRefusedDeleteCancelsAndAsksAgain();
  testConfirmBoxSwallowsClicks();
  console.log('audit1020_delete_confirm.test.js passed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
