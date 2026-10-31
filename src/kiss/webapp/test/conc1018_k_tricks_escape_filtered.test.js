// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

/* global require, console */

// Escape on the promptlet sheet while the open editor row is filtered
// out by the search box: the editor is not on screen, so there is no
// inner popup for Escape to close first; it closes the sheet.  The
// editor's draft is latched, not discarded: it comes back once the sheet
// is reopened and the query cleared.  With the editor on screen, Escape
// still closes the editor first and the sheet second.

const assert = require('assert');
const {makePage, click, typeInto} = require('./simplify2_harness');

function rows(doc) {
  return Array.from(doc.getElementById('tricks-list').children);
}

function escape(win, el) {
  el.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key: 'Escape',
      bubbles: true,
      cancelable: true,
    }),
  );
}

function editorOf(doc) {
  return doc.querySelector('#tricks-list .tricks-edit-input');
}

function main() {
  const {win} = makePage();
  const doc = win.document;
  win.__TRICKS__ = ['Mine one', 'Mine two', 'Bundled alpha', 'Bundled beta'];
  win.__MY_TRICKS_COUNT__ = 2;
  const panel = doc.getElementById('tricks-panel');
  const search = doc.getElementById('tricks-search');

  doc.getElementById('tricks-btn').click();
  assert.ok(panel.classList.contains('open'));
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  const ta = editorOf(doc);
  assert.ok(ta, 'editor open on "Mine one"');
  typeInto(win, ta, 'My unsaved draft');

  // The editor is on screen: Escape closes it, not the sheet.
  escape(win, ta);
  assert.ok(!editorOf(doc), 'the editor closed');
  assert.ok(panel.classList.contains('open'), 'the sheet stayed open');

  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  typeInto(win, editorOf(doc), 'My unsaved draft');
  search.focus();
  typeInto(win, search, 'bundled');
  assert.ok(!editorOf(doc), 'the editor row is filtered out');

  escape(win, search);
  assert.ok(
    !panel.classList.contains('open'),
    'Escape closes the sheet while the editor is off screen',
  );

  // The draft is latched: reopen, clear the query, it is back.
  doc.getElementById('tricks-btn').click();
  assert.ok(panel.classList.contains('open'));
  typeInto(win, search, '');
  const back = editorOf(doc);
  assert.ok(back, 'the editor row is back');
  assert.strictEqual(back.value, 'My unsaved draft');
  win.close();
  console.log(
    '  ok - Escape closes the promptlet sheet when its editor is filtered out',
  );
}

main();
