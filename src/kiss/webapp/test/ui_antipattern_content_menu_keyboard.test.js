// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// UI anti-pattern fix A12 / A15 for the content context menu
// (media/contentContextMenu.js): the Copy / Paste / Select All menu is
// keyboard operable like treeContextMenu.js already is.  Items carry a
// roving tabIndex and aria-disabled, focus moves to the first enabled item
// when the menu opens, Arrow / Home / End move, Enter / Space activate,
// Escape closes, and focus returns to the element that had it before.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {MEDIA} = require('./simplify2_harness');
const {makeRunner, rightClick: open, key} = require('./ui_antipattern_harness');

// Every page a test builds; the runner closes them when the test has
// ended, pass or fail, so nothing the menu armed keeps node alive.
const windows = [];
const {test, report} = makeRunner({windows});

function makeDom() {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body>' +
      '<p id="para">plain text</p>' +
      '<a id="link" href="https://example.com/x">link</a>' +
      '<textarea id="ta">hello world</textarea>' +
      '</body></html>',
    {
      runScripts: 'dangerously',
      pretendToBeVisual: true,
      url: 'https://localhost/',
    },
  );
  const win = dom.window;
  win.eval(fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8'));
  assert.ok(
    win.ContentContextMenu,
    'contentContextMenu.js must install window.ContentContextMenu',
  );
  windows.push(win);
  return win;
}

function items(win) {
  return Array.from(
    win.document.querySelectorAll(
      '#sorcar-content-context-menu .sorcar-ctx-item',
    ),
  );
}

function active(win) {
  return win.document.activeElement;
}


test('items expose tabIndex / aria-disabled and the first enabled item is focused on open', () => {
  const win = makeDom();
  win.ContentContextMenu.installContentContextMenu(win.document);
  // No selection: Copy is disabled; on a link, Copy Link Address is enabled.
  open(win, win.document.getElementById('link'));
  const rows = items(win);
  const byAction = Object.fromEntries(rows.map(r => [r.dataset.action, r]));
  assert.ok(byAction.copy.classList.contains('disabled'));
  assert.strictEqual(byAction.copy.getAttribute('aria-disabled'), 'true');
  assert.strictEqual(byAction.copy.tabIndex, -1);
  assert.strictEqual(byAction['copy-link'].getAttribute('aria-disabled'), null);
  assert.strictEqual(byAction['copy-link'].tabIndex, 0);
  assert.strictEqual(byAction['select-all'].tabIndex, 0);
  assert.strictEqual(
    active(win),
    byAction['copy-link'],
    'first ENABLED item has focus',
  );
});

test('ArrowDown / ArrowUp wrap over enabled items; Home / End jump', () => {
  const win = makeDom();
  win.ContentContextMenu.installContentContextMenu(win.document);
  open(win, win.document.getElementById('link'));
  const enabled = items(win).filter(r => !r.classList.contains('disabled'));
  assert.strictEqual(enabled.length, 2, 'Copy Link Address, Select All');
  key(win, win.document, 'ArrowDown');
  assert.strictEqual(active(win), enabled[1]);
  key(win, win.document, 'ArrowDown');
  assert.strictEqual(active(win), enabled[0], 'wraps to the first');
  key(win, win.document, 'ArrowUp');
  assert.strictEqual(active(win), enabled[1], 'wraps to the last');
  key(win, win.document, 'Home');
  assert.strictEqual(active(win), enabled[0]);
  key(win, win.document, 'End');
  assert.strictEqual(active(win), enabled[1]);
});

test('Enter activates the focused item and Escape closes', () => {
  const win = makeDom();
  win.ContentContextMenu.installContentContextMenu(win.document);
  const ta = win.document.getElementById('ta');
  open(win, ta);
  const enabled = items(win).filter(r => !r.classList.contains('disabled'));
  const selectAll = enabled.find(r => r.dataset.action === 'select-all');
  key(win, win.document, 'End');
  assert.strictEqual(active(win), selectAll);
  key(win, win.document, 'Enter');
  assert.strictEqual(items(win).length, 0, 'the menu closes on activation');
  assert.strictEqual(ta.selectionStart, 0);
  assert.strictEqual(
    ta.selectionEnd,
    'hello world'.length,
    'Select All ran on the field',
  );
  open(win, ta);
  assert.ok(items(win).length > 0);
  key(win, win.document, 'Escape');
  assert.strictEqual(items(win).length, 0, 'Escape closes');
});

test('Space activates too, and focus returns to the opener on close', () => {
  const win = makeDom();
  win.ContentContextMenu.installContentContextMenu(win.document);
  const ta = win.document.getElementById('ta');
  ta.focus();
  assert.strictEqual(active(win), ta);
  open(win, win.document.getElementById('para'));
  assert.notStrictEqual(active(win), ta, 'focus moved into the menu');
  key(win, win.document, 'Escape');
  assert.strictEqual(
    active(win),
    ta,
    'Escape hands focus back to the textarea',
  );
  open(win, ta);
  key(win, win.document, 'End');
  key(win, win.document, ' ');
  assert.strictEqual(items(win).length, 0, 'Space activates');
  assert.strictEqual(ta.selectionEnd, 'hello world'.length);
  assert.strictEqual(active(win), ta, 'the field keeps focus after Select All');
});

test('keys do nothing while the menu is closed', () => {
  const win = makeDom();
  win.ContentContextMenu.installContentContextMenu(win.document);
  const ta = win.document.getElementById('ta');
  ta.focus();
  key(win, win.document, 'ArrowDown');
  key(win, win.document, 'Enter');
  assert.strictEqual(active(win), ta);
  assert.strictEqual(items(win).length, 0);
});

// The browser focuses a clicked control on mousedown, before the click
// reaches the document handler that closes the menu; jsdom has no such
// default action, so the test performs it.
function pointerClick(win, el) {
  el.dispatchEvent(
    new win.MouseEvent('mousedown', {bubbles: true, cancelable: true}),
  );
  if (typeof el.focus === 'function' && el.tabIndex >= 0) el.focus();
  el.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true, detail: 1}),
  );
}

test('R3-1: dismissing the menu by clicking another field keeps focus (and typing) there', () => {
  const win = makeDom();
  const doc = win.document;
  doc.body.insertAdjacentHTML(
    'beforeend',
    '<input id="opener" value="one"><input id="destination" value="two">',
  );
  const opener = doc.getElementById('opener');
  const destination = doc.getElementById('destination');
  win.ContentContextMenu.installContentContextMenu(doc);
  opener.focus();
  open(win, opener);
  assert.ok(items(win).length > 0, 'right-click opened the menu');
  assert.ok(
    items(win).includes(active(win)),
    'focus is on the first menu item',
  );

  pointerClick(win, destination);
  assert.strictEqual(items(win).length, 0, 'the outside click closed the menu');
  assert.strictEqual(
    active(win),
    destination,
    'focus stays on the field the user clicked, not the old opener',
  );

  // Keyboard dismissal and clicks on the menu itself still hand focus
  // back to the opener.
  opener.focus();
  open(win, opener);
  key(win, win.document, 'Escape');
  assert.strictEqual(active(win), opener, 'Escape restores the opener');
  open(win, opener);
  const selectAll = items(win).find(r => r.dataset.action === 'select-all');
  pointerClick(win, selectAll);
  assert.strictEqual(items(win).length, 0, 'clicking an item closes the menu');
  assert.strictEqual(active(win), opener, 'the opener has focus afterwards');
  // A click on inert text leaves focus nowhere (body): restore it.
  open(win, opener);
  doc.activeElement.blur();
  pointerClick(win, doc.getElementById('para'));
  assert.strictEqual(items(win).length, 0);
  assert.strictEqual(
    active(win),
    opener,
    'a click on non-focusable text still returns focus to the opener',
  );
});

// Tab / Shift+Tab leave the menu by the browser's own focus move (there
// is no focus trap), so the menu closes then.  Left open, its
// document-level key handler went on intercepting Arrow / Home / End /
// Enter in whatever control the focus had landed on.  jsdom performs
// no default focus move for Tab, so the test moves focus as the
// browser would, on from the opener the closing menu handed it to.
test('Tab / Shift+Tab out of the menu closes it; the keys then stay with the new control', () => {
  const win = makeDom();
  const doc = win.document;
  const ta = doc.getElementById('ta');
  win.ContentContextMenu.installContentContextMenu(doc);
  for (const shiftKey of [true, false]) {
    ta.focus();
    open(win, doc.getElementById('para'));
    assert.ok(items(win).includes(active(win)), 'focus is on a menu item');
    const tab = new win.KeyboardEvent('keydown', {
      key: 'Tab',
      shiftKey,
      bubbles: true,
      cancelable: true,
    });
    active(win).dispatchEvent(tab);
    assert.strictEqual(tab.defaultPrevented, false, 'the browser moves focus');
    assert.strictEqual(items(win).length, 0, 'Tab closed the menu');
    assert.strictEqual(active(win), ta, 'focus is back on the opener');
    const link = doc.getElementById('link');
    link.focus();
    const arrow = new win.KeyboardEvent('keydown', {
      key: 'ArrowDown',
      bubbles: true,
      cancelable: true,
    });
    link.dispatchEvent(arrow);
    assert.strictEqual(arrow.defaultPrevented, false, 'ArrowDown stays');
    assert.strictEqual(active(win), link, 'focus stays on the control');
    assert.strictEqual(items(win).length, 0);
  }
});

report('ui_antipattern_content_menu_keyboard');
