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

const MEDIA = path.join(__dirname, '..', 'media');

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
  return win;
}

function key(win, name, init) {
  win.document.dispatchEvent(
    new win.KeyboardEvent(
      'keydown',
      Object.assign({key: name, bubbles: true, cancelable: true}, init),
    ),
  );
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

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  ok - ${name}`);
  } catch (err) {
    failures.push({name, err});
    console.log(`  not ok - ${name}`);
  }
}

test('items expose tabIndex / aria-disabled and the first enabled item is focused on open', () => {
  const win = makeDom();
  const ctl = win.ContentContextMenu.installContentContextMenu(win.document);
  // No selection: Copy is disabled; on a link, Copy Link Address is enabled.
  ctl.open(10, 10, win.document.getElementById('link'));
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
  ctl.dispose();
});

test('ArrowDown / ArrowUp wrap over enabled items; Home / End jump', () => {
  const win = makeDom();
  const ctl = win.ContentContextMenu.installContentContextMenu(win.document);
  ctl.open(10, 10, win.document.getElementById('link'));
  const enabled = items(win).filter(r => !r.classList.contains('disabled'));
  assert.strictEqual(enabled.length, 2, 'Copy Link Address, Select All');
  key(win, 'ArrowDown');
  assert.strictEqual(active(win), enabled[1]);
  key(win, 'ArrowDown');
  assert.strictEqual(active(win), enabled[0], 'wraps to the first');
  key(win, 'ArrowUp');
  assert.strictEqual(active(win), enabled[1], 'wraps to the last');
  key(win, 'Home');
  assert.strictEqual(active(win), enabled[0]);
  key(win, 'End');
  assert.strictEqual(active(win), enabled[1]);
  ctl.dispose();
});

test('Enter activates the focused item and Escape closes', () => {
  const win = makeDom();
  const ctl = win.ContentContextMenu.installContentContextMenu(win.document);
  const ta = win.document.getElementById('ta');
  ctl.open(10, 10, ta);
  const enabled = items(win).filter(r => !r.classList.contains('disabled'));
  const selectAll = enabled.find(r => r.dataset.action === 'select-all');
  key(win, 'End');
  assert.strictEqual(active(win), selectAll);
  key(win, 'Enter');
  assert.strictEqual(items(win).length, 0, 'the menu closes on activation');
  assert.strictEqual(ta.selectionStart, 0);
  assert.strictEqual(
    ta.selectionEnd,
    'hello world'.length,
    'Select All ran on the field',
  );
  ctl.open(10, 10, ta);
  assert.ok(items(win).length > 0);
  key(win, 'Escape');
  assert.strictEqual(items(win).length, 0, 'Escape closes');
  ctl.dispose();
});

test('Space activates too, and focus returns to the opener on close', () => {
  const win = makeDom();
  const ctl = win.ContentContextMenu.installContentContextMenu(win.document);
  const ta = win.document.getElementById('ta');
  ta.focus();
  assert.strictEqual(active(win), ta);
  ctl.open(10, 10, win.document.getElementById('para'));
  assert.notStrictEqual(active(win), ta, 'focus moved into the menu');
  key(win, 'Escape');
  assert.strictEqual(
    active(win),
    ta,
    'Escape hands focus back to the textarea',
  );
  ctl.open(10, 10, ta);
  key(win, 'End');
  key(win, ' ');
  assert.strictEqual(items(win).length, 0, 'Space activates');
  assert.strictEqual(ta.selectionEnd, 'hello world'.length);
  assert.strictEqual(active(win), ta, 'the field keeps focus after Select All');
  ctl.dispose();
});

test('keys do nothing while the menu is closed', () => {
  const win = makeDom();
  const ctl = win.ContentContextMenu.installContentContextMenu(win.document);
  const ta = win.document.getElementById('ta');
  ta.focus();
  key(win, 'ArrowDown');
  key(win, 'Enter');
  assert.strictEqual(active(win), ta);
  assert.strictEqual(items(win).length, 0);
  ctl.dispose();
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

function typeInto(win, text) {
  const el = active(win);
  el.value = String(el.value || '') + text;
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
  const ctl = win.ContentContextMenu.installContentContextMenu(doc);
  opener.focus();
  opener.dispatchEvent(
    new win.MouseEvent('contextmenu', {
      bubbles: true,
      cancelable: true,
      clientX: 10,
      clientY: 10,
    }),
  );
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
  typeInto(win, 'Z');
  assert.strictEqual(destination.value, 'twoZ');
  assert.strictEqual(
    opener.value,
    'one',
    'typing did not go back to the opener',
  );

  // Keyboard dismissal and clicks on the menu itself still hand focus
  // back to the opener.
  opener.focus();
  ctl.open(10, 10, opener);
  key(win, 'Escape');
  assert.strictEqual(active(win), opener, 'Escape restores the opener');
  ctl.open(10, 10, opener);
  const selectAll = items(win).find(r => r.dataset.action === 'select-all');
  pointerClick(win, selectAll);
  assert.strictEqual(items(win).length, 0, 'clicking an item closes the menu');
  assert.strictEqual(active(win), opener, 'the opener has focus afterwards');
  // A click on inert text leaves focus nowhere (body): restore it.
  ctl.open(10, 10, opener);
  doc.activeElement.blur();
  pointerClick(win, doc.getElementById('para'));
  assert.strictEqual(items(win).length, 0);
  assert.strictEqual(
    active(win),
    opener,
    'a click on non-focusable text still returns focus to the opener',
  );
  ctl.dispose();
});

console.log(`\n${passed} passed, ${failures.length} failed`);
for (const f of failures) {
  console.error(`\n${f.name}\n${f.err && f.err.stack}`);
}
if (failures.length) process.exit(1);
