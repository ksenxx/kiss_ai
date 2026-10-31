// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The content menu (media/contentContextMenu.js) and the sidebar tree menu
// (media/treeContextMenu.js) share one keyboard model (`menuKeyNav`) and one
// viewport clamp (`clampMenu`), defined in contentContextMenu.js.  The
// content menu also runs inside the sandboxed iframe of an opened .html file
// from the `SOURCE` text alone, so that text must carry both helpers: this
// suite installs the menu from `SOURCE` in a page that never loaded the
// module, and drives both menus by keyboard at a pointer past the viewport.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {makeRunner} = require('./ui_antipattern_harness');

const MEDIA = path.join(__dirname, '..', 'media');
const runner = makeRunner();
const windows = [];

function test(name, fn) {
  try {
    runner.test(name, fn);
  } finally {
    while (windows.length) windows.pop().close();
  }
}

function makeDom(scripts) {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body><a id="link" href="https://example.com/x">link</a></body></html>',
    {runScripts: 'dangerously', pretendToBeVisual: true, url: 'https://localhost/'},
  );
  const win = dom.window;
  windows.push(win);
  for (const name of scripts) {
    win.eval(fs.readFileSync(path.join(MEDIA, name), 'utf8'));
  }
  return win;
}

function key(win, name) {
  win.document.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: name, bubbles: true, cancelable: true}),
  );
}

function enabledItems(win, selector) {
  return Array.from(win.document.querySelectorAll(selector + ':not(.disabled)'));
}

/** Assert the open menu *el* sits inside *win*'s viewport. */
function assertClamped(win, el, width, height) {
  assert.strictEqual(el.style.left, win.innerWidth - width - 4 + 'px');
  assert.strictEqual(el.style.top, win.innerHeight - height - 4 + 'px');
}

test('the iframe copy (SOURCE) opens, clamps and navigates by keyboard', () => {
  const host = makeDom(['contentContextMenu.js']);
  const source = host.ContentContextMenu.installContentContextMenu.SOURCE;
  const win = makeDom([]);
  assert.strictEqual(win.ContentContextMenu, undefined);
  win.eval(source);
  const link = win.document.getElementById('link');
  link.dispatchEvent(
    new win.MouseEvent('contextmenu', {
      bubbles: true,
      cancelable: true,
      clientX: 5000,
      clientY: 5000,
    }),
  );
  const menu = win.document.getElementById('sorcar-content-context-menu');
  assert.ok(menu, 'the menu opened');
  // Copy (disabled), Copy Link Address, Select All: 3 * 24 + 8 high.
  assertClamped(win, menu, 180, 3 * 24 + 8);
  const items = enabledItems(win, '#sorcar-content-context-menu .sorcar-ctx-item');
  assert.strictEqual(items.length, 2);
  assert.strictEqual(win.document.activeElement, items[0]);
  key(win, 'End');
  assert.strictEqual(win.document.activeElement, items[1]);
  key(win, 'ArrowDown');
  assert.strictEqual(win.document.activeElement, items[0], 'wraps to the first');
  key(win, 'ArrowUp');
  assert.strictEqual(win.document.activeElement, items[1], 'wraps to the last');
  key(win, 'Home');
  assert.strictEqual(win.document.activeElement, items[0]);
  key(win, 'Escape');
  assert.strictEqual(win.document.getElementById('sorcar-content-context-menu'), null);
});

test('the tree menu clamps and navigates with the shared helpers', () => {
  const win = makeDom(['contentContextMenu.js', 'treeContextMenu.js']);
  const ran = [];
  const items = [
    {label: 'One', run: () => ran.push('one')},
    {label: 'Off', enabled: false},
    {label: 'Two', run: () => ran.push('two')},
  ];
  const menu = win.TreeContextMenu.show(win.document, 5000, 5000, items);
  assertClamped(win, menu, 200, 24 * items.length);
  const rows = enabledItems(win, '#sidebar-context-menu .tree-ctx-item');
  assert.strictEqual(win.document.activeElement, rows[0]);
  key(win, 'ArrowDown');
  assert.strictEqual(win.document.activeElement, rows[1], 'skips the disabled row');
  key(win, 'Home');
  key(win, ' ');
  assert.deepStrictEqual(ran, ['one']);
  assert.ok(menu.hidden, 'running an item closes the menu');
});

runner.report('contextMenuSharedNav_wave3');
