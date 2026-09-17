// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// JSDOM test for media/treeContextMenu.js: the shared VS Code-style
// context menu the Explorer and Source Control trees open on
// right-click.  Covers item rendering (labels, keybindings, disabled
// rows, separator squashing), viewport clamping, keyboard navigation
// (Up / Down / Home / End / Enter / Space / Escape), outside-click and
// window-blur dismissal, and the `closed` callback.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeDom() {
  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.eval(fs.readFileSync(path.join(MEDIA, 'treeContextMenu.js'), 'utf8'));
  assert.ok(
    win.TreeContextMenu,
    'treeContextMenu.js must install window.TreeContextMenu',
  );
  return win;
}

function key(win, name) {
  win.document.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key: name,
      bubbles: true,
      cancelable: true,
    }),
  );
}

function rows(win) {
  return Array.from(
    win.document.querySelectorAll('#sidebar-context-menu .tree-ctx-item'),
  );
}

function testRenderingAndSeparators() {
  const win = makeDom();
  const ran = [];
  const el = win.TreeContextMenu.show(win.document, 10, 20, [
    {separator: true},
    {id: 'newFile', label: 'New File...', run: () => ran.push('newFile')},
    {separator: true},
    {separator: true},
    {label: 'Rename...', key: 'F2', run: () => ran.push('rename')},
    {
      label: 'Paste',
      key: 'Ctrl+V',
      enabled: false,
      run: () => ran.push('paste'),
    },
    null,
    {separator: true},
  ]);
  assert.strictEqual(el.id, 'sidebar-context-menu');
  assert.strictEqual(el.getAttribute('role'), 'menu');
  assert.ok(!el.hidden, 'menu must be visible after show()');
  assert.ok(win.TreeContextMenu.isOpen());
  // Leading, doubled and trailing separators are squashed like VS Code.
  const children = Array.from(el.children).map(c => c.className);
  assert.deepStrictEqual(children, [
    'tree-ctx-item',
    'tree-ctx-sep',
    'tree-ctx-item',
    'tree-ctx-item disabled',
  ]);
  const items = rows(win);
  assert.strictEqual(items[0].dataset.action, 'newFile');
  assert.strictEqual(
    items[0].querySelector('.tree-ctx-label').textContent,
    'New File...',
  );
  assert.strictEqual(
    items[0].querySelector('.tree-ctx-key'),
    null,
    'no key span without a key',
  );
  assert.strictEqual(items[1].querySelector('.tree-ctx-key').textContent, 'F2');
  assert.strictEqual(items[2].getAttribute('aria-disabled'), 'true');
  assert.strictEqual(items[2].tabIndex, -1);
  assert.strictEqual(el.style.left, '10px');
  assert.strictEqual(el.style.top, '20px');
  assert.strictEqual(
    win.document.activeElement,
    items[0],
    'first enabled item takes focus',
  );

  // A disabled item cannot be run and does not close the menu.
  items[2].click();
  assert.deepStrictEqual(ran, []);
  assert.ok(
    win.TreeContextMenu.isOpen(),
    'clicking a disabled item keeps the menu open',
  );

  // Clicking an enabled item closes the menu first, then runs it.
  items[1].click();
  assert.deepStrictEqual(ran, ['rename']);
  assert.ok(!win.TreeContextMenu.isOpen());
  assert.ok(el.hidden);
  assert.strictEqual(el.childElementCount, 0, 'closing empties the menu');
  console.log(
    'ok: items, keybindings, disabled rows and separators render like VS Code',
  );
}

function testViewportClamping() {
  const win = makeDom();
  // jsdom reports offsetWidth/Height as 0, so the menu falls back to a
  // 200px width and 24px per item.
  const el = win.TreeContextMenu.show(
    win.document,
    win.innerWidth + 50,
    win.innerHeight + 50,
    [
      {label: 'A', run: () => {}},
      {label: 'B', run: () => {}},
    ],
  );
  assert.strictEqual(el.style.left, win.innerWidth - 200 - 4 + 'px');
  assert.strictEqual(el.style.top, win.innerHeight - 48 - 4 + 'px');
  // Negative coordinates clamp to 0.
  win.TreeContextMenu.show(win.document, -30, -30, [
    {label: 'A', run: () => {}},
  ]);
  assert.strictEqual(el.style.left, '0px');
  assert.strictEqual(el.style.top, '0px');
  win.TreeContextMenu.close();
  console.log('ok: menu is kept inside the viewport');
}

function testKeyboardNavigation() {
  const win = makeDom();
  const ran = [];
  let closedCalls = 0;
  win.TreeContextMenu.show(
    win.document,
    0,
    0,
    [
      {label: 'One', run: () => ran.push('one')},
      {label: 'Off', enabled: false, run: () => ran.push('off')},
      {label: 'Two', run: () => ran.push('two')},
      {label: 'Three', run: () => ran.push('three')},
    ],
    () => closedCalls++,
  );
  const items = rows(win);
  const enabled = [items[0], items[2], items[3]];
  const active = () => win.document.activeElement;
  assert.strictEqual(active(), enabled[0]);
  key(win, 'ArrowDown');
  assert.strictEqual(active(), enabled[1], 'ArrowDown skips the disabled row');
  key(win, 'ArrowDown');
  assert.strictEqual(active(), enabled[2]);
  key(win, 'ArrowDown');
  assert.strictEqual(active(), enabled[0], 'ArrowDown wraps to the first item');
  key(win, 'ArrowUp');
  assert.strictEqual(active(), enabled[2], 'ArrowUp wraps to the last item');
  key(win, 'ArrowUp');
  assert.strictEqual(active(), enabled[1]);
  key(win, 'Home');
  assert.strictEqual(active(), enabled[0]);
  key(win, 'End');
  assert.strictEqual(active(), enabled[2]);
  // Focus lost from the menu (e.g. the user tabbed away): ArrowDown
  // starts from the first item again.
  enabled[2].blur();
  key(win, 'ArrowDown');
  assert.strictEqual(active(), enabled[0]);
  key(win, 'ArrowDown');
  key(win, 'Enter');
  assert.deepStrictEqual(ran, ['two'], 'Enter runs the focused item');
  assert.strictEqual(closedCalls, 1, 'the closed callback fires once');
  assert.ok(!win.TreeContextMenu.isOpen());

  // Space also runs; Escape only closes.
  win.TreeContextMenu.show(win.document, 0, 0, [
    {label: 'Go', run: () => ran.push('go')},
  ]);
  key(win, ' ');
  assert.deepStrictEqual(ran, ['two', 'go']);
  win.TreeContextMenu.show(win.document, 0, 0, [
    {label: 'Go', run: () => ran.push('go2')},
  ]);
  key(win, 'Escape');
  assert.ok(!win.TreeContextMenu.isOpen(), 'Escape closes the menu');
  assert.deepStrictEqual(ran, ['two', 'go'], 'Escape runs nothing');
  // Keys with no menu open are ignored, and Enter with nothing focused
  // runs nothing.
  key(win, 'Enter');
  win.TreeContextMenu.show(win.document, 0, 0, [
    {label: 'Go', run: () => ran.push('go3')},
  ]);
  win.document.activeElement.blur();
  key(win, 'Enter');
  assert.deepStrictEqual(ran, ['two', 'go']);
  // A menu of only disabled rows ignores navigation keys.
  win.TreeContextMenu.show(win.document, 0, 0, [
    {label: 'Off', enabled: false},
  ]);
  key(win, 'ArrowDown');
  assert.ok(win.TreeContextMenu.isOpen());
  win.TreeContextMenu.close();
  console.log('ok: keyboard navigation matches VS Code menus');
}

function testDismissal() {
  const win = makeDom();
  let closedCalls = 0;
  const el = win.TreeContextMenu.show(
    win.document,
    0,
    0,
    [{label: 'A', run: () => {}}],
    () => closedCalls++,
  );
  // Pressing inside the menu keeps it open.
  el.firstElementChild.dispatchEvent(
    new win.MouseEvent('mousedown', {bubbles: true}),
  );
  assert.ok(
    win.TreeContextMenu.isOpen(),
    'mousedown inside the menu keeps it open',
  );
  // Pressing anywhere else closes it.
  win.document.body.dispatchEvent(
    new win.MouseEvent('mousedown', {bubbles: true}),
  );
  assert.ok(!win.TreeContextMenu.isOpen(), 'mousedown outside closes the menu');
  assert.strictEqual(closedCalls, 1);
  // A right-click elsewhere closes it too (the tree then opens a new one).
  win.TreeContextMenu.show(win.document, 0, 0, [{label: 'A', run: () => {}}]);
  win.document.body.dispatchEvent(
    new win.MouseEvent('contextmenu', {bubbles: true}),
  );
  assert.ok(!win.TreeContextMenu.isOpen());
  // Window blur / resize close it.
  win.TreeContextMenu.show(win.document, 0, 0, [{label: 'A', run: () => {}}]);
  win.dispatchEvent(new win.Event('blur'));
  assert.ok(!win.TreeContextMenu.isOpen(), 'window blur closes the menu');
  win.TreeContextMenu.show(win.document, 0, 0, [{label: 'A', run: () => {}}]);
  win.dispatchEvent(new win.Event('resize'));
  assert.ok(!win.TreeContextMenu.isOpen(), 'window resize closes the menu');
  // Showing again reuses the single menu element and replaces its items.
  win.TreeContextMenu.show(win.document, 0, 0, [{label: 'X', run: () => {}}]);
  win.TreeContextMenu.show(win.document, 0, 0, [{label: 'Y', run: () => {}}]);
  assert.strictEqual(
    win.document.querySelectorAll('#sidebar-context-menu').length,
    1,
  );
  assert.strictEqual(rows(win).length, 1);
  assert.strictEqual(rows(win)[0].textContent, 'Y');
  // close() when nothing is open is a no-op; a closed callback that is
  // not a function is ignored.
  win.TreeContextMenu.close();
  win.TreeContextMenu.close();
  win.TreeContextMenu.show(
    win.document,
    0,
    0,
    [{label: 'Z'}],
    'not-a-function',
  );
  rows(win)[0].click();
  assert.ok(!win.TreeContextMenu.isOpen());
  console.log('ok: outside clicks, blur and resize dismiss the menu');
}

function main() {
  testRenderingAndSeparators();
  testViewportClamping();
  testKeyboardNavigation();
  testDismissal();
  console.log('All tests passed');
}

try {
  main();
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
