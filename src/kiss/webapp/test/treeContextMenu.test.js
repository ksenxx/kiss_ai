// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// JSDOM test for media/treeContextMenu.js: the shared VS Code-style
// context menu the Explorer and Source Control trees open on
// right-click.  Covers item rendering (labels, keybindings, disabled
// rows, separator squashing), viewport clamping, keyboard navigation
// (Up / Down / Home / End / Enter / Space / Escape / Tab), outside-click
// and window-blur dismissal, the `closed` callback, and submenus with
// radio items (pointer, keyboard, focus, dismissal, placement).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {MEDIA} = require('./simplify2_harness');
const {key} = require('./ui_antipattern_harness');

function makeDom() {
  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  // The tree menu's keyboard model and clamp are contentContextMenu.js's,
  // loaded first as in chat.html.
  win.eval(fs.readFileSync(path.join(MEDIA, 'contentContextMenu.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'treeContextMenu.js'), 'utf8'));
  assert.ok(
    win.TreeContextMenu,
    'treeContextMenu.js must install window.TreeContextMenu',
  );
  return win;
}

function isOpen(win) {
  const el = win.document.getElementById('sidebar-context-menu');
  return !!el && !el.hidden;
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
  assert.ok(isOpen(win));
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
    isOpen(win),
    'clicking a disabled item keeps the menu open',
  );

  // Clicking an enabled item closes the menu first, then runs it.
  items[1].click();
  assert.deepStrictEqual(ran, ['rename']);
  assert.ok(!isOpen(win));
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
  key(win, win.document, 'Escape');
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
  key(win, win.document, 'ArrowDown');
  assert.strictEqual(active(), enabled[1], 'ArrowDown skips the disabled row');
  key(win, win.document, 'ArrowDown');
  assert.strictEqual(active(), enabled[2]);
  key(win, win.document, 'ArrowDown');
  assert.strictEqual(active(), enabled[0], 'ArrowDown wraps to the first item');
  key(win, win.document, 'ArrowUp');
  assert.strictEqual(active(), enabled[2], 'ArrowUp wraps to the last item');
  key(win, win.document, 'ArrowUp');
  assert.strictEqual(active(), enabled[1]);
  key(win, win.document, 'Home');
  assert.strictEqual(active(), enabled[0]);
  key(win, win.document, 'End');
  assert.strictEqual(active(), enabled[2]);
  // Focus lost from the menu without a key (the page blurred the
  // item): ArrowDown starts from the first item again.
  enabled[2].blur();
  key(win, win.document, 'ArrowDown');
  assert.strictEqual(active(), enabled[0]);
  key(win, win.document, 'ArrowDown');
  key(win, win.document, 'Enter');
  assert.deepStrictEqual(ran, ['two'], 'Enter runs the focused item');
  assert.strictEqual(closedCalls, 1, 'the closed callback fires once');
  assert.ok(!isOpen(win));

  // Space also runs; Escape only closes.
  win.TreeContextMenu.show(win.document, 0, 0, [
    {label: 'Go', run: () => ran.push('go')},
  ]);
  key(win, win.document, ' ');
  assert.deepStrictEqual(ran, ['two', 'go']);
  win.TreeContextMenu.show(win.document, 0, 0, [
    {label: 'Go', run: () => ran.push('go2')},
  ]);
  key(win, win.document, 'Escape');
  assert.ok(!isOpen(win), 'Escape closes the menu');
  assert.deepStrictEqual(ran, ['two', 'go'], 'Escape runs nothing');
  // Keys with no menu open are ignored, and Enter with nothing focused
  // runs nothing.
  key(win, win.document, 'Enter');
  win.TreeContextMenu.show(win.document, 0, 0, [
    {label: 'Go', run: () => ran.push('go3')},
  ]);
  win.document.activeElement.blur();
  key(win, win.document, 'Enter');
  assert.deepStrictEqual(ran, ['two', 'go']);
  // A menu of only disabled rows ignores navigation keys.
  win.TreeContextMenu.show(win.document, 0, 0, [
    {label: 'Off', enabled: false},
  ]);
  key(win, win.document, 'ArrowDown');
  assert.ok(isOpen(win));
  key(win, win.document, 'Escape');
  console.log('ok: keyboard navigation matches VS Code menus');
}

/**
 * Tab leaves the menu with the browser's own focus move, so the menu
 * closes: left open, its document-level key handler went on taking the
 * arrow keys of whatever the user tabbed into and pulled the focus back
 * onto the first menu item.
 */
function testTabLeavesTheMenu() {
  const win = makeDom();
  const editor = win.document.createElement('textarea');
  win.document.body.appendChild(editor);
  let closedCalls = 0;
  win.TreeContextMenu.show(
    win.document,
    0,
    0,
    [
      {label: 'Rename', run: () => {}},
      {label: 'Delete', run: () => {}},
    ],
    () => closedCalls++,
  );
  const press = (target, name) => {
    const ev = new win.KeyboardEvent('keydown', {
      key: name,
      bubbles: true,
      cancelable: true,
    });
    target.dispatchEvent(ev);
    return ev;
  };
  const tab = press(win.document.activeElement, 'Tab');
  assert.strictEqual(
    tab.defaultPrevented,
    false,
    'Tab keeps its default: the browser moves the focus on',
  );
  assert.ok(!isOpen(win), 'Tab closes the menu');
  assert.strictEqual(closedCalls, 1, 'the closed callback fires once');
  // jsdom does not move the focus on Tab; the browser would.
  editor.focus();
  const arrow = press(editor, 'ArrowDown');
  assert.strictEqual(
    arrow.defaultPrevented,
    false,
    'ArrowDown in the editor is the editor\'s: the caret moves',
  );
  assert.strictEqual(
    win.document.activeElement,
    editor,
    'the focus stays in the editor',
  );
  console.log('ok: Tab closes the menu and hands the keys back');
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
    isOpen(win),
    'mousedown inside the menu keeps it open',
  );
  // Pressing anywhere else closes it.
  win.document.body.dispatchEvent(
    new win.MouseEvent('mousedown', {bubbles: true}),
  );
  assert.ok(!isOpen(win), 'mousedown outside closes the menu');
  assert.strictEqual(closedCalls, 1);
  // A right-click elsewhere closes it too (the tree then opens a new one).
  win.TreeContextMenu.show(win.document, 0, 0, [{label: 'A', run: () => {}}]);
  win.document.body.dispatchEvent(
    new win.MouseEvent('contextmenu', {bubbles: true}),
  );
  assert.ok(!isOpen(win));
  // Window blur / resize close it.
  win.TreeContextMenu.show(win.document, 0, 0, [{label: 'A', run: () => {}}]);
  win.dispatchEvent(new win.Event('blur'));
  assert.ok(!isOpen(win), 'window blur closes the menu');
  win.TreeContextMenu.show(win.document, 0, 0, [{label: 'A', run: () => {}}]);
  win.dispatchEvent(new win.Event('resize'));
  assert.ok(!isOpen(win), 'window resize closes the menu');
  // Showing again reuses the single menu element and replaces its items.
  win.TreeContextMenu.show(win.document, 0, 0, [{label: 'X', run: () => {}}]);
  win.TreeContextMenu.show(win.document, 0, 0, [{label: 'Y', run: () => {}}]);
  assert.strictEqual(
    win.document.querySelectorAll('#sidebar-context-menu').length,
    1,
  );
  assert.strictEqual(rows(win).length, 1);
  assert.strictEqual(rows(win)[0].textContent, 'Y');
  // Escape when nothing is open is ignored; a closed callback that is
  // not a function is ignored.
  key(win, win.document, 'Escape');
  key(win, win.document, 'Escape');
  win.TreeContextMenu.show(
    win.document,
    0,
    0,
    [{label: 'Z'}],
    'not-a-function',
  );
  rows(win)[0].click();
  assert.ok(!isOpen(win));
  console.log('ok: outside clicks, blur and resize dismiss the menu');
}

const SUBMENU = '#sidebar-context-submenu';

function subRows(win) {
  return Array.from(win.document.querySelectorAll(SUBMENU + ' .tree-ctx-item'));
}

function subOpen(win) {
  const el = win.document.querySelector(SUBMENU);
  return !!el && !el.hidden;
}

function hover(win, el, pointerType = 'mouse') {
  el.dispatchEvent(new win.PointerEvent('pointerenter', {pointerType}));
}

/**
 * Open New File... / Sort > (Name, Off, Date) / Rename... and return its
 * rows.  The Sort submenu holds radio items, a disabled one, separators
 * to squash and an item that names a submenu of its own.
 */
function showSortMenu(win, ran, closed) {
  const sort = [
    {separator: true},
    {id: 'by-name', label: 'Name', checked: true, run: () => ran.push('name')},
    {label: 'Off', checked: false, enabled: false, run: () => ran.push('off')},
    {separator: true},
    {separator: true},
    {
      label: 'Date',
      checked: false,
      submenu: [{label: 'Deeper'}],
      run: () => ran.push('date'),
    },
    {separator: true},
  ];
  win.TreeContextMenu.show(
    win.document,
    0,
    0,
    [
      {label: 'New File...', run: () => ran.push('new')},
      {id: 'sort', label: 'Sort', key: 'S', submenu: sort},
      {label: 'Rename...', run: () => ran.push('rename')},
    ],
    closed,
  );
  return rows(win);
}

function testSubmenuWithPointer() {
  const win = makeDom();
  const ran = [];
  let closedCalls = 0;
  const [first, sort, rename] = showSortMenu(win, ran, () => closedCalls++);
  assert.strictEqual(sort.getAttribute('role'), 'menuitem');
  assert.strictEqual(sort.getAttribute('aria-haspopup'), 'menu');
  assert.strictEqual(sort.getAttribute('aria-expanded'), 'false');
  assert.deepStrictEqual(
    Array.from(sort.children, c => c.getAttribute('class')),
    ['tree-ctx-label', 'tree-ctx-key', 'tree-ctx-chevron'],
    'label, shortcut, then the chevron',
  );
  assert.ok(!rename.hasAttribute('aria-haspopup'));
  assert.ok(!rename.hasAttribute('aria-checked'), 'a plain item is no radio');
  assert.ok(!subOpen(win), 'the submenu opens on demand');

  // A finger landing on the item opens nothing yet: its tap's click
  // does, and must not reach a submenu item placed under the finger.
  hover(win, sort, 'touch');
  assert.ok(!subOpen(win), 'a touch waits for its click');

  // The mouse over the item opens the submenu; the focus stays put.
  hover(win, sort);
  assert.ok(subOpen(win));
  assert.strictEqual(sort.getAttribute('aria-expanded'), 'true');
  assert.strictEqual(win.document.activeElement, first);
  const sub = win.document.querySelector(SUBMENU);
  assert.strictEqual(sub.getAttribute('role'), 'menu');
  assert.strictEqual(sub.getAttribute('aria-label'), 'Sort');
  assert.deepStrictEqual(
    Array.from(sub.children, c => c.getAttribute('role')),
    ['menuitemradio', 'menuitemradio', 'separator', 'menuitemradio'],
    'separators are squashed in a submenu too',
  );
  const [name, off, date] = subRows(win);
  assert.strictEqual(name.dataset.action, 'by-name');
  assert.deepStrictEqual(
    [name, off, date].map(r => r.getAttribute('aria-checked')),
    ['true', 'false', 'false'],
  );
  assert.deepStrictEqual(
    [name, off, date].map(r => r.querySelectorAll('.tree-ctx-check').length),
    [1, 0, 0],
    'only the checked item carries the check mark',
  );
  assert.ok(!date.hasAttribute('aria-haspopup'), 'one level of submenus');
  assert.strictEqual(date.querySelector('.tree-ctx-chevron'), null);
  // Over the item again: the open submenu stays as it is.
  hover(win, sort);
  assert.strictEqual(subRows(win)[0], name);

  // Pressing inside the submenu keeps both menus open; a disabled item
  // runs nothing.
  name.dispatchEvent(new win.MouseEvent('mousedown', {bubbles: true}));
  off.click();
  assert.ok(isOpen(win) && subOpen(win));
  assert.deepStrictEqual(ran, []);
  // A click on the item moves the focus into the open submenu.
  sort.click();
  assert.strictEqual(win.document.activeElement, name);
  assert.ok(subOpen(win), 'the focus inside the submenu keeps it open');
  // Picking an item runs it and closes the whole menu.
  date.click();
  assert.deepStrictEqual(ran, ['date']);
  assert.ok(!isOpen(win) && !subOpen(win));
  assert.strictEqual(sub.childElementCount, 0, 'closing empties the submenu');
  assert.strictEqual(closedCalls, 1);
  console.log('ok: a submenu opens under the pointer and runs its items');
}

function testSubmenuWithKeyboard() {
  const win = makeDom();
  const ran = [];
  let closedCalls = 0;
  const active = () => win.document.activeElement;
  const [first, sort] = showSortMenu(win, ran, () => closedCalls++);
  // Right on an item without a submenu does nothing.
  key(win, win.document, 'ArrowRight');
  assert.ok(!subOpen(win));
  assert.strictEqual(active(), first);
  key(win, win.document, 'ArrowDown');
  assert.strictEqual(active(), sort);
  // Right opens the submenu on its first enabled item.
  const right = new win.KeyboardEvent('keydown', {
    key: 'ArrowRight',
    bubbles: true,
    cancelable: true,
  });
  win.document.dispatchEvent(right);
  assert.ok(right.defaultPrevented);
  const [name, , date] = subRows(win);
  assert.strictEqual(active(), name);
  assert.strictEqual(sort.getAttribute('aria-expanded'), 'true');
  // The arrow keys stay inside the submenu and skip its disabled item.
  key(win, win.document, 'ArrowDown');
  assert.strictEqual(active(), date);
  key(win, win.document, 'ArrowDown');
  assert.strictEqual(active(), name, 'ArrowDown wraps inside the submenu');
  key(win, win.document, 'End');
  assert.strictEqual(active(), date);
  key(win, win.document, 'ArrowRight');
  assert.strictEqual(active(), date, 'Right in the submenu does nothing');
  // Left closes the submenu alone and returns the focus to its item.
  key(win, win.document, 'ArrowLeft');
  assert.ok(!subOpen(win) && isOpen(win));
  assert.strictEqual(active(), sort);
  assert.strictEqual(sort.getAttribute('aria-expanded'), 'false');
  // Enter opens it too; Escape closes the submenu first, the menu next.
  key(win, win.document, 'Enter');
  assert.strictEqual(active(), subRows(win)[0]);
  key(win, win.document, 'Escape');
  assert.ok(!subOpen(win) && isOpen(win));
  assert.strictEqual(active(), sort);
  assert.strictEqual(closedCalls, 0);
  // Space opens it; Enter runs the focused submenu item and closes all.
  key(win, win.document, ' ');
  key(win, win.document, 'Enter');
  assert.deepStrictEqual(ran, ['name']);
  assert.ok(!isOpen(win) && !subOpen(win));
  assert.strictEqual(closedCalls, 1);
  // Tab inside the submenu leaves, and closes, the whole menu.
  showSortMenu(win, ran)[1].click();
  key(win, win.document, 'Tab');
  assert.ok(!isOpen(win) && !subOpen(win));
  // Escape with the menu alone open still closes it.
  showSortMenu(win, ran);
  key(win, win.document, 'Escape');
  assert.ok(!isOpen(win));
  assert.deepStrictEqual(ran, ['name']);
  console.log('ok: a submenu opens, navigates and closes by keyboard');
}

function testSubmenuFocusAndDismissal() {
  const win = makeDom();
  const ran = [];
  let [, sort, rename] = showSortMenu(win, ran);
  // Held open by the pointer, the submenu stays while the focus reaches
  // its item and closes when the focus moves on to another item.
  hover(win, sort);
  key(win, win.document, 'ArrowDown');
  assert.strictEqual(win.document.activeElement, sort);
  assert.ok(subOpen(win));
  key(win, win.document, 'ArrowDown');
  assert.strictEqual(win.document.activeElement, rename);
  assert.ok(!subOpen(win), 'the focus on another item closes the submenu');
  assert.strictEqual(sort.getAttribute('aria-expanded'), 'false');
  // Left with the submenu held open by the pointer focuses its item.
  hover(win, sort);
  key(win, win.document, 'ArrowLeft');
  assert.ok(!subOpen(win) && isOpen(win));
  assert.strictEqual(win.document.activeElement, sort);

  // Every way the menu closes takes the open submenu with it.
  const dismissals = [
    () =>
      win.document.body.dispatchEvent(
        new win.MouseEvent('mousedown', {bubbles: true}),
      ),
    () => win.dispatchEvent(new win.Event('blur')),
    () => win.dispatchEvent(new win.Event('resize')),
    () => win.TreeContextMenu.show(win.document, 0, 0, [{label: 'Other'}]),
  ];
  dismissals.forEach(dismiss => {
    [, sort] = showSortMenu(win, ran);
    hover(win, sort);
    assert.ok(subOpen(win));
    dismiss();
    assert.ok(!subOpen(win));
    assert.strictEqual(subRows(win).length, 0);
  });
  assert.deepStrictEqual(rows(win).map(r => r.textContent), ['Other']);

  // A disabled item opens no submenu; of two items with submenus the
  // one opened last shows; a submenu without an enabled item opens but
  // takes no focus.
  win.TreeContextMenu.show(win.document, 0, 0, [
    {label: 'Off', enabled: false, submenu: [{label: 'Never'}]},
    {label: 'One', submenu: [{label: 'First'}]},
    {label: 'Two', submenu: [{label: 'Second', enabled: false}]},
  ]);
  const [off, one, two] = rows(win);
  assert.ok(off.querySelector('.tree-ctx-chevron'));
  hover(win, off);
  off.click();
  assert.ok(!subOpen(win));
  hover(win, one);
  assert.deepStrictEqual(subRows(win).map(r => r.textContent), ['First']);
  two.click();
  assert.deepStrictEqual(subRows(win).map(r => r.textContent), ['Second']);
  assert.strictEqual(one.getAttribute('aria-expanded'), 'false');
  assert.strictEqual(two.getAttribute('aria-expanded'), 'true');
  assert.strictEqual(win.document.activeElement, one, 'no item to focus');
  key(win, win.document, 'Escape');
  key(win, win.document, 'Escape');
  assert.ok(!isOpen(win));
  assert.deepStrictEqual(ran, []);
  console.log('ok: a submenu follows the focus and closes with the menu');
}

function testSubmenuPlacement() {
  const win = makeDom();
  // jsdom lays nothing out, so the test gives the item its box; the
  // submenu falls back to a 200px width and 24px per item.
  function openAt(left, right, top) {
    win.TreeContextMenu.show(win.document, 0, 0, [
      {label: 'Sort', submenu: [{label: 'A'}, {label: 'B'}]},
    ]);
    const owner = rows(win)[0];
    owner.getBoundingClientRect = () => ({left, right, top});
    owner.click();
    return win.document.querySelector(SUBMENU);
  }
  // Room on the right: against the item's right edge, level with it.
  let sub = openAt(100, 320, 50);
  assert.deepStrictEqual([sub.style.left, sub.style.top], ['320px', '50px']);
  // The viewport ends first and the left has the room: flipped.
  sub = openAt(700, 920, 50);
  assert.strictEqual(sub.style.left, '500px');
  // No room on either side, nor below: kept inside the viewport.
  sub = openAt(100, 900, win.innerHeight);
  assert.strictEqual(sub.style.left, win.innerWidth - 200 - 4 + 'px');
  assert.strictEqual(sub.style.top, win.innerHeight - 48 - 4 + 'px');
  // Laid out, the submenu is placed by its own width and raised by the
  // menu's border and padding, so its first item is level with the item.
  const menuEl = win.document.getElementById('sidebar-context-menu');
  Object.defineProperty(sub, 'offsetWidth', {value: 300});
  Object.defineProperty(menuEl, 'clientTop', {value: 1});
  Object.defineProperty(win.HTMLElement.prototype, 'offsetTop', {value: 4});
  sub = openAt(700, 1000, 50);
  assert.deepStrictEqual([sub.style.left, sub.style.top], ['400px', '45px']);
  key(win, win.document, 'Escape');
  key(win, win.document, 'Escape');
  console.log('ok: a submenu opens beside its item, inside the viewport');
}

function main() {
  testRenderingAndSeparators();
  testViewportClamping();
  testKeyboardNavigation();
  testTabLeavesTheMenu();
  testDismissal();
  testSubmenuWithPointer();
  testSubmenuWithKeyboard();
  testSubmenuFocusAndDismissal();
  testSubmenuPlacement();
  console.log('All tests passed');
}

try {
  main();
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
