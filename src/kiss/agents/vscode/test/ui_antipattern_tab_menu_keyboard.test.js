// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for anti-patterns A5/A12 in media/main.js:
// the chat-tab context menu is keyboard operable like the sidebar tree
// menus.  Shift+F10 / the ContextMenu key open it from a focused tab,
// its items are focusable menuitems reached with the arrows, Home and
// End, Enter picks one, and Escape closes it and hands focus back to
// the tab.  Each test fails on the old code.
'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

const {test, report} = h.makeRunner();

function tabEl(win, tabId) {
  return win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
}

function menu(win) {
  return h.byId(win, 'tab-context-menu');
}

function items(win) {
  return h.all(win, '#tab-context-menu .tab-ctx-item');
}

function twoTabs() {
  const {win, posted} = h.makeWebview();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.endLaunch();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabA, tabB);
  return {win, posted, tabA, tabB};
}

async function main() {
  await test('Shift+F10 on a focused tab opens the menu with the first item focused', () => {
    const {win, tabA} = twoTabs();
    const tab = tabEl(win, tabA);
    tab.focus();
    h.key(win, tab, 'F10', {shiftKey: true});
    assert.ok(menu(win).classList.contains('open'));
    assert.strictEqual(menu(win).getAttribute('role'), 'menu');
    const first = items(win)[0];
    assert.strictEqual(first.getAttribute('role'), 'menuitem');
    assert.strictEqual(first.tabIndex, -1);
    assert.strictEqual(win.document.activeElement, first);
    win.close();
  });

  await test('arrows, Home and End move between the items (wrapping)', () => {
    const {win, tabA} = twoTabs();
    const tab = tabEl(win, tabA);
    tab.focus();
    h.rightClick(win, tab);
    const list = items(win);
    assert.strictEqual(list.length, 4);
    assert.strictEqual(win.document.activeElement, list[0]);
    h.key(win, list[0], 'ArrowDown');
    assert.strictEqual(win.document.activeElement, list[1]);
    h.key(win, list[1], 'End');
    assert.strictEqual(win.document.activeElement, list[3]);
    h.key(win, list[3], 'ArrowDown');
    assert.strictEqual(win.document.activeElement, list[0], 'wraps to the top');
    h.key(win, list[0], 'ArrowUp');
    assert.strictEqual(win.document.activeElement, list[3], 'wraps to the end');
    h.key(win, list[3], 'Home');
    assert.strictEqual(win.document.activeElement, list[0]);
    win.close();
  });

  await test('Enter picks the focused item (Close closes the tab)', () => {
    const {win, posted, tabA} = twoTabs();
    const tab = tabEl(win, tabA);
    tab.focus();
    h.rightClick(win, tab);
    const close = items(win).find(el => el.textContent === 'Close');
    close.focus();
    h.key(win, close, 'Enter');
    assert.ok(!menu(win).classList.contains('open'), 'the menu closed');
    assert.ok(!tabEl(win, tabA), 'the tab is gone');
    assert.ok(
      h.ofType(posted, 'closeTab').some(m => m.tabId === tabA),
      'the close was posted',
    );
    win.close();
  });

  await test('Escape closes the menu and returns focus to the tab', () => {
    const {win, tabA} = twoTabs();
    const tab = tabEl(win, tabA);
    tab.focus();
    h.rightClick(win, tab);
    assert.notStrictEqual(win.document.activeElement, tab, 'precondition');
    h.key(win, win.document.activeElement, 'Escape');
    assert.ok(!menu(win).classList.contains('open'));
    assert.strictEqual(win.document.activeElement, tab);
    win.close();
  });

  report('ui_antipattern_tab_menu_keyboard');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
