// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for anti-patterns A5/A12 in media/main.js:
// the tab context menu of the group strip (#tab-list: the chat on
// screen and its sub-agent tabs) is keyboard operable like the sidebar
// tree menus.  Shift+F10 / the ContextMenu key open it from a focused
// tab, its items are focusable menuitems reached with the arrows, Home
// and End, Enter picks one, and Escape closes it and hands focus back
// to the tab.  Each test fails on the old code.
'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');
const {tabEl} = require('./subagent_tab_helpers');

const {test, report} = h.makeRunner();

function menu(win) {
  return h.byId(win, 'tab-context-menu');
}

function items(win) {
  return h.all(win, '#tab-context-menu .tab-ctx-item');
}

// Two chats; tabA is on screen with a sub-agent tab, so the strip is
// shown (a lone chat has no strip) and tabA's entry can take focus.
function twoTabs() {
  const {win, posted} = h.makePage();
  const tabA = win._testApi.getActiveTabId();
  win._testApi.endLaunch();
  win._testApi.createNewTab();
  const tabB = win._testApi.getActiveTabId();
  assert.notStrictEqual(tabA, tabB);
  win._testApi.switchToTab(tabA);
  h.send(win, {
    type: 'openSubagentTab',
    tab_id: 'sub-a',
    parent_tab_id: tabA,
    description: 'a sub-agent',
    task_id: 'sub-task-a',
    taskIndex: 0,
    isSubagentTab: true,
  });
  assert.strictEqual(win._testApi.getActiveTabId(), tabA, 'tabA is on screen');
  assert.notStrictEqual(
    h.byId(win, 'tab-bar').style.display,
    'none',
    'the group strip is shown',
  );
  return {win, posted, tabA};
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

  // The menu is no focus trap: Tab closes it and keeps the browser's own
  // focus move (the keydown is not default-prevented).
  await test('Tab leaves the menu: it closes without trapping the key', () => {
    const {win, tabA} = twoTabs();
    const tab = tabEl(win, tabA);
    tab.focus();
    h.rightClick(win, tab);
    const first = items(win)[0];
    assert.strictEqual(win.document.activeElement, first, 'precondition');
    assert.strictEqual(
      h.key(win, first, 'ArrowDown'),
      false,
      'an arrow is the menu\'s own: default-prevented',
    );
    assert.strictEqual(
      h.key(win, items(win)[1], 'Tab'),
      true,
      'Tab keeps its browser default (the focus move)',
    );
    assert.ok(!menu(win).classList.contains('open'), 'the menu closed');
    win.close();
  });

  // The content tabs' menu bar (File / Edit / ... above an editor) runs
  // on the same keyboard model as the context menus: the arrows wrap
  // through the enabled rows, Home / End jump to the ends, and Tab
  // closes the dropdown and hands focus back to its menu button.
  await test('the content menu bar dropdown shares the keyboard model', async () => {
    const ctx = h.makePage();
    const {win} = ctx;
    await h.openDirtyContentTab(ctx);
    const fileBtn = win.document.querySelector('.content-menubar .content-menu-btn');
    assert.ok(fileBtn && fileBtn.textContent === 'File', 'the File menu button');
    assert.strictEqual(h.key(win, fileBtn, 'ArrowDown'), false, 'ArrowDown opens it');
    const dropdown = () => win.document.querySelector('.content-menu-dropdown');
    const rows = () =>
      h.all(win, '.content-menu-dropdown .content-menu-item[tabindex]');
    assert.ok(dropdown(), 'the dropdown is open');
    assert.strictEqual(rows().length, 2, 'Save (dirty tab) and Close Editor');
    assert.strictEqual(win.document.activeElement, rows()[0], 'first row focused');
    assert.strictEqual(h.key(win, rows()[0], 'ArrowDown'), false);
    assert.strictEqual(win.document.activeElement, rows()[1]);
    assert.strictEqual(h.key(win, rows()[1], 'ArrowDown'), false);
    assert.strictEqual(win.document.activeElement, rows()[0], 'wraps to the top');
    assert.strictEqual(h.key(win, rows()[0], 'ArrowUp'), false);
    assert.strictEqual(win.document.activeElement, rows()[1], 'wraps to the end');
    assert.strictEqual(h.key(win, rows()[1], 'Home'), false);
    assert.strictEqual(win.document.activeElement, rows()[0]);
    assert.strictEqual(h.key(win, rows()[0], 'End'), false);
    assert.strictEqual(win.document.activeElement, rows()[1]);
    // A key the model does not own passes through untouched.
    assert.strictEqual(h.key(win, rows()[1], 'a'), true, 'not default-prevented');
    assert.ok(dropdown(), 'and the dropdown stays open');
    assert.strictEqual(
      h.key(win, rows()[1], 'Tab'),
      true,
      'Tab keeps its browser default (the focus move)',
    );
    assert.ok(!dropdown(), 'the dropdown closed');
    assert.strictEqual(win.document.activeElement, fileBtn, 'focus is back on the button');
    assert.strictEqual(fileBtn.getAttribute('aria-expanded'), 'false');
    win.close();
  });

  // The dropdown hangs off the body, so nothing takes it away when the
  // pane folds under it: a run starting in the chat on screen (from
  // another surface, say) folds the pane with no pointer press here.
  await test('a run that folds the content pane closes its open menu', async () => {
    const ctx = h.makePage();
    const {win} = ctx;
    await h.openDirtyContentTab(ctx);
    const fileBtn = win.document.querySelector('.content-menubar .content-menu-btn');
    h.click(win, fileBtn);
    assert.ok(
      win.document.querySelector('.content-menu-dropdown'),
      'precondition: the menu is open',
    );
    h.send(win, {
      type: 'status',
      running: true,
      tabId: win._testApi.getActiveTabId(),
    });
    assert.ok(
      win.document.body.classList.contains('content-pane-folded'),
      'precondition: the run folded the pane',
    );
    assert.strictEqual(win.document.querySelector('.content-menu-dropdown'), null);
    assert.strictEqual(fileBtn.getAttribute('aria-expanded'), 'false');
    win.close();
  });

  report('ui_antipattern_tab_menu_keyboard');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
