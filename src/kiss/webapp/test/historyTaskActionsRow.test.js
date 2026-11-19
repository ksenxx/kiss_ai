// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (jsdom) tests for the actions of a task panel inside the
// task-history panel.
//
// A task panel (``.sidebar-item.running-item`` inside ``#history-list``)
// is ONE line: its text and, at the right end, a "…" button.  The
// favourite / copy / details / delete buttons, the "... ago" label and
// the machine the task ran on are the lines of the menu that button
// opens (the panel's ``.sidebar-item-actions`` strip, a popover).  They
// used to sit on a line of their own under the text.
//
// jsdom has no Popover API, so the menu cannot be opened here: these
// tests cover what the page builds and what the menu's buttons do, and
// ``tests/webapp/test_history_row_menu.py`` covers the open menu in a
// real browser.
//
// The page is booted the way ``web_server.py`` serves ``chat.html``
// (``<body class="remote-chat">`` plus ``remote-codex.css``).

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  loadRemoteCss,
  stubClipboard,
} = require('./simplify2_harness');
const {loadHistory, makeSession, historyRow} = require('./history_helpers');

const ICON_PX = 12;

function makePage() {
  return makeChatPage({beforeScripts: win => loadRemoteCss(win, true)});
}

function px(value) {
  const n = parseFloat(String(value));
  return isNaN(n) ? 0 : n;
}

/** The class that names each child of *el*, in DOM order. */
function childKinds(el, kinds) {
  return Array.from(el.children).map(
    child =>
      kinds.find(kind => child.classList.contains(kind)) || child.tagName,
  );
}

// ---------------------------------------------------------------------------
// 1. The panel is one line: text, then the "…" button; the buttons are
//    the menu's.  (Tests 1-2 read and then click the same rendered row,
//    so they share one page.)
// ---------------------------------------------------------------------------

function testPanelIsOneLineWithMenu(win) {
  const row = historyRow(win);
  assert.ok(row, 'history row rendered');
  assert.deepStrictEqual(
    childKinds(row, [
      'sidebar-item-text',
      'sidebar-item-more',
      'sidebar-item-actions',
      'running-item-info',
    ]),
    [
      'sidebar-item-text',
      'sidebar-item-more',
      'sidebar-item-actions',
      'running-item-info',
    ],
    'the panel holds its text, the "…" button, its menu and its details',
  );
  const text = row.querySelector('.sidebar-item-text');
  const more = row.querySelector('.sidebar-item-more');
  const actions = row.querySelector('.sidebar-item-actions');

  // One line: the text takes the room the button leaves and is cut
  // with an ellipsis; only the details break onto lines of their own.
  const textStyle = win.getComputedStyle(text);
  assert.ok(px(textStyle.flexGrow) >= 1, 'the task text fills its line');
  assert.notStrictEqual(
    textStyle.flexBasis,
    '100%',
    'the task text shares its line with the "…" button',
  );
  assert.strictEqual(
    win.getComputedStyle(row.querySelector('.running-item-info')).flexBasis,
    '100%',
    'the details take lines of their own below',
  );

  // The strip is the menu of the "…" button: a popover (closed, so not
  // displayed) that the button targets, both named for a screen reader.
  assert.strictEqual(more.tagName, 'BUTTON');
  assert.strictEqual(more.type, 'button');
  assert.strictEqual(more.getAttribute('aria-label'), 'Task actions');
  assert.strictEqual(more.popoverTargetElement, actions);
  assert.strictEqual(actions.getAttribute('popover'), 'auto');
  assert.strictEqual(actions.getAttribute('role'), 'group');
  assert.strictEqual(actions.getAttribute('aria-label'), 'Task actions');
  assert.strictEqual(
    win.getComputedStyle(actions).display,
    'none',
    'the menu is not part of the row until the "…" button opens it',
  );

  // The menu's lines, top to bottom: the buttons (the destructive one
  // last), then when the task was launched and on which machine.  Each
  // button carries the name its line shows (main.css draws it from the
  // aria-label).
  assert.deepStrictEqual(
    childKinds(actions, [
      'sidebar-item-favorite',
      'sidebar-item-copy',
      'sidebar-item-collapse',
      'sidebar-item-delete',
      'sidebar-item-confirm',
      'sidebar-item-launched',
      'sidebar-item-machine',
    ]),
    [
      'sidebar-item-favorite',
      'sidebar-item-copy',
      'sidebar-item-collapse',
      'sidebar-item-delete',
      'sidebar-item-confirm',
      'sidebar-item-launched',
      'sidebar-item-machine',
    ],
  );
  assert.deepStrictEqual(
    Array.from(actions.querySelectorAll(':scope > button')).map(btn =>
      btn.getAttribute('aria-label'),
    ),
    [
      'Favourite task',
      'Copy task to clipboard',
      'Expand task details',
      'Delete task',
    ],
  );
  assert.match(
    actions.querySelector('.sidebar-item-launched').textContent,
    / ago$|^just now$/,
    'the menu says when the task was launched',
  );
  assert.strictEqual(
    actions.querySelector('.sidebar-item-machine').textContent,
    'this machine',
    'and where it ran: here, on a daemon that has not said its name yet',
  );
  console.log('  ok - a task panel is one line with a "…" menu');
}

// ---------------------------------------------------------------------------
// 2. The menu's buttons act on the task, never on the row under them,
//    and rename themselves with their state.
// ---------------------------------------------------------------------------

function testMenuButtonsAct(win, posted) {
  const row = historyRow(win);
  const actions = row.querySelector('.sidebar-item-actions');
  const opened = () =>
    posted.filter(m => m.type === 'resumeSession' || m.type === 'openTab');
  const before = opened().length;

  const fav = actions.querySelector('.sidebar-item-favorite');
  fav.click();
  assert.ok(fav.classList.contains('favorited'), 'the star favourites');
  assert.strictEqual(fav.getAttribute('aria-label'), 'Unfavourite task');
  assert.ok(
    posted.some(m => m.type === 'setFavorite'),
    'favouriting reaches the backend',
  );
  fav.click();
  assert.ok(!fav.classList.contains('favorited'), 'and unfavourites');
  assert.strictEqual(fav.getAttribute('aria-label'), 'Favourite task');

  const toggle = actions.querySelector('.sidebar-item-collapse');
  toggle.click();
  assert.ok(!row.classList.contains('collapsed'), 'the chevron unfolds');
  assert.strictEqual(
    toggle.getAttribute('aria-label'),
    'Collapse task details',
  );

  // A click on a line that is no button (the age) does nothing either.
  actions
    .querySelector('.sidebar-item-launched')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    opened().length,
    before,
    'nothing clicked in the menu opens the chat of the row',
  );

  // The copy button swaps in a check mark after a successful copy.
  const copy = actions.querySelector('.sidebar-item-copy');
  const clip = stubClipboard(win);
  copy.click();
  // The flash follows PanelCopy.copyText's promise chain: wait a macrotask.
  return new Promise(r => setTimeout(r, 0)).then(() => {
    assert.strictEqual(clip.last(), 'refactor the parser');
    assert.ok(copy.classList.contains('copied'), 'the copy line shows it');
    assert.strictEqual(opened().length, before);
    win.close();
    console.log('  ok - the menu buttons act on the task only');
  });
}

// ---------------------------------------------------------------------------
// 3. Rows without a task id have a menu with the details toggle alone.
// ---------------------------------------------------------------------------

function testRowWithoutTaskIdKeepsMenu() {
  const {win} = makePage();
  loadHistory(win, [
    {
      id: 'chat-2',
      title: 'old imported chat',
      preview: 'old imported chat',
      has_events: false,
      timestamp: 1700000000,
    },
  ]);
  const row = historyRow(win);
  const actions = row.querySelector('.sidebar-item-actions');
  assert.strictEqual(
    row.querySelector('.sidebar-item-more').popoverTargetElement,
    actions,
    'a row without a task id has its "…" menu too',
  );
  assert.deepStrictEqual(
    Array.from(actions.querySelectorAll('button')).map(btn =>
      btn.getAttribute('aria-label'),
    ),
    ['Expand task details'],
    'with no favourite, copy or delete line',
  );
  win.close();
  console.log('  ok - task-id-less rows keep the menu');
}

// ---------------------------------------------------------------------------
// 4. Only the *history* panels have a menu - the promptlets list keeps
//    its compact inline buttons (a 12px icon in a "2px 0" padded box).
// ---------------------------------------------------------------------------

function testTricksListUnaffected() {
  const {win} = makePage();
  win.__TRICKS__ = ['run the tests'];
  win.document.getElementById('tricks-btn').click();
  const item = win.document.querySelector('#tricks-list .sidebar-item');
  assert.ok(item, 'promptlet row rendered');
  assert.ok(
    !item.classList.contains('running-item'),
    'promptlet rows are not history task panels',
  );
  assert.strictEqual(
    item.querySelector('.sidebar-item-more, [popover]'),
    null,
    'promptlet rows have no "…" menu',
  );
  const copy = item.querySelector('.sidebar-item-copy');
  assert.ok(copy, 'promptlet row has a copy button');
  const svg = copy.querySelector('svg');
  const width = win.getComputedStyle(svg).width;
  assert.ok(
    width === '' || width === 'auto',
    `no CSS rule resizes a promptlet icon; got width=${width}`,
  );
  assert.strictEqual(px(svg.getAttribute('width')), ICON_PX);
  const btnStyle = win.getComputedStyle(copy);
  assert.strictEqual(
    px(btnStyle.minWidth),
    0,
    'promptlet buttons keep their compact intrinsic box',
  );
  assert.strictEqual(
    px(btnStyle.paddingLeft),
    0,
    'promptlet buttons keep the compact "2px 0" padding',
  );
  win.close();
  console.log('  ok - promptlet rows keep their compact buttons');
}

async function main() {
  const {win, posted} = makePage();
  loadHistory(win, [makeSession()]);
  testPanelIsOneLineWithMenu(win);
  await testMenuButtonsAct(win, posted);
  testRowWithoutTaskIdKeepsMenu();
  testTricksListUnaffected();
  console.log('All historyTaskActionsRow tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
