// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (jsdom) tests for the action-button strip of a task panel
// inside the task-history panel.
//
// A task panel (``.sidebar-item.running-item`` inside ``#history-list``)
// used to squeeze its favourite / copy / collapse buttons onto the same
// line as the task text, which left the text almost no room on a narrow
// sidebar.  The buttons must sit on their own full-width line *below*
// the task text.
//
// They were briefly drawn 50% bigger than the compact buttons used by
// every other sidebar list.  That size bump has been reverted: a task
// panel now uses exactly the same 12x12 icon in a 12x16 box as the rest
// of the sidebar, so only the *placement* differs.
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
const {loadHistory, makeSession} = require('./history_helpers');

// The compact sidebar button is a 12x16 box: a 12px icon with "2px 0"
// padding and no explicit min-width/min-height.  A history task panel
// must use exactly that geometry - the 18x24 / 18x18 enlargement it
// briefly carried has been reverted.
const ICON_PX = 12;
const BUTTON_W_PX = 12;
const BUTTON_H_PX = 16;
const REVERTED_ICON_PX = 18;
const REVERTED_BUTTON_W_PX = 18;
const REVERTED_BUTTON_H_PX = 24;

const ACTION_BUTTON_SELECTORS = [
  '.sidebar-item-favorite',
  '.sidebar-item-copy',
  '.sidebar-item-collapse',
];

// A compact icon takes its size from its own markup attributes, so no
// stylesheet rule may set a width for it.  An icon left at its intrinsic
// size therefore computes to the initial value, which jsdom reports as
// 'auto' (or '' for an <svg> outside any layout).
function assertCompactIcon(win, svg, what) {
  const width = win.getComputedStyle(svg).width;
  assert.ok(
    width === '' || width === 'auto',
    `${what}: no CSS rule may resize the icon away from its intrinsic ` +
      `${ICON_PX}px; the reverted rule forced ${REVERTED_ICON_PX}px, ` +
      `got width=${width}`,
  );
  assert.strictEqual(
    px(svg.getAttribute('width')),
    ICON_PX,
    `${what}: the icon markup must stay ${ICON_PX}px`,
  );
  assert.strictEqual(
    px(svg.getAttribute('height')),
    ICON_PX,
    `${what}: the icon markup must stay ${ICON_PX}px tall`,
  );
}

function makePage() {
  return makeChatPage({beforeScripts: win => loadRemoteCss(win, true)});
}

function historyRow(win) {
  return win.document.querySelector('#history-list .sidebar-item');
}

function px(value) {
  const n = parseFloat(String(value));
  return isNaN(n) ? 0 : n;
}

// ---------------------------------------------------------------------------
// 1. The buttons live on their own line.  (Tests 1-3 read and then click
//    the same rendered row, so they share one page.)
// ---------------------------------------------------------------------------

function testActionsOnSeparateLine(win) {
  const row = historyRow(win);
  assert.ok(row, `history row rendered`);

  const actions = row.querySelector('.sidebar-item-actions');
  assert.ok(actions, `action strip rendered`);

  const text = row.querySelector('.sidebar-item-text');
  assert.ok(text, `task text rendered`);

  // The row is a wrapping flex container; a child only gets its own line
  // when it is forced to span the full content width.
  const rowStyle = win.getComputedStyle(row);
  assert.strictEqual(
    rowStyle.display,
    'flex',
    `the task panel must stay a flex container`,
  );
  assert.strictEqual(
    rowStyle.flexWrap,
    'wrap',
    `the task panel must wrap so the actions can break to a new line`,
  );

  const actionsStyle = win.getComputedStyle(actions);
  assert.strictEqual(
    actionsStyle.flexBasis,
    '100%',
    `the action strip must claim a full-width line of its own; ` +
      `got flex-basis=${actionsStyle.flexBasis}`,
  );
  assert.strictEqual(
    px(actionsStyle.marginLeft),
    0,
    `the action strip must not keep the inline left gutter; ` +
      `got margin-left=${actionsStyle.marginLeft}`,
  );

  // The task text must no longer have to share its line with the buttons:
  // it keeps flex:1 while the actions wrap away onto the next line.
  const textStyle = win.getComputedStyle(text);
  assert.ok(
    px(textStyle.flexGrow) >= 1,
    `the task text must still grow to fill its line`,
  );
  // Regression guard: the title needs its own full-width basis too. With the
  // default flex-basis of 0 it is squeezed to zero width beside the
  // full-width action strip and the task title becomes invisible.
  assert.strictEqual(
    textStyle.flexBasis,
    '100%',
    `the task text must claim a full-width line of its own, ` +
      `otherwise the full-width action strip collapses it; ` +
      `got flex-basis=${textStyle.flexBasis}`,
  );

  // DOM order decides which line comes first when nothing reorders them:
  // text -> actions -> info.
  const children = Array.prototype.slice.call(row.children);
  assert.ok(
    children.indexOf(text) < children.indexOf(actions),
    `the action strip must come after the task text`,
  );
  const info = row.querySelector('.running-item-info');
  assert.ok(
    children.indexOf(actions) < children.indexOf(info),
    `the action strip must come before the metadata block`,
  );
  assert.strictEqual(
    win.getComputedStyle(actions).order,
    win.getComputedStyle(text).order,
    `no CSS 'order' trickery may re-shuffle the action strip`,
  );

  console.log(`  ok - action buttons sit on their own line`);
}

// ---------------------------------------------------------------------------
// 2. The buttons keep the compact sidebar size.
// ---------------------------------------------------------------------------

function testActionsKeepCompactSize(win) {
  const row = historyRow(win);
  const compact = win.document.createElement('button');
  compact.className = 'sidebar-item-copy';
  win.document.body.appendChild(compact);
  const compactStyle = win.getComputedStyle(compact);

  ACTION_BUTTON_SELECTORS.forEach(sel => {
    const btn = row.querySelector(sel);
    assert.ok(btn, `${sel} rendered`);

    const btnStyle = win.getComputedStyle(btn);
    // No min-width/min-height override may survive: the box has to be
    // sized by its 12px icon plus the shared "2px 0" padding.
    assert.strictEqual(
      px(btnStyle.minWidth),
      0,
      `${sel} must not force a wider box; the reverted rule set ` +
        `min-width=${REVERTED_BUTTON_W_PX}px, got ${btnStyle.minWidth}`,
    );
    assert.strictEqual(
      px(btnStyle.minHeight),
      0,
      `${sel} must not force a taller box; the reverted rule set ` +
        `min-height=${REVERTED_BUTTON_H_PX}px, got ${btnStyle.minHeight}`,
    );
    assert.strictEqual(
      px(btnStyle.paddingTop),
      2,
      `${sel} must keep the compact "2px 0" padding so the box ` +
        `stays ${BUTTON_H_PX}px tall; got padding-top=${btnStyle.paddingTop}`,
    );
    assert.strictEqual(
      px(btnStyle.paddingLeft),
      0,
      `${sel} must keep the compact "2px 0" padding so the box ` +
        `stays ${BUTTON_W_PX}px wide; got padding-left=${btnStyle.paddingLeft}`,
    );
    // Every declaration must match a plain compact sidebar button, so no
    // leftover of the enlargement (border-radius, justify-content, ...)
    // can distinguish a history panel's button from the others.
    ['borderRadius', 'justifyContent', 'alignItems', 'display'].forEach(
      prop => {
        assert.strictEqual(
          btnStyle[prop],
          compactStyle[prop],
          `${sel} must render exactly like a compact sidebar ` +
            `button; ${prop} differs (${btnStyle[prop]} vs ${compactStyle[prop]})`,
        );
      },
    );

    const svg = btn.querySelector('svg');
    assert.ok(svg, `${sel} renders an icon`);
    assertCompactIcon(win, svg, `${sel}`);
  });

  console.log(`  ok - action buttons keep the compact size`);
}

// ---------------------------------------------------------------------------
// 3. The compact icon size survives every innerHTML rewrite.
// ---------------------------------------------------------------------------

function testIconSizeSurvivesStateChanges(win, posted) {
  const row = historyRow(win);

  // Favouriting swaps the outline star for the filled star via innerHTML.
  const fav = row.querySelector('.sidebar-item-favorite');
  fav.click();
  assert.ok(
    fav.classList.contains('favorited'),
    `clicking the star favourites the task`,
  );
  assert.ok(
    posted.some(m => m.type === 'setFavorite'),
    `favouriting still reaches the backend`,
  );
  assertCompactIcon(win, fav.querySelector('svg'), `filled star`);

  fav.click();
  assert.ok(
    !fav.classList.contains('favorited'),
    `clicking again unfavourites the task`,
  );
  assertCompactIcon(win, fav.querySelector('svg'), `outline star`);

  // The copy button swaps in a check mark after a successful copy.
  const copy = row.querySelector('.sidebar-item-copy');
  const clip = stubClipboard(win);
  copy.click();
  // The flash follows PanelCopy.copyText's promise chain: wait a macrotask.
  return new Promise(r => setTimeout(r, 0)).then(() => {
    assert.strictEqual(
      clip.last(),
      'refactor the parser',
      `the copy button still copies the task text`,
    );
    assert.ok(
      copy.classList.contains('copied'),
      `the copy button flashes its success state`,
    );
    assertCompactIcon(win, copy.querySelector('svg'), `check mark`);
    win.close();
    console.log(`  ok - icon size survives every state swap`);
  });
}

// ---------------------------------------------------------------------------
// 4. Rows without a task id only show the collapse toggle - still on its
//    own, full-width line.
// ---------------------------------------------------------------------------

function testRowWithoutTaskIdKeepsOwnLine() {
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
  assert.ok(actions, `action strip rendered without a task id`);
  assert.strictEqual(
    row.querySelector('.sidebar-item-favorite'),
    null,
    `no favourite button without a task id`,
  );
  assert.strictEqual(
    row.querySelector('.sidebar-item-copy'),
    null,
    `no copy button without a task id`,
  );
  assert.strictEqual(
    win.getComputedStyle(actions).flexBasis,
    '100%',
    `the lone collapse toggle still gets its own line`,
  );
  const toggle = actions.querySelector('.sidebar-item-collapse');
  assertCompactIcon(win, toggle.querySelector('svg'), `lone collapse toggle`);
  assert.strictEqual(
    px(win.getComputedStyle(toggle).minHeight),
    0,
    `the lone collapse toggle keeps the compact box too`,
  );
  win.close();
  console.log(`  ok - task-id-less rows keep the layout`);
}

// ---------------------------------------------------------------------------
// 5. Only the *history* task panels change - the promptlets list keeps
//    its compact inline buttons.
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
  const copy = item.querySelector('.sidebar-item-copy');
  assert.ok(copy, 'promptlet row has a copy button');
  const svg = copy.querySelector('svg');
  assertCompactIcon(win, svg, 'promptlet copy');
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
  testActionsOnSeparateLine(win);
  testActionsKeepCompactSize(win);
  await testIconSizeSurvivesStateChanges(win, posted);
  testRowWithoutTaskIdKeepsOwnLine();
  testTricksListUnaffected();
  console.log('All historyTaskActionsRow tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
