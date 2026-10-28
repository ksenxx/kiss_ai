// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The task-info panel's resize handle: the shared docked-panel tests
// (panel_resize_cases.js) plus what is specific to the right-hand
// panel: each drag cap reserves the OTHER panel's actual width, two huge
// persisted widths cannot crush the chat, the first keyboard step grows
// from the default, and an untouched panel keeps following 20vw.

'use strict';

const assert = require('assert');
const {makeResizePage: makePage, panelW, drag} = require('./panel_resize_helpers');
const {
  runPanelResizeTests,
  setWindowWidth,
  key,
  MIN_W,
  WINDOW_W,
  DEFAULT_W,
  CHAT_MIN,
} = require('./panel_resize_cases');

function metaW(win) {
  return panelW(win, '--meta-w');
}

function sidebarW(win) {
  return panelW(win, '--sidebar-w');
}

// The task-info panel's handle rides its LEFT edge, so a pointer at
// clientX asks for a width of (window.innerWidth - clientX).  The cap
// only ever comes from the chat column between the two panels keeping
// --chat-min-w; with the history panel at its untouched 20vw default:
const maxFor = w => w - CHAT_MIN - Math.round(w * 0.2);

// clientX that asks the meta panel for width w.
const xFor = w => WINDOW_W - w;

const META = {
  label: 'meta panel',
  resizerId: 'meta-resizer',
  parentId: 'meta-panel',
  cssVar: '--meta-w',
  otherCssVar: '--sidebar-w',
  storageKey: 'kiss-meta-w',
  maxFor,
  xFor,
  growKey: 'ArrowLeft',
  shrinkKey: 'ArrowRight',
};

function testCapsReserveTheOtherPanelsActualWidth() {
  const {win} = makePage({desktopMatches: true});
  const metaResizer = win.document.getElementById('meta-resizer');
  const sidebarResizer = win.document.getElementById('sidebar-resizer');
  // Widen the history panel to 400px: the task-info panel may now only
  // grow to 1024 - 360 - 400 = 264px.
  drag(win, sidebarResizer, 300, 400);
  assert.strictEqual(sidebarW(win), '400px');
  assert.strictEqual(
    metaResizer.getAttribute('aria-valuemax'),
    String(WINDOW_W - CHAT_MIN - 400),
    'widening the history panel must shrink the meta drag cap',
  );
  drag(win, metaResizer, xFor(DEFAULT_W), xFor(900));
  assert.strictEqual(
    metaW(win),
    `${WINDOW_W - CHAT_MIN - 400}px`,
    'the meta cap must reserve the history panel ACTUAL width, not 20vw',
  );
  // And the other way round: collapse the task-info panel to the
  // sliver and the history panel may take almost everything.
  drag(win, metaResizer, xFor(264), xFor(2));
  assert.strictEqual(metaW(win), `${MIN_W}px`);
  assert.strictEqual(
    sidebarResizer.getAttribute('aria-valuemax'),
    String(WINDOW_W - CHAT_MIN - MIN_W),
    'collapsing the task-info panel must widen the sidebar drag cap',
  );
  drag(win, sidebarResizer, 400, 1000);
  assert.strictEqual(
    sidebarW(win),
    `${WINDOW_W - CHAT_MIN - MIN_W}px`,
    'the sidebar may now grow into the space the meta sliver freed',
  );
  win.close();
  console.log('PASS each drag cap reserves the OTHER panel actual width');
}

function testBothPersistedWidthsKeepTheChatUsable() {
  // Both panels persisted huge: the history panel loads first and is
  // clamped against the meta default; the meta panel then clamps
  // against the history panel's ACTUAL width, so the chat column
  // keeps --chat-min-w.
  const {win} = makePage({
    desktopMatches: true,
    stored: {'kiss-meta-w': '900', 'kiss-sidebar-w': '900'},
  });
  const sb = parseInt(sidebarW(win), 10);
  const mt = parseInt(metaW(win), 10);
  assert.ok(
    sb + mt <= WINDOW_W - CHAT_MIN,
    `two huge persisted widths must leave the chat ${CHAT_MIN}px: ` +
      `sidebar ${sb}px + meta ${mt}px on a ${WINDOW_W}px window`,
  );
  win.close();
  console.log('PASS two huge persisted widths cannot crush the chat');
}

function testKeyboardBaselineIsTheDefaultWidth() {
  const {win} = makePage({desktopMatches: true});
  const resizer = win.document.getElementById('meta-resizer');
  key(win, resizer, 'ArrowLeft');
  assert.strictEqual(
    metaW(win),
    `${DEFAULT_W + 16}px`,
    'the first keyboard step must grow from the default width',
  );
  win.close();
  console.log('PASS meta keyboard resize starts from the default width');
}

function testUntouchedPanelStaysFluid() {
  const {win} = makePage({desktopMatches: true});
  const resizer = win.document.getElementById('meta-resizer');
  assert.strictEqual(
    metaW(win),
    '',
    'no inline --meta-w until the user resizes (CSS 20vw rules)',
  );
  setWindowWidth(win, 1500);
  assert.strictEqual(
    metaW(win),
    '',
    'an untouched panel must keep following the fluid 20vw default',
  );
  assert.strictEqual(
    resizer.getAttribute('aria-valuenow'),
    String(Math.round(1500 * 0.2)),
    'aria-valuenow must follow the fluid default across resizes',
  );
  win.close();
  console.log('PASS an untouched meta panel stays fluid at 20vw');
}

runPanelResizeTests(META);
testCapsReserveTheOtherPanelsActualWidth();
testBothPersistedWidthsKeepTheChatUsable();
testKeyboardBaselineIsTheDefaultWidth();
testUntouchedPanelStaysFluid();
console.log('All remoteMetaResize tests passed.');
