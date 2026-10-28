// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The history sidebar's resize handle: the shared docked-panel tests
// (panel_resize_cases.js) plus what is specific to the left-hand panel:
// a drag keeps it docked open, a wide window lets it grow past the old
// cap, and keyboard steps start from the rendered (clamped) width.

'use strict';

const assert = require('assert');
const {makeResizePage: makePage, panelW, pointer, drag} = require('./panel_resize_helpers');
const {
  runPanelResizeTests,
  setWindowWidth,
  key,
  WINDOW_W,
  CHAT_MIN,
} = require('./panel_resize_cases');

function sidebarW(win) {
  return panelW(win, '--sidebar-w');
}

// A drag has no upper cap of its own: the panel may grow as long as the
// chat column, between it and the docked task-info panel (20vw), keeps
// CHAT_MIN.
const maxFor = w => Math.floor(w * 0.8) - CHAT_MIN;
const MAX_W = maxFor(WINDOW_W);

const SIDEBAR = {
  label: 'sidebar',
  resizerId: 'sidebar-resizer',
  parentId: 'sidebar',
  cssVar: '--sidebar-w',
  otherCssVar: '--meta-w',
  storageKey: 'kiss-sidebar-w',
  maxFor,
  xFor: w => w,
  growKey: 'ArrowRight',
  shrinkKey: 'ArrowLeft',
};

function testDragKeepsDockAndMarksBody() {
  const {win} = makePage({desktopMatches: true});
  const sidebar = win.document.getElementById('sidebar');
  const resizer = win.document.getElementById('sidebar-resizer');
  pointer(win, resizer, 'pointerdown', {clientX: 300, pointerId: 1});
  assert.ok(
    win.document.body.classList.contains('sidebar-resizing'),
    'body must carry sidebar-resizing during the drag',
  );
  pointer(win, resizer, 'pointermove', {clientX: 350, pointerId: 1});
  assert.ok(
    sidebar.classList.contains('open'),
    'the sidebar must stay docked open while resizing',
  );
  pointer(win, resizer, 'pointerup', {clientX: 350, pointerId: 1});
  assert.ok(
    !win.document.body.classList.contains('sidebar-resizing'),
    'sidebar-resizing must be removed on pointerup',
  );
  assert.ok(sidebar.classList.contains('open'));
  win.close();
  console.log('PASS drag marks the body and keeps the sidebar docked');
}

function testWideWindowAllowsBeyondDefaultCap() {
  const {win} = makePage({desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  setWindowWidth(win, 2500);
  drag(win, resizer, 300, 1500);
  assert.strictEqual(
    sidebarW(win),
    '1500px',
    'a drag may pass the old 820px cap; only the chat minimum limits it',
  );
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), '1500');
  drag(win, resizer, 1500, 2400);
  assert.strictEqual(
    sidebarW(win),
    `${maxFor(2500)}px`,
    'the chat column (beside the 20vw task-info panel) keeps its minimum',
  );
  // The DEFAULT width follows the window: one fifth of 2500px.
  resizer.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
  assert.strictEqual(
    sidebarW(win),
    `${Math.round(2500 * 0.2)}px`,
    'double-click default is one fifth of the window',
  );
  win.close();
  console.log('PASS wide windows may drag the panel past the default cap');
}

function testKeyboardStepsFromRenderedWidth() {
  const {win} = makePage({
    desktopMatches: true,
    stored: {'kiss-sidebar-w': '1500'},
  });
  const resizer = win.document.getElementById('sidebar-resizer');
  assert.strictEqual(sidebarW(win), `${MAX_W}px`);
  key(win, resizer, 'ArrowLeft');
  assert.strictEqual(
    sidebarW(win),
    `${MAX_W - 16}px`,
    'ArrowLeft must step from the RENDERED width the user sees, not ' +
      'from the (clamped-away) stored preference',
  );
  assert.strictEqual(
    win.localStorage.getItem('kiss-sidebar-w'),
    String(MAX_W - 16),
    'the keyboard resize persists exactly what is on screen',
  );
  win.close();
  console.log('PASS keyboard resize steps from the rendered width');
}

runPanelResizeTests(SIDEBAR);
testDragKeepsDockAndMarksBody();
testWideWindowAllowsBeyondDefaultCap();
testKeyboardStepsFromRenderedWidth();
console.log('All remoteSidebarResize tests passed.');
