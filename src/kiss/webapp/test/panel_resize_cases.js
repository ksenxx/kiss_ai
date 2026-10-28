// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// The resize tests the two docked remote panels (the history sidebar,
// the task-info panel) share, written once over a panel description:
//
// * The handle is an accessible ARIA separator inside its panel.
// * A drag resizes the panel, captures the pointer, persists the width
//   and leaves the other panel alone; it clamps to the sliver floor and
//   to the cap that keeps the chat column usable.
// * A persisted width is restored, sanitised, and springs back once a
//   window grown wide enough holds it.
// * Arrow keys step the width by 16px (and stop at the floor), a
//   double-click resets it, a narrower window shrinks the panel.
// * On a narrow (mobile) window drag and keys are inert, and a
//   pointercancel ends a drag like pointerup.
//
// A panel is `{label, resizerId, parentId, cssVar, otherCssVar,
// storageKey, maxFor, xFor, growKey, shrinkKey}`: `maxFor(windowW)` is
// the drag cap with the other panel at its 20vw default, `xFor(w)` the
// clientX that asks for width `w` (the task-info handle rides its left
// edge), `growKey` / `shrinkKey` the arrow keys widening / narrowing it.
//
// Resize bounds declared as --sidebar-min-w / --chat-min-w in
// remote-codex.css.  A drag may take a panel down to a 10px sliver; the
// DEFAULT width is one fifth of the browser window (the 20vw fallback).
// jsdom never loads that stylesheet, so main.js falls back to the same
// numbers, and reports window.innerWidth === 1024.

const assert = require('assert');
const {makeResizePage: makePage, panelW, pointer, drag} = require('./panel_resize_helpers');

const MIN_W = 10;
const WINDOW_W = 1024;
const DEFAULT_W = Math.round(WINDOW_W * 0.2);
const CHAT_MIN = 360;

function key(win, el, name) {
  el.dispatchEvent(new win.KeyboardEvent('keydown', {key: name, bubbles: true}));
}

function resizerOf(win, panel) {
  return win.document.getElementById(panel.resizerId);
}

function widthOf(win, panel) {
  return panelW(win, panel.cssVar);
}

function testResizerExistsAndIsAccessible(panel) {
  const {win} = makePage({desktopMatches: true});
  const resizer = resizerOf(win, panel);
  assert.ok(resizer, `#${panel.resizerId} handle must exist in chat.html`);
  assert.strictEqual(resizer.getAttribute('role'), 'separator', 'an ARIA separator');
  assert.strictEqual(
    resizer.getAttribute('aria-orientation'),
    'vertical',
    'window splitter separators are aria-orientation=vertical',
  );
  assert.strictEqual(resizer.getAttribute('tabindex'), '0', 'keyboard focusable');
  assert.strictEqual(
    win.innerWidth,
    WINDOW_W,
    'these bounds assume jsdom reports a 1024px-wide window',
  );
  assert.strictEqual(resizer.getAttribute('aria-valuemin'), String(MIN_W));
  assert.strictEqual(resizer.getAttribute('aria-valuemax'), String(panel.maxFor(WINDOW_W)));
  assert.strictEqual(
    resizer.getAttribute('aria-valuenow'),
    String(DEFAULT_W),
    'the default width is reflected in aria-valuenow',
  );
  assert.strictEqual(
    resizer.parentElement.id,
    panel.parentId,
    'the handle lives inside its panel',
  );
  win.close();
  console.log(`PASS ${panel.label} resizer exists and is an accessible ARIA separator`);
}

function testDragResizesPanel(panel) {
  const {win, captured, released} = makePage({desktopMatches: true});
  const resizer = resizerOf(win, panel);
  drag(win, resizer, panel.xFor(DEFAULT_W), panel.xFor(420));
  assert.strictEqual(widthOf(win, panel), '420px', `a drag sets ${panel.cssVar}: 420px`);
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), '420', 'aria-valuenow tracks the width');
  assert.ok(captured.length >= 1, 'the drag captures the pointer so fast drags do not escape');
  assert.ok(released.length >= 1, 'pointerup releases the capture');
  assert.strictEqual(
    win.localStorage.getItem(panel.storageKey),
    '420',
    'pointerup persists the width to localStorage',
  );
  assert.strictEqual(
    panelW(win, panel.otherCssVar),
    '',
    'resizing one panel does not touch the other',
  );
  win.close();
  console.log(`PASS dragging the handle resizes the docked ${panel.label}`);
}

function testDragClampsWidth(panel) {
  const {win} = makePage({desktopMatches: true});
  const resizer = resizerOf(win, panel);
  const maxW = panel.maxFor(WINDOW_W);
  drag(win, resizer, panel.xFor(DEFAULT_W), panel.xFor(80));
  assert.strictEqual(widthOf(win, panel), '80px', 'a drag may collapse the panel far below the default');
  drag(win, resizer, panel.xFor(80), panel.xFor(2));
  assert.strictEqual(widthOf(win, panel), `${MIN_W}px`, `a drag clamps to the ${MIN_W}px sliver floor`);
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), String(MIN_W));
  drag(win, resizer, panel.xFor(MIN_W), panel.xFor(900));
  assert.strictEqual(
    widthOf(win, panel),
    `${maxW}px`,
    `a drag clamps to ${maxW}px so the chat keeps ${CHAT_MIN}px`,
  );
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), String(maxW));
  win.close();
  console.log(`PASS ${panel.label} drag width is clamped to [${MIN_W}px, ${maxW}px]`);
}

function testWidthPersistsAndRestores(panel) {
  const first = makePage({desktopMatches: true});
  drag(first.win, resizerOf(first.win, panel), panel.xFor(DEFAULT_W), panel.xFor(430));
  assert.strictEqual(first.win.localStorage.getItem(panel.storageKey), '430');
  first.win.close();
  const second = makePage({desktopMatches: true, stored: {[panel.storageKey]: '430'}});
  assert.strictEqual(widthOf(second.win, panel), '430px', 'a persisted width is restored on load');
  assert.strictEqual(resizerOf(second.win, panel).getAttribute('aria-valuenow'), '430');
  second.win.close();
  console.log(`PASS ${panel.label} width persists to localStorage and restores on load`);
}

function testPersistedGarbageSanitized(panel) {
  const cases = [
    ['abc', '', 'a non-numeric persisted width is ignored (the CSS default applies)'],
    ['9999', `${panel.maxFor(WINDOW_W)}px`, 'an over-wide persisted width is clamped down'],
    ['240', '240px', 'a persisted sliver width above the 10px floor is honoured as-is'],
    ['4', `${MIN_W}px`, 'a persisted width below the sliver floor is clamped up'],
  ];
  for (const [stored, expected, message] of cases) {
    const {win} = makePage({desktopMatches: true, stored: {[panel.storageKey]: stored}});
    assert.strictEqual(widthOf(win, panel), expected, message);
    win.close();
  }
  console.log(`PASS garbage / out-of-range persisted ${panel.label} widths are sanitized`);
}

function testKeyboardResize(panel) {
  const {win} = makePage({desktopMatches: true});
  const resizer = resizerOf(win, panel);
  drag(win, resizer, panel.xFor(DEFAULT_W), panel.xFor(400));
  key(win, resizer, panel.growKey);
  assert.strictEqual(widthOf(win, panel), '416px', `${panel.growKey} grows by 16px`);
  key(win, resizer, panel.shrinkKey);
  key(win, resizer, panel.shrinkKey);
  assert.strictEqual(widthOf(win, panel), '384px', `${panel.shrinkKey} shrinks by 16px`);
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), '384');
  assert.strictEqual(
    win.localStorage.getItem(panel.storageKey),
    '384',
    'a keyboard resize persists too',
  );
  drag(win, resizer, panel.xFor(384), panel.xFor(20));
  assert.strictEqual(widthOf(win, panel), '20px');
  key(win, resizer, panel.shrinkKey);
  key(win, resizer, panel.shrinkKey);
  assert.strictEqual(widthOf(win, panel), `${MIN_W}px`, `${panel.shrinkKey} stops at the sliver floor`);
  win.close();
  console.log(`PASS arrow keys resize the ${panel.label} by 16px steps`);
}

function testDoubleClickResets(panel) {
  const {win} = makePage({desktopMatches: true});
  const resizer = resizerOf(win, panel);
  drag(win, resizer, panel.xFor(DEFAULT_W), panel.xFor(400));
  assert.strictEqual(widthOf(win, panel), '400px');
  resizer.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
  assert.strictEqual(widthOf(win, panel), `${DEFAULT_W}px`, 'a double-click resets to the default width');
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), String(DEFAULT_W));
  assert.strictEqual(
    win.localStorage.getItem(panel.storageKey),
    null,
    'a double-click clears the persisted width',
  );
  win.close();
  console.log(`PASS ${panel.label} double-click resets the width and clears persistence`);
}

function setWindowWidth(win, width) {
  Object.defineProperty(win, 'innerWidth', {value: width, configurable: true});
  win.dispatchEvent(new win.Event('resize'));
}

function testShrinkingWindowNarrowsThePanel(panel) {
  const {win} = makePage({desktopMatches: true});
  const resizer = resizerOf(win, panel);
  const maxW = panel.maxFor(WINDOW_W);
  drag(win, resizer, panel.xFor(DEFAULT_W), panel.xFor(maxW));
  assert.strictEqual(widthOf(win, panel), `${maxW}px`);
  setWindowWidth(win, 940);
  assert.strictEqual(
    widthOf(win, panel),
    `${panel.maxFor(940)}px`,
    'a narrower window shrinks the panel so the chat stays usable',
  );
  assert.strictEqual(
    resizer.getAttribute('aria-valuemax'),
    String(panel.maxFor(940)),
    'aria-valuemax follows the narrower window',
  );
  setWindowWidth(win, 700);
  assert.strictEqual(
    widthOf(win, panel),
    `${panel.maxFor(700)}px`,
    'even below the default width the chat keeps its minimum',
  );
  win.close();
  console.log(`PASS shrinking the window narrows the docked ${panel.label}`);
}

function testWidePreferenceSurvivesNarrowWindow(panel) {
  const {win} = makePage({desktopMatches: true, stored: {[panel.storageKey]: '1500'}});
  assert.strictEqual(
    widthOf(win, panel),
    `${panel.maxFor(WINDOW_W)}px`,
    'a persisted width wider than the cap renders clamped',
  );
  setWindowWidth(win, 2500);
  assert.strictEqual(
    widthOf(win, panel),
    '1500px',
    'the persisted preference springs back once the window holds it',
  );
  assert.strictEqual(
    win.localStorage.getItem(panel.storageKey),
    '1500',
    'no user action happened: the stored preference stays untouched',
  );
  win.close();
  console.log(`PASS a wide persisted ${panel.label} width survives a narrow window`);
}

function testMobileDragInert(panel) {
  const {win} = makePage({desktopMatches: false});
  const resizer = resizerOf(win, panel);
  assert.ok(resizer, 'the handle exists in the shared markup');
  drag(win, resizer, panel.xFor(DEFAULT_W), panel.xFor(420));
  assert.strictEqual(widthOf(win, panel), '', `a drag on a narrow window does not set ${panel.cssVar}`);
  assert.strictEqual(win.localStorage.getItem(panel.storageKey), null, 'no persistence on mobile');
  key(win, resizer, panel.growKey);
  assert.strictEqual(widthOf(win, panel), '', 'keyboard resize is inert on mobile');
  win.close();
  console.log(`PASS ${panel.label} resize is inert on narrow (mobile) remote windows`);
}

function testPointerCancelEndsDrag(panel) {
  const {win} = makePage({desktopMatches: true});
  const resizer = resizerOf(win, panel);
  pointer(win, resizer, 'pointerdown', {clientX: panel.xFor(DEFAULT_W), pointerId: 1});
  assert.ok(
    win.document.body.classList.contains('sidebar-resizing'),
    'body carries sidebar-resizing during the drag',
  );
  pointer(win, resizer, 'pointermove', {clientX: panel.xFor(440), pointerId: 1});
  assert.strictEqual(widthOf(win, panel), '440px');
  pointer(win, resizer, 'pointercancel', {clientX: panel.xFor(440), pointerId: 1});
  pointer(win, resizer, 'pointermove', {clientX: panel.xFor(200), pointerId: 1});
  assert.strictEqual(widthOf(win, panel), '440px', 'moves after pointercancel are ignored (drag ended)');
  assert.ok(
    !win.document.body.classList.contains('sidebar-resizing'),
    'the sidebar-resizing body class is cleared on cancel',
  );
  win.close();
  console.log(`PASS pointercancel ends the ${panel.label} drag like pointerup`);
}

/** Run every shared resize test over `panel`. */
function runPanelResizeTests(panel) {
  testResizerExistsAndIsAccessible(panel);
  testDragResizesPanel(panel);
  testDragClampsWidth(panel);
  testWidthPersistsAndRestores(panel);
  testPersistedGarbageSanitized(panel);
  testKeyboardResize(panel);
  testDoubleClickResets(panel);
  testShrinkingWindowNarrowsThePanel(panel);
  testWidePreferenceSurvivesNarrowWindow(panel);
  testMobileDragInert(panel);
  testPointerCancelEndsDrag(panel);
}

module.exports = {
  runPanelResizeTests,
  setWindowWidth,
  key,
  MIN_W,
  WINDOW_W,
  DEFAULT_W,
  CHAT_MIN,
};
