// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests: horizontally RESIZABLE docked history sidebar in
// the REMOTE webapp desktop layout.
//
// Feature under test: on desktop-wide remote windows (body.remote-chat
// + body.remote-desktop, matchMedia '(min-width: 900px)') the docked
// agent-history sidebar (#sidebar) must be horizontally resizable via
// a drag handle (#sidebar-resizer) on its right edge:
//
//   * the handle exists in chat.html and is an accessible ARIA window
//     splitter (role=separator, aria-orientation=vertical, focusable,
//     aria-valuenow/min/max reflecting the width);
//   * dragging with pointer events resizes the sidebar by setting the
//     --sidebar-w custom property (which drives BOTH the sidebar width
//     and #app's margin-left in remote-codex.css, so they never
//     desync);
//   * the width is clamped to [220px, 600px];
//   * the width persists across reloads (localStorage) and garbage or
//     out-of-range persisted values are sanitized;
//   * ArrowLeft/ArrowRight on the focused handle resize by 16px steps
//     (W3C window-splitter keyboard pattern);
//   * double-click resets to the 300px default and clears persistence;
//   * NONE of this activates on mobile-width windows nor inside the
//     VS Code extension webview (no remote-chat class), and jsdom-less
//     pointer-capture APIs must never crash main.js.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/remoteSidebarResize.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom webview running the real chat.html + panelCopy.js +
 * main.js, with a controllable matchMedia stub and recorded
 * pointer-capture calls (jsdom implements neither natively).
 *
 * @param {object} [opts]
 * @param {boolean} [opts.remote=true] add class="remote-chat" to body
 * @param {boolean} [opts.desktopMatches=true] initial
 *     matchMedia('(min-width: 900px)').matches
 * @param {string|null} [opts.storedWidth=null] pre-seed
 *     localStorage['kiss-sidebar-w'] BEFORE main.js runs
 * @returns {{win: object, posted: Array, fireChange: function(boolean),
 *     captured: Array, released: Array}}
 */
function makeWebview(opts) {
  const {remote = true, desktopMatches = true, storedWidth = null} =
    opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  if (remote) {
    html = html.replace('<body', '<body class="remote-chat"');
  }
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  // jsdom has no pointer-capture: record the calls main.js makes.
  const captured = [];
  const released = [];
  win.Element.prototype.setPointerCapture = function (id) {
    captured.push(id);
  };
  win.Element.prototype.releasePointerCapture = function (id) {
    released.push(id);
  };
  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: (msg) => posted.push(msg),
      getState: () => state,
      setState: (s) => {
        state = s;
      },
    };
  };
  if (storedWidth !== null) {
    win.localStorage.setItem('kiss-sidebar-w', storedWidth);
  }
  const listeners = [];
  const mql = {
    matches: desktopMatches === true,
    media: '(min-width: 900px)',
    addEventListener: (ev, fn) => {
      if (ev === 'change') listeners.push(fn);
    },
    removeEventListener: () => {},
    addListener: (fn) => listeners.push(fn),
    removeListener: () => {},
  };
  win.matchMedia = function (query) {
    if (query === '(min-width: 900px)') return mql;
    return {
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  function fireChange(matches) {
    mql.matches = matches;
    listeners.forEach((fn) => fn(mql));
  }
  return {win, posted, fireChange, captured, released};
}

/** Current --sidebar-w custom property value ('' when unset). */
function sidebarW(win) {
  return win.document.documentElement.style.getPropertyValue('--sidebar-w');
}

/** Dispatch a pointer-type event on *el* (jsdom: MouseEvent carrier). */
function pointer(win, el, type, props) {
  const ev = new win.MouseEvent(type, {
    bubbles: true,
    cancelable: true,
    button: 0,
    ...props,
  });
  if (props && props.pointerId !== undefined) {
    Object.defineProperty(ev, 'pointerId', {value: props.pointerId});
  }
  el.dispatchEvent(ev);
  return ev;
}

/** Perform a full drag of the resizer: down at x0, move to x1, up. */
function drag(win, resizer, x0, x1) {
  pointer(win, resizer, 'pointerdown', {clientX: x0, pointerId: 1});
  pointer(win, resizer, 'pointermove', {clientX: x1, pointerId: 1});
  pointer(win, resizer, 'pointerup', {clientX: x1, pointerId: 1});
}

// ---------------------------------------------------------------------------
// 1. The resize handle exists and is an accessible ARIA window
//    splitter on the docked desktop sidebar.
// ---------------------------------------------------------------------------
function testResizerExistsAndIsAccessible() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  assert.ok(resizer, '#sidebar-resizer handle must exist in chat.html');
  assert.strictEqual(
    resizer.getAttribute('role'),
    'separator',
    'resizer must be an ARIA separator',
  );
  assert.strictEqual(
    resizer.getAttribute('aria-orientation'),
    'vertical',
    'window splitter separators are aria-orientation=vertical',
  );
  assert.strictEqual(
    resizer.getAttribute('tabindex'),
    '0',
    'resizer must be keyboard focusable',
  );
  assert.strictEqual(resizer.getAttribute('aria-valuemin'), '220');
  assert.strictEqual(resizer.getAttribute('aria-valuemax'), '600');
  assert.strictEqual(
    resizer.getAttribute('aria-valuenow'),
    '300',
    'default width 300 must be reflected in aria-valuenow',
  );
  assert.strictEqual(
    resizer.parentElement.id,
    'sidebar',
    'the handle must live inside the sidebar (its right edge)',
  );
  win.close();
  console.log('PASS resizer exists and is an accessible ARIA separator');
}

// ---------------------------------------------------------------------------
// 2. Dragging the handle resizes the sidebar via --sidebar-w.
// ---------------------------------------------------------------------------
function testDragResizesSidebar() {
  const {win, captured, released} = makeWebview({
    remote: true,
    desktopMatches: true,
  });
  const resizer = win.document.getElementById('sidebar-resizer');
  drag(win, resizer, 300, 420);
  assert.strictEqual(
    sidebarW(win),
    '420px',
    'dragging to x=420 must set --sidebar-w: 420px',
  );
  assert.strictEqual(
    resizer.getAttribute('aria-valuenow'),
    '420',
    'aria-valuenow must track the width',
  );
  assert.ok(
    captured.length >= 1,
    'the drag must capture the pointer so fast drags do not escape',
  );
  assert.ok(released.length >= 1, 'pointerup must release the capture');
  win.close();
  console.log('PASS dragging the handle resizes the docked sidebar');
}

// ---------------------------------------------------------------------------
// 3. The width is clamped to [220, 600].
// ---------------------------------------------------------------------------
function testDragClampsWidth() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  drag(win, resizer, 300, 80);
  assert.strictEqual(sidebarW(win), '220px', 'drag far left clamps to 220px');
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), '220');
  drag(win, resizer, 220, 900);
  assert.strictEqual(sidebarW(win), '600px', 'drag far right clamps to 600px');
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), '600');
  win.close();
  console.log('PASS drag width is clamped to [220px, 600px]');
}

// ---------------------------------------------------------------------------
// 4. The chosen width persists (localStorage) and is restored on load.
// ---------------------------------------------------------------------------
function testWidthPersistsAndRestores() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  drag(win, resizer, 300, 450);
  assert.strictEqual(
    win.localStorage.getItem('kiss-sidebar-w'),
    '450',
    'pointerup must persist the width to localStorage',
  );
  win.close();
  // A fresh page with a persisted width restores it on load.
  const second = makeWebview({
    remote: true,
    desktopMatches: true,
    storedWidth: '450',
  });
  assert.strictEqual(
    sidebarW(second.win),
    '450px',
    'persisted width must be restored on load',
  );
  assert.strictEqual(
    second.win.document
      .getElementById('sidebar-resizer')
      .getAttribute('aria-valuenow'),
    '450',
  );
  second.win.close();
  console.log('PASS width persists to localStorage and restores on load');
}

// ---------------------------------------------------------------------------
// 5. Garbage / out-of-range persisted values are sanitized.
// ---------------------------------------------------------------------------
function testPersistedGarbageSanitized() {
  const garbage = makeWebview({
    remote: true,
    desktopMatches: true,
    storedWidth: 'abc',
  });
  assert.strictEqual(
    sidebarW(garbage.win),
    '',
    'non-numeric persisted width must be ignored (default 300 via CSS)',
  );
  garbage.win.close();
  const huge = makeWebview({
    remote: true,
    desktopMatches: true,
    storedWidth: '9999',
  });
  assert.strictEqual(
    sidebarW(huge.win),
    '600px',
    'out-of-range persisted width must be clamped',
  );
  huge.win.close();
  console.log('PASS garbage / out-of-range persisted widths are sanitized');
}

// ---------------------------------------------------------------------------
// 6. Keyboard: ArrowRight/ArrowLeft resize by 16px and persist.
// ---------------------------------------------------------------------------
function testKeyboardResize() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true}),
  );
  assert.strictEqual(sidebarW(win), '316px', 'ArrowRight grows by 16px');
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true}),
  );
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true}),
  );
  assert.strictEqual(sidebarW(win), '284px', 'ArrowLeft shrinks by 16px');
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), '284');
  assert.strictEqual(
    win.localStorage.getItem('kiss-sidebar-w'),
    '284',
    'keyboard resize must persist too',
  );
  win.close();
  console.log('PASS ArrowLeft/ArrowRight resize the sidebar by 16px steps');
}

// ---------------------------------------------------------------------------
// 7. Double-click resets to the 300px default and clears persistence.
// ---------------------------------------------------------------------------
function testDoubleClickResets() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  drag(win, resizer, 300, 500);
  assert.strictEqual(sidebarW(win), '500px');
  resizer.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
  assert.strictEqual(
    sidebarW(win),
    '300px',
    'double-click must reset to the 300px default',
  );
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), '300');
  assert.strictEqual(
    win.localStorage.getItem('kiss-sidebar-w'),
    null,
    'double-click must clear the persisted width',
  );
  win.close();
  console.log('PASS double-click resets the width and clears persistence');
}

// ---------------------------------------------------------------------------
// 8. Mobile (narrow) remote windows: dragging must do nothing (the
//    drawer is full-flow there; the handle is display:none via CSS and
//    inert via the remote-desktop guard in JS).
// ---------------------------------------------------------------------------
function testMobileDragInert() {
  const {win} = makeWebview({remote: true, desktopMatches: false});
  const resizer = win.document.getElementById('sidebar-resizer');
  assert.ok(resizer, 'handle exists in the shared markup');
  drag(win, resizer, 300, 420);
  assert.strictEqual(
    sidebarW(win),
    '',
    'dragging on a narrow window must not set --sidebar-w',
  );
  assert.strictEqual(
    win.localStorage.getItem('kiss-sidebar-w'),
    null,
    'no persistence on mobile',
  );
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true}),
  );
  assert.strictEqual(sidebarW(win), '', 'keyboard resize inert on mobile');
  win.close();
  console.log('PASS resize is inert on narrow (mobile) remote windows');
}

// ---------------------------------------------------------------------------
// 9. VS Code extension webview isolation: no remote-chat class → the
//    resizer never activates and main.js must not crash.
// ---------------------------------------------------------------------------
function testVsCodeWebviewIsolation() {
  const {win, posted} = makeWebview({remote: false, desktopMatches: true});
  assert.ok(
    posted.find((m) => m.type === 'ready'),
    'webview must boot normally',
  );
  const resizer = win.document.getElementById('sidebar-resizer');
  drag(win, resizer, 300, 420);
  assert.strictEqual(
    sidebarW(win),
    '',
    'the VS Code webview must never gain --sidebar-w',
  );
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true}),
  );
  assert.strictEqual(sidebarW(win), '');
  win.close();
  console.log('PASS VS Code webview (no remote-chat) is unaffected');
}

// ---------------------------------------------------------------------------
// 10. pointercancel ends the drag exactly like pointerup (touch
//     interruptions, e.g. an incoming call, must not leave a stuck
//     drag that keeps resizing on later moves).
// ---------------------------------------------------------------------------
function testPointerCancelEndsDrag() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  pointer(win, resizer, 'pointerdown', {clientX: 300, pointerId: 1});
  pointer(win, resizer, 'pointermove', {clientX: 400, pointerId: 1});
  assert.strictEqual(sidebarW(win), '400px');
  pointer(win, resizer, 'pointercancel', {clientX: 400, pointerId: 1});
  // A stray move AFTER the cancel must not resize any further.
  pointer(win, resizer, 'pointermove', {clientX: 550, pointerId: 1});
  assert.strictEqual(
    sidebarW(win),
    '400px',
    'moves after pointercancel must be ignored (drag ended)',
  );
  assert.ok(
    !win.document.body.classList.contains('sidebar-resizing'),
    'the sidebar-resizing body class must be cleared on cancel',
  );
  win.close();
  console.log('PASS pointercancel ends the drag like pointerup');
}

// ---------------------------------------------------------------------------
// 11. During a drag the body carries sidebar-resizing (disables text
//     selection via CSS) and the sidebar stays docked open.
// ---------------------------------------------------------------------------
function testDragKeepsDockAndMarksBody() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
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

testResizerExistsAndIsAccessible();
testDragResizesSidebar();
testDragClampsWidth();
testWidthPersistsAndRestores();
testPersistedGarbageSanitized();
testKeyboardResize();
testDoubleClickResets();
testMobileDragInert();
testVsCodeWebviewIsolation();
testPointerCancelEndsDrag();
testDragKeepsDockAndMarksBody();
console.log('All remoteSidebarResize tests passed.');
