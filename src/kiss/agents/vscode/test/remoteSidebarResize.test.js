// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  function fireChange(matches) {
    mql.matches = matches;
    listeners.forEach((fn) => fn(mql));
  }
  return {win, posted, fireChange, captured, released};
}

function sidebarW(win) {
  return win.document.documentElement.style.getPropertyValue('--sidebar-w');
}

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

function drag(win, resizer, x0, x1) {
  pointer(win, resizer, 'pointerdown', {clientX: x0, pointerId: 1});
  pointer(win, resizer, 'pointermove', {clientX: x1, pointerId: 1});
  pointer(win, resizer, 'pointerup', {clientX: x1, pointerId: 1});
}

// Resize bounds declared as --sidebar-min-w / --sidebar-max-w /
// --chat-min-w in remote-codex.css.  The minimum is the width at which
// every history filter toggle fits on a single line; jsdom never loads
// that stylesheet, so main.js falls back to the same numbers.  The
// effective maximum also leaves CHAT_MIN for the chat column, and
// jsdom reports window.innerWidth === 1024.
const MIN_W = 520;
const HARD_MAX_W = 820;
const CHAT_MIN = 360;
const WINDOW_W = 1024;
const MAX_W = Math.min(HARD_MAX_W, WINDOW_W - CHAT_MIN);

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
  assert.strictEqual(
    win.innerWidth,
    WINDOW_W,
    'these bounds assume jsdom reports a 1024px-wide window',
  );
  assert.strictEqual(resizer.getAttribute('aria-valuemin'), String(MIN_W));
  assert.strictEqual(resizer.getAttribute('aria-valuemax'), String(MAX_W));
  assert.strictEqual(
    resizer.getAttribute('aria-valuenow'),
    String(MIN_W),
    'default width must be reflected in aria-valuenow',
  );
  assert.strictEqual(
    resizer.parentElement.id,
    'sidebar',
    'the handle must live inside the sidebar (its right edge)',
  );
  win.close();
  console.log('PASS resizer exists and is an accessible ARIA separator');
}

function testDragResizesSidebar() {
  const {win, captured, released} = makeWebview({
    remote: true,
    desktopMatches: true,
  });
  const resizer = win.document.getElementById('sidebar-resizer');
  drag(win, resizer, 600, 620);
  assert.strictEqual(
    sidebarW(win),
    '620px',
    'dragging to x=620 must set --sidebar-w: 620px',
  );
  assert.strictEqual(
    resizer.getAttribute('aria-valuenow'),
    '620',
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

function testDragClampsWidth() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  drag(win, resizer, 600, 80);
  assert.strictEqual(
    sidebarW(win),
    `${MIN_W}px`,
    `drag far left clamps to ${MIN_W}px so the filter toggles stay on ` +
      'one line',
  );
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), String(MIN_W));
  drag(win, resizer, MIN_W, 1600);
  assert.strictEqual(
    sidebarW(win),
    `${MAX_W}px`,
    `drag far right clamps to ${MAX_W}px`,
  );
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), String(MAX_W));
  win.close();
  console.log(`PASS drag width is clamped to [${MIN_W}px, ${MAX_W}px]`);
}

function testWidthPersistsAndRestores() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  drag(win, resizer, 600, 650);
  assert.strictEqual(
    win.localStorage.getItem('kiss-sidebar-w'),
    '650',
    'pointerup must persist the width to localStorage',
  );
  win.close();
  const second = makeWebview({
    remote: true,
    desktopMatches: true,
    storedWidth: '650',
  });
  assert.strictEqual(
    sidebarW(second.win),
    '650px',
    'persisted width must be restored on load',
  );
  assert.strictEqual(
    second.win.document
      .getElementById('sidebar-resizer')
      .getAttribute('aria-valuenow'),
    '650',
  );
  second.win.close();
  console.log('PASS width persists to localStorage and restores on load');
}

function testPersistedGarbageSanitized() {
  const garbage = makeWebview({
    remote: true,
    desktopMatches: true,
    storedWidth: 'abc',
  });
  assert.strictEqual(
    sidebarW(garbage.win),
    '',
    'non-numeric persisted width must be ignored (CSS default applies)',
  );
  garbage.win.close();
  const huge = makeWebview({
    remote: true,
    desktopMatches: true,
    storedWidth: '9999',
  });
  assert.strictEqual(
    sidebarW(huge.win),
    `${MAX_W}px`,
    'over-wide persisted width must be clamped down',
  );
  huge.win.close();
  const tiny = makeWebview({
    remote: true,
    desktopMatches: true,
    storedWidth: '240',
  });
  assert.strictEqual(
    sidebarW(tiny.win),
    `${MIN_W}px`,
    'a width persisted before the panel was widened must be clamped up',
  );
  tiny.win.close();
  console.log('PASS garbage / out-of-range persisted widths are sanitized');
}

function testKeyboardResize() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  drag(win, resizer, 600, 600);
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true}),
  );
  assert.strictEqual(sidebarW(win), '616px', 'ArrowRight grows by 16px');
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true}),
  );
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true}),
  );
  assert.strictEqual(sidebarW(win), '584px', 'ArrowLeft shrinks by 16px');
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), '584');
  assert.strictEqual(
    win.localStorage.getItem('kiss-sidebar-w'),
    '584',
    'keyboard resize must persist too',
  );
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true}),
  );
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true}),
  );
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true}),
  );
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true}),
  );
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true}),
  );
  assert.strictEqual(
    sidebarW(win),
    `${MIN_W}px`,
    'ArrowLeft must stop at the one-line filter width',
  );
  win.close();
  console.log('PASS ArrowLeft/ArrowRight resize the sidebar by 16px steps');
}

function testDoubleClickResets() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  drag(win, resizer, 600, 640);
  assert.strictEqual(sidebarW(win), '640px');
  resizer.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
  assert.strictEqual(
    sidebarW(win),
    `${MIN_W}px`,
    'double-click must reset to the default width',
  );
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), String(MIN_W));
  assert.strictEqual(
    win.localStorage.getItem('kiss-sidebar-w'),
    null,
    'double-click must clear the persisted width',
  );
  win.close();
  console.log('PASS double-click resets the width and clears persistence');
}

function testShrinkingWindowNarrowsThePanel() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  drag(win, resizer, 600, MAX_W);
  assert.strictEqual(sidebarW(win), `${MAX_W}px`);
  Object.defineProperty(win, 'innerWidth', {value: 940, configurable: true});
  win.dispatchEvent(new win.Event('resize'));
  assert.strictEqual(
    sidebarW(win),
    `${940 - CHAT_MIN}px`,
    'a narrower window must shrink the panel so the chat stays usable',
  );
  assert.strictEqual(
    resizer.getAttribute('aria-valuemax'),
    String(940 - CHAT_MIN),
    'aria-valuemax must follow the narrower window',
  );
  Object.defineProperty(win, 'innerWidth', {value: 700, configurable: true});
  win.dispatchEvent(new win.Event('resize'));
  assert.strictEqual(
    sidebarW(win),
    `${MIN_W}px`,
    'the one-line filter width is still the hard floor',
  );
  win.close();
  console.log('PASS shrinking the window narrows the docked panel');
}

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

function testPointerCancelEndsDrag() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  pointer(win, resizer, 'pointerdown', {clientX: 600, pointerId: 1});
  pointer(win, resizer, 'pointermove', {clientX: 640, pointerId: 1});
  assert.strictEqual(sidebarW(win), '640px');
  pointer(win, resizer, 'pointercancel', {clientX: 640, pointerId: 1});
  pointer(win, resizer, 'pointermove', {clientX: 750, pointerId: 1});
  assert.strictEqual(
    sidebarW(win),
    '640px',
    'moves after pointercancel must be ignored (drag ended)',
  );
  assert.ok(
    !win.document.body.classList.contains('sidebar-resizing'),
    'the sidebar-resizing body class must be cleared on cancel',
  );
  win.close();
  console.log('PASS pointercancel ends the drag like pointerup');
}

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
testShrinkingWindowNarrowsThePanel();
testMobileDragInert();
testVsCodeWebviewIsolation();
testPointerCancelEndsDrag();
testDragKeepsDockAndMarksBody();
console.log('All remoteSidebarResize tests passed.');
