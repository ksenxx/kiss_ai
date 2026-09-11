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
  const {
    remote = true,
    desktopMatches = true,
    storedWidth = null,
    storedSidebarWidth = null,
  } = opts || {};
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
    win.localStorage.setItem('kiss-meta-w', storedWidth);
  }
  if (storedSidebarWidth !== null) {
    win.localStorage.setItem('kiss-sidebar-w', storedSidebarWidth);
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

function metaW(win) {
  return win.document.documentElement.style.getPropertyValue('--meta-w');
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

// The task-info panel's handle rides its LEFT edge, so a pointer at
// clientX asks for a width of (window.innerWidth - clientX).  Same
// bounds as the history panel: a drag may collapse the panel to the
// --sidebar-min-w sliver, the DEFAULT width is one fifth of the
// window (the 20vw --meta-panel-w fallback), and the cap only ever
// comes from the chat column between the two panels keeping
// --chat-min-w.  jsdom reports window.innerWidth === 1024.
const MIN_W = 10;
const WINDOW_W = 1024;
const DEFAULT_W = Math.round(WINDOW_W * 0.2);
const CHAT_MIN = 360;
// With the history panel at its untouched 20vw default:
const maxFor = w => w - CHAT_MIN - Math.round(w * 0.2);
const MAX_W = maxFor(WINDOW_W);

// clientX that asks the meta panel for width w.
const xFor = w => WINDOW_W - w;

function testResizerExistsAndIsAccessible() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('meta-resizer');
  assert.ok(resizer, '#meta-resizer handle must exist in chat.html');
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
    resizer.parentElement.id,
    'meta-panel',
    'the handle must live inside the task-info panel (its left edge)',
  );
  assert.strictEqual(resizer.getAttribute('aria-valuemin'), String(MIN_W));
  assert.strictEqual(resizer.getAttribute('aria-valuemax'), String(MAX_W));
  assert.strictEqual(
    resizer.getAttribute('aria-valuenow'),
    String(DEFAULT_W),
    'default width must be reflected in aria-valuenow',
  );
  win.close();
  console.log('PASS meta resizer exists and is an accessible ARIA separator');
}

function testDragResizesMetaPanel() {
  const {win, captured, released} = makeWebview({
    remote: true,
    desktopMatches: true,
  });
  const resizer = win.document.getElementById('meta-resizer');
  drag(win, resizer, xFor(DEFAULT_W), xFor(420));
  assert.strictEqual(
    metaW(win),
    '420px',
    'dragging the handle 420px from the right edge must set --meta-w: 420px',
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
  assert.strictEqual(
    win.localStorage.getItem('kiss-meta-w'),
    '420',
    'pointerup must persist the width to localStorage',
  );
  assert.strictEqual(
    sidebarW(win),
    '',
    'resizing the task-info panel must not touch the history panel',
  );
  win.close();
  console.log('PASS dragging the handle resizes the docked task-info panel');
}

function testDragClampsWidth() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('meta-resizer');
  drag(win, resizer, xFor(DEFAULT_W), xFor(80));
  assert.strictEqual(
    metaW(win),
    '80px',
    'a drag may collapse the panel far below the default width',
  );
  drag(win, resizer, xFor(80), xFor(2));
  assert.strictEqual(
    metaW(win),
    `${MIN_W}px`,
    `drag far right clamps to the ${MIN_W}px sliver floor`,
  );
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), String(MIN_W));
  drag(win, resizer, xFor(MIN_W), xFor(900));
  assert.strictEqual(
    metaW(win),
    `${MAX_W}px`,
    `drag far left clamps to ${MAX_W}px so the chat keeps ${CHAT_MIN}px`,
  );
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), String(MAX_W));
  win.close();
  console.log(`PASS meta drag width is clamped to [${MIN_W}px, ${MAX_W}px]`);
}

function testCapsReserveTheOtherPanelsActualWidth() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
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

function testWidthPersistsAndRestores() {
  const first = makeWebview({remote: true, desktopMatches: true});
  const resizer = first.win.document.getElementById('meta-resizer');
  drag(first.win, resizer, xFor(DEFAULT_W), xFor(430));
  assert.strictEqual(first.win.localStorage.getItem('kiss-meta-w'), '430');
  first.win.close();
  const second = makeWebview({
    remote: true,
    desktopMatches: true,
    storedWidth: '430',
  });
  assert.strictEqual(
    metaW(second.win),
    '430px',
    'persisted width must be restored on load',
  );
  assert.strictEqual(
    second.win.document
      .getElementById('meta-resizer')
      .getAttribute('aria-valuenow'),
    '430',
  );
  second.win.close();
  console.log('PASS meta width persists to localStorage and restores on load');
}

function testPersistedGarbageSanitized() {
  const garbage = makeWebview({
    remote: true,
    desktopMatches: true,
    storedWidth: 'abc',
  });
  assert.strictEqual(
    metaW(garbage.win),
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
    metaW(huge.win),
    `${MAX_W}px`,
    'over-wide persisted width must be clamped down',
  );
  huge.win.close();
  const tiny = makeWebview({
    remote: true,
    desktopMatches: true,
    storedWidth: '4',
  });
  assert.strictEqual(
    metaW(tiny.win),
    `${MIN_W}px`,
    'a persisted width below the sliver floor must be clamped up',
  );
  tiny.win.close();
  console.log('PASS garbage / out-of-range persisted meta widths sanitized');
}

function testBothPersistedWidthsKeepTheChatUsable() {
  // Both panels persisted huge: the history panel loads first and is
  // clamped against the meta default; the meta panel then clamps
  // against the history panel's ACTUAL width, so the chat column
  // keeps --chat-min-w.
  const {win} = makeWebview({
    remote: true,
    desktopMatches: true,
    storedWidth: '900',
    storedSidebarWidth: '900',
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

function testKeyboardResize() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('meta-resizer');
  drag(win, resizer, xFor(DEFAULT_W), xFor(400));
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true}),
  );
  assert.strictEqual(
    metaW(win),
    '416px',
    'ArrowLeft grows the RIGHT-hand panel by 16px',
  );
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true}),
  );
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true}),
  );
  assert.strictEqual(
    metaW(win),
    '384px',
    'ArrowRight shrinks the RIGHT-hand panel by 16px',
  );
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), '384');
  assert.strictEqual(
    win.localStorage.getItem('kiss-meta-w'),
    '384',
    'keyboard resize must persist too',
  );
  win.close();
  console.log('PASS ArrowLeft/ArrowRight resize the meta panel by 16px');
}

function testKeyboardBaselineIsTheDefaultWidth() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('meta-resizer');
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true}),
  );
  assert.strictEqual(
    metaW(win),
    `${DEFAULT_W + 16}px`,
    'the first keyboard step must grow from the default width',
  );
  win.close();
  console.log('PASS meta keyboard resize starts from the default width');
}

function testDoubleClickResets() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('meta-resizer');
  drag(win, resizer, xFor(DEFAULT_W), xFor(400));
  assert.strictEqual(metaW(win), '400px');
  resizer.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
  assert.strictEqual(
    metaW(win),
    `${DEFAULT_W}px`,
    'double-click must reset to the default width',
  );
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), String(DEFAULT_W));
  assert.strictEqual(
    win.localStorage.getItem('kiss-meta-w'),
    null,
    'double-click must clear the persisted width',
  );
  win.close();
  console.log('PASS double-click resets the width and clears persistence');
}

function testShrinkingWindowNarrowsThePanel() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('meta-resizer');
  drag(win, resizer, xFor(DEFAULT_W), xFor(900));
  assert.strictEqual(metaW(win), `${MAX_W}px`);
  Object.defineProperty(win, 'innerWidth', {value: 940, configurable: true});
  win.dispatchEvent(new win.Event('resize'));
  assert.strictEqual(
    metaW(win),
    `${maxFor(940)}px`,
    'a narrower window must shrink the panel so the chat stays usable',
  );
  assert.strictEqual(
    resizer.getAttribute('aria-valuemax'),
    String(maxFor(940)),
    'aria-valuemax must follow the narrower window',
  );
  win.close();
  console.log('PASS shrinking the window narrows the docked meta panel');
}

function testUntouchedPanelStaysFluid() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('meta-resizer');
  assert.strictEqual(
    metaW(win),
    '',
    'no inline --meta-w until the user resizes (CSS 20vw rules)',
  );
  Object.defineProperty(win, 'innerWidth', {value: 1500, configurable: true});
  win.dispatchEvent(new win.Event('resize'));
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

function testWidePreferenceSurvivesNarrowWindow() {
  const {win} = makeWebview({
    remote: true,
    desktopMatches: true,
    storedWidth: '600',
  });
  assert.strictEqual(
    metaW(win),
    `${MAX_W}px`,
    'a persisted width wider than the cap renders clamped',
  );
  Object.defineProperty(win, 'innerWidth', {value: 2500, configurable: true});
  win.dispatchEvent(new win.Event('resize'));
  assert.strictEqual(
    metaW(win),
    '600px',
    'the persisted preference must spring back once the window holds it',
  );
  assert.strictEqual(
    win.localStorage.getItem('kiss-meta-w'),
    '600',
    'no user action happened: the stored preference stays untouched',
  );
  win.close();
  console.log('PASS a wide persisted meta width survives a narrow window');
}

function testPointerCancelEndsDrag() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('meta-resizer');
  pointer(win, resizer, 'pointerdown', {clientX: xFor(DEFAULT_W), pointerId: 1});
  assert.ok(
    win.document.body.classList.contains('sidebar-resizing'),
    'body must carry the resizing class during a meta drag',
  );
  pointer(win, resizer, 'pointermove', {clientX: xFor(440), pointerId: 1});
  assert.strictEqual(metaW(win), '440px');
  pointer(win, resizer, 'pointercancel', {clientX: xFor(440), pointerId: 1});
  pointer(win, resizer, 'pointermove', {clientX: xFor(200), pointerId: 1});
  assert.strictEqual(
    metaW(win),
    '440px',
    'moves after pointercancel must be ignored (drag ended)',
  );
  assert.ok(
    !win.document.body.classList.contains('sidebar-resizing'),
    'the resizing body class must be cleared on cancel',
  );
  win.close();
  console.log('PASS pointercancel ends the meta drag like pointerup');
}

function testMobileDragInert() {
  const {win} = makeWebview({remote: true, desktopMatches: false});
  const resizer = win.document.getElementById('meta-resizer');
  assert.ok(resizer, 'handle exists in the shared markup');
  drag(win, resizer, xFor(DEFAULT_W), xFor(420));
  assert.strictEqual(
    metaW(win),
    '',
    'dragging on a narrow window must not set --meta-w',
  );
  assert.strictEqual(
    win.localStorage.getItem('kiss-meta-w'),
    null,
    'no persistence on mobile',
  );
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true}),
  );
  assert.strictEqual(metaW(win), '', 'keyboard resize inert on mobile');
  win.close();
  console.log('PASS meta resize is inert on narrow (mobile) remote windows');
}

function testVsCodeWebviewIsolation() {
  const {win, posted} = makeWebview({remote: false, desktopMatches: true});
  assert.ok(
    posted.find((m) => m.type === 'ready'),
    'webview must boot normally',
  );
  const resizer = win.document.getElementById('meta-resizer');
  assert.strictEqual(
    resizer.getAttribute('aria-valuenow'),
    null,
    'no ARIA seeding inside the VS Code webview',
  );
  drag(win, resizer, xFor(DEFAULT_W), xFor(420));
  assert.strictEqual(
    metaW(win),
    '',
    'the VS Code webview must never gain --meta-w',
  );
  win.close();
  console.log('PASS VS Code webview (no remote-chat) is unaffected');
}

testResizerExistsAndIsAccessible();
testDragResizesMetaPanel();
testDragClampsWidth();
testCapsReserveTheOtherPanelsActualWidth();
testWidthPersistsAndRestores();
testPersistedGarbageSanitized();
testBothPersistedWidthsKeepTheChatUsable();
testKeyboardResize();
testKeyboardBaselineIsTheDefaultWidth();
testDoubleClickResets();
testShrinkingWindowNarrowsThePanel();
testUntouchedPanelStaysFluid();
testWidePreferenceSurvivesNarrowWindow();
testPointerCancelEndsDrag();
testMobileDragInert();
testVsCodeWebviewIsolation();
console.log('All remoteMetaResize tests passed.');
