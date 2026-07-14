// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests: desktop-width defaults for the REMOTE webapp.
//
// Feature under test — on desktop browsers the remote webapp must:
//
//   1. Give the docked history panel a DEFAULT width of 1/4 of the
//      browser screen (25vw, clamped to the resize range [220, 600]):
//        * the CSS fallback for --sidebar-w is a 25vw-based clamp();
//        * main.js seeds the resize logic (aria-valuenow, keyboard
//          baseline, dblclick reset) from 25% of window.innerWidth
//          instead of a fixed 300px;
//        * an explicitly dragged/persisted width still wins.
//   2. Let the chat panels (children of #output), the pinned task
//      panel (#task-panel), and the composer (#input-container) span
//      90% of the chat webview (#app column) instead of the old
//      768px / 85% / 75% caps.
//
// jsdom performs no layout, so the width RULES are asserted from the
// parsed remote-codex.css and the dynamic behavior (defaults, drag,
// reset, keyboard, persistence) from the real chat.html + main.js
// running in jsdom (window.innerWidth = 1024 → 25% = 256px).
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/remoteDesktopWidths.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom webview running the real chat.html + panelCopy.js +
 * main.js, with a controllable matchMedia stub and pointer-capture
 * stubs (jsdom implements neither natively).
 *
 * @param {object} [opts]
 * @param {boolean} [opts.remote=true] add class="remote-chat" to body
 * @param {boolean} [opts.desktopMatches=true] initial
 *     matchMedia('(min-width: 900px)').matches
 * @param {string|null} [opts.storedWidth=null] pre-seed
 *     localStorage['kiss-sidebar-w'] BEFORE main.js runs
 * @returns {{win: object, posted: Array, fireChange: function(boolean)}}
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
  win.Element.prototype.setPointerCapture = function () {};
  win.Element.prototype.releasePointerCapture = function () {};
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
  return {win, posted, fireChange};
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

const CSS = fs.readFileSync(path.join(MEDIA, 'remote-codex.css'), 'utf8');

/** Declaration body of the LAST body.remote-chat rule for selector. */
function cssRule(selector) {
  const source = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(source + String.raw`\s*(?:,[^{]*)?\{([^}]*)\}`, 'g');
  let body = null;
  let m;
  while ((m = re.exec(CSS)) !== null) body = m[1];
  assert.ok(body !== null, `CSS rule for ${selector} missing`);
  return body;
}

// jsdom default innerWidth is 1024 → 25% of the screen is 256px.
const QUARTER = 256;

// ---------------------------------------------------------------------------
// 1. CSS: the docked sidebar defaults to 1/4 of the browser screen.
//    Both the sidebar width and #app's margin use the SAME 25vw-based
//    clamp() fallback inside var(--sidebar-w, ...).
// ---------------------------------------------------------------------------
function testCssSidebarQuarterScreenDefault() {
  const sidebar = cssRule('body.remote-chat.remote-desktop #sidebar');
  const app = cssRule('body.remote-chat.remote-desktop #app');
  const fallback = 'var(--sidebar-w, clamp(220px, 25vw, 600px))';
  assert.ok(
    sidebar.includes(`width: ${fallback}`),
    `docked sidebar default width must be 25vw (1/4 screen) clamped ` +
      `to the resize range — got: ${sidebar.trim()}`,
  );
  assert.ok(
    app.includes(`margin-left: ${fallback}`),
    '#app margin must be driven by the SAME 25vw-based fallback',
  );
  console.log('PASS CSS defaults the docked sidebar to 1/4 screen (25vw)');
}

// ---------------------------------------------------------------------------
// 2. CSS: chat panels streamed into #output span 90% of the chat
//    webview column (was min(85%, 768px)).
// ---------------------------------------------------------------------------
function testCssChatPanelsNinetyPercent() {
  const rule = cssRule('body.remote-chat #output > *:not(#welcome)');
  assert.ok(
    /max-width:\s*90%/.test(rule),
    `chat panels must span 90% of the chat webview — got: ${rule.trim()}`,
  );
  assert.ok(!rule.includes('768px'), 'the old 768px cap must be gone');
  console.log('PASS CSS chat panels span 90% of the chat webview');
}

// ---------------------------------------------------------------------------
// 3. CSS: the pinned task panel spans 90% of the chat webview (was
//    75%) and stays a right-aligned bubble.
// ---------------------------------------------------------------------------
function testCssTaskPanelNinetyPercent() {
  const rule = cssRule('body.remote-chat #task-panel');
  assert.ok(
    /max-width:\s*90%/.test(rule),
    `the fixed task panel must span 90% — got: ${rule.trim()}`,
  );
  assert.ok(
    rule.includes('margin-left: auto'),
    'task panel must stay right-aligned',
  );
  console.log('PASS CSS fixed task panel spans 90% of the chat webview');
}

// ---------------------------------------------------------------------------
// 4. CSS: the composer card spans 90% of the chat webview (was 768px)
//    and stays centered.
// ---------------------------------------------------------------------------
function testCssComposerNinetyPercent() {
  const rule = cssRule('body.remote-chat #input-container');
  assert.ok(
    /max-width:\s*90%/.test(rule),
    `the composer must span 90% of the chat webview — got: ${rule.trim()}`,
  );
  assert.ok(!rule.includes('768px'), 'the old 768px cap must be gone');
  assert.ok(rule.includes('margin: 0 auto'), 'composer stays centered');
  console.log('PASS CSS composer spans 90% of the chat webview');
}

// ---------------------------------------------------------------------------
// 5. JS: with no persisted width the resize logic seeds from 1/4 of
//    the window width (jsdom: 1024 → 256), reflected in aria-valuenow.
// ---------------------------------------------------------------------------
function testDefaultSeededFromQuarterWindow() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  assert.strictEqual(
    resizer.getAttribute('aria-valuenow'),
    String(QUARTER),
    'default aria-valuenow must be 25% of the window width',
  );
  assert.strictEqual(
    sidebarW(win),
    '',
    'no inline --sidebar-w until the user resizes (CSS fallback rules)',
  );
  win.close();
  console.log('PASS resize logic seeds its default from 1/4 window width');
}

// ---------------------------------------------------------------------------
// 6. JS: the keyboard baseline starts from the quarter-width default
//    (ArrowRight = 256 + 16 = 272).
// ---------------------------------------------------------------------------
function testKeyboardBaselineQuarterWindow() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true}),
  );
  assert.strictEqual(
    sidebarW(win),
    `${QUARTER + 16}px`,
    'ArrowRight must grow from the quarter-screen default',
  );
  win.close();
  console.log('PASS keyboard resize starts from the quarter-width default');
}

// ---------------------------------------------------------------------------
// 7. JS: double-click resets the width back to 1/4 of the CURRENT
//    window width (not a fixed 300px) and clears persistence.
// ---------------------------------------------------------------------------
function testDoubleClickResetsToQuarterWindow() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  drag(win, resizer, 300, 500);
  assert.strictEqual(sidebarW(win), '500px');
  resizer.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
  assert.strictEqual(
    sidebarW(win),
    `${QUARTER}px`,
    'dblclick must reset to 1/4 of the window width',
  );
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), String(QUARTER));
  assert.strictEqual(win.localStorage.getItem('kiss-sidebar-w'), null);
  win.close();
  console.log('PASS double-click resets to 1/4 of the window width');
}

// ---------------------------------------------------------------------------
// 8. JS: an explicitly persisted width still beats the quarter-screen
//    default, and drag clamping stays [220, 600].
// ---------------------------------------------------------------------------
function testPersistedWidthStillWins() {
  const stored = makeWebview({
    remote: true,
    desktopMatches: true,
    storedWidth: '450',
  });
  assert.strictEqual(
    sidebarW(stored.win),
    '450px',
    'persisted width must override the quarter-screen default',
  );
  assert.strictEqual(
    stored.win.document
      .getElementById('sidebar-resizer')
      .getAttribute('aria-valuenow'),
    '450',
  );
  stored.win.close();
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  drag(win, resizer, 300, 80);
  assert.strictEqual(sidebarW(win), '220px', 'min clamp unchanged');
  drag(win, resizer, 220, 900);
  assert.strictEqual(sidebarW(win), '600px', 'max clamp unchanged');
  win.close();
  console.log('PASS persisted width wins; clamp range unchanged');
}

// ---------------------------------------------------------------------------
// 9. Isolation: no quarter-screen seeding leaks into the VS Code
//    extension webview (no remote-chat class).
// ---------------------------------------------------------------------------
function testVsCodeWebviewIsolation() {
  const {win, posted} = makeWebview({remote: false, desktopMatches: true});
  assert.ok(
    posted.find((m) => m.type === 'ready'),
    'webview must boot normally',
  );
  const resizer = win.document.getElementById('sidebar-resizer');
  assert.strictEqual(
    resizer.getAttribute('aria-valuenow'),
    null,
    'no ARIA seeding inside the VS Code webview',
  );
  assert.strictEqual(sidebarW(win), '');
  win.close();
  console.log('PASS VS Code webview (no remote-chat) is unaffected');
}

testCssSidebarQuarterScreenDefault();
testCssChatPanelsNinetyPercent();
testCssTaskPanelNinetyPercent();
testCssComposerNinetyPercent();
testDefaultSeededFromQuarterWindow();
testKeyboardBaselineQuarterWindow();
testDoubleClickResetsToQuarterWindow();
testPersistedWidthStillWins();
testVsCodeWebviewIsolation();
console.log('All remoteDesktopWidths tests passed.');
