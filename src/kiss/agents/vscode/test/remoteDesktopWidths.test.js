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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  function fireChange(matches) {
    mql.matches = matches;
    listeners.forEach((fn) => fn(mql));
  }
  return {win, posted, fireChange};
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

const CSS = fs.readFileSync(path.join(MEDIA, 'remote-codex.css'), 'utf8');

function cssRule(selector) {
  const source = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(source + String.raw`\s*(?:,[^{]*)?\{([^}]*)\}`, 'g');
  let body = null;
  let m;
  while ((m = re.exec(CSS)) !== null) body = m[1];
  assert.ok(body !== null, `CSS rule for ${selector} missing`);
  return body;
}

const QUARTER = 256;

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

function testCssChatPanelsNotRestyled() {
  // The chat thread and the fixed task panel must render exactly like
  // the VS Code extension webview (main.css), so remote-codex.css must
  // not target #output children or #task-panel at all.
  const stripped = CSS.replace(/\/\*[\s\S]*?\*\//g, '');
  assert.ok(
    !stripped.includes('#output'),
    'remote-codex.css must not restyle #output or its children',
  );
  assert.ok(
    !stripped.includes('#task-panel'),
    'remote-codex.css must not restyle the fixed task panel',
  );
  console.log('PASS CSS chat panels and task panel keep the extension look');
}

function testCssComposerFullWidth() {
  const rule = cssRule('body.remote-chat #input-container');
  assert.ok(
    !/max-width/.test(rule),
    `the composer must have NO width cap so it is as wide as the ` +
      `chat webview — got: ${rule.trim()}`,
  );
  assert.ok(
    !rule.includes('margin: 0 auto'),
    'the composer must not be a centered narrow column',
  );
  assert.ok(!rule.includes('768px'), 'the old 768px cap must be gone');
  console.log('PASS CSS composer spans the full chat webview width');
}

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
testCssChatPanelsNotRestyled();
testCssComposerFullWidth();
testDefaultSeededFromQuarterWindow();
testKeyboardBaselineQuarterWindow();
testDoubleClickResetsToQuarterWindow();
testPersistedWidthStillWins();
testVsCodeWebviewIsolation();
console.log('All remoteDesktopWidths tests passed.');
