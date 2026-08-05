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

// jsdom reports window.innerWidth = 1024, so 34vw is 348px and the
// --sidebar-min-w floor (the width at which every history filter
// toggle fits on one line) wins.
const DEFAULT_W = 520;

// The panel may never take so much of the window that the chat column
// drops below --chat-min-w (360px), so on jsdom's 1024px window the
// effective maximum is 664px rather than the 820px hard bound.
const MAX_W = 1024 - 360;

function testCssSidebarWidthFitsTheFilterToggles() {
  const vars = cssRule('body.remote-chat');
  assert.ok(
    vars.includes(`--sidebar-min-w: ${DEFAULT_W}px`),
    `the docked panel floor must be the one-line filter width ` +
      `(${DEFAULT_W}px) — got: ${vars.trim()}`,
  );
  assert.ok(
    /--sidebar-default-w:\s*clamp\(\s*var\(--sidebar-min-w\),[^;]*var\(--sidebar-max-w\)\)/.test(
      vars,
    ),
    'the default width must clamp between the shared min/max bounds',
  );
  const sidebar = cssRule('body.remote-chat.remote-desktop #sidebar.open');
  const app = cssRule(
    'body.remote-chat.remote-desktop:has(#sidebar.open) #app',
  );
  const fallback = 'var(--sidebar-w, var(--sidebar-default-w))';
  assert.ok(
    sidebar.includes(`width: ${fallback}`),
    `the docked sidebar width must fall back to --sidebar-default-w — ` +
      `got: ${sidebar.trim()}`,
  );
  assert.ok(
    app.includes(`margin-left: ${fallback}`),
    '#app margin must be driven by the SAME width fallback',
  );
  console.log('PASS CSS sizes the docked sidebar to fit the filter toggles');
}

function testCssDockedRulesRequireAnOpenSidebar() {
  // The burger button only toggles #sidebar.open, so every docked rule
  // must depend on that class — otherwise the panel stays glued to the
  // screen and the chat never reclaims its width.
  const app = cssRule('body.remote-chat.remote-desktop #app');
  assert.ok(
    /margin-left:\s*0/.test(app),
    `with the panel hidden #app must have no left offset — ` +
      `got: ${app.trim()}`,
  );
  const stripped = CSS.replace(/\/\*[\s\S]*?\*\//g, '');
  const dockedWithoutOpen = new RegExp(
    String.raw`body\.remote-chat\.remote-desktop #sidebar\s*\{`,
  );
  assert.ok(
    !dockedWithoutOpen.test(stripped),
    'docked #sidebar rules must be scoped to #sidebar.open',
  );
  console.log('PASS CSS docks the sidebar only while it is open');
}

function testCssSettingsPanelStaysNarrow() {
  const settings = cssRule('body.remote-chat.remote-desktop #settings-panel');
  assert.ok(
    settings.includes('width: min(90vw, var(--settings-panel-w))'),
    `the settings drawer must stay as narrow as the VS Code sidebar — ` +
      `got: ${settings.trim()}`,
  );
  const vars = cssRule('body.remote-chat');
  assert.ok(
    /--settings-panel-w:\s*\d+px/.test(vars),
    'the narrow settings width must be a fixed pixel bound',
  );
  console.log('PASS CSS keeps the desktop settings panel narrow');
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

function testDefaultSeededFromTheOneLineFloor() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  assert.strictEqual(
    resizer.getAttribute('aria-valuenow'),
    String(DEFAULT_W),
    'default aria-valuenow must be the one-line filter width',
  );
  assert.strictEqual(
    sidebarW(win),
    '',
    'no inline --sidebar-w until the user resizes (CSS fallback rules)',
  );
  win.close();
  console.log('PASS resize logic seeds its default from the one-line floor');
}

function testKeyboardBaselineIsTheDefaultWidth() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  resizer.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true}),
  );
  assert.strictEqual(
    sidebarW(win),
    `${DEFAULT_W + 16}px`,
    'ArrowRight must grow from the default width',
  );
  win.close();
  console.log('PASS keyboard resize starts from the default width');
}

function testDoubleClickResetsToTheDefaultWidth() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  drag(win, resizer, 300, 640);
  assert.strictEqual(sidebarW(win), '640px');
  resizer.dispatchEvent(new win.MouseEvent('dblclick', {bubbles: true}));
  assert.strictEqual(
    sidebarW(win),
    `${DEFAULT_W}px`,
    'dblclick must reset to the default width',
  );
  assert.strictEqual(resizer.getAttribute('aria-valuenow'), String(DEFAULT_W));
  assert.strictEqual(win.localStorage.getItem('kiss-sidebar-w'), null);
  win.close();
  console.log('PASS double-click resets to the default width');
}

function testPersistedWidthStillWins() {
  const stored = makeWebview({
    remote: true,
    desktopMatches: true,
    storedWidth: '640',
  });
  assert.strictEqual(
    sidebarW(stored.win),
    '640px',
    'persisted width must override the default',
  );
  assert.strictEqual(
    stored.win.document
      .getElementById('sidebar-resizer')
      .getAttribute('aria-valuenow'),
    '640',
  );
  stored.win.close();
  const narrow = makeWebview({
    remote: true,
    desktopMatches: true,
    storedWidth: '240',
  });
  assert.strictEqual(
    sidebarW(narrow.win),
    `${DEFAULT_W}px`,
    'a width persisted before the widening must be clamped back up so ' +
      'the filter toggles stay on one line',
  );
  narrow.win.close();
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  drag(win, resizer, 300, 80);
  assert.strictEqual(sidebarW(win), `${DEFAULT_W}px`, 'min clamp');
  drag(win, resizer, 520, 1600);
  assert.strictEqual(sidebarW(win), `${MAX_W}px`, 'max clamp');
  win.close();
  console.log('PASS persisted width wins but is clamped to the new range');
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

testCssSidebarWidthFitsTheFilterToggles();
testCssDockedRulesRequireAnOpenSidebar();
testCssSettingsPanelStaysNarrow();
testCssChatPanelsNotRestyled();
testCssComposerFullWidth();
testDefaultSeededFromTheOneLineFloor();
testKeyboardBaselineIsTheDefaultWidth();
testDoubleClickResetsToTheDefaultWidth();
testPersistedWidthStillWins();
testVsCodeWebviewIsolation();
console.log('All remoteDesktopWidths tests passed.');
