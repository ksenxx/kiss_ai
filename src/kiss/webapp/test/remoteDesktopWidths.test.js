// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {
  makePage: makeChatPage,
  fakeMatchMedia,
  MEDIA,
} = require('./simplify2_harness');
const {inlineDesignTokens} = require('./designTokens');

function makePage(opts) {
  const {desktopMatches = true} = opts || {};
  return makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts: win => fakeMatchMedia(win, desktopMatches === true),
  });
}

function sidebarW(win) {
  return win.document.documentElement.style.getPropertyValue('--sidebar-w');
}

const CSS = inlineDesignTokens(
  fs.readFileSync(path.join(MEDIA, 'remote-codex.css'), 'utf8'),
);

function cssRule(selector) {
  const source = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(source + String.raw`\s*(?:,[^{]*)?\{([^}]*)\}`, 'g');
  let body = null;
  let m;
  while ((m = re.exec(CSS)) !== null) body = m[1];
  assert.ok(body !== null, `CSS rule for ${selector} missing`);
  return body;
}

// jsdom reports window.innerWidth = 1024; the default docked width is
// one fifth of the window (the 20vw fallback of --sidebar-default-w),
// so 205px.  Drags, clamping and persistence of the width are covered
// by remoteSidebarResize.test.js.
const MIN_W = 10;
const DEFAULT_W = Math.round(1024 * 0.2);

function testCssSidebarWidthIsOneFifthOfTheWindow() {
  const vars = cssRule('body.remote-chat');
  assert.ok(
    vars.includes(`--sidebar-min-w: ${MIN_W}px`),
    `the docked panel resize floor must be the ${MIN_W}px sliver — ` +
      `got: ${vars.trim()}`,
  );
  assert.ok(
    /--sidebar-default-w:\s*20vw/.test(vars),
    'the default width must be one fifth of the browser window (20vw)',
  );
  const sidebar = cssRule('body.remote-chat.remote-desktop #sidebar.open');
  assert.ok(
    /padding-left:\s*min\(16px,[^;]*\/\s*4\)/.test(sidebar) &&
      /padding-right:\s*min\(16px,[^;]*\/\s*4\)/.test(sidebar),
    "the docked panel's horizontal padding must collapse with the " +
      'panel so a 10px drag really renders 10px (border-box padding ' +
      `would otherwise floor it at 32px) — got: ${sidebar.trim()}`,
  );
  const app = cssRule(
    'body.remote-chat.remote-desktop:has(#sidebar.open) #app',
  );
  const fallback = 'var(--sidebar-w, var(--sidebar-default-w))';
  assert.ok(
    sidebar.includes(`width: ${fallback}`),
    'the docked sidebar width must fall back to --sidebar-default-w — ' +
      `got: ${sidebar.trim()}`,
  );
  assert.ok(
    app.includes(`margin-left: ${fallback}`),
    '#app margin must be driven by the SAME width fallback',
  );
  console.log('PASS CSS sizes the docked sidebar to a fifth of the window');
}

function testCssDockedRulesRequireAnOpenSidebar() {
  // The burger button only toggles #sidebar.open, so every docked rule
  // must depend on that class — otherwise the panel stays glued to the
  // screen and the chat never reclaims its width.
  const app = cssRule('body.remote-chat.remote-desktop #app');
  assert.ok(
    /margin-left:\s*0/.test(app),
    'with the panel hidden #app must have no left offset — ' +
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
    'the settings drawer must stay a narrow fixed-width column — ' +
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
  // The chat thread and the fixed task panel are styled by main.css
  // alone, so remote-codex.css must not target #output children or
  // #task-panel at all.
  const stripped = CSS.replace(/\/\*[\s\S]*?\*\//g, '');
  assert.ok(
    !stripped.includes('#output'),
    'remote-codex.css must not restyle #output or its children',
  );
  assert.ok(
    !stripped.includes('#task-panel'),
    'remote-codex.css must not restyle the fixed task panel',
  );
  console.log('PASS CSS chat panels and task panel keep the main.css look');
}

function testCssComposerFullWidth() {
  const rule = cssRule('body.remote-chat #input-container');
  assert.ok(
    !/max-width/.test(rule),
    'the composer must have NO width cap so it is as wide as the ' +
      `chat page — got: ${rule.trim()}`,
  );
  assert.ok(
    !rule.includes('margin: 0 auto'),
    'the composer must not be a centered narrow column',
  );
  assert.ok(!rule.includes('768px'), 'the old 768px cap must be gone');
  console.log('PASS CSS composer spans the full chat page width');
}

function testDefaultSeededFromOneFifthOfTheWindow() {
  const {win} = makePage({desktopMatches: true});
  const resizer = win.document.getElementById('sidebar-resizer');
  assert.strictEqual(
    resizer.getAttribute('aria-valuenow'),
    String(DEFAULT_W),
    'default aria-valuenow must be one fifth of the window width',
  );
  assert.strictEqual(
    sidebarW(win),
    '',
    'no inline --sidebar-w until the user resizes (CSS fallback rules)',
  );
  win.close();
  console.log('PASS resize logic seeds its default from a fifth of the window');
}

function testKeyboardBaselineIsTheDefaultWidth() {
  const {win} = makePage({desktopMatches: true});
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

testCssSidebarWidthIsOneFifthOfTheWindow();
testCssDockedRulesRequireAnOpenSidebar();
testCssSettingsPanelStaysNarrow();
testCssChatPanelsNotRestyled();
testCssComposerFullWidth();
testDefaultSeededFromOneFifthOfTheWindow();
testKeyboardBaselineIsTheDefaultWidth();
console.log('All remoteDesktopWidths tests passed.');
