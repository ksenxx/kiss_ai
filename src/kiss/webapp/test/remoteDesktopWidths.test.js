// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage: makeChatPage, stubDesktop, send, byId} = require('./simplify2_harness');
const {makeResizePage: makePage, panelW} = require('./panel_resize_helpers');
const {installTokenisedCss} = require('./designTokens');

function sidebarW(win) {
  return panelW(win, '--sidebar-w');
}

// jsdom reports window.innerWidth = 1024; the default docked width is
// one fifth of the window (the 20vw fallback of --sidebar-default-w),
// so 205px.  Drags, clamping and persistence of the width are covered
// by remoteSidebarResize.test.js.
const MIN_W = 10;
const DEFAULT_W = Math.round(1024 * 0.2);

// The CSS checks read the computed styles of live elements.  jsdom
// matches no width feature of an @media prelude, so each layout's width
// blocks are applied by name; it resolves no var(), min() or vw either,
// so such a value comes back as the cascade picked it (design tokens
// inlined).
const DESKTOP = {name: 'desktop', desktop: true, media: /\(width >= 900px\)/};
const PHONE = {
  name: 'phone',
  desktop: false,
  media: /\(width < 900px\)|\(width <= 600px\)/,
};
const WIDTH_FALLBACK = 'var(--sidebar-w, var(--sidebar-default-w))';

// One of each kind of transcript panel the chat thread renders.
const TRANSCRIPT = [
  {type: 'prompt', text: 'go see src/a.py'},
  {type: 'tool_call', name: 'Read', path: '/tmp/a.txt'},
  {type: 'tool_result', name: 'Read', content: 'data'},
  {type: 'thinking_start'},
  {type: 'thinking_delta', text: 'hmm'},
  {type: 'thinking_end'},
  {type: 'text_delta', text: '# Title\n\nSome **bold** `code`'},
  {type: 'text_end'},
  {type: 'tool_call', name: 'Bash', command: 'ls', description: 'list'},
  {type: 'tool_result', name: 'Bash', content: 'a\nb', is_error: true},
  {type: 'tool_call', name: 'summary', description: '- one\n- two'},
  {type: 'result', text: 'done', success: true},
];

/**
 * A remote chat page in main.js's desktop mode (sidebar docked open) or
 * phone mode, styled by main.css plus remote-codex.css (main.css alone
 * with `mainCssOnly`) with the width blocks `media` matches applied
 * (none for a null `media`).
 */
function styledPage(desktop, media, mainCssOnly = false) {
  const {win} = makeChatPage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js'],
    beforeScripts: w => stubDesktop(w, desktop),
  });
  const sheets = mainCssOnly ? ['main.css'] : ['main.css', 'remote-codex.css'];
  for (const name of sheets) installTokenisedCss(win, name, {applyMedia: media});
  return win;
}

function computed(win, id) {
  return win.getComputedStyle(byId(win, id));
}

/** Every property of *el*'s computed style, as `{property: value}`. */
function styleOf(win, el) {
  const cs = win.getComputedStyle(el);
  const style = {};
  for (let i = 0; i < cs.length; i++) {
    style[cs.item(i)] = cs.getPropertyValue(cs.item(i));
  }
  return style;
}

/** The computed style of #output and of every element inside it. */
function transcriptStyles(win) {
  const output = byId(win, 'output');
  return [output, ...output.querySelectorAll('*')].map(el => ({
    el: el.tagName.toLowerCase() + '.' + (el.getAttribute('class') || ''),
    style: styleOf(win, el),
  }));
}

function testCssSidebarWidthIsOneFifthOfTheWindow() {
  const win = styledPage(true, DESKTOP.media);
  const vars = win.getComputedStyle(win.document.body);
  assert.strictEqual(
    vars.getPropertyValue('--sidebar-min-w').trim(),
    `${MIN_W}px`,
    `the docked panel resize floor must be the ${MIN_W}px sliver`,
  );
  assert.strictEqual(
    vars.getPropertyValue('--sidebar-default-w').trim(),
    '20vw',
    'the default width must be one fifth of the browser window (20vw)',
  );
  assert.ok(
    byId(win, 'sidebar').classList.contains('open'),
    'the desktop docks the panel open',
  );
  const sidebar = computed(win, 'sidebar');
  for (const padding of [sidebar.paddingLeft, sidebar.paddingRight]) {
    assert.match(
      padding,
      /^min\(16px,.*\/\s*4\)/,
      "the docked panel's horizontal padding must collapse with the " +
        'panel so a 10px drag really renders 10px (border-box padding ' +
        `would otherwise floor it at 32px) — got: ${padding}`,
    );
  }
  assert.strictEqual(
    sidebar.width,
    WIDTH_FALLBACK,
    'the docked sidebar width must fall back to --sidebar-default-w',
  );
  assert.strictEqual(
    computed(win, 'app').marginLeft,
    WIDTH_FALLBACK,
    '#app margin must be driven by the SAME width fallback',
  );
  win.close();
  console.log('PASS CSS sizes the docked sidebar to a fifth of the window');
}

function testCssDockedRulesRequireAnOpenSidebar() {
  // The burger button only toggles #sidebar.open, so every docked rule
  // must depend on that class — otherwise the panel stays glued to the
  // screen and the chat never reclaims its width.  The closed panel
  // computes exactly as it does with the desktop blocks left out.
  const docked = styledPage(true, DESKTOP.media);
  const undocked = styledPage(true, null);
  for (const win of [docked, undocked]) {
    byId(win, 'sidebar').classList.remove('open');
  }
  assert.strictEqual(
    computed(docked, 'app').marginLeft,
    '0px',
    'with the panel hidden #app must have no left offset',
  );
  assert.deepStrictEqual(
    styleOf(docked, byId(docked, 'sidebar')),
    styleOf(undocked, byId(undocked, 'sidebar')),
    'docked #sidebar rules must be scoped to #sidebar.open',
  );
  docked.close();
  undocked.close();
  console.log('PASS CSS docks the sidebar only while it is open');
}

function testCssSettingsPanelStaysNarrow() {
  const win = styledPage(true, DESKTOP.media);
  assert.strictEqual(
    computed(win, 'settings-panel').width,
    'min(90vw, var(--settings-panel-w))',
    'the settings drawer must stay a narrow fixed-width column',
  );
  assert.match(
    win.getComputedStyle(win.document.body).getPropertyValue('--settings-panel-w'),
    /^\s*\d+px\s*$/,
    'the narrow settings width must be a fixed pixel bound',
  );
  win.close();
  console.log('PASS CSS keeps the desktop settings panel narrow');
}

function testCssChatPanelsNotRestyled() {
  // The chat thread is styled by main.css alone: adding remote-codex.css
  // leaves every element of a rendered transcript computing the same
  // style, on either layout.
  for (const layout of [DESKTOP, PHONE]) {
    const plain = styledPage(layout.desktop, layout.media, true);
    const remote = styledPage(layout.desktop, layout.media);
    for (const win of [plain, remote]) {
      for (const ev of TRANSCRIPT) send(win, ev);
    }
    const expected = transcriptStyles(plain);
    assert.ok(
      plain.document.querySelector('#output .tc.tc-summary'),
      'the transcript must render its panels',
    );
    assert.deepStrictEqual(
      transcriptStyles(remote),
      expected,
      `remote-codex.css must not restyle the chat thread (${layout.name})`,
    );
    plain.close();
    remote.close();
  }
  console.log('PASS CSS chat panels keep the main.css look');
}

function testCssComposerFullWidth() {
  for (const layout of [DESKTOP, PHONE]) {
    const win = styledPage(layout.desktop, layout.media);
    const box = computed(win, 'input-container');
    assert.strictEqual(
      box.maxWidth,
      'none',
      'the composer must have NO width cap so it is as wide as the ' +
        `chat page (${layout.name})`,
    );
    assert.strictEqual(box.width, 'auto', 'the composer has no fixed width');
    assert.strictEqual(
      box.marginLeft + ' ' + box.marginRight,
      '0px 0px',
      `the composer must not be a centered narrow column (${layout.name})`,
    );
    win.close();
  }
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
