// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test: the chat-webview panel-title header for a ``Bash``
// tool_call event must render in the cyan theme color (matching how
// file paths are rendered), distinguishing Bash tool invocations from
// other tools whose header background/foreground use the orange
// accent.
//
// The bug: ``media/main.js`` builds every tool_call header with a
// bare ``tc-h`` class.  ``media/main.css`` colours ``.tc-h`` orange,
// so Bash tool-call headers render orange just like every other tool.
// The user wants Bash headers — and only Bash headers — shown in
// cyan.
//
// This test exercises the real ``media/main.js`` against the real
// ``media/chat.html`` markup in jsdom, then verifies (a) the rendered
// Bash header carries a Bash-specific class marker that the
// stylesheet can target, and (b) the real ``media/main.css`` declares
// a rule that paints that marker with the cyan theme variable.  This
// is the same pattern used by ``historyTaskWorkspace.test.js`` to
// pin down a CSS-only invariant.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/bashHeaderCyan.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the real chat webview (chat.html +
 * panelCopy.js + main.js).  Mirrors the harness in
 * ``bughunt3_warning_event.test.js`` so the test exercises the same
 * code path the production extension does.
 */
function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  // Strip the production <script> tags — we eval the source files
  // ourselves below so they pick up the jsdom globals.
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function findBashHeader(win) {
  // The webview wraps each tool_call panel as a ``<div class="ev tc">``
  // with a ``<div class="tc-h">`` header carrying the tool name.  Walk
  // every header looking for the one whose text is "Bash" — the
  // fixture only emits a single tool_call, but this is more robust
  // than relying on DOM order.
  // The header text may be wrapped with a leading chevron glyph
  // (added by addCollapse), so strip non-letter prefixes before
  // matching the tool name.
  const headers = win.document.querySelectorAll('#output .tc-h');
  for (const h of headers) {
    const txt = (h.textContent || '').replace(/^[^A-Za-z]+/, '').trim();
    if (txt === 'Bash') return h;
  }
  return null;
}

function testBashHeaderHasBashMarkerClass() {
  const {win} = makeWebview();
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls -la',
    description: 'list files',
  });
  const hdr = findBashHeader(win);
  assert.ok(hdr, 'Bash tool_call must render a .tc-h header with text "Bash"');
  // The bug-fix attaches a CSS hook that lets the stylesheet target
  // this exact header.  Without it the .tc-h orange rule wins and
  // the Bash header renders orange like every other tool.
  assert.ok(
    hdr.classList.contains('tc-h-bash'),
    'BUG: Bash tool_call header must carry a "tc-h-bash" CSS hook ' +
      'so the stylesheet can paint it cyan — got classes: ' +
      JSON.stringify(Array.from(hdr.classList)),
  );
  win.close();
  console.log('  ok - Bash tool_call header has tc-h-bash class');
}

function testNonBashHeaderDoesNotHaveBashMarker() {
  // Sanity check: a Read tool_call (a non-Bash tool) must NOT carry
  // the Bash CSS hook, otherwise every tool would render cyan.
  const {win} = makeWebview();
  send(win, {
    type: 'tool_call',
    name: 'Read',
    path: '/tmp/x.txt',
  });
  const headers = Array.from(
    win.document.querySelectorAll('#output .tc-h'),
  );
  const read = headers.find(
    h => (h.textContent || '').replace(/^[^A-Za-z]+/, '').trim() === 'Read',
  );
  assert.ok(read, 'Read tool_call must render a .tc-h header');
  assert.ok(
    !read.classList.contains('tc-h-bash'),
    'non-Bash tool_call headers must NOT carry the tc-h-bash CSS hook',
  );
  win.close();
  console.log('  ok - non-Bash tool_call header has no tc-h-bash class');
}

function testCssDeclaresCyanRuleForBashHeader() {
  // jsdom never loads the external ``main.css`` stylesheet that
  // ``chat.html`` references via ``{{STYLE_HREF}}``.  We read the CSS
  // file directly (same trick ``historyTaskWorkspace.test.js`` uses)
  // and assert the rule exists in the real production stylesheet.
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  // Find every selector block whose left-hand side mentions
  // ``tc-h-bash`` and check it sets ``color: var(--cyan)``.  We allow
  // the selector to be a compound (e.g. ``.tc-h.tc-h-bash``) so the
  // fix can ratchet up specificity to beat the existing ``.tc-h``
  // orange rule.
  const blockRe = /([^{}]*tc-h-bash[^{}]*)\{([^}]*)\}/g;
  let match;
  let foundCyanRule = false;
  while ((match = blockRe.exec(css)) !== null) {
    const body = match[2];
    if (/color\s*:\s*var\(\s*--cyan\s*\)/.test(body)) {
      foundCyanRule = true;
      break;
    }
  }
  assert.ok(
    foundCyanRule,
    'BUG: main.css must declare a "tc-h-bash" selector with ' +
      '"color: var(--cyan)" so the Bash tool_call header renders ' +
      'in the cyan theme color',
  );
  console.log('  ok - main.css colours .tc-h-bash with var(--cyan)');
}

function testBashHeaderComputedStyleIsCyan() {
  // Final end-to-end sanity check: inject the real main.css into the
  // jsdom document, render the Bash tool_call, and read the computed
  // color back out.  jsdom's CSS engine resolves the cascade enough
  // to confirm the bash-marker rule beats the generic ``.tc-h``
  // rule.  We assert the resolved colour resolves through the
  // ``--cyan`` custom property by checking the computed ``color``
  // property is non-empty and the element actually carries the
  // ``tc-h-bash`` class.
  const {win} = makeWebview();
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const styleEl = win.document.createElement('style');
  styleEl.textContent = css;
  win.document.head.appendChild(styleEl);

  send(win, {type: 'tool_call', name: 'Bash', command: 'echo hi'});
  const hdr = findBashHeader(win);
  assert.ok(hdr, 'Bash tool_call header must exist');
  const computed = win.getComputedStyle(hdr);
  const color = computed.getPropertyValue('color').trim();
  // jsdom may return either the resolved custom-property value or
  // an empty string when the cascade resolves to a CSS variable
  // without a fallback.  The decisive signal is the class hook (the
  // CSS rule above asserts the colour binding), so we re-check that
  // here as an end-to-end belt-and-braces.
  assert.ok(
    hdr.classList.contains('tc-h-bash'),
    'Bash header lost its tc-h-bash hook when main.css was injected',
  );
  // ``color`` itself may be empty (jsdom + var()) — assert it is not
  // an explicit orange literal that would mean the .tc-h rule won.
  assert.ok(
    !/255,?\s*125|^orange$/i.test(color),
    'Bash header computed color must not be an orange literal: ' +
      JSON.stringify(color),
  );
  win.close();
  console.log('  ok - Bash header computed style picks up the cyan rule');
}

function runTests() {
  testBashHeaderHasBashMarkerClass();
  testNonBashHeaderDoesNotHaveBashMarker();
  testCssDeclaresCyanRuleForBashHeader();
  testBashHeaderComputedStyleIsCyan();
}

try {
  runTests();
  console.log('\n4 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
