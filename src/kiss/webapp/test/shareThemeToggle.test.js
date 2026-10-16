// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end tests for the shared chat page's light/dark mode toggle.
//
// _build_share_page (src/kiss/server/web_server.py) writes a page with
// a floating #share-theme-btn, two inlined highlight.js style elements
// (#hljs-style-dark active, #hljs-style-light disabled via
// media="not all") and an html.light-theme variable block; share.js
// wires the button: it toggles the `light-theme` class on <html>,
// flips the active hljs stylesheet, relabels itself ("Switch to
// light mode" / "Switch to dark mode") and persists the choice in
// localStorage.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const SHARE_JS = fs.readFileSync(path.join(MEDIA, 'share.js'), 'utf8');

/**
 * Build the exact document shape _build_share_page writes (theme
 * button + hljs style pair + #app) and run share.js in it.
 *
 * @param {string} [initialTheme] Pre-seeded localStorage theme.
 * @returns {Window} The page's window.
 */
function makeSharePage(initialTheme) {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><head>' +
      '<style id="hljs-style-dark">.hljs{color:#fff}</style>' +
      '<style id="hljs-style-light" media="not all">.hljs{color:#000}' +
      '</style></head><body>' +
      '<button id="share-theme-btn" type="button" ' +
      'title="Switch to light mode" ' +
      'aria-label="Switch to light mode"></button>' +
      '<div id="app"><div id="output"></div></div>' +
      '</body></html>',
    {runScripts: 'dangerously', pretendToBeVisual: true, url: 'https://x/'},
  );
  const win = dom.window;
  if (initialTheme) {
    win.localStorage.setItem('kissShareTheme', initialTheme);
  }
  win.eval(SHARE_JS + '\n//# sourceURL=share.js');
  return win;
}

function click(el) {
  el.dispatchEvent(
    new el.ownerDocument.defaultView.MouseEvent('click', {
      bubbles: true,
      cancelable: true,
    }),
  );
}

function assertDark(win) {
  const doc = win.document;
  assert.ok(
    !doc.documentElement.classList.contains('light-theme'),
    'dark mode: <html> must not carry light-theme',
  );
  assert.strictEqual(
    doc.getElementById('hljs-style-dark').getAttribute('media'),
    'all',
  );
  assert.strictEqual(
    doc.getElementById('hljs-style-light').getAttribute('media'),
    'not all',
  );
  const btn = doc.getElementById('share-theme-btn');
  assert.strictEqual(btn.getAttribute('aria-label'), 'Switch to light mode');
  assert.strictEqual(btn.title, 'Switch to light mode');
  assert.ok(btn.querySelector('svg'), 'button shows an icon');
}

function assertLight(win) {
  const doc = win.document;
  assert.ok(
    doc.documentElement.classList.contains('light-theme'),
    'light mode: <html> must carry light-theme',
  );
  assert.strictEqual(
    doc.getElementById('hljs-style-dark').getAttribute('media'),
    'not all',
  );
  assert.strictEqual(
    doc.getElementById('hljs-style-light').getAttribute('media'),
    'all',
  );
  const btn = doc.getElementById('share-theme-btn');
  assert.strictEqual(btn.getAttribute('aria-label'), 'Switch to dark mode');
  assert.strictEqual(btn.title, 'Switch to dark mode');
  assert.ok(btn.querySelector('svg'), 'button shows an icon');
}

function testDefaultsToDark() {
  const win = makeSharePage();
  assertDark(win);
  win.close();
}

function testClickSwitchesToLightAndBack() {
  const win = makeSharePage();
  const btn = win.document.getElementById('share-theme-btn');
  click(btn);
  assertLight(win);
  assert.strictEqual(win.localStorage.getItem('kissShareTheme'), 'light');
  click(btn);
  assertDark(win);
  assert.strictEqual(win.localStorage.getItem('kissShareTheme'), 'dark');
  win.close();
}

function testSavedLightThemeIsRestoredOnLoad() {
  const win = makeSharePage('light');
  assertLight(win);
  win.close();
}

function testSavedBogusThemeFallsBackToDark() {
  const win = makeSharePage('sepia');
  assertDark(win);
  win.close();
}

function testPageWithoutThemeChromeStillLoads() {
  // A legacy shared page (written before the toggle existed) carries
  // neither the button nor the style pair; share.js must still load
  // and keep its collapse/expand behaviour working.
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body><div id="app"><div id="output">' +
      '<div class="ev tc collapsible">' +
      '<div class="tc-h collapse-header">Bash' +
      '<span class="collapse-preview"></span></div>' +
      '<pre>ls</pre></div>' +
      '</div></div></body></html>',
    {runScripts: 'dangerously', pretendToBeVisual: true, url: 'https://x/'},
  );
  const win = dom.window;
  win.eval(SHARE_JS + '\n//# sourceURL=share.js');
  const panel = win.document.querySelector('.collapsible');
  click(win.document.querySelector('.collapse-header'));
  assert.ok(
    panel.classList.contains('collapsed'),
    'collapse still works without the theme chrome',
  );
  win.close();
}

function main() {
  testDefaultsToDark();
  console.log('  ok - shared page defaults to dark mode');
  testClickSwitchesToLightAndBack();
  console.log('  ok - toggle switches light/dark and persists the choice');
  testSavedLightThemeIsRestoredOnLoad();
  console.log('  ok - saved light theme is restored on load');
  testSavedBogusThemeFallsBackToDark();
  console.log('  ok - unknown saved value falls back to dark');
  testPageWithoutThemeChromeStillLoads();
  console.log('  ok - legacy page without theme chrome still works');
  console.log('shareThemeToggle.test.js: all tests passed');
}

main();
