// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The welcome page: the KISS Sorcar logo (transparent art, a light-theme
// and a dark-theme colouring), the greeting and the tagline; no
// suggested-prompt chips and no remote URL / password block.  Checks
// the page markup (chat.html + main.js in JSDOM): both logos are
// present, the #suggestions and #welcome-config containers are gone,
// opening a new chat asks the daemon only for the welcome info, and the
// highlight.js sheet follows the page theme (light sheet with
// body.light-theme, swapped live by the theme toggle).  The served
// HTML's logo URLs and content hashes are checked by
// src/kiss/tests/webapp/test_welcome_logo_html.py.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {makePage} = require('./simplify2_harness.js');

const MEDIA = path.join(__dirname, '..', 'media');

function testMarkupHasLogosAndNoSuggestions() {
  const {win} = makePage();
  const doc = win.document;
  for (const [id, file] of [
    ['welcome-logo', 'welcome-logo.png'],
    ['welcome-logo-dark', 'welcome-logo-dark.png'],
  ]) {
    const logo = doc.querySelector(`#welcome img#${id}.welcome-logo`);
    assert.ok(logo, `the welcome page must show the ${id} image`);
    assert.strictEqual(
      logo.getAttribute('alt'),
      '',
      'the logo is decorative: the heading next to it names the product',
    );
    const bytes = fs.readFileSync(path.join(MEDIA, file));
    assert.strictEqual(
      bytes.subarray(1, 4).toString('latin1'),
      'PNG',
      `media/${file} must be a PNG image`,
    );
    // Transparent art: an RGBA PNG (colour type 6 in the IHDR chunk),
    // so the logo sits directly on the theme background.
    assert.strictEqual(bytes[25], 6, `media/${file} must be an RGBA PNG`);
  }
  assert.ok(
    doc.querySelector('#welcome h2').textContent.includes('Welcome to'),
    'the greeting heading stays under the logo',
  );
  assert.strictEqual(doc.getElementById('suggestions'), null);
  assert.strictEqual(
    doc.getElementById('welcome-config'),
    null,
    'the remote URL / password block is not on the welcome page',
  );
  win.close();
  console.log('  ok - the page renders both logos and no config block');
}

function testNewChatRequestsWelcomeInfoOnly() {
  const {win, posted} = makePage();
  const welcome = win.document.getElementById('welcome');
  assert.ok(welcome.querySelector('#welcome-logo'), 'logo is on the page');

  // "+" keeps a chat that is still new; the daemon's task-text echo
  // makes the boot chat "used" so "+" opens a new one.
  win.dispatchEvent(
    new win.MessageEvent('message', {
      data: {type: 'setTaskText', text: 'first task'},
    }),
  );
  posted.length = 0;
  win.document
    .getElementById('new-chat-btn')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const types = posted.map(m => m.type);
  assert.ok(types.includes('newChat'), `new chat posts newChat: ${types}`);
  assert.ok(
    types.includes('getWelcomeInfo'),
    `new chat asks for the welcome info (remote URL): ${types}`,
  );
  assert.ok(!types.includes('getWelcomeSuggestions'), `${types}`);

  // A daemon that still speaks the old protocol must not bring the
  // chips back.
  win.dispatchEvent(
    new win.MessageEvent('message', {
      data: {type: 'welcome_suggestions', suggestions: [{text: 'old chip'}]},
    }),
  );
  const shown = win.document.getElementById('welcome');
  assert.ok(!shown.textContent.includes('old chip'));
  assert.ok(!shown.textContent.includes('Suggested prompt'));
  assert.strictEqual(win.document.querySelector('.suggestion-chip'), null);
  assert.ok(shown.querySelector('#welcome-logo'), 'logo survives a new chat');
  win.close();
  console.log('  ok - new chat requests getWelcomeInfo and shows no chips');
}

// The highlight sheet follows the page theme: the saved theme selects
// the sheet at start-up and the theme toggle swaps it live, both ways.
function testHighlightSheetFollowsTheTheme() {
  const hljs = {dark: '/m/dark.css', light: '/m/light.css'};
  const {win} = makePage({
    beforeScripts: w => {
      w.__HLJS_THEME_CSS__ = hljs;
      w.localStorage.setItem('kissRemoteTheme', 'light');
    },
  });
  const link = win.document.getElementById('hljs-theme');
  assert.ok(link, 'chat.html links the highlight sheet as #hljs-theme');
  assert.ok(
    win.document.body.classList.contains('light-theme'),
    'the saved light theme is applied at start',
  );
  assert.strictEqual(link.getAttribute('href'), '/m/light.css', 'light at start');
  const btn = win.document.getElementById('theme-btn');
  assert.ok(btn, 'the theme toggle exists');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(!win.document.body.classList.contains('light-theme'));
  assert.strictEqual(link.getAttribute('href'), '/m/dark.css', 'dark after toggle');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(win.document.body.classList.contains('light-theme'));
  assert.strictEqual(link.getAttribute('href'), '/m/light.css', 'light again');
  win.close();
  console.log('  ok - the highlight sheet follows the page theme');
}

testMarkupHasLogosAndNoSuggestions();
testNewChatRequestsWelcomeInfoOnly();
testHighlightSheetFollowsTheTheme();
