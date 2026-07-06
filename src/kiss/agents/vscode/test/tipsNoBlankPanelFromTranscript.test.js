// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test: transcript text must never instantiate a Tips panel.
//
// Bug locked in:
//
//   After ``./scripts/build-extension.sh`` the reloaded chat webview
//   showed TWO Tips windows — one blank, one with the real tips.  The
//   blank one came from the transcript: a chat message (e.g. a task
//   result summary describing the tips feature) contained the literal
//   text ``<kiss-tips-panel>``.  ``kissSanitize`` in ``media/main.js``
//   only stripped a blocklist of dangerous tags (script, iframe, ...),
//   so the tag survived ``marked.parse`` → ``innerHTML``, and because
//   ``media/tips.js`` registers ``<kiss-tips-panel>`` as a custom
//   element the browser upgraded it into a REAL panel: its constructor
//   builds the header/footer chrome but no tips were ever assigned, so
//   a blank full-viewport overlay appeared on top of (or under) the
//   legitimate auto-shown Tips window.
//
// Contract locked in here:
//
//   * Rendering a chat event whose markdown contains a raw
//     ``<kiss-tips-panel>`` tag must NOT mount a second panel — the
//     sanitizer strips every custom element (any hyphenated tag).
//   * A ``<kiss-tips-panel>`` mentioned inside a code span stays
//     visible as literal text in the transcript.
//   * Defense in depth: a ``<kiss-tips-panel>`` element upgraded from
//     parsed HTML (bypassing the sanitizer) self-removes on connect —
//     only ``showTipsPanel()``, which assigns ``.tips`` before
//     mounting, may show a panel.  An empty-tips panel keeps
//     working because ``showTipsPanel([])`` still assigns ``.tips``.
//
// This test drives the real ``media/main.js``, ``media/tips.js`` and
// ``media/marked.min.js`` (plus the real ``media/chat.html`` markup
// and ``media/panelCopy.js``) inside jsdom, exactly like
// ``bughunt3_warning_event.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/tipsNoBlankPanelFromTranscript.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const TIPS = ['## First tip\n\nUse **KISS Sorcar** like a pro.'];

/**
 * Build a jsdom window running the production chat webview: the real
 * ``chat.html`` body (placeholders blanked), ``marked.min.js``,
 * ``tips.js`` (with ``window.__TIPS__`` injected first, exactly like
 * the inline script emitted by ``buildChatHtml``), ``panelCopy.js``
 * and ``main.js`` evaluated in the window, and a recording
 * ``acquireVsCodeApi`` stub (the only host API the webview has).
 */
function makeWebview({show}) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
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

  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(`window.__TIPS__ = ${JSON.stringify({tips: TIPS, show})};`);
  win.eval(fs.readFileSync(path.join(MEDIA, 'tips.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** All mounted ``<kiss-tips-panel>`` elements anywhere in the page. */
function panels(win) {
  return Array.from(win.document.querySelectorAll('kiss-tips-panel'));
}

/** The rendered tip body text of a mounted panel (shadow DOM). */
function panelBodyText(panel) {
  const body = panel.shadowRoot && panel.shadowRoot.querySelector('.tips-body');
  return body ? body.textContent.trim() : '';
}

function testResultSummaryCannotSpawnBlankPanel() {
  const {win} = makeWebview({show: true});

  assert.strictEqual(
    panels(win).length,
    1,
    'fresh install: exactly one auto-shown tips panel',
  );
  assert.ok(
    panelBodyText(panels(win)[0]).includes('Use KISS Sorcar like a pro.'),
    'the auto-shown panel renders the tip contents',
  );

  // A task result whose summary mentions the tag raw (not in a code
  // span) — exactly what a session summary describing the tips
  // feature looks like in the transcript.
  send(win, {
    type: 'result',
    summary: 'Read media/tips.js fully: <kiss-tips-panel>; auto-show wired.',
    total_tokens: 10,
    cost: '$0.01',
  });

  const after = panels(win);
  assert.strictEqual(
    after.length,
    1,
    'BUG: a result summary containing a raw <kiss-tips-panel> tag ' +
      'mounted a second (blank) Tips window',
  );
  assert.ok(
    panelBodyText(after[0]).includes('Use KISS Sorcar like a pro.'),
    'the surviving panel must be the real tips window, not the blank one',
  );
  win.close();
  console.log('  ok - result summary cannot spawn a blank tips panel');
}

function testPromptTextCannotSpawnBlankPanel() {
  const {win} = makeWebview({show: false});

  send(win, {
    type: 'prompt',
    text: 'why does <kiss-tips-panel> show up blank after a rebuild?',
  });

  assert.strictEqual(
    panels(win).length,
    0,
    'BUG: a prompt containing a raw <kiss-tips-panel> tag mounted a ' +
      'blank Tips window',
  );
  win.close();
  console.log('  ok - prompt text cannot spawn a blank tips panel');
}

function testCodeSpanMentionStaysVisibleText() {
  const {win} = makeWebview({show: false});

  send(win, {
    type: 'result',
    summary: 'The Tips window is the `<kiss-tips-panel>` web component.',
    total_tokens: 10,
    cost: '$0.01',
  });

  assert.strictEqual(panels(win).length, 0, 'no panel from a code span');
  const output = win.document.getElementById('output');
  assert.ok(
    output.textContent.includes('<kiss-tips-panel>'),
    'a code-span mention must stay visible as literal text',
  );
  win.close();
  console.log('  ok - code-span mention stays visible literal text');
}

function testParsedPanelSelfRemovesButProgrammaticPanelSurvives() {
  const {win} = makeWebview({show: false});

  // Defense in depth: HTML injected through any path that bypasses the
  // sanitizer must still not produce a live panel.
  const div = win.document.createElement('div');
  div.innerHTML = '<kiss-tips-panel></kiss-tips-panel>';
  win.document.body.appendChild(div);
  assert.strictEqual(
    panels(win).length,
    0,
    'BUG: a <kiss-tips-panel> upgraded from parsed HTML (no tips ever ' +
      'assigned) must self-remove instead of covering the chat as a ' +
      'blank overlay',
  );

  // The legit programmatic path still works — including an
  // empty-tips panel (``showTipsPanel([])`` assigns ``.tips``).
  const empty = win.__kissShowTipsPanel([]);
  assert.strictEqual(panels(win).length, 1, 'empty panel still mounts');
  empty.shadowRoot.querySelector('.tips-close').click();
  assert.strictEqual(panels(win).length, 0, 'empty panel closes');

  win.__kissShowTipsPanel(TIPS);
  assert.strictEqual(panels(win).length, 1, 'real panel still mounts');
  assert.ok(panelBodyText(panels(win)[0]).includes('Use KISS Sorcar'));
  win.close();
  console.log('  ok - parsed panel self-removes; programmatic panels work');
}

testResultSummaryCannotSpawnBlankPanel();
testPromptTextCannotSpawnBlankPanel();
testCodeSpanMentionStaysVisibleText();
testParsedPanelSelfRemovesButProgrammaticPanelSurvives();
console.log('tipsNoBlankPanelFromTranscript: all tests passed');
