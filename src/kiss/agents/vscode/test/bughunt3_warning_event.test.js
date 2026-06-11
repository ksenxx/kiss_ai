// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Bug-hunt 3 integration test: backend ``warning`` events must be
// rendered by the chat webview.
//
// Bug locked in:
//
//   ``WorktreeSorcarAgent._flush_warnings`` (worktree_sorcar_agent.py)
//   broadcasts ``{"type": "warning", "message": ...}`` for stash-pop
//   failures and merge conflicts — warnings the user MUST see because
//   their uncommitted changes may be sitting in a git stash.  The
//   extension's SorcarSidebarView forwards every backend event to the
//   webview verbatim, but ``media/main.js`` has no ``case 'warning'``
//   handler (and ``src/types.ts`` lacks the event type), so the
//   warning is silently dropped: no banner, nothing in the transcript.
//
// This test drives the real ``media/main.js`` (plus the real
// ``media/chat.html`` markup and ``media/panelCopy.js``) inside jsdom,
// exactly like ``bughunt2_status_timer.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/bughunt3_warning_event.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview: the real
 * ``chat.html`` body (placeholders blanked), ``panelCopy.js`` and
 * ``main.js`` evaluated in the window, and a recording
 * ``acquireVsCodeApi`` stub (the only host API the webview has).
 */
function makeWebview() {
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

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function testWarningEventIsRendered() {
  const {win} = makeWebview();

  const msg =
    'Failed to restore your uncommitted changes: `git stash pop` ' +
    'conflicted. Your changes are preserved in the git stash.';
  send(win, {type: 'warning', message: msg});

  const output = win.document.getElementById('output');
  const text = output ? output.textContent : '';
  assert.ok(
    text.includes('git stash pop'),
    'BUG: a backend warning event (stash-pop failure) was silently ' +
      'dropped by the webview — the user never learns their ' +
      'uncommitted changes are stuck in the stash',
  );
  win.close();
  console.log('  ok - warning event renders in the transcript');
}

function testWarningEscapesHtml() {
  const {win} = makeWebview();

  send(win, {
    type: 'warning',
    message: 'conflict in <img src=x onerror=alert(1)> & "branch"',
  });

  const output = win.document.getElementById('output');
  assert.ok(output, 'output container must exist');
  assert.strictEqual(
    output.querySelector('img'),
    null,
    'BUG: warning message HTML was not escaped (XSS)',
  );
  assert.ok(
    output.textContent.includes('<img src=x onerror=alert(1)> & "branch"'),
    'warning text must be shown verbatim (escaped)',
  );
  win.close();
  console.log('  ok - warning message is HTML-escaped');
}

function testForeignTabWarningNotRendered() {
  const {win} = makeWebview();

  send(win, {
    type: 'warning',
    message: 'foreign-window stash warning',
    tabId: 'some-other-window-tab',
  });

  const output = win.document.getElementById('output');
  const text = output ? output.textContent : '';
  assert.ok(
    !text.includes('foreign-window stash warning'),
    'a warning stamped for another tab must not render in this tab',
  );
  win.close();
  console.log('  ok - foreign-tab warning is not rendered');
}

function runTests() {
  testWarningEventIsRendered();
  testWarningEscapesHtml();
  testForeignTabWarningNotRendered();
}

try {
  runTests();
  console.log('\n3 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
