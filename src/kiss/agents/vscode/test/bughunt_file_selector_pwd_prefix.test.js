// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test: the REAL ``media/main.js`` webview, driven inside
// jsdom, must NOT prepend ``PWD/`` when the user picks a file from the
// ``@``-mention file selector autocomplete that appears in the task
// input box.
//
// CONTEXT (from the task): a previous fix (commit 1f8945d3, "use
// relative path notation instead of PWD prefix") replaced ``PWD/`` with
// ``./`` for drag-and-drop file insertion and for the editor-selection
// "append to input" path.  But the ``@``-mention file selector path —
// ``insertAtMention()`` — was missed, so files chosen from the @-picker
// are still inserted as ``PWD/<path>`` instead of ``./<path>``.
//
// This test drives the production ``chat.html`` + ``panelCopy.js`` +
// ``main.js`` (no mocks of project code) through the entire @-mention
// flow:
//   1. user types ``@`` into the task input,
//   2. webview posts ``getFiles`` to the backend,
//   3. backend replies with a ``files`` event listing matches,
//   4. user clicks the autocomplete entry for ``src/foo.py``.
//
// REQUIREMENT: after the click, the task input must contain
// ``./src/foo.py`` (relative-path notation, consistent with the prior
// drag-and-drop fix), NOT ``PWD/src/foo.py``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/bughunt_file_selector_pwd_prefix.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview, capturing
 * every message the webview posts back to the (mock) extension host.
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
  let persisted;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => persisted,
      setState: s => {
        persisted = s;
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

/**
 * Type ``text`` into the task input at the current caret position,
 * firing the ``input`` event the webview listens for to open the
 * autocomplete.  Returns the input element.
 */
function typeInto(win, text) {
  const inp = win.document.getElementById('task-input');
  const start = inp.selectionStart || 0;
  const before = inp.value.substring(0, start);
  const after = inp.value.substring(start);
  inp.value = before + text + after;
  const np = start + text.length;
  inp.setSelectionRange(np, np);
  inp.focus();
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  return inp;
}

async function runTests() {
  const wv = makeWebview();
  const {win, posted} = wv;

  // Wait for the webview's ``ready`` so the tab is fully initialized.
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');

  // ---- 1. user types ``@`` in the task input -------------------------
  posted.length = 0;
  const inp = typeInto(win, '@');

  // The webview should ask the extension for matching files.
  const getFiles = posted.find(m => m.type === 'getFiles');
  assert.ok(
    getFiles,
    'typing "@" must post a getFiles request (was: ' +
      JSON.stringify(posted.map(m => m.type)) +
      ')',
  );
  assert.strictEqual(
    getFiles.prefix,
    '',
    'getFiles prefix must be the @-mention query (empty here)',
  );

  // ---- 2. backend replies with a ``files`` event ---------------------
  const PICKED = 'src/foo.py';
  send(win, {
    type: 'files',
    prefix: '',
    files: [
      {type: 'file', text: PICKED},
      {type: 'file', text: 'src/bar.py'},
    ],
  });

  // The autocomplete dropdown should now be visible with the entries.
  const ac = win.document.getElementById('autocomplete');
  assert.strictEqual(
    ac.style.display,
    'block',
    'autocomplete must be visible after files reply',
  );
  const items = ac.querySelectorAll('.ac-item');
  assert.ok(items.length >= 2, 'autocomplete must list the returned files');

  // ---- 3. user clicks the first entry --------------------------------
  let target = null;
  items.forEach(it => {
    if (it.dataset.text === PICKED) target = it;
  });
  assert.ok(target, 'autocomplete must contain the picked file entry');
  target.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  // ---- 4. the task input must NOT have a ``PWD/`` prefix -------------
  assert.ok(
    !inp.value.includes('PWD/'),
    'BUG: file selector must not prepend "PWD/" to the picked file path ' +
      '(input value was: ' +
      JSON.stringify(inp.value) +
      '). This regression came back after the prior fix removed PWD/ ' +
      'from drag-and-drop and editor-selection paths but missed the ' +
      '@-mention insertAtMention() path.',
  );

  // And it must use the agreed relative-path notation ``./<file>``.
  assert.ok(
    inp.value.startsWith('./' + PICKED),
    'file selector must insert the picked file as "./<path>" (got: ' +
      JSON.stringify(inp.value) +
      ')',
  );

  // ---- 5. also verify the recordFileUsage message carries the raw ----
  // ----    file path (no PWD/ prefix), matching backend expectations.
  const rec = posted.find(m => m.type === 'recordFileUsage');
  assert.ok(rec, 'clicking an autocomplete entry must post recordFileUsage');
  assert.strictEqual(
    rec.path,
    PICKED,
    'recordFileUsage.path must be the raw picked path (no PWD/ prefix)',
  );

  // ---- 6. picking a file when the @-mention has a non-empty query ----
  // ----    and there is trailing text must still avoid the PWD prefix.
  inp.value = 'please open @sr';
  inp.setSelectionRange(inp.value.length, inp.value.length);
  inp.focus();
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));

  send(win, {
    type: 'files',
    prefix: 'sr',
    files: [{type: 'file', text: 'src/baz.py'}],
  });
  const ac2 = win.document.getElementById('autocomplete');
  const items2 = ac2.querySelectorAll('.ac-item');
  assert.ok(items2.length >= 1, 'autocomplete must populate for "sr" prefix');
  items2[0].dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  assert.ok(
    !inp.value.includes('PWD/'),
    'BUG: file selector must not prepend "PWD/" when picking a file ' +
      'with a non-empty @-mention query (input value was: ' +
      JSON.stringify(inp.value) +
      ')',
  );
  assert.ok(
    inp.value.includes('./src/baz.py'),
    'file selector must insert "./<path>" into the mid-line @-mention ' +
      '(got: ' +
      JSON.stringify(inp.value) +
      ')',
  );

  win.close();
}

runTests().then(
  () => {
    console.log('\nAll tests passed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    process.exit(1);
  },
);
