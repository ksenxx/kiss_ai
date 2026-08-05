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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

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

  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');

  posted.length = 0;
  const inp = typeInto(win, '@');

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

  const PICKED = 'src/foo.py';
  send(win, {
    type: 'files',
    prefix: '',
    files: [
      {type: 'file', text: PICKED},
      {type: 'file', text: 'src/bar.py'},
    ],
  });

  const ac = win.document.getElementById('autocomplete');
  assert.strictEqual(
    ac.style.display,
    'block',
    'autocomplete must be visible after files reply',
  );
  const items = ac.querySelectorAll('.ac-item');
  assert.ok(items.length >= 2, 'autocomplete must list the returned files');

  let target = null;
  items.forEach(it => {
    if (it.dataset.text === PICKED) target = it;
  });
  assert.ok(target, 'autocomplete must contain the picked file entry');
  target.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));

  assert.ok(
    !inp.value.includes('PWD/'),
    'BUG: file selector must not prepend "PWD/" to the picked file path ' +
      '(input value was: ' +
      JSON.stringify(inp.value) +
      '). This regression came back after the prior fix removed PWD/ ' +
      'from drag-and-drop and editor-selection paths but missed the ' +
      '@-mention insertAtMention() path.',
  );

  assert.ok(
    inp.value.startsWith('./' + PICKED),
    'file selector must insert the picked file as "./<path>" (got: ' +
      JSON.stringify(inp.value) +
      ')',
  );

  const rec = posted.find(m => m.type === 'recordFileUsage');
  assert.ok(rec, 'clicking an autocomplete entry must post recordFileUsage');
  assert.strictEqual(
    rec.path,
    PICKED,
    'recordFileUsage.path must be the raw picked path (no PWD/ prefix)',
  );

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
