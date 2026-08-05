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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

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
