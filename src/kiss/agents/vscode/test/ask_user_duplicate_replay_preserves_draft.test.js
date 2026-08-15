// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The server re-emits a still-pending askUser question on every session
// replay so a client that connects or reloads mid-question also shows the
// modal.  Already-connected clients receive that duplicate too: it must be
// idempotent — re-initializing the modal would wipe the answer the user is
// typing just because another client reloaded.  A genuinely new question
// (which always follows an askUserDone) must still replace the modal.

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
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function testDuplicateAskPreservesTypedDraft() {
  const {win} = makeWebview();
  const api = win._testApi;
  assert.ok(api, '_testApi must be exposed by main.js');
  const tabId = api.getActiveTabId();

  send(win, {type: 'askUser', question: 'Deploy to production?', tabId});
  const modal = win.document.getElementById('ask-user-modal');
  assert.strictEqual(modal.style.display, 'flex', 'modal must open');
  const input = modal.querySelector('.ask-user-input');
  assert.ok(input, 'ask-user input must be mounted');
  input.value = 'yes, but only eu-west';

  // Another client reloads: its ready pipeline replays the session and the
  // server re-broadcasts the same pending question to every client.
  send(win, {type: 'askUser', question: 'Deploy to production?', tabId});

  assert.strictEqual(
    modal.style.display,
    'flex',
    'modal must stay open after a duplicate replay delivery',
  );
  const inputAfter = modal.querySelector('.ask-user-input');
  assert.strictEqual(
    inputAfter.value,
    'yes, but only eu-west',
    'BUG: a duplicate askUser replay must not wipe the typed draft',
  );

  win.close();
  console.log('  ok - duplicate askUser replay preserves the typed draft');
}

function testNewQuestionAfterDoneReplacesModal() {
  const {win} = makeWebview();
  const api = win._testApi;
  const tabId = api.getActiveTabId();

  send(win, {type: 'askUser', question: 'First question?', tabId});
  const modal = win.document.getElementById('ask-user-modal');
  const input = modal.querySelector('.ask-user-input');
  input.value = 'draft for the first question';
  send(win, {type: 'askUserDone', tabId});
  assert.notStrictEqual(
    modal.style.display,
    'flex',
    'askUserDone must close the modal',
  );

  send(win, {type: 'askUser', question: 'First question?', tabId});
  assert.strictEqual(
    modal.style.display,
    'flex',
    'a repeated question after askUserDone is a NEW question and must show',
  );
  assert.strictEqual(
    modal.querySelector('.ask-user-input').value,
    '',
    'a new question must start with an empty answer box',
  );

  win.close();
  console.log('  ok - identical question after askUserDone shows a fresh modal');
}

function runTests() {
  testDuplicateAskPreservesTypedDraft();
  testNewQuestionAfterDoneReplacesModal();
}

try {
  runTests();
  console.log('\n2 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
