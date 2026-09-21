// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The server re-emits a still-pending askUser question on every session
// replay so a client that connects or reloads mid-question also shows the
// composer.  Already-connected clients receive that duplicate too: it must be
// idempotent — resetting the answer mode would wipe the answer the user is
// typing just because another client reloaded.  A genuinely new question
// (which always follows an askUserDone) must still put the composer back
// into answer mode.

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

function answering(win) {
  return win.document.body.classList.contains('ask-answering');
}

function testDuplicateAskPreservesTypedDraft() {
  const {win} = makeWebview();
  const api = win._testApi;
  assert.ok(api, '_testApi must be exposed by main.js');
  const tabId = api.getActiveTabId();

  send(win, {type: 'askUser', question: 'Deploy to production?', tabId});
  assert.ok(answering(win), 'the composer must enter answer mode');
  const input = win.document.getElementById('task-input');
  input.value = 'yes, but only eu-west';

  // Another client reloads: its ready pipeline replays the session and the
  // server re-broadcasts the same pending question to every client.
  send(win, {type: 'askUser', question: 'Deploy to production?', tabId});

  assert.ok(
    answering(win),
    'the composer must stay in answer mode after a duplicate replay delivery',
  );
  assert.strictEqual(
    input.value,
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
  const input = win.document.getElementById('task-input');
  input.value = 'draft for the first question';
  send(win, {type: 'askUserDone', tabId});
  assert.ok(
    !answering(win),
    'askUserDone must take the composer out of answer mode',
  );
  assert.strictEqual(
    input.value,
    'draft for the first question',
    'text the user typed is theirs: another client answering must not wipe it',
  );

  send(win, {type: 'askUser', question: 'First question?', tabId});
  assert.ok(
    answering(win),
    'a repeated question after askUserDone is a NEW question and must show',
  );

  win.close();
  console.log('  ok - identical question after askUserDone asks again');
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
