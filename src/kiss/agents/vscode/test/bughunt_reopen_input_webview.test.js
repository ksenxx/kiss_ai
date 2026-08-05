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

let persistedState;

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
    return {
      postMessage: msg => posted.push(msg),
      getState: () => persistedState,
      setState: s => {
        persistedState = s;
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

function typeAndSend(win, text) {
  const inp = win.document.getElementById('task-input');
  const sendBtn = win.document.getElementById('send-btn');
  inp.value = text;
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  sendBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

async function runTests() {
  const wv1 = makeWebview();
  const ready1 = wv1.posted.find(m => m.type === 'ready');
  assert.ok(ready1 && ready1.tabId, 'webview must post ready with a tabId');
  const TAB = ready1.tabId;

  typeAndSend(wv1.win, 'do a long task');
  const submit = wv1.posted.find(m => m.type === 'submit');
  assert.ok(submit && submit.tabId === TAB, 'first send must be a submit');

  send(wv1.win, {type: 'clear', chat_id: 'chat-1', tabId: TAB});
  send(wv1.win, {
    type: 'status',
    running: true,
    tabId: TAB,
    startTs: Date.now(),
  });

  typeAndSend(wv1.win, 'mid-run note before close');
  assert.ok(
    wv1.posted.some(m => m.type === 'appendUserMessage'),
    'sanity: the launching tab sends appendUserMessage while running',
  );
  assert.ok(persistedState, 'tab state must have been persisted');
  wv1.win.close();

  const wv2 = makeWebview();
  const ready2 = wv2.posted.find(m => m.type === 'ready');
  assert.ok(ready2, 're-opened webview must post ready');
  assert.strictEqual(
    JSON.stringify(ready2.restoredTabs),
    JSON.stringify([{tabId: TAB, chatId: 'chat-1'}]),
    'BUG: re-opened webview must restore the running tab so the ' +
      'extension can resumeSession it (got ' +
      JSON.stringify(ready2.restoredTabs) +
      ')',
  );

  send(wv2.win, {
    type: 'status',
    running: true,
    tabId: TAB,
    startTs: Date.now(),
  });
  send(wv2.win, {
    type: 'task_events',
    events: [],
    task: 'do a long task',
    tabId: TAB,
    chat_id: 'chat-1',
  });

  wv2.posted.length = 0;
  typeAndSend(wv2.win, 'please also update the docs');
  const duringMsgs = wv2.posted.filter(
    m => m.type === 'appendUserMessage' || m.type === 'submit',
  );
  assert.deepStrictEqual(
    duringMsgs.map(m => m.type),
    ['appendUserMessage'],
    'BUG: after reopen, a mid-run message must be sent as ' +
      'appendUserMessage (was: ' +
      JSON.stringify(duringMsgs.map(m => m.type)) +
      '). The re-opened tab never re-learned the task is running.',
  );

  send(wv2.win, {type: 'status', running: false, tabId: TAB});
  wv2.posted.length = 0;
  typeAndSend(wv2.win, 'now do a follow-up task');
  const afterMsgs = wv2.posted.filter(
    m => m.type === 'appendUserMessage' || m.type === 'submit',
  );
  assert.deepStrictEqual(
    afterMsgs.map(m => m.type),
    ['submit'],
    'after the task finishes, a typed message must be a submit (was: ' +
      JSON.stringify(afterMsgs.map(m => m.type)) +
      ')',
  );

  wv2.win.close();
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
