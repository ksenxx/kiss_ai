// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage: makeChatPage, send, readyTabId} = require('./simplify2_harness');

// The state the previous page persisted, carried into the next page as
// a reload of the same surface would.
let persistedState;

function makePage() {
  return makeChatPage({initialState: persistedState});
}

function typeAndSend(win, text) {
  const inp = win.document.getElementById('task-input');
  const sendBtn = win.document.getElementById('send-btn');
  inp.value = text;
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  sendBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

async function runTests() {
  const wv1 = makePage();
  const TAB = readyTabId(wv1.posted);

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
  persistedState = wv1.getState();
  assert.ok(persistedState, 'tab state must have been persisted');
  wv1.win.close();

  const wv2 = makePage();
  const ready2 = wv2.posted.find(m => m.type === 'ready');
  assert.ok(ready2, 're-opened page must post ready');
  assert.strictEqual(
    JSON.stringify(ready2.restoredTabs),
    JSON.stringify([]),
    'tabs are server-canonical: the re-opened page announces no ' +
      'locally persisted tabs (got ' +
      JSON.stringify(ready2.restoredTabs) +
      ')',
  );

  // The daemon answers `ready` with the shared-registry snapshot (the
  // running task's tab is still registered) and replays the session.
  send(wv2.win, {
    type: 'tabs_state',
    tabs: [
      {tabId: TAB, chatId: 'chat-1', title: 'do a long task', workDir: ''},
    ],
  });
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
