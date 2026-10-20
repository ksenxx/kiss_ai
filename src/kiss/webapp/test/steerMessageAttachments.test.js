// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (jsdom) test: a message sent while a task is running
// carries the files attached to it.  The composer used to refuse such a
// send ("A message added to a running task is text only"); now the
// ``appendUserMessage`` command carries the attachments in the same
// wire shape as ``submit`` and the composer is cleared like after any
// send.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  waitFor,
} = require('./simplify2_harness');

function makePage() {
  const {win, posted} = makeChatPage({
    beforeScripts: stubRaf,
    extraScripts: ['marked.min.js', 'voice.js'],
  });
  const tab = posted.find(m => m.type === 'ready').tabId;
  send(win, {type: 'clear', chat_id: 'chat-a', tabId: tab});
  send(win, {type: 'status', running: true, tabId: tab, startTs: Date.now()});
  send(win, {type: 'daemonStatus', connected: true});
  win._testApi.endLaunch();
  return {win, posted, tab};
}

function pastePng(win, bytes) {
  const inp = win.document.getElementById('task-input');
  const png = new win.File([new Uint8Array(bytes)], 'shot.png', {
    type: 'image/png',
  });
  const paste = new win.Event('paste', {bubbles: true, cancelable: true});
  Object.defineProperty(paste, 'clipboardData', {
    value: {items: [{kind: 'file', getAsFile: () => png}]},
  });
  inp.dispatchEvent(paste);
}

function pressEnter(win) {
  const inp = win.document.getElementById('task-input');
  inp.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
}

async function testSteeringMessageCarriesTheAttachment() {
  const {win, posted, tab} = makePage();
  const inp = win.document.getElementById('task-input');
  pastePng(win, [137, 80, 78, 71, 13, 10, 26, 10]);
  await waitFor(
    () => win.document.querySelector('.file-chip'),
    'the photo chip is shown',
  );
  inp.value = 'use this screenshot';
  posted.length = 0;
  pressEnter(win);
  await waitFor(
    () => posted.some(m => m.type === 'appendUserMessage'),
    'the steering message was never sent',
  );
  const sent = posted.filter(m => m.type === 'appendUserMessage');
  assert.strictEqual(
    sent.length,
    1,
    'the running task gets the message: ' + JSON.stringify(posted),
  );
  assert.strictEqual(sent[0].prompt, 'use this screenshot');
  assert.strictEqual(sent[0].tabId, tab);
  assert.strictEqual(sent[0].attachments.length, 1, 'the photo rides along');
  assert.strictEqual(sent[0].attachments[0].name, 'shot.png');
  assert.strictEqual(sent[0].attachments[0].mimeType, 'image/png');
  assert.strictEqual(
    sent[0].attachments[0].data,
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).toString('base64'),
    'the bytes travel base64-encoded, as for a submit',
  );
  assert.strictEqual(posted.filter(m => m.type === 'submit').length, 0);
  assert.strictEqual(inp.value, '', 'the composer is cleared');
  assert.strictEqual(
    win.document.querySelector('.file-chip'),
    null,
    'the chip is gone',
  );
  assert.ok(
    !Array.from(win.document.querySelectorAll('.notification, .notice')).some(
      el => /text only/.test(el.textContent),
    ),
    'no "text only" refusal',
  );
  win.close();
  console.log('  ok - a message to a running task carries its attachment');
}

async function testTextOnlySteeringMessageSendsNoAttachments() {
  const {win, posted} = makePage();
  const inp = win.document.getElementById('task-input');
  inp.value = 'just text';
  posted.length = 0;
  pressEnter(win);
  const sent = posted.filter(m => m.type === 'appendUserMessage');
  assert.strictEqual(sent.length, 1);
  assert.ok(
    Array.isArray(sent[0].attachments) && sent[0].attachments.length === 0,
  );
  win.close();
  console.log('  ok - a text-only message carries an empty attachment list');
}

(async () => {
  await testSteeringMessageCarriesTheAttachment();
  await testTextOnlySteeringMessageSendsNoAttachments();
  console.log('steerMessageAttachments.test.js passed');
})().catch(err => {
  console.error(err);
  process.exit(1);
});
