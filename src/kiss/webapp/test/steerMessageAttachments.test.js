// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (jsdom) test: a text-only message sent while a task is
// running goes out as an ``appendUserMessage`` with an empty attachment
// list, in the same wire shape as ``submit``.  The case with a pasted
// photo riding along is pinned by
// audit1012_mainjs3_append_drops_attachments.test.js.

'use strict';

const assert = require('assert');
const {makePage: makeChatPage, send, stubRaf} = require('./simplify2_harness');
const {typeAndEnter} = require('./attachment_helpers');

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

function testTextOnlySteeringMessageSendsNoAttachments() {
  const {win, posted, tab} = makePage();
  posted.length = 0;
  typeAndEnter(win, 'just text');
  const sent = posted.filter(m => m.type === 'appendUserMessage');
  assert.strictEqual(sent.length, 1, JSON.stringify(posted));
  assert.strictEqual(sent[0].prompt, 'just text');
  assert.strictEqual(sent[0].tabId, tab);
  // The array is the jsdom realm's, so no deepStrictEqual against [].
  assert.ok(
    Array.isArray(sent[0].attachments) && sent[0].attachments.length === 0,
  );
  assert.strictEqual(posted.filter(m => m.type === 'submit').length, 0);
  assert.strictEqual(win.document.getElementById('task-input').value, '');
  win.close();
  console.log('  ok - a text-only message carries an empty attachment list');
}

testTextOnlySteeringMessageSendsNoAttachments();
console.log('steerMessageAttachments.test.js passed');
