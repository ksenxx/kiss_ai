// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage: makeChatPage, send, stubRaf} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {installAudio} = require('./speech_helpers');
const {test, report} = makeRunner();

const B64 = 'SUQzBAAAAAAAAA==';

function makePage() {
  let played;
  const {win, posted} = makeChatPage({
    beforeScripts: win => {
      stubRaf(win);
      played = installAudio(win);
    },
  });
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'page must post ready with its tab id');
  return {win, posted, played, tabId: ready.tabId};
}

test('one stamped copy for an own tab plays exactly once', () => {
  const {win, played, tabId} = makePage();
  send(win, {
    type: 'talk',
    language: 'en-US',
    text: 'hello there',
    talkId: 't-1',
    taskId: 7,
    tabId,
    audioB64: B64,
  });
  assert.strictEqual(played.length, 1, JSON.stringify(played));
  assert.strictEqual(played[0].src, 'data:audio/mpeg;base64,' + B64);
});

test('duplicate stamped copies of one talk() call play only once', () => {
  const {win, played, tabId} = makePage();
  send(win, {
    type: 'talk',
    language: 'en-US',
    text: 'do not repeat me',
    talkId: 't-dup',
    taskId: 7,
    tabId,
    audioB64: B64,
  });
  send(win, {
    type: 'talk',
    language: 'en-US',
    text: 'do not repeat me',
    talkId: 't-dup',
    taskId: 7,
    tabId,
    audioB64: B64,
  });
  assert.strictEqual(
    played.length,
    1,
    `played ${played.length} times: ${JSON.stringify(played)}`,
  );
});

test('copies stamped for two DIFFERENT owned tabs play only once', () => {
  const {win, posted, played, tabId} = makePage();
  // "+" keeps a chat that is still new; createNewTab() always opens one.
  win._testApi.createNewTab();
  const newChat = posted.filter(m => m.type === 'newChat').pop();
  assert.ok(newChat && newChat.tabId && newChat.tabId !== tabId);
  send(win, {
    type: 'talk',
    language: 'en-US',
    text: 'once across tabs',
    talkId: 't-two-tabs',
    taskId: 7,
    tabId,
    audioB64: B64,
  });
  send(win, {
    type: 'talk',
    language: 'en-US',
    text: 'once across tabs',
    talkId: 't-two-tabs',
    taskId: 7,
    tabId: newChat.tabId,
    audioB64: B64,
  });
  assert.strictEqual(
    played.length,
    1,
    `played ${played.length} times: ${JSON.stringify(played)}`,
  );
});

test('a copy for an own BACKGROUND tab still plays', () => {
  const {win, posted, played} = makePage();
  // Open a second chat without "+": "+" retires the idle chat it
  // leaves, and a closed tab's talk is rightly silent.
  win._testApi.createNewTab();
  const newChat = posted.filter(m => m.type === 'newChat').pop();
  assert.ok(newChat && newChat.tabId, 'new tab must post newChat');
  const firstTabId = posted.find(m => m.type === 'ready').tabId;
  assert.notStrictEqual(newChat.tabId, firstTabId);
  send(win, {
    type: 'talk',
    language: 'fr-FR',
    text: 'arri\u00e8re-plan',
    talkId: 't-3',
    taskId: 9,
    tabId: firstTabId,
    audioB64: B64,
  });
  assert.strictEqual(played.length, 1, JSON.stringify(played));
  assert.strictEqual(played[0].src, 'data:audio/mpeg;base64,' + B64);
});

test('legacy talk event without talkId or tabId still plays', () => {
  const {win, played} = makePage();
  send(win, {type: 'talk', language: 'es', text: 'hola', audioB64: B64});
  assert.strictEqual(played.length, 1, JSON.stringify(played));
  assert.strictEqual(played[0].src, 'data:audio/mpeg;base64,' + B64);
});

/** Send talk call number *i* (talkId `t-many-<i>`) to own tab *tabId*. */
function sendNumberedTalk(win, tabId, i) {
  send(win, {
    type: 'talk',
    language: 'en-US',
    text: `utterance ${i}`,
    talkId: `t-many-${i}`,
    taskId: 7,
    tabId,
    audioB64: B64,
  });
}

test('dedupe memory is bounded (oldest talkIds forgotten, recent kept)', () => {
  const {win, played, tabId} = makePage();
  for (let i = 0; i < 600; i++) {
    sendNumberedTalk(win, tabId, i);
    played[played.length - 1].onended();
  }
  assert.strictEqual(played.length, 600);
  // The 250 most recent ids are still remembered...
  sendNumberedTalk(win, tabId, 599);
  sendNumberedTalk(win, tabId, 350);
  assert.strictEqual(played.length, 600, 'recent talkIds must stay deduped');
  // ...and the oldest were dropped: the memory does not grow for ever.
  sendNumberedTalk(win, tabId, 0);
  assert.strictEqual(played.length, 601, 'the oldest talkId must be forgotten');
});

report('talkSpeaksOnce');
// The pages are left with a clip "playing": its stall sampler
// (main.js TALK_STALL_CHECK_MS interval) would keep node alive.
process.exit(0);
