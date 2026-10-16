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

const B64 = 'SUQzBAAAAAAAAA==';

function makePage() {
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
  win.acquireTransport = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  const played = [];
  win.Audio = function Audio(src) {
    played.push(this);
    this.src = src;
    this.play = () => Promise.resolve();
  };

  const spoken = [];
  win.SpeechSynthesisUtterance = function (text) {
    this.text = text;
  };
  win.speechSynthesis = {
    speak: utter => spoken.push(utter),
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'page must post ready with its tab id');
  return {win, posted, played, spoken, tabId: ready.tabId};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.message}`);
  }
}

test('one stamped copy for an own tab plays exactly once', () => {
  const {win, played, spoken, tabId} = makePage();
  send(win, {
    type: 'talk', language: 'en-US', text: 'hello there',
    talkId: 't-1', taskId: 7, tabId, audioB64: B64,
  });
  assert.strictEqual(played.length, 1, JSON.stringify(played));
  assert.strictEqual(played[0].src, 'data:audio/mpeg;base64,' + B64);
  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');
});

test('duplicate stamped copies of one talk() call play only once', () => {
  const {win, played, spoken, tabId} = makePage();
  send(win, {
    type: 'talk', language: 'en-US', text: 'do not repeat me',
    talkId: 't-dup', taskId: 7, tabId, audioB64: B64,
  });
  send(win, {
    type: 'talk', language: 'en-US', text: 'do not repeat me',
    talkId: 't-dup', taskId: 7, tabId, audioB64: B64,
  });
  assert.strictEqual(
    played.length, 1,
    `played ${played.length} times: ${JSON.stringify(played)}`,
  );
  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');
});

test('copies stamped for two DIFFERENT owned tabs play only once', () => {
  const {win, posted, played, spoken, tabId} = makePage();
  // "+" keeps a chat that is still new; createNewTab() always opens one.
  win._testApi.createNewTab();
  const newChat = posted.filter(m => m.type === 'newChat').pop();
  assert.ok(newChat && newChat.tabId && newChat.tabId !== tabId);
  assert.strictEqual(spoken.length, 0, 'no unlock primer any more');
  send(win, {
    type: 'talk', language: 'en-US', text: 'once across tabs',
    talkId: 't-two-tabs', taskId: 7, tabId, audioB64: B64,
  });
  send(win, {
    type: 'talk', language: 'en-US', text: 'once across tabs',
    talkId: 't-two-tabs', taskId: 7, tabId: newChat.tabId, audioB64: B64,
  });
  assert.strictEqual(
    played.length, 1,
    `played ${played.length} times: ${JSON.stringify(played)}`,
  );
  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');
});

test('a copy for an own BACKGROUND tab still plays', () => {
  const {win, posted, played, spoken} = makePage();
  // Open a second chat without "+": "+" retires the idle chat it
  // leaves, and a closed tab's talk is rightly silent.
  win._testApi.createNewTab();
  const newChat = posted.filter(m => m.type === 'newChat').pop();
  assert.ok(newChat && newChat.tabId, 'new tab must post newChat');
  const firstTabId = posted.find(m => m.type === 'ready').tabId;
  assert.notStrictEqual(newChat.tabId, firstTabId);
  send(win, {
    type: 'talk', language: 'fr-FR', text: 'arri\u00e8re-plan',
    talkId: 't-3', taskId: 9, tabId: firstTabId, audioB64: B64,
  });
  assert.strictEqual(played.length, 1, JSON.stringify(played));
  assert.strictEqual(played[0].src, 'data:audio/mpeg;base64,' + B64);
  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');
});

test('legacy talk event without talkId or tabId still plays', () => {
  const {win, played, spoken} = makePage();
  send(win, {type: 'talk', language: 'es', text: 'hola', audioB64: B64});
  assert.strictEqual(played.length, 1, JSON.stringify(played));
  assert.strictEqual(played[0].src, 'data:audio/mpeg;base64,' + B64);
  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');
});

test('dedupe memory is bounded (many talk calls keep working)', () => {
  const {win, played, tabId} = makePage();
  for (let i = 0; i < 600; i++) {
    send(win, {
      type: 'talk', language: 'en-US', text: `utterance ${i}`,
      talkId: `t-many-${i}`, taskId: 7, tabId, audioB64: B64,
    });
    played[played.length - 1].onended();
  }
  assert.strictEqual(played.length, 600);
  send(win, {
    type: 'talk', language: 'en-US', text: 'utterance 599',
    talkId: 't-many-599', taskId: 7, tabId, audioB64: B64,
  });
  assert.strictEqual(played.length, 600, 'latest talkId must stay deduped');
});

console.log(`\n${passed} passed, ${failures.length} failed`);
// The pages are left with a clip "playing": its stall sampler
// (main.js TALK_STALL_CHECK_MS interval) would keep node alive.
process.exit(failures.length > 0 ? 1 : 0);
