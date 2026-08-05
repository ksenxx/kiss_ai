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

  const played = [];
  win.Audio = function Audio(src) {
    this.src = src;
    played.push(this);
    this.play = () => {
      if (typeof this.onended === 'function') this.onended({type: 'ended'});
      return Promise.resolve();
    };
  };

  const spoken = [];
  win.SpeechSynthesisUtterance = function (text) {
    this.text = text;
    spoken.push(this);
  };
  win.speechSynthesis = {
    speak: utter => spoken.push(utter),
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with its tab id');
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

test('a muted talk copy stays silent', () => {
  const {win, played, spoken, tabId} = makeWebview();
  send(win, {
    type: 'talk', language: 'en-US', text: 'played on the terminal',
    audioB64: B64, talkId: 't-muted-1', taskId: 7, tabId, muted: true,
  });
  assert.strictEqual(
    played.length, 0,
    `muted copy must not play its clip, played: ${played.length}`,
  );
  assert.strictEqual(spoken.length, 0, 'Web Speech must never be used');
});

test('a muted copy does not block a later playable copy', () => {
  const {win, played, spoken, tabId} = makeWebview();
  send(win, {
    type: 'talk', language: 'en-US', text: 'first muted',
    audioB64: B64, talkId: 't-muted-2', taskId: 7, tabId, muted: true,
  });
  send(win, {
    type: 'talk', language: 'en-US', text: 'now audible',
    audioB64: B64, talkId: 't-muted-2', taskId: 7, tabId,
  });
  assert.strictEqual(played.length, 1, `played ${played.length} clips`);
  assert.strictEqual(played[0].src, 'data:audio/mpeg;base64,' + B64);
  assert.strictEqual(spoken.length, 0, 'Web Speech must never be used');
});

test('an unmuted copy still plays exactly once', () => {
  const {win, played, spoken, tabId} = makeWebview();
  send(win, {
    type: 'talk', language: 'en-US', text: 'normal talk',
    audioB64: B64, talkId: 't-plain', taskId: 7, tabId,
  });
  assert.strictEqual(played.length, 1, `played ${played.length} clips`);
  assert.strictEqual(played[0].src, 'data:audio/mpeg;base64,' + B64);
  assert.strictEqual(spoken.length, 0, 'Web Speech must never be used');
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  for (const f of failures) {
    console.error(`FAILED: ${f.name}`);
    console.error(f.error && f.error.stack ? f.error.stack : f.error);
  }
  process.exit(1);
}
