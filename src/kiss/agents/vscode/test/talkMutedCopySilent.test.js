// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test: a ``talk`` copy stamped ``muted: true``
// by the daemon must NOT be played by the webview.
//
// The daemon mutes a talk copy when another player on the SAME
// machine already owns the utterance — e.g. a task launched with the
// ``sorcar`` CLI plays the clip on the terminal speakers and forwards
// the event to the daemon, whose UDS peers (this webview among them)
// are all on that same machine.  Before this arbitration the webview
// played the relayed copy too: the same clip sounded twice, slightly
// offset — distorted, overlapping speech.
//
// A muted copy must also NOT poison the talkId dedupe set: a later
// playable (unmuted) copy of the same talkId must still speak.
//
// Runs the REAL production ``media/main.js`` in jsdom (only the
// vscode host API and the Web Speech API are recording stubs, as in
// every webview test).  Run with:
//
//     node test/talkMutedCopySilent.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview with a
 * recording Web Speech API stub (same harness as
 * talkSpeaksOnce.test.js).
 */
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

  const spoken = [];
  win.SpeechSynthesisUtterance = function (text) {
    this.text = text;
  };
  win.speechSynthesis = {
    speak: utter => spoken.push(utter),
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with its tab id');
  return {win, posted, spoken, tabId: ready.tabId};
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

// ---------------------------------------------------------------------------

test('a muted talk copy stays silent', () => {
  const {win, spoken, tabId} = makeWebview();
  send(win, {
    type: 'talk', language: 'en-US', text: 'played on the terminal',
    talkId: 't-muted-1', taskId: 7, tabId, muted: true,
  });
  assert.strictEqual(
    spoken.length, 0,
    `muted copy must not speak, spoke: ${JSON.stringify(spoken)}`,
  );
});

test('a muted copy does not block a later playable copy', () => {
  // The mute decision is per COPY, not per utterance: a muted copy
  // must not add its talkId to the spoken set, otherwise a playable
  // copy that legitimately reaches this webview later (e.g. after a
  // subscription change) would be wrongly suppressed.
  const {win, spoken, tabId} = makeWebview();
  send(win, {
    type: 'talk', language: 'en-US', text: 'first muted',
    talkId: 't-muted-2', taskId: 7, tabId, muted: true,
  });
  send(win, {
    type: 'talk', language: 'en-US', text: 'now audible',
    talkId: 't-muted-2', taskId: 7, tabId,
  });
  assert.strictEqual(spoken.length, 1, JSON.stringify(spoken));
  assert.strictEqual(spoken[0].text, 'now audible');
});

test('an unmuted copy still speaks exactly once', () => {
  const {win, spoken, tabId} = makeWebview();
  send(win, {
    type: 'talk', language: 'en-US', text: 'normal talk',
    talkId: 't-plain', taskId: 7, tabId,
  });
  assert.strictEqual(spoken.length, 1, JSON.stringify(spoken));
  assert.strictEqual(spoken[0].text, 'normal talk');
});

// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  for (const f of failures) {
    console.error(`FAILED: ${f.name}`);
    console.error(f.error && f.error.stack ? f.error.stack : f.error);
  }
  process.exit(1);
}
