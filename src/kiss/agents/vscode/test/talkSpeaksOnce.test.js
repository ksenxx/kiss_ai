// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test: the agent's ``talk`` tool must speak
// each utterance EXACTLY ONCE per device.
//
// The backend fans out one ``{type: 'talk'}`` copy per subscribed
// viewer tab of the task, and every stamped copy is delivered to
// EVERY connected webview (the frontend is responsible for tabId
// routing).  The webview must therefore:
//
//  * speak a talk event only when its ``tabId`` belongs to one of
//    THIS webview's tabs (a copy stamped for another window's tab
//    must stay silent here — that window speaks it);
//  * speak each utterance at most once, even when several stamped
//    copies land on this webview (two open tabs of the same task,
//    a stale subscription left over from a reload, ...) — copies of
//    one ``talk()`` call share the same ``talkId``;
//  * still speak when the task's tab is open but in the BACKGROUND;
//  * keep speaking legacy events that carry no ``talkId`` / ``tabId``.
//
// Runs the REAL production ``media/main.js`` in jsdom (only the
// vscode host API and the Web Speech API are recording stubs, as in
// every webview test).  Run with:
//
//     node test/talkSpeaksOnce.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview with a
 * recording Web Speech API stub.  Returns ``{win, posted, spoken,
 * tabId}`` where ``spoken`` records every utterance passed to
 * ``speechSynthesis.speak`` and ``tabId`` is the webview's initial
 * (active) chat tab id, taken from the ``ready`` handshake.
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

  // Recording Web Speech API stub (jsdom has none).  The production
  // handler reads ``window.speechSynthesis`` at event time, so the
  // stub is honoured for every dispatched talk event.
  const spoken = [];
  win.SpeechSynthesisUtterance = function (text) {
    this.text = text;
  };
  win.speechSynthesis = {
    speak: utter => spoken.push({text: utter.text, lang: utter.lang}),
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

test('one stamped copy for an own tab speaks exactly once', () => {
  const {win, spoken, tabId} = makeWebview();
  send(win, {
    type: 'talk', language: 'en-US', text: 'hello there',
    talkId: 't-1', taskId: 7, tabId,
  });
  assert.strictEqual(spoken.length, 1, JSON.stringify(spoken));
  assert.strictEqual(spoken[0].text, 'hello there');
  assert.strictEqual(spoken[0].lang, 'en-US');
});

test('duplicate stamped copies of one talk() call speak only once', () => {
  // Two viewer tabs of the same task on this device (or one live tab
  // plus a stale subscription from before a reload): the backend
  // sends one stamped copy per tab, all sharing the same talkId.
  const {win, spoken, tabId} = makeWebview();
  send(win, {
    type: 'talk', language: 'en-US', text: 'do not repeat me',
    talkId: 't-dup', taskId: 7, tabId,
  });
  send(win, {
    type: 'talk', language: 'en-US', text: 'do not repeat me',
    talkId: 't-dup', taskId: 7, tabId,
  });
  assert.strictEqual(
    spoken.length, 1,
    `spoken ${spoken.length} times: ${JSON.stringify(spoken)}`,
  );
});

test('copies stamped for two DIFFERENT owned tabs speak only once', () => {
  // The production duplication: this device has TWO tabs open on the
  // task (e.g. one live tab plus one reopened from history), so the
  // backend stamps one copy per tab — same talkId, different tabIds.
  const {win, posted, spoken, tabId} = makeWebview();
  win.document.querySelector('.chat-tab-add').click();
  const newChat = posted.filter(m => m.type === 'newChat').pop();
  assert.ok(newChat && newChat.tabId && newChat.tabId !== tabId);
  // The click above is this device's first user gesture: the iOS
  // speech unlock (``unlockSpeechSynthesis`` in main.js) must speak
  // exactly one EMPTY — inaudible — primer utterance inside it.
  assert.deepStrictEqual(
    spoken.map(s => s.text), [''],
    'first user gesture must speak the empty iOS unlock primer',
  );
  spoken.length = 0;
  send(win, {
    type: 'talk', language: 'en-US', text: 'once across tabs',
    talkId: 't-two-tabs', taskId: 7, tabId,
  });
  send(win, {
    type: 'talk', language: 'en-US', text: 'once across tabs',
    talkId: 't-two-tabs', taskId: 7, tabId: newChat.tabId,
  });
  assert.strictEqual(
    spoken.length, 1,
    `spoken ${spoken.length} times: ${JSON.stringify(spoken)}`,
  );
  assert.strictEqual(spoken[0].text, 'once across tabs');
});

test("a copy stamped for another window's tab stays silent", () => {
  const {win, spoken} = makeWebview();
  send(win, {
    type: 'talk', language: 'en-US', text: 'for the other window',
    talkId: 't-2', taskId: 7, tabId: 'some-other-windows-tab',
  });
  assert.strictEqual(spoken.length, 0, JSON.stringify(spoken));
});

test('a copy for an own BACKGROUND tab still speaks', () => {
  const {win, posted, spoken} = makeWebview();
  // Open a second tab (becomes active); the original tab is now in
  // the background.  createNewTab posts ``newChat`` with the new id.
  win.document.querySelector('.chat-tab-add').click();
  const newChat = posted.filter(m => m.type === 'newChat').pop();
  assert.ok(newChat && newChat.tabId, 'new tab must post newChat');
  const firstTabId = posted.find(m => m.type === 'ready').tabId;
  assert.notStrictEqual(newChat.tabId, firstTabId);
  // Drop the empty iOS unlock primer spoken inside the click's user
  // gesture (inaudible on a real device; asserted in the two-tabs
  // test above) so the counts below cover talk speech only.
  assert.deepStrictEqual(spoken.map(s => s.text), ['']);
  spoken.length = 0;
  send(win, {
    type: 'talk', language: 'fr-FR', text: 'arri\u00e8re-plan',
    talkId: 't-3', taskId: 9, tabId: firstTabId,
  });
  assert.strictEqual(spoken.length, 1, JSON.stringify(spoken));
  assert.strictEqual(spoken[0].text, 'arri\u00e8re-plan');
  assert.strictEqual(spoken[0].lang, 'fr-FR');
});

test('two DIFFERENT talk() calls both speak', () => {
  const {win, spoken, tabId} = makeWebview();
  send(win, {
    type: 'talk', language: 'en-US', text: 'first thing',
    talkId: 't-a', taskId: 7, tabId,
  });
  send(win, {
    type: 'talk', language: 'en-US', text: 'second thing',
    talkId: 't-b', taskId: 7, tabId,
  });
  assert.deepStrictEqual(
    spoken.map(s => s.text),
    ['first thing', 'second thing'],
  );
});

test('legacy talk event without talkId or tabId still speaks', () => {
  const {win, spoken} = makeWebview();
  send(win, {type: 'talk', language: 'es', text: 'hola'});
  assert.strictEqual(spoken.length, 1, JSON.stringify(spoken));
  assert.strictEqual(spoken[0].text, 'hola');
  assert.strictEqual(spoken[0].lang, 'es');
});

test('empty text never speaks', () => {
  const {win, spoken, tabId} = makeWebview();
  send(win, {type: 'talk', language: 'en-US', text: '', talkId: 't-4', tabId});
  send(win, {type: 'talk', language: 'en-US', talkId: 't-5', tabId});
  assert.strictEqual(spoken.length, 0, JSON.stringify(spoken));
});

test('missing Web Speech API is a silent no-op', () => {
  const {win, spoken, tabId} = makeWebview();
  win.speechSynthesis = undefined;
  send(win, {
    type: 'talk', language: 'en-US', text: 'no synth',
    talkId: 't-6', tabId,
  });
  assert.strictEqual(spoken.length, 0);
});

test('dedupe memory is bounded (many talk calls keep working)', () => {
  const {win, spoken, tabId} = makeWebview();
  for (let i = 0; i < 600; i++) {
    send(win, {
      type: 'talk', language: 'en-US', text: `utterance ${i}`,
      talkId: `t-many-${i}`, taskId: 7, tabId,
    });
  }
  assert.strictEqual(spoken.length, 600);
  // A duplicate of the LATEST utterance is still suppressed after
  // the pruning threshold has been crossed.
  send(win, {
    type: 'talk', language: 'en-US', text: 'utterance 599',
    talkId: 't-many-599', taskId: 7, tabId,
  });
  assert.strictEqual(spoken.length, 600, 'latest talkId must stay deduped');
});

// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
