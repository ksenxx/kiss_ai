// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the agent ``talk`` tool playback: a backend
// ``{type: 'talk', ...}`` event plays ONLY the GPT-synthesized clip
// (``audioB64``) through an Audio element on EVERY client with a tab
// open for the running task — even when that tab is not the active
// tab.  The robotic Web Speech (speechSynthesis) fallback is gone
// for good: an event without audio, a device without the Audio API,
// or a failing clip degrades to SILENCE while the serialized talk
// queue still advances so later talks keep playing.  A copy stamped
// for a tab this webview does not own belongs to another window and
// must stay silent here (see talkSpeaksOnce.test.js for the full
// per-device dedupe contract).
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/talkTool.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview: the real
 * ``chat.html`` body (placeholders blanked), ``panelCopy.js`` and
 * ``main.js`` evaluated in the window, and a recording
 * ``acquireVsCodeApi`` stub (the only host API the webview has).
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

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

/**
 * Install a TRIPWIRE Web Speech API on *win* (jsdom has none).  The
 * production code must NEVER call it any more — the returned array
 * records any (forbidden) utterance so tests can assert it stays
 * empty.
 */
function installSpeech(win) {
  const spoken = [];
  win.SpeechSynthesisUtterance = function SpeechSynthesisUtterance(text) {
    this.text = text;
    this.lang = '';
  };
  win.speechSynthesis = {
    speak: u => spoken.push(u),
  };
  return spoken;
}

/**
 * Install a recording Audio constructor on *win* (jsdom's own media
 * elements cannot play).  Returns the array of created sources;
 * ``play()`` resolves and each element's ``onended`` can be fired by
 * the test to complete the serialized talk queue.
 */
function installAudio(win) {
  const created = [];
  win.Audio = function Audio(src) {
    created.push(this);
    this.src = src;
    this.play = () => Promise.resolve();
  };
  return created;
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

const B64 = 'SUQzBAAAAAAAAA=='; // decodes to "ID3..." — an MP3 tag header

function testAudiolessTalkIsSilentAndQueueAdvances() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  const created = installAudio(win);
  const activeTab = win._demoApi.getActiveTabId();

  // No audioB64: the old code spoke this through the robotic Web
  // Speech engine; the new code stays silent and completes the talk
  // immediately.
  send(win, {type: 'talk', language: 'es', text: 'hola usuario',
             talkId: 'tt-noaudio', tabId: activeTab});

  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');
  assert.strictEqual(created.length, 0, 'no clip to play — stays silent');

  // The silent talk must have RELEASED the serialized talk queue: a
  // later talk carrying a good clip plays right away.
  send(win, {type: 'talk', language: 'es', text: 'con audio',
             talkId: 'tt-audio', tabId: activeTab,
             audioB64: B64, audioMime: 'audio/mpeg'});

  assert.strictEqual(created.length, 1, 'queue advanced to the clip talk');
  assert.strictEqual(created[0].src, 'data:audio/mpeg;base64,' + B64);
  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');
  console.log('PASS: audio-less talk is silent and the queue advances');
}

function testTalkForOtherWindowsTabStaysSilent() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  const created = installAudio(win);

  // The backend stamps one copy per subscribed viewer tab and sends
  // every copy to every connected webview.  A copy stamped for a tab
  // this webview does NOT own belongs to another window / device —
  // that window plays it.  Playing it here too made every utterance
  // play twice on the same speakers.
  send(win, {type: 'talk', language: 'de', text: 'hallo',
             tabId: 'some-other-windows-tab', audioB64: B64});

  assert.strictEqual(created.length, 0, "another window's copy is silent");
  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');
  console.log("PASS: talk copy for another window's tab stays silent");
}

function testTalkWithoutLanguagePlaysClip() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  const created = installAudio(win);

  // ``language`` only steers server-side synthesis; playback of the
  // clip must not depend on it.
  send(win, {type: 'talk', text: 'plain default', talkId: 'tt-nolang',
             audioB64: B64});

  assert.strictEqual(created.length, 1, 'clip plays without a language');
  assert.strictEqual(created[0].src, 'data:audio/mpeg;base64,' + B64);
  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');
  console.log('PASS: talk without language still plays the clip');
}

function testTalkEmptyTextIsIgnored() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  const created = installAudio(win);

  send(win, {type: 'talk', language: 'en'});
  send(win, {type: 'talk', language: 'en', text: ''});
  send(win, {type: 'talk', language: 'en', text: '', audioB64: B64});

  assert.strictEqual(created.length, 0, 'empty text must not play');
  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');
  console.log('PASS: talk with empty/missing text is ignored');
}

function testTalkWithoutAudioApiDoesNotCrashAndQueueAdvances() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  win.Audio = undefined; // device without the Audio API

  // Must be a silent no-op (never crash, never go robotic) AND must
  // not wedge the serialized talk queue.
  send(win, {type: 'talk', language: 'en', text: 'hello',
             talkId: 'tt-noapi', audioB64: B64});
  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');

  const created = installAudio(win);
  send(win, {type: 'talk', language: 'en', text: 'after silence',
             talkId: 'tt-after', audioB64: B64});
  assert.strictEqual(created.length, 1, 'queue advanced past silent talk');
  console.log('PASS: talk without the Audio API is silent, queue advances');
}

function testThrowingAudioConstructorIsSwallowedAndQueueAdvances() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  win.Audio = function Audio() {
    throw new Error('data URLs forbidden');
  };

  send(win, {type: 'talk', language: 'en', text: 'hello',
             talkId: 'tt-throw', audioB64: B64});
  assert.strictEqual(spoken.length, 0, 'Web Speech must never speak');

  // The failure degraded to silence and released the queue.
  const created = installAudio(win);
  send(win, {type: 'talk', language: 'en', text: 'next one',
             talkId: 'tt-next', audioB64: B64});
  assert.strictEqual(created.length, 1, 'queue advanced past the failure');
  console.log('PASS: a throwing Audio constructor is swallowed silently');
}

testAudiolessTalkIsSilentAndQueueAdvances();
testTalkForOtherWindowsTabStaysSilent();
testTalkWithoutLanguagePlaysClip();
testTalkEmptyTextIsIgnored();
testTalkWithoutAudioApiDoesNotCrashAndQueueAdvances();
testThrowingAudioConstructorIsSwallowedAndQueueAdvances();
console.log('All talkTool tests passed.');
