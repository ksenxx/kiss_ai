// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the agent ``talk`` tool playback: a backend
// ``{type: 'talk', language, text}`` event must be spoken through the
// device's default speaker via the Web Speech API on EVERY client
// with a tab open for the running task — even when that tab is not
// the active tab — and must never crash a browser without speech
// support.  A copy stamped for a tab this webview does not own
// belongs to another window and must stay silent here (see
// talkSpeaksOnce.test.js for the full per-device dedupe contract).
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
 * Install a recording Web Speech API on *win* (jsdom has none).
 * Returns the array of spoken utterances.
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

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function testTalkSpeaksTextWithLanguage() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);
  const activeTab = win._demoApi.getActiveTabId();

  send(win, {type: 'talk', language: 'es', text: 'hola usuario',
             tabId: activeTab});

  assert.strictEqual(spoken.length, 1, 'exactly one utterance spoken');
  assert.strictEqual(spoken[0].text, 'hola usuario');
  assert.strictEqual(spoken[0].lang, 'es');
  console.log('PASS: talk event speaks text with the requested language');
}

function testTalkForOtherWindowsTabStaysSilent() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);

  // The backend stamps one copy per subscribed viewer tab and sends
  // every copy to every connected webview.  A copy stamped for a tab
  // this webview does NOT own belongs to another window / device —
  // that window plays it.  Speaking it here too made every utterance
  // play twice on the same speakers.
  send(win, {type: 'talk', language: 'de', text: 'hallo',
             tabId: 'some-other-windows-tab'});

  assert.strictEqual(spoken.length, 0, "another window's copy is silent");
  console.log("PASS: talk copy for another window's tab stays silent");
}

function testTalkWithoutLanguageUsesDefaultVoiceLang() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);

  send(win, {type: 'talk', text: 'plain default'});

  assert.strictEqual(spoken.length, 1);
  assert.strictEqual(spoken[0].text, 'plain default');
  assert.strictEqual(spoken[0].lang, '', 'lang untouched without language');
  console.log('PASS: talk without language leaves utterance lang default');
}

function testTalkEmptyTextIsIgnored() {
  const {win} = makeWebview();
  const spoken = installSpeech(win);

  send(win, {type: 'talk', language: 'en'});
  send(win, {type: 'talk', language: 'en', text: ''});

  assert.strictEqual(spoken.length, 0, 'empty text must not be spoken');
  console.log('PASS: talk with empty/missing text is ignored');
}

function testTalkWithoutSpeechSupportDoesNotCrash() {
  const {win} = makeWebview();
  // No installSpeech: jsdom has neither speechSynthesis nor
  // SpeechSynthesisUtterance — the handler must be a silent no-op.
  send(win, {type: 'talk', language: 'en', text: 'hello'});
  console.log('PASS: talk without Web Speech support does not crash');
}

function testTalkSpeakFailureIsSwallowed() {
  const {win} = makeWebview();
  win.SpeechSynthesisUtterance = function SpeechSynthesisUtterance(text) {
    this.text = text;
    this.lang = '';
  };
  win.speechSynthesis = {
    speak: () => {
      throw new Error('speech blocked by browser policy');
    },
  };
  send(win, {type: 'talk', language: 'en', text: 'hello'});
  console.log('PASS: a throwing speechSynthesis.speak is swallowed');
}

testTalkSpeaksTextWithLanguage();
testTalkForOtherWindowsTabStaysSilent();
testTalkWithoutLanguageUsesDefaultVoiceLang();
testTalkEmptyTextIsIgnored();
testTalkWithoutSpeechSupportDoesNotCrash();
testTalkSpeakFailureIsSwallowed();
console.log('All talkTool tests passed.');
