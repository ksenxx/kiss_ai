// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for natural-voice selection in the agent ``talk``
// tool playback: when the device offers several system voices, the
// webview must speak with the most natural-sounding (neural) voice
// matching the requested language instead of the often robotic
// browser default.  Ranking: exact BCP-47 tag > same base language,
// then quality markers in the voice name ("natural" > "neural" >
// "premium" > "enhanced" > "siri" > "google" > "online").  Playback
// must still work (with the default voice) when the voice list is
// empty, missing, or throws.
//
// Runs the REAL production ``media/main.js`` in jsdom (only the
// vscode host API and the Web Speech API are recording stubs, as in
// every webview test).  Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/talkNaturalVoice.test.js

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

  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win};
}

/**
 * Install a recording Web Speech API on *win* whose ``getVoices``
 * returns *voices* (jsdom has none).  Returns the array of spoken
 * utterances.
 */
function installSpeech(win, voices) {
  const spoken = [];
  win.SpeechSynthesisUtterance = function SpeechSynthesisUtterance(text) {
    this.text = text;
    this.lang = '';
  };
  win.speechSynthesis = {
    getVoices: () => voices,
    speak: u => spoken.push(u),
  };
  return spoken;
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Send one talk event stamped for this webview's active tab. */
function talk(win, language, text) {
  send(win, {
    type: 'talk',
    language,
    text,
    tabId: win._demoApi.getActiveTabId(),
  });
}

const VOICES = [
  {name: 'Fred', lang: 'en-US'},
  {name: 'Google US English', lang: 'en-US'},
  {name: 'Microsoft Aria Online (Natural) - English (United States)',
   lang: 'en-US'},
  {name: 'Samantha (Enhanced)', lang: 'en-US'},
  {name: 'Amelie (Premium)', lang: 'fr-CA'},
  {name: 'Anna', lang: 'de-DE'},
];

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

test('picks the highest-quality natural voice for an exact language match',
     () => {
  const {win} = makeWebview();
  const spoken = installSpeech(win, VOICES);
  talk(win, 'en-US', 'hello there');
  assert.strictEqual(spoken.length, 1);
  assert.ok(spoken[0].voice, 'a voice must be selected');
  assert.strictEqual(
      spoken[0].voice.name,
      'Microsoft Aria Online (Natural) - English (United States)',
      '"Natural" outranks "Enhanced", "Google" and the plain default');
});

test('falls back to a same-base-language natural voice', () => {
  const {win} = makeWebview();
  const spoken = installSpeech(win, VOICES);
  talk(win, 'fr-FR', 'bonjour');
  assert.strictEqual(spoken.length, 1);
  assert.ok(spoken[0].voice, 'a voice must be selected');
  assert.strictEqual(spoken[0].voice.name, 'Amelie (Premium)',
                     'fr-CA is the only French voice for fr-FR');
});

test('exact plain voice beats a natural voice of another region', () => {
  const {win} = makeWebview();
  const spoken = installSpeech(win, [
    {name: 'Thomas', lang: 'fr-FR'},
    {name: 'Amelie (Premium)', lang: 'fr-CA'},
  ]);
  talk(win, 'fr-FR', 'bonjour');
  assert.strictEqual(spoken.length, 1);
  assert.strictEqual(spoken[0].voice.name, 'Thomas',
                     'exact language tag outranks quality markers');
});

test('picks the best-quality voice overall when no language is given',
     () => {
  const {win} = makeWebview();
  const spoken = installSpeech(win, VOICES);
  talk(win, '', 'hello');
  assert.strictEqual(spoken.length, 1);
  assert.ok(spoken[0].voice, 'a voice must be selected');
  assert.strictEqual(
      spoken[0].voice.name,
      'Microsoft Aria Online (Natural) - English (United States)');
});

test('leaves the default voice when no voice matches the language', () => {
  const {win} = makeWebview();
  const spoken = installSpeech(win, VOICES);
  talk(win, 'ja-JP', 'konnichiwa');
  assert.strictEqual(spoken.length, 1, 'still speaks with the default');
  assert.strictEqual(spoken[0].voice, undefined);
});

test('leaves the default voice when the voice list is empty', () => {
  const {win} = makeWebview();
  const spoken = installSpeech(win, []);
  talk(win, 'en-US', 'hello');
  assert.strictEqual(spoken.length, 1, 'still speaks with the default');
  assert.strictEqual(spoken[0].voice, undefined);
});

test('still speaks when getVoices is missing (legacy engines)', () => {
  const {win} = makeWebview();
  const spoken = [];
  win.SpeechSynthesisUtterance = function (text) {
    this.text = text;
    this.lang = '';
  };
  win.speechSynthesis = {speak: u => spoken.push(u)};
  talk(win, 'en-US', 'hello');
  assert.strictEqual(spoken.length, 1);
  assert.strictEqual(spoken[0].voice, undefined);
});

test('still speaks when getVoices throws', () => {
  const {win} = makeWebview();
  const spoken = [];
  win.SpeechSynthesisUtterance = function (text) {
    this.text = text;
    this.lang = '';
  };
  win.speechSynthesis = {
    getVoices: () => {
      throw new Error('boom');
    },
    speak: u => spoken.push(u),
  };
  talk(win, 'en-US', 'hello');
  assert.strictEqual(spoken.length, 1);
  assert.strictEqual(spoken[0].voice, undefined);
});

// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  for (const f of failures) {
    console.error(`FAIL: ${f.name}`);
    console.error(f.error && f.error.stack ? f.error.stack : f.error);
  }
  process.exit(1);
}
console.log('PASS: talkNaturalVoice.test.js');
