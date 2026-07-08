// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for male-voice parity of the agent ``talk`` tool in
// the chat webview.  The sorcar CLI REPL always speaks with a male
// voice: it plays the server-synthesized clip (speech_synthesis.py,
// DEFAULT_TTS_VOICE="cedar" — a deep male narrator).  The chat webview
// plays the same clip when it can, but whenever it takes the Web
// Speech API fallback path (no synthesized audio in the event,
// synthesis failed server-side, or clip playback was rejected by the
// autoplay policy) it must STILL sound like the same male narrator —
// so ``pickNaturalVoice`` must prefer a male system voice over a
// female one of equal or better "natural" quality, while language
// match keeps the highest priority and quality still ranks among male
// voices (and among all voices when no male voice exists).
//
// Runs the REAL production ``media/main.js`` in jsdom (only the
// vscode host API, the Web Speech API, and the Audio API are
// recording stubs, as in every webview test).  Run directly with
// ``node``:
//
//     node src/kiss/agents/vscode/test/talkMaleVoice.test.js

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

/** Send one plain-text talk event (no synthesized clip). */
function talk(win, language, text) {
  send(win, {type: 'talk', language, text});
}

// A realistic Windows / Chrome voice list: the female neural voice
// (Aria) is listed BEFORE the male neural voice (Guy), the way real
// engines commonly order them.
const WINDOWS_VOICES = [
  {name: 'Microsoft Aria Online (Natural) - English (United States)',
   lang: 'en-US'},
  {name: 'Microsoft Guy Online (Natural) - English (United States)',
   lang: 'en-US'},
  {name: 'Microsoft Zira - English (United States)', lang: 'en-US'},
  {name: 'Google US English', lang: 'en-US'},
];

// A realistic macOS voice list: the only "quality-marked" en-US voice
// (Samantha (Enhanced)) is female; Alex and Fred are the male voices.
const MACOS_VOICES = [
  {name: 'Samantha (Enhanced)', lang: 'en-US'},
  {name: 'Alex', lang: 'en-US'},
  {name: 'Fred', lang: 'en-US'},
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

async function asyncTest(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.message}`);
  }
}

// ---------------------------------------------------------------------------

test('CLI-parity: prefers the male neural voice over the female one',
     () => {
  const {win} = makeWebview();
  const spoken = installSpeech(win, WINDOWS_VOICES);
  talk(win, 'en-US', 'hello there');
  assert.strictEqual(spoken.length, 1);
  assert.ok(spoken[0].voice, 'a voice must be selected');
  assert.strictEqual(
      spoken[0].voice.name,
      'Microsoft Guy Online (Natural) - English (United States)',
      'the male narrator (CLI parity) must outrank the female voice');
});

test('prefers a plain male voice over a female quality-marked voice',
     () => {
  const {win} = makeWebview();
  const spoken = installSpeech(win, MACOS_VOICES);
  talk(win, 'en-US', 'hello there');
  assert.strictEqual(spoken.length, 1);
  assert.ok(spoken[0].voice, 'a voice must be selected');
  assert.strictEqual(spoken[0].voice.name, 'Alex',
                     'a male voice must beat "Samantha (Enhanced)"');
});

test('"Male" in the voice name wins; "Female" must not match "male"',
     () => {
  const {win} = makeWebview();
  const spoken = installSpeech(win, [
    {name: 'Google UK English Female', lang: 'en-GB'},
    {name: 'Google UK English Male', lang: 'en-GB'},
  ]);
  talk(win, 'en-GB', 'hello there');
  assert.strictEqual(spoken.length, 1);
  assert.strictEqual(spoken[0].voice.name, 'Google UK English Male');
});

test('prefers the highest-quality voice among several male voices',
     () => {
  const {win} = makeWebview();
  const spoken = installSpeech(win, [
    {name: 'Fred', lang: 'en-US'},
    {name: 'Microsoft Guy Online (Natural) - English (United States)',
     lang: 'en-US'},
    {name: 'Alex', lang: 'en-US'},
  ]);
  talk(win, 'en-US', 'hello there');
  assert.strictEqual(spoken.length, 1);
  assert.strictEqual(
      spoken[0].voice.name,
      'Microsoft Guy Online (Natural) - English (United States)',
      'quality markers still rank among male voices');
});

test('prefers the male voice when no language is given', () => {
  const {win} = makeWebview();
  const spoken = installSpeech(win, WINDOWS_VOICES);
  talk(win, '', 'hello there');
  assert.strictEqual(spoken.length, 1);
  assert.strictEqual(
      spoken[0].voice.name,
      'Microsoft Guy Online (Natural) - English (United States)');
});

test('language match still outranks maleness', () => {
  const {win} = makeWebview();
  const spoken = installSpeech(win, [
    {name: 'Alex', lang: 'en-US'},
    {name: 'Amelie (Premium)', lang: 'fr-CA'},
  ]);
  talk(win, 'fr-FR', 'bonjour');
  assert.strictEqual(spoken.length, 1);
  assert.strictEqual(spoken[0].voice.name, 'Amelie (Premium)',
                     'a same-language female voice beats a male voice ' +
                     'of another language');
});

test('recognizes Edge concatenated multilingual male voice names', () => {
  const {win} = makeWebview();
  const spoken = installSpeech(win, [
    {name: 'Microsoft AvaMultilingual Online (Natural) - English ' +
       '(United States)', lang: 'en-US'},
    {name: 'Microsoft AndrewMultilingual Online (Natural) - English ' +
       '(United States)', lang: 'en-US'},
  ]);
  talk(win, 'en-US', 'hello there');
  assert.strictEqual(spoken.length, 1);
  assert.strictEqual(
      spoken[0].voice.name,
      'Microsoft AndrewMultilingual Online (Natural) - English ' +
        '(United States)',
      '"AndrewMultilingual" must be recognized as a male voice');
});

test('keeps the best natural voice when no male voice exists', () => {
  const {win} = makeWebview();
  const spoken = installSpeech(win, [
    {name: 'Google US English', lang: 'en-US'},
    {name: 'Samantha (Enhanced)', lang: 'en-US'},
  ]);
  talk(win, 'en-US', 'hello there');
  assert.strictEqual(spoken.length, 1);
  assert.strictEqual(spoken[0].voice.name, 'Samantha (Enhanced)',
                     'quality ranking is unchanged without male voices');
});

// The webview prefers the synthesized male clip (audioB64); when the
// autoplay policy rejects play(), the Web Speech fallback must keep
// the male character rather than switching to a female system voice.
async function rejectedClipFallsBackToMaleVoice() {
  const {win} = makeWebview();
  const spoken = installSpeech(win, WINDOWS_VOICES);
  win.Audio = function Audio(src) {
    this.src = src;
    this.play = () => Promise.reject(new Error('autoplay blocked'));
  };
  send(win, {
    type: 'talk',
    language: 'en-US',
    text: 'hello there',
    talkId: 'male-1',
    audioB64: 'SUQzBAAAAAAAAA==',
    audioMime: 'audio/mpeg',
  });
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.strictEqual(spoken.length, 1, 'speech fallback must speak');
  assert.strictEqual(
      spoken[0].voice.name,
      'Microsoft Guy Online (Natural) - English (United States)',
      'the rejected-clip fallback must keep the male voice');
}

// ---------------------------------------------------------------------------

async function main() {
  await asyncTest(
      'rejected clip playback falls back to the male system voice',
      rejectedClipFallsBackToMaleVoice);

  console.log(`\n${passed} passed, ${failures.length} failed`);
  if (failures.length) {
    for (const f of failures) {
      console.error(`FAIL: ${f.name}`);
      console.error(f.error && f.error.stack ? f.error.stack : f.error);
    }
    process.exit(1);
  }
  console.log('PASS: talkMaleVoice.test.js');
}

main();
