// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the voice-round UX added around the wake word:
//
//  - When the "Sorcar" wake word fires (mic button flashes RED), the
//    blinking "Listening ..." overlay over the task input appears:
//    #input-text-wrap gains the 'listening' class that main.css turns
//    into large blinking type.
//  - The overlay disappears on every non-red transition: yellow
//    voiceTranscribing, terminal voiceSpeech, and silence.
//  - After a voice-dictated task is inserted and submitted, voice.js
//    says "Working on it." — preferring the GPT-synthesized clip at
//    cfg.ackAudioUrl (window.Audio), and falling back to the Web
//    Speech API when the clip is missing or playback is rejected.
//  - An empty translation (silence) never speaks the ack.
//  - A page without the overlay element still works (no throw).
//
// This test runs the real ``media/voice.js`` against a real jsdom
// document (no mocks for the code under test).
//
// Run directly with ``node test/voiceListeningOverlay.test.js``.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const VOICE_JS_PATH = path.join(__dirname, '..', 'media', 'voice.js');

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

/**
 * Build a fresh jsdom window with the real input structure from
 * chat.html (#input-text-wrap wrapping the overlay and the textarea),
 * inject the webview-mode config plus optional Audio/speechSynthesis
 * stubs, and execute the real voice.js.
 */
function makeWindow(opts) {
  opts = opts || {};
  const overlayHtml = opts.noOverlay
    ? ''
    : '<div id="listening-overlay">Listening ...</div>';
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body>' +
      '<button id="voice-btn" class="toggle-btn"></button>' +
      '<div id="input-text-wrap">' +
      overlayHtml +
      '<textarea id="task-input"></textarea>' +
      '</div>' +
      '</body></html>',
    {runScripts: 'dangerously', url: 'https://localhost/'},
  );
  const win = dom.window;
  win.__VOICE__ = {mode: 'webview'};
  if (opts.ackAudioUrl) win.__VOICE__.ackAudioUrl = opts.ackAudioUrl;

  // Audio stub: records constructed URLs and play() calls.
  win.__audioPlays = [];
  if (opts.audio === 'reject') {
    win.Audio = function (url) {
      this.play = () => {
        win.__audioPlays.push({url, rejected: true});
        return Promise.reject(new Error('autoplay blocked'));
      };
    };
  } else if (opts.audio !== 'none') {
    win.Audio = function (url) {
      this.play = () => {
        win.__audioPlays.push({url, rejected: false});
        return Promise.resolve();
      };
    };
  } else {
    win.Audio = undefined;
  }

  // Web Speech API stub: records spoken utterances.
  win.__spoken = [];
  if (opts.speech !== 'none') {
    win.SpeechSynthesisUtterance = function (text) {
      this.text = text;
    };
    win.speechSynthesis = {
      speak(utter) {
        win.__spoken.push(utter.text);
      },
    };
  }

  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);

  return win;
}

function sendHostMessage(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function wrapListening(win) {
  return win.document
    .getElementById('input-text-wrap')
    .classList.contains('listening');
}

// ---------------------------------------------------------------------------

test('wake shows the blinking "Listening ..." overlay', () => {
  const win = makeWindow();
  assert.strictEqual(wrapListening(win), false, 'hidden before the wake');
  sendHostMessage(win, {type: 'voiceWake'});
  assert.strictEqual(
    wrapListening(win),
    true,
    'wake must add the listening class next to the red mic flash',
  );
  assert.strictEqual(
    win.document.getElementById('listening-overlay').textContent,
    'Listening ...',
  );
});

test('transcribing (yellow) hides the overlay again', () => {
  const win = makeWindow();
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceTranscribing'});
  assert.strictEqual(wrapListening(win), false);
});

test('translated speech hides the overlay', () => {
  const win = makeWindow({ackAudioUrl: 'https://localhost/ack.mp3'});
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceSpeech', text: 'Fix the parser bug'});
  assert.strictEqual(wrapListening(win), false);
});

test('silence hides the overlay and never speaks the ack', () => {
  const win = makeWindow({ackAudioUrl: 'https://localhost/ack.mp3'});
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceSpeech', text: ''});
  assert.strictEqual(wrapListening(win), false);
  assert.strictEqual(win.__audioPlays.length, 0, 'no ack clip on silence');
  assert.deepStrictEqual(win.__spoken, [], 'no fallback speech on silence');
});

test('submitted speech plays the GPT "Working on it." clip', () => {
  const win = makeWindow({ackAudioUrl: 'https://localhost/ack.mp3'});
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceSpeech', text: 'Fix the parser bug'});
  assert.strictEqual(win.__audioPlays.length, 1, 'ack clip must play once');
  assert.strictEqual(win.__audioPlays[0].url, 'https://localhost/ack.mp3');
  assert.deepStrictEqual(
    win.__spoken,
    [],
    'system voice must not double-speak when the clip plays',
  );
});

test('missing ackAudioUrl speaks the ack with the system voice', () => {
  const win = makeWindow();
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceSpeech', text: 'Fix the parser bug'});
  assert.strictEqual(win.__audioPlays.length, 0);
  assert.deepStrictEqual(win.__spoken, ['Working on it.']);
});

test('no speech output at all stays silent without throwing', () => {
  const win = makeWindow({audio: 'none', speech: 'none'});
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceSpeech', text: 'Fix the parser bug'});
  assert.strictEqual(
    win.document.getElementById('task-input').value,
    'Fix the parser bug',
  );
});

test('a page without the overlay element still handles the wake', () => {
  const win = makeWindow({noOverlay: true});
  sendHostMessage(win, {type: 'voiceWake'});
  const btn = win.document.getElementById('voice-btn');
  assert.ok(
    btn.classList.contains('voice-triggered'),
    'red mic flash must still work without the overlay',
  );
});

// ---------------------------------------------------------------------------

/** Async case: a rejected clip playback falls back to the system voice. */
async function rejectedPlaybackFallsBack() {
  const win = makeWindow({
    ackAudioUrl: 'https://localhost/ack.mp3',
    audio: 'reject',
  });
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {type: 'voiceSpeech', text: 'Fix the parser bug'});
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.strictEqual(win.__audioPlays.length, 1, 'clip playback attempted');
  assert.deepStrictEqual(win.__spoken, ['Working on it.']);
}

(async () => {
  const name = 'rejected clip playback falls back to the system voice';
  try {
    await rejectedPlaybackFallsBack();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.message}`);
  }
  console.log(`\n${passed} passed, ${failures.length} failed`);
  if (failures.length > 0) process.exit(1);
})();
