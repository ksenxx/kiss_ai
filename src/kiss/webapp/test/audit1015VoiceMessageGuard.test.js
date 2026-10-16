// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// voice.js's window 'message' listener must only act on the daemon's
// synthetic MessageEvents (dispatched by media/transport.js with no
// source window).  A sandboxed iframe on the page can postMessage to
// its parent, and such an event carries the iframe's window as
// event.source: it must not be able to type a fake transcript into the
// task input.  The listener also requires the documented wire shape
// ({type: 'voiceSpeech', text: string}).

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const VOICE_JS_PATH =
  process.env.KISS_VOICE_JS || path.join(__dirname, '..', 'media', 'voice.js');

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

function makeWindow() {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body>' +
      '<button id="voice-btn" class="toggle-btn"></button>' +
      '<textarea id="task-input"></textarea>' +
      '<iframe id="preview" sandbox="allow-scripts"></iframe>' +
      '</body></html>',
    {runScripts: 'dangerously', url: 'https://localhost/'},
  );
  const win = dom.window;
  win.__VOICE__ = {};
  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);
  return win;
}

function dispatch(win, init) {
  win.dispatchEvent(new win.MessageEvent('message', init));
}

function inputValue(win) {
  return win.document.getElementById('task-input').value;
}

test('a sourceless daemon event still types the transcript', () => {
  const win = makeWindow();
  dispatch(win, {data: {type: 'voiceSpeech', text: 'hello there'}});
  assert.strictEqual(inputValue(win), 'hello there');
});

test('an event with a source window is ignored', () => {
  const win = makeWindow();
  const iframeWin = win.document.getElementById('preview').contentWindow;
  dispatch(win, {
    data: {type: 'voiceSpeech', text: 'rm -rf everything'},
    source: iframeWin,
    origin: 'null',
  });
  dispatch(win, {data: {type: 'voiceSpeech', text: 'from self'}, source: win});
  assert.strictEqual(inputValue(win), '');
});

test('events whose data has no string text are ignored', () => {
  const win = makeWindow();
  dispatch(win, {data: {type: 'voiceSpeech', text: 42}});
  dispatch(win, {data: {type: 'voiceSpeech'}});
  dispatch(win, {data: 'voiceSpeech'});
  dispatch(win, {data: null});
  dispatch(win, {});
  assert.strictEqual(inputValue(win), '');
});

// A real iframe postMessage is not exercised here: jsdom's
// window.postMessage never sets event.source (a TODO in
// jsdom/lib/jsdom/browser/Window.js), so it looks exactly like the
// daemon's synthetic event.  The sourced MessageEvents above model what
// a browser delivers for a cross-window postMessage.

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
