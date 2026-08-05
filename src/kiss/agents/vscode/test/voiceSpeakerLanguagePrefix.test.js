// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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

function makeWindow() {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body>' +
      '<button id="voice-btn" class="toggle-btn"></button>' +
      '<textarea id="task-input"></textarea>' +
      '</body></html>',
    {runScripts: 'dangerously', url: 'https://localhost/'},
  );
  const win = dom.window;
  win.__VOICE__ = {mode: 'webview'};

  const counters = {input: 0, submit: 0};
  win.document
    .getElementById('task-input')
    .addEventListener('input', () => counters.input++);
  win.addEventListener('kiss-voice-submit', () => counters.submit++);

  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(VOICE_JS_PATH, 'utf-8');
  win.document.body.appendChild(script);

  return {win, counters};
}

function sendHostMessage(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

test('speaker + language payload inserts the language-aware prefix', () => {
  const {win, counters} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {
    type: 'voiceSpeech',
    text: 'Fix the bug',
    speaker: 1,
    language: 'fr',
  });
  assert.strictEqual(
    inp.value,
    'Speaker #1 says in the language fr that: Fix the bug',
  );
  assert.strictEqual(counters.input, 1);
  assert.strictEqual(counters.submit, 1);
});

test('each round carries its own speaker and language', () => {
  const {win, counters} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {
    type: 'voiceSpeech',
    text: 'first task',
    speaker: 1,
    language: 'en',
  });
  assert.strictEqual(
    inp.value,
    'Speaker #1 says in the language en that: first task',
  );
  inp.value = '';
  sendHostMessage(win, {
    type: 'voiceSpeech',
    text: 'second task',
    speaker: 2,
    language: 'es-mx',
  });
  assert.strictEqual(
    inp.value,
    'Speaker #2 says in the language es-mx that: second task',
  );
  sendHostMessage(win, {
    type: 'voiceSpeech',
    text: 'third task',
    speaker: 1,
    language: 'de',
  });
  assert.strictEqual(
    inp.value,
    'Speaker #2 says in the language es-mx that: second task ' +
      'Speaker #1 says in the language de that: third task',
  );
  assert.strictEqual(counters.submit, 3);
});

test('missing language falls back to the plain speaker prefix', () => {
  const {win, counters} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceSpeech', text: 'Fix it', speaker: 1});
  assert.strictEqual(inp.value, 'Speaker #1 says that: Fix it');
  assert.strictEqual(counters.submit, 1);
});

test('bogus language values fall back to the plain speaker prefix', () => {
  for (const language of ['', '   ', 42, null, undefined, {}, ['fr'], NaN]) {
    const {win, counters} = makeWindow();
    const inp = win.document.getElementById('task-input');
    sendHostMessage(win, {
      type: 'voiceSpeech',
      text: 'Do it',
      speaker: 2,
      language,
    });
    assert.strictEqual(
      inp.value,
      'Speaker #2 says that: Do it',
      `language=${String(language)} must fall back to the plain prefix`,
    );
    assert.strictEqual(counters.submit, 1);
  }
});

test('language tag is trimmed before prefixing', () => {
  const {win} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {
    type: 'voiceSpeech',
    text: 'Run the tests',
    speaker: 3,
    language: '  fr-ca \n',
  });
  assert.strictEqual(
    inp.value,
    'Speaker #3 says in the language fr-ca that: Run the tests',
  );
});

test('language without a speaker inserts bare text', () => {
  const {win, counters} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {
    type: 'voiceSpeech',
    text: 'Hello everyone',
    language: 'en',
  });
  assert.strictEqual(inp.value, 'Hello everyone');
  assert.strictEqual(counters.submit, 1);
});

test('language with a bogus speaker inserts bare text', () => {
  for (const speaker of [0, -1, 1.5, '2', null, NaN, Infinity]) {
    const {win, counters} = makeWindow();
    const inp = win.document.getElementById('task-input');
    sendHostMessage(win, {
      type: 'voiceSpeech',
      text: 'Do it',
      speaker,
      language: 'fr',
    });
    assert.strictEqual(
      inp.value,
      'Do it',
      `speaker=${String(speaker)} must not produce a prefix`,
    );
    assert.strictEqual(counters.submit, 1);
  }
});

test('prefixed speech appends to an existing draft with a space', () => {
  const {win} = makeWindow();
  const inp = win.document.getElementById('task-input');
  inp.value = 'draft text';
  sendHostMessage(win, {
    type: 'voiceSpeech',
    text: 'and more',
    speaker: 2,
    language: 'hi',
  });
  assert.strictEqual(
    inp.value,
    'draft text Speaker #2 says in the language hi that: and more',
  );
});

test('a language never turns an empty translation into an insert', () => {
  const {win, counters} = makeWindow();
  const inp = win.document.getElementById('task-input');
  inp.value = 'precious draft';
  sendHostMessage(win, {type: 'voiceWake'});
  sendHostMessage(win, {
    type: 'voiceSpeech',
    text: '',
    speaker: 1,
    language: 'fr',
  });
  sendHostMessage(win, {
    type: 'voiceSpeech',
    text: '   ',
    speaker: 2,
    language: 'en',
  });
  sendHostMessage(win, {type: 'voiceSpeech', language: 'en'});
  assert.strictEqual(inp.value, 'precious draft');
  assert.strictEqual(counters.input, 0);
  assert.strictEqual(counters.submit, 0);
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  process.exit(1);
}
