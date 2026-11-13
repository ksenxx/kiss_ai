// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makeRunner} = require('./ui_antipattern_harness');
const {send: sendHostMessage} = require('./simplify2_harness');
const {
  makeBareVoicePage,
  startPipeline,
  speakRound,
  dictate,
} = require('./voice_page_helpers');
const {test, report} = makeRunner();

// Every page built; closed after the run so node exits on its own.
const windows = [];

// A bare voice page with the pipeline live: voice.js types only the
// transcript of a round it posted, so every test dictates a real one.
async function makePage() {
  const counters = {input: 0, submit: 0};
  const page = makeBareVoicePage({}, w => {
    w.document
      .getElementById('task-input')
      .addEventListener('input', () => counters.input++);
    w.addEventListener('kiss-voice-submit', () => counters.submit++);
  });
  windows.push(page.win);
  await startPipeline(page);
  return {page, win: page.win, counters};
}

async function main() {
  await test('speaker + language payload inserts the language-aware prefix', async () => {
    const {page, win, counters} = await makePage();
    const inp = win.document.getElementById('task-input');
    dictate(page, 'Fix the bug', {speaker: 1, language: 'fr'});
    assert.strictEqual(
      inp.value,
      'Speaker #1 says in the language fr that: Fix the bug',
    );
    assert.strictEqual(counters.input, 1);
    assert.strictEqual(counters.submit, 1);
  });

  await test('each round carries its own speaker and language', async () => {
    const {page, win, counters} = await makePage();
    const inp = win.document.getElementById('task-input');
    dictate(page, 'first task', {speaker: 1, language: 'en'});
    assert.strictEqual(
      inp.value,
      'Speaker #1 says in the language en that: first task',
    );
    inp.value = '';
    dictate(page, 'second task', {speaker: 2, language: 'es-mx'});
    assert.strictEqual(
      inp.value,
      'Speaker #2 says in the language es-mx that: second task',
    );
    dictate(page, 'third task', {speaker: 1, language: 'de'});
    assert.strictEqual(
      inp.value,
      'Speaker #2 says in the language es-mx that: second task ' +
        'Speaker #1 says in the language de that: third task',
    );
    assert.strictEqual(counters.submit, 3);
  });

  await test('missing language falls back to the plain speaker prefix', async () => {
    const {page, win, counters} = await makePage();
    const inp = win.document.getElementById('task-input');
    dictate(page, 'Fix it', {speaker: 1});
    assert.strictEqual(inp.value, 'Speaker #1 says that: Fix it');
    assert.strictEqual(counters.submit, 1);
  });

  await test('bogus language values fall back to the plain speaker prefix', async () => {
    for (const language of ['', '   ', 42, null, undefined, {}, ['fr'], NaN]) {
      const {page, win, counters} = await makePage();
      const inp = win.document.getElementById('task-input');
      dictate(page, 'Do it', {speaker: 2, language});
      assert.strictEqual(
        inp.value,
        'Speaker #2 says that: Do it',
        `language=${String(language)} must fall back to the plain prefix`,
      );
      assert.strictEqual(counters.submit, 1);
    }
  });

  await test('language tag is trimmed before prefixing', async () => {
    const {page, win} = await makePage();
    const inp = win.document.getElementById('task-input');
    dictate(page, 'Run the tests', {speaker: 3, language: '  fr-ca \n'});
    assert.strictEqual(
      inp.value,
      'Speaker #3 says in the language fr-ca that: Run the tests',
    );
  });

  await test('language without a speaker inserts bare text', async () => {
    const {page, win, counters} = await makePage();
    const inp = win.document.getElementById('task-input');
    dictate(page, 'Hello everyone', {language: 'en'});
    assert.strictEqual(inp.value, 'Hello everyone');
    assert.strictEqual(counters.submit, 1);
  });

  await test('language with a bogus speaker inserts bare text', async () => {
    for (const speaker of [0, -1, 1.5, '2', null, NaN, Infinity]) {
      const {page, win, counters} = await makePage();
      const inp = win.document.getElementById('task-input');
      dictate(page, 'Do it', {speaker, language: 'fr'});
      assert.strictEqual(
        inp.value,
        'Do it',
        `speaker=${String(speaker)} must not produce a prefix`,
      );
      assert.strictEqual(counters.submit, 1);
    }
  });

  await test('prefixed speech appends to an existing draft with a space', async () => {
    const {page, win} = await makePage();
    const inp = win.document.getElementById('task-input');
    inp.value = 'draft text';
    dictate(page, 'and more', {speaker: 2, language: 'hi'});
    assert.strictEqual(
      inp.value,
      'draft text Speaker #2 says in the language hi that: and more',
    );
  });

  await test('a language never turns an empty translation into an insert', async () => {
    const {page, win, counters} = await makePage();
    const inp = win.document.getElementById('task-input');
    inp.value = 'precious draft';
    dictate(page, '', {speaker: 1, language: 'fr'});
    dictate(page, '   ', {speaker: 2, language: 'en'});
    // No text at all, on a round that is outstanding.
    const round = speakRound(page);
    sendHostMessage(win, {type: 'voiceSpeech', language: 'en', round});
    assert.strictEqual(inp.value, 'precious draft');
    assert.strictEqual(counters.input, 0);
    assert.strictEqual(counters.submit, 0);
  });

  windows.forEach(win => win.close());
  report('voiceSpeakerLanguagePrefix');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
