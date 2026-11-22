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
// No main.js runs here, so nothing submits and the words stay in the
// composer.  Rounds across conversations are conc2026_voice_rounds'.
// `errors` collects what the page's listeners throw: jsdom reports an
// exception of a listener to the window, not to the test that sent the
// message.
async function makePage() {
  const inputEvents = {count: 0};
  const errors = [];
  const page = makeBareVoicePage({}, w => {
    w.document
      .getElementById('task-input')
      .addEventListener('input', () => inputEvents.count++);
    w.addEventListener('error', event => errors.push(String(event.error)));
  });
  windows.push(page.win);
  await startPipeline(page);
  return {page, win: page.win, inputEvents, errors};
}

async function main() {
  await test('voiceSpeech types only the translated text', async () => {
    const {page, win, inputEvents} = await makePage();
    const inp = win.document.getElementById('task-input');
    assert.strictEqual(inp.value, '');
    dictate(page, 'Hello everyone');
    assert.strictEqual(inp.value, 'Hello everyone');
    assert.strictEqual(inputEvents.count, 1);
  });

  await test('voiceSpeech appends to an existing user draft with a space', async () => {
    const {page, win} = await makePage();
    const inp = win.document.getElementById('task-input');
    inp.value = 'draft text';
    dictate(page, 'and more');
    assert.strictEqual(inp.value, 'draft text and more');
  });

  await test('voiceSpeech trims surrounding whitespace from the translation', async () => {
    const {page, win} = await makePage();
    const inp = win.document.getElementById('task-input');
    dictate(page, '  Bonjour to you \n');
    assert.strictEqual(inp.value, 'Bonjour to you');
  });

  await test('empty voiceSpeech leaves the input empty', async () => {
    const {page, win, inputEvents} = await makePage();
    const inp = win.document.getElementById('task-input');
    assert.strictEqual(inp.value, '');
    dictate(page, '');
    assert.strictEqual(inp.value, '');
    assert.strictEqual(inputEvents.count, 0);
  });

  await test('blank voiceSpeech never touches other user text', async () => {
    const {page, win, inputEvents} = await makePage();
    const inp = win.document.getElementById('task-input');
    inp.value = 'precious draft';
    dictate(page, '   ');
    assert.strictEqual(inp.value, 'precious draft');
    assert.strictEqual(inputEvents.count, 0);
  });

  await test('non-string voiceSpeech payloads are treated as no speech', async () => {
    const {page, win, inputEvents, errors} = await makePage();
    const inp = win.document.getElementById('task-input');
    // Tagged with a round that is outstanding, so only the payload can
    // be what keeps them out.
    const round = speakRound(page);
    sendHostMessage(win, {type: 'voiceSpeech', text: 42, round});
    assert.strictEqual(inp.value, '');
    sendHostMessage(win, {type: 'voiceSpeech', round});
    assert.strictEqual(inp.value, '');
    assert.strictEqual(inputEvents.count, 0);
    // They were ignored, not crashed on (a listener that throws types
    // nothing either), and they did not answer the round: its
    // transcript is still typed, once.
    assert.deepStrictEqual(errors, []);
    sendHostMessage(win, {type: 'voiceSpeech', text: 'Hello everyone', round});
    assert.strictEqual(inp.value, 'Hello everyone');
    assert.strictEqual(inputEvents.count, 1);
    assert.deepStrictEqual(errors, []);
  });

  windows.forEach(win => win.close());
  report('voiceSpeechInsert');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
