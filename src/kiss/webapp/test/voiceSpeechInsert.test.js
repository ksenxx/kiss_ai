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
async function makePage() {
  const inputEvents = {count: 0};
  const page = makeBareVoicePage({}, w => {
    w.document
      .getElementById('task-input')
      .addEventListener('input', () => inputEvents.count++);
  });
  windows.push(page.win);
  await startPipeline(page);
  return {page, win: page.win, inputEvents};
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
    const {page, win, inputEvents} = await makePage();
    const inp = win.document.getElementById('task-input');
    // Tagged with a round that is outstanding, so only the payload can
    // be what keeps them out.
    const round = speakRound(page);
    sendHostMessage(win, {type: 'voiceSpeech', text: 42, round});
    assert.strictEqual(inp.value, '');
    sendHostMessage(win, {type: 'voiceSpeech', round});
    assert.strictEqual(inp.value, '');
    assert.strictEqual(inputEvents.count, 0);
  });

  windows.forEach(win => win.close());
  report('voiceSpeechInsert');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
