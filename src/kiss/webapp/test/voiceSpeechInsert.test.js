// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makeRunner} = require('./ui_antipattern_harness');
const {send: sendHostMessage} = require('./simplify2_harness');
const {makeBareVoiceWindow} = require('./voice_page_helpers');
const {test, report} = makeRunner();

// A page with no microphone, so no wake word ever opens a round: every
// voiceSpeech here takes the no-round path, which types the text into
// the composer.  The round path is covered by conc2026_voice_rounds.
function makeWindow() {
  const inputEvents = {count: 0};
  const win = makeBareVoiceWindow({}, w => {
    w.document
      .getElementById('task-input')
      .addEventListener('input', () => inputEvents.count++);
  });
  return {win, inputEvents};
}

test('voiceSpeech types only the translated text', () => {
  const {win, inputEvents} = makeWindow();
  const inp = win.document.getElementById('task-input');
  assert.strictEqual(inp.value, '');
  sendHostMessage(win, {type: 'voiceSpeech', text: 'Hello everyone'});
  assert.strictEqual(inp.value, 'Hello everyone');
  assert.strictEqual(inputEvents.count, 1);
});

test('voiceSpeech appends to an existing user draft with a space', () => {
  const {win} = makeWindow();
  const inp = win.document.getElementById('task-input');
  inp.value = 'draft text';
  sendHostMessage(win, {type: 'voiceSpeech', text: 'and more'});
  assert.strictEqual(inp.value, 'draft text and more');
});

test('voiceSpeech trims surrounding whitespace from the translation', () => {
  const {win} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceSpeech', text: '  Bonjour to you \n'});
  assert.strictEqual(inp.value, 'Bonjour to you');
});

test('empty voiceSpeech leaves the input empty', () => {
  const {win, inputEvents} = makeWindow();
  const inp = win.document.getElementById('task-input');
  assert.strictEqual(inp.value, '');
  sendHostMessage(win, {type: 'voiceSpeech', text: ''});
  assert.strictEqual(inp.value, '');
  assert.strictEqual(inputEvents.count, 0);
});

test('blank voiceSpeech never touches other user text', () => {
  const {win, inputEvents} = makeWindow();
  const inp = win.document.getElementById('task-input');
  inp.value = 'precious draft';
  sendHostMessage(win, {type: 'voiceSpeech', text: '   '});
  assert.strictEqual(inp.value, 'precious draft');
  assert.strictEqual(inputEvents.count, 0);
});

test('non-string voiceSpeech payloads are treated as no speech', () => {
  const {win, inputEvents} = makeWindow();
  const inp = win.document.getElementById('task-input');
  sendHostMessage(win, {type: 'voiceSpeech', text: 42});
  assert.strictEqual(inp.value, '');
  sendHostMessage(win, {type: 'voiceSpeech'});
  assert.strictEqual(inp.value, '');
  assert.strictEqual(inputEvents.count, 0);
});

report('voiceSpeechInsert');
