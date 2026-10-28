// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage: makeChatPage, send, stubRaf} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {test, report} = makeRunner();

function stubVoiceConfig(win) {
  stubRaf(win);
  win.__VOICE__ = {};
}

// Every page built; closed after the run so node exits on its own.
const windows = [];

function makePage() {
  const page = makeChatPage({
    beforeScripts: stubVoiceConfig,
    extraScripts: ['voice.js'],
  });
  windows.push(page.win);
  return page;
}

function input(win) {
  return win.document.getElementById('task-input');
}


test('spoken task is submitted to the idle highlighted tab', () => {
  const {win, posted} = makePage();
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'Fix the parser bug', speaker: 1});
  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(submits.length, 1, JSON.stringify(posted));
  assert.strictEqual(
    submits[0].prompt,
    'Speaker #1 says that: Fix the parser bug',
  );
  assert.strictEqual(
    posted.filter(m => m.type === 'appendUserMessage').length,
    0,
  );
  assert.strictEqual(input(win).value, '');
});

test('spoken task steers a running agent as a user message', () => {
  const {win, posted} = makePage();
  send(win, {type: 'status', running: true});
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'Also update the docs', speaker: 2});
  const steers = posted.filter(m => m.type === 'appendUserMessage');
  assert.strictEqual(steers.length, 1, JSON.stringify(posted));
  assert.strictEqual(
    steers[0].prompt,
    'Speaker #2 says that: Also update the docs',
  );
  assert.strictEqual(posted.filter(m => m.type === 'submit').length, 0);
  assert.strictEqual(input(win).value, '');
});

// The language-aware prefix ("says in the language xx that:") is
// voiceSpeakerLanguagePrefix's; submit-vs-steer routing does not depend
// on the language.

test('legacy speech without a speaker still submits unprefixed', () => {
  const {win, posted} = makePage();
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'Run all tests'});
  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(submits.length, 1, JSON.stringify(posted));
  assert.strictEqual(submits[0].prompt, 'Run all tests');
  assert.strictEqual(input(win).value, '');
});

test('empty translation submits nothing and keeps a draft intact', () => {
  const {win, posted} = makePage();
  input(win).value = 'precious draft';
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: '', speaker: 1});
  assert.strictEqual(
    posted.filter(m => m.type === 'submit' || m.type === 'appendUserMessage')
      .length,
    0,
    JSON.stringify(posted),
  );
  assert.strictEqual(input(win).value, 'precious draft');
});

test('a spoken task merges with an existing draft on submit', () => {
  const {win, posted} = makePage();
  input(win).value = 'Context: parser module.';
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'Fix it', speaker: 1});
  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(submits.length, 1, JSON.stringify(posted));
  assert.strictEqual(
    submits[0].prompt,
    'Context: parser module. Speaker #1 says that: Fix it',
  );
  assert.strictEqual(input(win).value, '');
});

test('two speakers submit two distinct prefixed tasks', () => {
  const {win, posted} = makePage();
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'first task', speaker: 1});
  send(win, {type: 'voiceSpeech', text: 'second task', speaker: 2});
  const prompts = posted.filter(m => m.type === 'submit').map(m => m.prompt);
  assert.deepStrictEqual(prompts, [
    'Speaker #1 says that: first task',
    'Speaker #2 says that: second task',
  ]);
});

windows.forEach(win => win.close());
report('voiceSubmitToAgent');
