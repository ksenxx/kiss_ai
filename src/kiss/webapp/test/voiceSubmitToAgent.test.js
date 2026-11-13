// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {send} = require('./simplify2_harness');
const {makeVoicePage, startPipeline, dictate} = require('./voice_page_helpers');
const {makeRunner} = require('./ui_antipattern_harness');
const {test, report} = makeRunner();

// Every page built; closed after the run so node exits on its own.
const windows = [];

// A chat page with the voice pipeline live: voice.js types only the
// transcript of a round it posted, so every test dictates a real one.
async function makePage() {
  const page = makeVoicePage();
  windows.push(page.win);
  await startPipeline(page);
  return page;
}

function input(win) {
  return win.document.getElementById('task-input');
}

async function main() {
  await test('spoken task is submitted to the idle highlighted tab', async () => {
    const page = await makePage();
    const {win, posted} = page;
    posted.length = 0;
    dictate(page, 'Fix the parser bug', {speaker: 1});
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

  await test('spoken task steers a running agent as a user message', async () => {
    const page = await makePage();
    const {win, posted} = page;
    send(win, {type: 'status', running: true});
    posted.length = 0;
    dictate(page, 'Also update the docs', {speaker: 2});
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

  await test('speech without a speaker still submits unprefixed', async () => {
    const page = await makePage();
    const {win, posted} = page;
    posted.length = 0;
    dictate(page, 'Run all tests');
    const submits = posted.filter(m => m.type === 'submit');
    assert.strictEqual(submits.length, 1, JSON.stringify(posted));
    assert.strictEqual(submits[0].prompt, 'Run all tests');
    assert.strictEqual(input(win).value, '');
  });

  await test('empty translation submits nothing and keeps a draft intact', async () => {
    const page = await makePage();
    const {win, posted} = page;
    input(win).value = 'precious draft';
    posted.length = 0;
    dictate(page, '', {speaker: 1});
    assert.strictEqual(
      posted.filter(m => m.type === 'submit' || m.type === 'appendUserMessage')
        .length,
      0,
      JSON.stringify(posted),
    );
    assert.strictEqual(input(win).value, 'precious draft');
  });

  await test('a spoken task merges with an existing draft on submit', async () => {
    const page = await makePage();
    const {win, posted} = page;
    input(win).value = 'Context: parser module.';
    posted.length = 0;
    dictate(page, 'Fix it', {speaker: 1});
    const submits = posted.filter(m => m.type === 'submit');
    assert.strictEqual(submits.length, 1, JSON.stringify(posted));
    assert.strictEqual(
      submits[0].prompt,
      'Context: parser module. Speaker #1 says that: Fix it',
    );
    assert.strictEqual(input(win).value, '');
  });

  await test('two speakers submit two distinct prefixed tasks', async () => {
    const page = await makePage();
    const {posted} = page;
    posted.length = 0;
    dictate(page, 'first task', {speaker: 1});
    dictate(page, 'second task', {speaker: 2});
    const prompts = posted.filter(m => m.type === 'submit').map(m => m.prompt);
    assert.deepStrictEqual(prompts, [
      'Speaker #1 says that: first task',
      'Speaker #2 says that: second task',
    ]);
  });

  windows.forEach(win => win.close());
  report('voiceSubmitToAgent');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
