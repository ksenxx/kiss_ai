// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  answering,
} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {
  makeVoicePage,
  micBtn,
  startPipeline,
  wake,
} = require('./voice_page_helpers');
const {test, report} = makeRunner();

function stubVoiceConfig(win) {
  stubRaf(win);
  win.__VOICE__ = {};
  win.localStorage.setItem('kissVoiceEnabled', '1');
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

function taskInput(win) {
  return win.document.getElementById('task-input');
}

async function main() {
  await test('a pending question puts the composer into answer mode', () => {
    const {win} = makePage();
    send(win, {type: 'askUser', question: 'Which color?'});
    assert.ok(answering(win), 'body.ask-answering must be set');
    assert.strictEqual(
      taskInput(win).placeholder,
      'Type your answer and press Enter',
    );
  });

  await test('speech while a question is pending is sent as the userAnswer', () => {
    const {win, posted} = makePage();
    send(win, {type: 'askUser', question: 'Which color?'});
    const tabId = win._testApi.getActiveTabId();
    posted.length = 0;
    send(win, {type: 'voiceSpeech', text: 'blue', speaker: 1, language: 'fr'});
    const answers = posted.filter(m => m.type === 'userAnswer');
    assert.strictEqual(answers.length, 1, JSON.stringify(posted));
    assert.strictEqual(
      answers[0].answer,
      'Speaker #1 says in the language fr that: blue',
    );
    assert.strictEqual(answers[0].tabId, tabId);
    assert.strictEqual(
      posted.filter(m => m.type === 'submit' || m.type === 'appendUserMessage')
        .length,
      0,
      'the answer must NOT be submitted as a task: ' + JSON.stringify(posted),
    );
    assert.strictEqual(
      taskInput(win).value,
      '',
      'composer is cleared after the answer',
    );
    assert.ok(!answering(win), 'answer mode must end');
  });

  // Speech with no question pending submitting a task is pinned by
  // voiceSubmitToAgent ('spoken task is submitted to the idle highlighted
  // tab') and, after a question closed, by 'after askUserDone later speech
  // goes back to the task input path' below.

  await test('an empty translation never answers and keeps the question open', () => {
    const {win, posted} = makePage();
    send(win, {type: 'askUser', question: 'Which color?'});
    taskInput(win).value = 'precious draft';
    posted.length = 0;
    send(win, {type: 'voiceSpeech', text: '   ', speaker: 1});
    assert.strictEqual(
      posted.filter(
        m =>
          m.type === 'userAnswer' ||
          m.type === 'submit' ||
          m.type === 'appendUserMessage',
      ).length,
      0,
      JSON.stringify(posted),
    );
    assert.ok(answering(win), 'the question must stay open');
    assert.strictEqual(taskInput(win).value, 'precious draft');
  });

  await test('spoken answer merges with a typed draft in the composer', () => {
    const {win, posted} = makePage();
    send(win, {type: 'askUser', question: 'Which color?'});
    taskInput(win).value = 'partial';
    posted.length = 0;
    send(win, {type: 'voiceSpeech', text: 'and blue', speaker: 2});
    const answers = posted.filter(m => m.type === 'userAnswer');
    assert.strictEqual(answers.length, 1, JSON.stringify(posted));
    assert.strictEqual(
      answers[0].answer,
      'partial Speaker #2 says that: and blue',
    );
  });

  await test('speech without a speaker answers with the raw text', () => {
    const {win, posted} = makePage();
    send(win, {type: 'askUser', question: 'Which color?'});
    posted.length = 0;
    send(win, {type: 'voiceSpeech', text: 'plain answer'});
    const answers = posted.filter(m => m.type === 'userAnswer');
    assert.strictEqual(answers.length, 1, JSON.stringify(posted));
    assert.strictEqual(answers[0].answer, 'plain answer');
  });

  await test('the wake word focuses the composer while a question is pending', async () => {
    const page = makeVoicePage();
    windows.push(page.win);
    const {win} = page;
    await startPipeline(page);
    send(win, {type: 'askUser', question: 'Which color?'});
    // The user has gone on to another control by the time they speak.
    micBtn(win).focus();
    assert.strictEqual(win.document.activeElement, micBtn(win));
    wake(page);
    assert.strictEqual(
      win.document.activeElement,
      taskInput(win),
      'wake must focus the composer, which is the answer box',
    );
    assert.ok(answering(win), 'the question stays open');
  });

  await test('after askUserDone later speech goes back to the task input path', () => {
    const {win, posted} = makePage();
    send(win, {type: 'askUser', question: 'Which color?'});
    send(win, {type: 'askUserDone'});
    assert.ok(!answering(win));
    assert.strictEqual(taskInput(win).placeholder, '');
    posted.length = 0;
    send(win, {type: 'voiceSpeech', text: 'next task', speaker: 1});
    const submits = posted.filter(m => m.type === 'submit');
    assert.strictEqual(submits.length, 1, JSON.stringify(posted));
    assert.strictEqual(submits[0].prompt, 'Speaker #1 says that: next task');
    assert.strictEqual(posted.filter(m => m.type === 'userAnswer').length, 0);
  });

  await test('answering by voice clears the pending question for the tab', () => {
    const {win, posted} = makePage();
    send(win, {type: 'askUser', question: 'Which color?'});
    send(win, {type: 'voiceSpeech', text: 'blue', speaker: 1});
    posted.length = 0;
    send(win, {type: 'voiceSpeech', text: 'now run tests', speaker: 1});
    assert.strictEqual(posted.filter(m => m.type === 'userAnswer').length, 0);
    assert.strictEqual(posted.filter(m => m.type === 'submit').length, 1);
  });

  windows.forEach(win => win.close());
  report('voiceAskUserAnswer');
}

main();
