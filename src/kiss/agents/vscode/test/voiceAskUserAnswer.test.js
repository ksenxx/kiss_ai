// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.__VOICE__ = {mode: 'webview'};
  win.localStorage.setItem('kissVoiceEnabled', '1');
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'voice.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function taskInput(win) {
  return win.document.getElementById('task-input');
}

// The composer is in answer mode while the tab on screen has a question.
function answering(win) {
  return win.document.body.classList.contains('ask-answering');
}

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

test('a pending question puts the composer into answer mode', () => {
  const {win} = makeWebview();
  send(win, {type: 'askUser', question: 'Which color?'});
  assert.ok(answering(win), 'body.ask-answering must be set');
  assert.strictEqual(
    taskInput(win).placeholder,
    'Type your answer and press Enter',
  );
});

test('speech while a question is pending is sent as the userAnswer', () => {
  const {win, posted} = makeWebview();
  send(win, {type: 'askUser', question: 'Which color?'});
  const tabId = win._testApi.getActiveTabId();
  posted.length = 0;
  send(win, {type: 'voiceWake'});
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

test('speech without a pending question still submits a task (regression)', () => {
  const {win, posted} = makeWebview();
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'Run tests', speaker: 1});
  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(submits.length, 1, JSON.stringify(posted));
  assert.strictEqual(submits[0].prompt, 'Speaker #1 says that: Run tests');
  assert.strictEqual(posted.filter(m => m.type === 'userAnswer').length, 0);
});

test('an empty translation never answers and keeps the question open', () => {
  const {win, posted} = makeWebview();
  send(win, {type: 'askUser', question: 'Which color?'});
  taskInput(win).value = 'precious draft';
  posted.length = 0;
  send(win, {type: 'voiceWake'});
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

test('spoken answer merges with a typed draft in the composer', () => {
  const {win, posted} = makeWebview();
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

test('speech without a speaker answers with the raw text', () => {
  const {win, posted} = makeWebview();
  send(win, {type: 'askUser', question: 'Which color?'});
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'plain answer'});
  const answers = posted.filter(m => m.type === 'userAnswer');
  assert.strictEqual(answers.length, 1, JSON.stringify(posted));
  assert.strictEqual(answers[0].answer, 'plain answer');
});

test('the wake word focuses the composer while a question is pending', () => {
  const {win} = makeWebview();
  send(win, {type: 'askUser', question: 'Which color?'});
  send(win, {type: 'voiceWake'});
  assert.strictEqual(
    win.document.activeElement,
    taskInput(win),
    'wake must focus the composer, which is the answer box',
  );
});

test('after askUserDone later speech goes back to the task input path', () => {
  const {win, posted} = makeWebview();
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

test('answering by voice clears the pending question for the tab', () => {
  const {win, posted} = makeWebview();
  send(win, {type: 'askUser', question: 'Which color?'});
  send(win, {type: 'voiceSpeech', text: 'blue', speaker: 1});
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'now run tests', speaker: 1});
  assert.strictEqual(posted.filter(m => m.type === 'userAnswer').length, 0);
  assert.strictEqual(posted.filter(m => m.type === 'submit').length, 1);
});

console.log(`\n${passed} passed, ${failures.length} failed`);
process.exit(failures.length > 0 ? 1 : 0);
