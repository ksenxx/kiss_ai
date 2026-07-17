// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end integration test: when the agent asks the user a question
// (ask-user modal), speech spoken after the "Sorcar" wake word must be
// sent as the ANSWER to that question (a ``userAnswer`` message) rather
// than as a new task, and the ask-user panel must show its own mic
// button whose state mirrors the main #voice-btn and whose click
// toggles wake-word listening exactly like the main mic.
//
// Runs the REAL production ``media/main.js`` and ``media/voice.js``
// together in one jsdom webview (only the vscode host API is a
// recording stub, as in every webview test).  Run with:
//
//     node test/voiceAskUserAnswer.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview plus the
 * production voice.js in webview mode.  Returns ``{win, posted}``
 * where ``posted`` records every message posted to the vscode host.
 */
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
  // The mic is closed by default (fresh install); seed the explicit
  // opt-in so wake-word listening auto-enables for these tests.
  win.localStorage.setItem('kissVoiceEnabled', '1');
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
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

function modal(win) {
  return win.document.getElementById('ask-user-modal');
}

function askInput(win) {
  return modal(win).querySelector('.ask-user-input');
}

function askMic(win) {
  return modal(win).querySelector('.ask-user-mic');
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

// ---------------------------------------------------------------------------

test('the ask-user panel shows a mic button along with question/input/submit', () => {
  const {win} = makeWebview();
  send(win, {type: 'askUser', question: 'Which color?'});
  assert.strictEqual(modal(win).style.display, 'flex');
  assert.ok(
    modal(win).querySelector('.ask-user-question'),
    'question element must stay',
  );
  assert.ok(askInput(win), 'answer textarea must stay');
  assert.ok(
    modal(win).querySelector('.ask-user-submit'),
    'submit button must stay',
  );
  const mic = askMic(win);
  assert.ok(mic, 'ask-user panel must contain a .ask-user-mic button');
  assert.strictEqual(mic.tagName, 'BUTTON');
  assert.ok(mic.querySelector('svg'), 'mic button must show the mic icon');
});

test('speech while a question is pending is sent as the userAnswer', () => {
  const {win, posted} = makeWebview();
  send(win, {type: 'askUser', question: 'Which color?'});
  const tabId = win._demoApi.getActiveTabId();
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
  assert.strictEqual(taskInput(win).value, '', 'task input must stay empty');
  assert.strictEqual(modal(win).style.display, 'none', 'modal must close');
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

test('an empty translation never answers and keeps the modal open', () => {
  const {win, posted} = makeWebview();
  send(win, {type: 'askUser', question: 'Which color?'});
  askInput(win).value = 'precious draft';
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
  assert.strictEqual(modal(win).style.display, 'flex', 'modal must stay open');
  assert.strictEqual(askInput(win).value, 'precious draft');
});

test('spoken answer merges with a typed draft in the answer box', () => {
  const {win, posted} = makeWebview();
  send(win, {type: 'askUser', question: 'Which color?'});
  askInput(win).value = 'partial';
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

test('clicking the ask-user mic toggles wake-word listening like the main mic', () => {
  const {win, posted} = makeWebview();
  send(win, {type: 'askUser', question: 'Which color?'});
  // The mic auto-enables at load, so the first click must turn it OFF.
  posted.length = 0;
  askMic(win).click();
  let toggles = posted.filter(m => m.type === 'voiceToggle');
  assert.strictEqual(toggles.length, 1, JSON.stringify(posted));
  assert.strictEqual(toggles[0].enabled, false);
  posted.length = 0;
  askMic(win).click();
  toggles = posted.filter(m => m.type === 'voiceToggle');
  assert.strictEqual(toggles.length, 1, JSON.stringify(posted));
  assert.strictEqual(toggles[0].enabled, true);
});

test('the ask-user mic mirrors listening / wake / transcribing states', () => {
  const {win} = makeWebview();
  send(win, {type: 'askUser', question: 'Which color?'});
  const btn = win.document.getElementById('voice-btn');
  const mic = askMic(win);
  send(win, {type: 'voiceState', listening: true});
  assert.ok(btn.classList.contains('voice-listening'));
  assert.ok(
    mic.classList.contains('voice-listening'),
    'ask mic must mirror voice-listening',
  );
  send(win, {type: 'voiceWake'});
  assert.ok(btn.classList.contains('voice-triggered'));
  assert.ok(
    mic.classList.contains('voice-triggered'),
    'ask mic must mirror the red wake flash',
  );
  send(win, {type: 'voiceTranscribing'});
  assert.ok(btn.classList.contains('voice-transcribing'));
  assert.ok(
    mic.classList.contains('voice-transcribing'),
    'ask mic must mirror the yellow transcribing flash',
  );
  assert.ok(!mic.classList.contains('voice-triggered'));
});

test('a mic mounted after the state changed still shows the current state', () => {
  const {win} = makeWebview();
  send(win, {type: 'voiceState', listening: true});
  send(win, {type: 'askUser', question: 'Which color?'});
  const mic = askMic(win);
  assert.ok(
    mic.classList.contains('voice-listening'),
    'a late-mounted ask mic must sync to the live voice state',
  );
});

test('the wake word focuses the answer box while a question is pending', () => {
  const {win} = makeWebview();
  send(win, {type: 'askUser', question: 'Which color?'});
  send(win, {type: 'voiceWake'});
  assert.strictEqual(
    win.document.activeElement,
    askInput(win),
    'wake must focus the ask-user answer box, not the task input',
  );
});

test('after askUserDone later speech goes back to the task input path', () => {
  const {win, posted} = makeWebview();
  send(win, {type: 'askUser', question: 'Which color?'});
  send(win, {type: 'askUserDone'});
  assert.strictEqual(modal(win).style.display, 'none');
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
  // A second utterance must be a fresh task, not another answer.
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'now run tests', speaker: 1});
  assert.strictEqual(posted.filter(m => m.type === 'userAnswer').length, 0);
  assert.strictEqual(posted.filter(m => m.type === 'submit').length, 1);
});

// ---------------------------------------------------------------------------

console.log(`\n${passed} passed, ${failures.length} failed`);
// main.js starts webview timers (status ticks) that keep the node
// event loop alive; exit explicitly once the verdict is printed.
process.exit(failures.length > 0 ? 1 : 0);
