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
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'voice.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function input(win) {
  return win.document.getElementById('task-input');
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

test('spoken task is submitted to the idle highlighted tab', () => {
  const {win, posted} = makeWebview();
  posted.length = 0;
  send(win, {type: 'voiceWake'});
  send(win, {type: 'voiceSpeech', text: 'Fix the parser bug', speaker: 1});
  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(submits.length, 1, JSON.stringify(posted));
  assert.strictEqual(
    submits[0].prompt,
    'Speaker #1 says that: Fix the parser bug',
  );
  assert.strictEqual(posted.filter(m => m.type === 'appendUserMessage').length, 0);
  assert.strictEqual(input(win).value, '');
});

test('spoken task steers a running agent as a user message', () => {
  const {win, posted} = makeWebview();
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

test('spoken task with a language submits the language-aware prompt', () => {
  const {win, posted} = makeWebview();
  posted.length = 0;
  send(win, {type: 'voiceWake'});
  send(win, {
    type: 'voiceSpeech',
    text: 'Fix the parser bug',
    speaker: 1,
    language: 'fr',
  });
  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(submits.length, 1, JSON.stringify(posted));
  assert.strictEqual(
    submits[0].prompt,
    'Speaker #1 says in the language fr that: Fix the parser bug',
  );
  assert.strictEqual(input(win).value, '');
});

test('spoken task with a language steers a running agent', () => {
  const {win, posted} = makeWebview();
  send(win, {type: 'status', running: true});
  posted.length = 0;
  send(win, {
    type: 'voiceSpeech',
    text: 'Also update the docs',
    speaker: 2,
    language: 'es',
  });
  const steers = posted.filter(m => m.type === 'appendUserMessage');
  assert.strictEqual(steers.length, 1, JSON.stringify(posted));
  assert.strictEqual(
    steers[0].prompt,
    'Speaker #2 says in the language es that: Also update the docs',
  );
  assert.strictEqual(posted.filter(m => m.type === 'submit').length, 0);
  assert.strictEqual(input(win).value, '');
});

test('legacy speech without a speaker still submits unprefixed', () => {
  const {win, posted} = makeWebview();
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'Run all tests'});
  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(submits.length, 1, JSON.stringify(posted));
  assert.strictEqual(submits[0].prompt, 'Run all tests');
  assert.strictEqual(input(win).value, '');
});

test('empty translation submits nothing and keeps a draft intact', () => {
  const {win, posted} = makeWebview();
  input(win).value = 'precious draft';
  posted.length = 0;
  send(win, {type: 'voiceWake'});
  send(win, {type: 'voiceSpeech', text: '', speaker: 1});
  assert.strictEqual(
    posted.filter(
      m => m.type === 'submit' || m.type === 'appendUserMessage',
    ).length,
    0,
    JSON.stringify(posted),
  );
  assert.strictEqual(input(win).value, 'precious draft');
});

test('a spoken task merges with an existing draft on submit', () => {
  const {win, posted} = makeWebview();
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
  const {win, posted} = makeWebview();
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'first task', speaker: 1});
  send(win, {type: 'voiceSpeech', text: 'second task', speaker: 2});
  const prompts = posted
    .filter(m => m.type === 'submit')
    .map(m => m.prompt);
  assert.deepStrictEqual(prompts, [
    'Speaker #1 says that: first task',
    'Speaker #2 says that: second task',
  ]);
});

console.log(`\n${passed} passed, ${failures.length} failed`);
process.exit(failures.length > 0 ? 1 : 0);
