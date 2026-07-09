// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the demo-mode host hooks in ``media/main.js``
// (``window._demoApi``): demo replay must be able to
//
//   * read a user prompt aloud (``speakText``) through the Web Speech
//     API,
//   * actually play a replayed ``talk`` tool call (``playTalkEvent``),
//   * actually materialise sub-agent tabs for a replayed
//     ``run_parallel`` fan-out (``openSubagentTab``), and
//   * cancel queued speech (``stopSpeech``) when the demo is stopped.
//
// Drives the real ``chat.html`` + ``main.js`` inside jsdom, exactly
// like talkTool.test.js (no mocks of project code).
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/demoApiHooks.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview: the real
 * ``chat.html`` body (placeholders blanked), ``panelCopy.js`` and
 * ``main.js`` evaluated in the window, and a recording
 * ``acquireVsCodeApi`` stub (the only host API the webview has).
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

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

/**
 * Install a recording Web Speech API on *win* (jsdom has none).  Each
 * spoken utterance's ``onend`` fires immediately so the serialized
 * talk queue advances to the next job.  Returns {spoken, cancels}.
 */
function installSpeech(win, opts) {
  const complete = !opts || opts.complete !== false;
  const spoken = [];
  const cancels = [];
  win.SpeechSynthesisUtterance = function SpeechSynthesisUtterance(text) {
    this.text = text;
    this.lang = '';
  };
  win.speechSynthesis = {
    speak: u => {
      spoken.push(u);
      if (complete && typeof u.onend === 'function') u.onend();
    },
    cancel: () => cancels.push(1),
  };
  return {spoken, cancels};
}

function spokenText(spoken) {
  return spoken.map(u => u.text).join(' ');
}

function testSpeakTextReadsUserPrompt() {
  const {win} = makeWebview();
  const {spoken} = installSpeech(win);

  win._demoApi.speakText('User said plan my trip', 'en-US');

  assert.ok(spoken.length >= 1, 'speakText must speak at least once');
  assert.ok(
    spokenText(spoken).includes('User said plan my trip'),
    'the prompt narration must be spoken verbatim',
  );
  assert.strictEqual(spoken[0].lang, 'en-US');
  console.log('PASS: speakText narrates the user prompt aloud');
}

function testPlayTalkEventSpeaksRecordedTalk() {
  const {win} = makeWebview();
  const {spoken} = installSpeech(win);

  win._demoApi.playTalkEvent({
    text: 'Hello there, I am on it.',
    language: 'en-GB',
    emotion: 'cheerful',
  });

  assert.ok(spoken.length >= 1, 'playTalkEvent must speak');
  assert.ok(spokenText(spoken).includes('Hello there'));
  assert.strictEqual(spoken[0].lang, 'en-GB');
  console.log('PASS: playTalkEvent speaks a replayed talk tool call');
}

function testQueuedSpeechIsSerialized() {
  const {win} = makeWebview();
  const {spoken} = installSpeech(win);

  win._demoApi.speakText('User said first task');
  win._demoApi.playTalkEvent({text: 'Second utterance here.'});

  const all = spokenText(spoken);
  assert.ok(all.includes('first task'), 'first narration spoken');
  assert.ok(all.includes('Second utterance'), 'queued talk spoken after');
  assert.ok(
    all.indexOf('first task') < all.indexOf('Second utterance'),
    'talk queue must preserve order',
  );
  console.log('PASS: demo speech is serialized through the talk queue');
}

function testStopSpeechCancelsQueue() {
  const {win} = makeWebview();
  // Speech that never completes — utterances pile up in the queue.
  const {spoken, cancels} = installSpeech(win, {complete: false});

  win._demoApi.speakText('User said a very long prompt');
  win._demoApi.playTalkEvent({text: 'Never reached.'});
  assert.ok(spoken.length >= 1, 'first job started');

  win._demoApi.stopSpeech();
  assert.ok(cancels.length >= 1, 'stopSpeech must cancel speech synthesis');

  // The queue must be RELEASED, not just cleared: engines may fire
  // neither onend nor onerror for cancelled utterances, so a stuck
  // busy flag would silence every future talk playback.
  const before = spoken.length;
  win._demoApi.speakText('User said speak again after cancel');
  assert.ok(
    spoken.length > before,
    'talk queue must accept and start new speech after stopSpeech',
  );
  assert.ok(
    spokenText(spoken).includes('speak again after cancel'),
    'post-cancel narration must actually be spoken',
  );
  console.log('PASS: stopSpeech cancels queued demo speech');
}

function testOpenSubagentTabMaterialisesTab() {
  const {win} = makeWebview();
  installSpeech(win);
  const parentId = win._demoApi.getActiveTabId();

  // Demo replay always sets the running state before processing
  // events (see _startDemoReplay) — a non-running task renders its
  // panels collapsed, and a collapsed fan-out panel keeps its
  // sub-agent tabs closed by design.
  win._demoApi.setRunningState(true);

  // Render the fan-out's tool-call panel first (like demo replay does),
  // then materialise the sub-agent tabs through the real handler.
  win._demoApi.processEvent({
    type: 'tool_call',
    name: 'run_parallel',
    extras: {tasks: '["research topic A", "summarize topic B"]'},
  });
  win._demoApi.openSubagentTab({
    type: 'openSubagentTab',
    tab_id: 'demo-sub-1-0',
    parent_tab_id: parentId,
    description: 'research topic A',
    taskIndex: 0,
    isDone: false,
  });
  win._demoApi.openSubagentTab({
    type: 'openSubagentTab',
    tab_id: 'demo-sub-1-1',
    parent_tab_id: parentId,
    description: 'summarize topic B',
    taskIndex: 1,
    isDone: false,
  });

  const subTabs = win.document.querySelectorAll('.subagent-tab');
  assert.strictEqual(subTabs.length, 2, 'two sub-agent tabs in the tab bar');
  const titles = Array.from(subTabs).map(t => t.textContent);
  assert.ok(
    titles.some(t => t.includes('1. research topic A')),
    'first sub-agent tab titled with its 1-based task index',
  );
  assert.ok(
    titles.some(t => t.includes('2. summarize topic B')),
    'second sub-agent tab titled with its 1-based task index',
  );

  // The fan-out panel exists and owns the run_parallel accent class.
  const panel = win.document.querySelector('.tc-run-parallel');
  assert.ok(panel, 'run_parallel tool-call panel rendered');
  console.log('PASS: openSubagentTab materialises sub-agent tabs');
}

function testOpenSubagentTabUnknownParentIgnored() {
  const {win} = makeWebview();
  installSpeech(win);
  win._demoApi.setRunningState(true);

  win._demoApi.openSubagentTab({
    type: 'openSubagentTab',
    tab_id: 'demo-sub-9-0',
    parent_tab_id: 'not-a-local-tab',
    description: 'phantom',
    taskIndex: 0,
    isDone: false,
  });

  assert.strictEqual(
    win.document.querySelectorAll('.subagent-tab').length,
    0,
    'a fan-out for an unknown parent tab must not create phantom tabs',
  );
  console.log('PASS: openSubagentTab ignores unknown parent tabs');
}

function runTests() {
  testSpeakTextReadsUserPrompt();
  testPlayTalkEventSpeaksRecordedTalk();
  testQueuedSpeechIsSerialized();
  testStopSpeechCancelsQueue();
  testOpenSubagentTabMaterialisesTab();
  testOpenSubagentTabUnknownParentIgnored();
  console.log('All demoApiHooks tests passed.');
  // setRunningState(true) starts webview timers that keep the node
  // event loop alive — exit explicitly once every assertion passed.
  process.exit(0);
}

runTests();
