// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test that the demo replay no longer performs "Step A"
// (prompt display + narration) before replaying a task's events:
//
//   1. The recorded task text must NOT be typed into the input box
//      (no ``setInput`` call with the task text) and must NOT be
//      cleared afterwards (no ``clearInput`` call).
//   2. No "User said ..." narration must be spoken (no ``speakText``
//      call at all).
//   3. The replay must request the session events promptly — without
//      the old 2-second prompt display pause.
//   4. The replay still runs to completion: the result panel streams
//      and the demo deactivates.
//
// Drives the real ``media/demo.js`` inside jsdom (no mocks of project
// code; the ``window._demoApi`` host shim that main.js normally
// provides is stubbed, exactly like demoPauseOnTalk.test.js).
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/demoNoPromptDisplay.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const DEMO_PATH = path.join(__dirname, '..', 'media', 'demo.js');

function sleep(ms) {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

/** Poll until *pred* returns true or *timeoutMs* elapses. */
async function waitFor(pred, timeoutMs, what) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (pred()) return;
    await sleep(20);
  }
  throw new Error('timed out waiting for ' + what);
}

/**
 * Build a jsdom window with the real ``demo.js`` evaluated and a
 * ``window._demoApi`` host shim. ``calls`` records every interesting
 * host-api invocation in order (with a timestamp for pacing checks).
 * Speech hooks resolve immediately so the replay never blocks on
 * audio in this test.
 */
function makeDemoWindow(events) {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body><div id="output"></div></body></html>',
    {runScripts: 'dangerously', pretendToBeVisual: true},
  );
  const win = dom.window;
  let active = false;
  const calls = [];
  const api = {
    get active() {
      return active;
    },
    set active(v) {
      active = !!v;
    },
    resolveEvents: null,
    setInput(text) {
      calls.push({fn: 'setInput', text, t: Date.now()});
    },
    clearInput() {
      calls.push({fn: 'clearInput', t: Date.now()});
    },
    clearForReplay() {},
    resetOutputState() {},
    processEvent(ev) {
      calls.push({fn: 'processEvent', ev, t: Date.now()});
    },
    setTaskText() {},
    updateTabTitle() {},
    hideWelcome() {},
    scrollToBottom() {},
    getActiveTabId() {
      return 'demo-tab';
    },
    sendMessage(msg) {
      calls.push({fn: 'sendMessage', msg, t: Date.now()});
      if (msg && msg.type === 'resumeSession') {
        setTimeout(() => {
          if (api.resolveEvents) api.resolveEvents(events);
        }, 10);
      }
    },
    collapsePanels() {
      calls.push({fn: 'collapsePanels', t: Date.now()});
    },
    setRunningState() {},
    showSpinner() {},
    removeSpinner() {},
    speakText(text, language) {
      calls.push({fn: 'speakText', text, language, t: Date.now()});
      return Promise.resolve();
    },
    playTalkEvent(ev) {
      calls.push({fn: 'playTalkEvent', ev, t: Date.now()});
      return Promise.resolve();
    },
    openSubagentTab(ev) {
      calls.push({fn: 'openSubagentTab', ev, t: Date.now()});
    },
    stopSpeech() {
      calls.push({fn: 'stopSpeech', t: Date.now()});
    },
  };
  win._demoApi = api;
  win.eval(fs.readFileSync(DEMO_PATH, 'utf8'));
  return {win, api, calls};
}

/** One session: an LLM text panel followed by a result. */
function sessionEvents() {
  return [
    {type: 'text_delta', text: 'Working on it...'},
    {
      type: 'result',
      success: true,
      summary: 'REPLAY-FINISHED-MARKER',
      total_tokens: 3,
      cost: '$0.00',
    },
  ];
}

const TASK_TEXT = 'refactor the parser and add tests';

function startReplay(win) {
  return win._startDemoReplay([
    {id: 1, has_events: true, preview: TASK_TEXT, timestamp: 1},
  ]);
}

function resultRendered(win) {
  return win.document
    .getElementById('output')
    .textContent.includes('REPLAY-FINISHED-MARKER');
}

async function testNoPromptDisplayOrNarration() {
  const {win, api, calls} = makeDemoWindow(sessionEvents());
  const start = Date.now();
  const replay = startReplay(win);

  // The session events must be requested promptly — the old Step A
  // held the replay for a 2-second prompt display (plus narration)
  // before sending ``resumeSession``.
  await waitFor(
    () =>
      calls.some(c => c.fn === 'sendMessage' && c.msg.type === 'resumeSession'),
    5000,
    'resumeSession request',
  );
  const resume = calls.find(
    c => c.fn === 'sendMessage' && c.msg.type === 'resumeSession',
  );
  const elapsed = resume.t - start;
  assert.ok(
    elapsed < 1500,
    'events must be requested without the 2s prompt pause (took ' +
      elapsed +
      'ms)',
  );

  await replay;

  // The task text must never be typed into the input box or cleared.
  assert.strictEqual(
    calls.filter(c => c.fn === 'setInput').length,
    0,
    'replay must not type the task text into the input box',
  );
  assert.strictEqual(
    calls.filter(c => c.fn === 'clearInput').length,
    0,
    'replay must not clear the input box',
  );

  // No "User said ..." narration (nor any other speakText call).
  assert.strictEqual(
    calls.filter(c => c.fn === 'speakText').length,
    0,
    'replay must not narrate the prompt',
  );

  // The replay still completes normally.
  assert.ok(resultRendered(win), 'replay streams the result to completion');
  assert.strictEqual(api.active, false, 'demo deactivates after the replay');
  win.close();
  console.log('  ok - replay skips prompt display and narration entirely');
}

async function runTests() {
  await testNoPromptDisplayOrNarration();
}

runTests().then(
  () => {
    console.log('\n1 passed, 0 failed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    process.exit(1);
  },
);
