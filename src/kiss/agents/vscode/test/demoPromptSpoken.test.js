// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test that the demo replay handles ``prompt`` events
// specially: whenever a replayed session contains a ``prompt`` event
// (a follow-up message the user sent while the task ran), the demo
// must READ THE PROMPT ALOUD — exactly like a replayed ``talk`` tool
// call — prefixed with the words "User says ":
//
//   1. A ``prompt`` event must be narrated via ``playTalkEvent`` with
//      text ``"User says " + prompt text`` (and still rendered in the
//      output via ``processEvent`` like any other event).
//   2. The replay must PAUSE at the prompt panel until the narration
//      promise resolves — no panel collapse, no result streaming
//      while the speech is playing (same contract as ``talk``).
//   3. Cancelling the demo during an in-flight prompt narration
//      resolves the pending speech promise via ``stopSpeech`` so the
//      paused replay exits immediately.
//   4. A ``prompt`` event with empty text is not spoken (and does not
//      hang the replay).
//   5. ``groupEventsIntoPanels`` gives a ``prompt`` its own panel and
//      the following thinking/text still starts a fresh LLM panel.
//
// Drives the real ``media/demo.js`` inside jsdom (no mocks of project
// code; the ``window._demoApi`` host shim that main.js normally
// provides is stubbed, exactly like demoPauseOnTalk.test.js).
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/demoPromptSpoken.test.js

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
 * ``window._demoApi`` host shim whose speech hooks return promises
 * the test resolves by hand:
 *
 *   - ``playTalkEvent`` returns a pending promise and stashes its
 *     resolver in ``pending`` (unless ``opts.instantSpeech``, in
 *     which case it resolves immediately);
 *   - ``stopSpeech`` resolves every pending promise, mirroring the
 *     real discard behavior in main.js.
 *
 * ``calls`` records every interesting host-api invocation in order.
 */
function makeDemoWindow(events, opts) {
  const instantSpeech = !!(opts && opts.instantSpeech);
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body><div id="output"></div></body></html>',
    {runScripts: 'dangerously', pretendToBeVisual: true},
  );
  const win = dom.window;
  let active = false;
  const calls = [];
  const pending = [];
  const api = {
    get active() {
      return active;
    },
    set active(v) {
      active = !!v;
    },
    resolveEvents: null,
    setInput() {},
    clearInput() {},
    clearForReplay() {},
    resetOutputState() {},
    processEvent(ev) {
      calls.push({fn: 'processEvent', ev});
    },
    setTaskText() {},
    updateTabTitle() {},
    hideWelcome() {},
    scrollToBottom() {},
    getActiveTabId() {
      return 'demo-tab';
    },
    sendMessage(msg) {
      calls.push({fn: 'sendMessage', msg});
      if (msg && msg.type === 'resumeSession') {
        setTimeout(() => {
          if (api.resolveEvents) api.resolveEvents(events);
        }, 10);
      }
    },
    collapsePanels() {
      calls.push({fn: 'collapsePanels'});
    },
    setRunningState() {},
    showSpinner() {},
    removeSpinner() {},
    speakText(text, language) {
      calls.push({fn: 'speakText', text, language});
      return Promise.resolve();
    },
    playTalkEvent(ev) {
      calls.push({fn: 'playTalkEvent', ev});
      if (instantSpeech) return Promise.resolve();
      return new Promise(resolve => {
        pending.push({kind: 'playTalkEvent', ev, resolve});
      });
    },
    openSubagentTab(ev) {
      calls.push({fn: 'openSubagentTab', ev});
    },
    stopSpeech() {
      calls.push({fn: 'stopSpeech'});
      // Mirror main.js: discarding queued + in-flight jobs resolves
      // the promises the paused replay is awaiting.
      while (pending.length) pending.shift().resolve();
    },
  };
  win._demoApi = api;
  win.eval(fs.readFileSync(DEMO_PATH, 'utf8'));
  return {win, api, calls, pending};
}

const PROMPT_TEXT = 'please also add documentation';

/**
 * One session: an LLM text panel, a user follow-up prompt, another
 * LLM text panel, then a result.
 */
function promptEvents() {
  return [
    {type: 'text_delta', text: 'Working on the parser...'},
    {type: 'prompt', text: PROMPT_TEXT},
    {type: 'text_delta', text: 'Sure, adding documentation too.'},
    {
      type: 'result',
      success: true,
      summary: 'REPLAY-FINISHED-MARKER',
      total_tokens: 3,
      cost: '$0.00',
    },
  ];
}

function startReplay(win, preview) {
  return win._startDemoReplay([
    {id: 1, has_events: true, preview: preview, timestamp: 1},
  ]);
}

function resultRendered(win) {
  return win.document
    .getElementById('output')
    .textContent.includes('REPLAY-FINISHED-MARKER');
}

async function testPromptSpokenWithUserSaysPrefix() {
  const {win, api, calls} = makeDemoWindow(promptEvents(), {
    instantSpeech: true,
  });
  await startReplay(win, 'prompt narration');

  const spoken = calls.filter(c => c.fn === 'playTalkEvent');
  assert.strictEqual(
    spoken.length,
    1,
    'exactly one narration for the single prompt event (got ' +
      spoken.length +
      ')',
  );
  assert.strictEqual(
    spoken[0].ev.text,
    'User says ' + PROMPT_TEXT,
    'prompt narration must be prefixed with "User says "',
  );
  // The prompt event is still rendered in the output like any other.
  assert.ok(
    calls.some(c => c.fn === 'processEvent' && c.ev.type === 'prompt'),
    'prompt event must still be rendered via processEvent',
  );
  assert.ok(resultRendered(win), 'replay completes after narrating');
  assert.strictEqual(api.active, false, 'demo deactivates after the replay');
  win.close();
  console.log('  ok - prompt event narrated with "User says " prefix');
}

async function testReplayPausesUntilPromptNarrationEnds() {
  const {win, api, calls, pending} = makeDemoWindow(promptEvents());
  const replay = startReplay(win, 'pause on prompt');

  // The prompt narration starts playing...
  await waitFor(
    () => calls.some(c => c.fn === 'playTalkEvent'),
    5000,
    'prompt narration to start',
  );
  // ...and while its promise is unresolved the replay must stay
  // paused: no result streaming — even long after the usual 500ms
  // panel pause would have elapsed.
  await sleep(1200);
  assert.ok(
    !resultRendered(win),
    'replay must not stream the result while the prompt narration plays',
  );

  // End the narration — the replay resumes and runs to completion.
  await waitFor(() => pending.length >= 1, 1000, 'narration registration');
  assert.strictEqual(pending[0].ev.text, 'User says ' + PROMPT_TEXT);
  pending.shift().resolve();
  await replay;
  assert.ok(resultRendered(win), 'replay completes after the narration ends');
  assert.strictEqual(api.active, false);
  win.close();
  console.log('  ok - replay pauses at a prompt until the narration ends');
}

async function testCancelDuringPromptNarrationExitsImmediately() {
  const {win, api, calls, pending} = makeDemoWindow(promptEvents());
  const replay = startReplay(win, 'cancel mid prompt');

  await waitFor(
    () => calls.some(c => c.fn === 'playTalkEvent'),
    5000,
    'prompt narration to start',
  );

  // Cancel while the replay is paused awaiting the narration:
  // _cancelDemoReplay -> stopSpeech resolves the pending promise, so
  // the paused coroutine wakes up, sees cancelRequested and exits.
  win._cancelDemoReplay();
  await replay;
  assert.ok(
    calls.some(c => c.fn === 'stopSpeech'),
    'cancel must stop the in-flight prompt narration',
  );
  assert.ok(
    !resultRendered(win),
    'cancelled replay must not stream the result',
  );
  assert.strictEqual(api.active, false, 'cancelled replay clears active');
  assert.strictEqual(pending.length, 0, 'no speech promise left dangling');
  win.close();
  console.log('  ok - cancel during a prompt narration exits immediately');
}

async function testEmptyPromptNotSpoken() {
  const events = [
    {type: 'prompt', text: ''},
    {
      type: 'result',
      success: true,
      summary: 'REPLAY-FINISHED-MARKER',
      total_tokens: 3,
      cost: '$0.00',
    },
  ];
  const {win, api, calls} = makeDemoWindow(events);
  await startReplay(win, 'empty prompt');
  assert.strictEqual(
    calls.filter(c => c.fn === 'playTalkEvent').length,
    0,
    'an empty prompt must not be narrated',
  );
  assert.ok(resultRendered(win), 'replay completes without hanging');
  assert.strictEqual(api.active, false);
  win.close();
  console.log('  ok - empty prompt is not narrated and never hangs');
}

async function testPromptGetsItsOwnPanel() {
  const {win} = makeDemoWindow([]);
  const groups = win._groupEventsIntoPanels(promptEvents());
  assert.strictEqual(
    groups.length,
    4,
    'expected [llm][prompt][llm][result] panels, got ' + groups.length,
  );
  assert.strictEqual(groups[0].length, 1);
  assert.strictEqual(groups[0][0].type, 'text_delta');
  assert.strictEqual(groups[1].length, 1, 'prompt must be its own panel');
  assert.strictEqual(groups[1][0].type, 'prompt');
  assert.strictEqual(
    groups[2][0].type,
    'text_delta',
    'thinking/text after a prompt must start a fresh LLM panel',
  );
  assert.strictEqual(groups[3][0].type, 'result');
  win.close();
  console.log('  ok - a prompt event is grouped into its own panel');
}

async function runTests() {
  await testPromptSpokenWithUserSaysPrefix();
  await testReplayPausesUntilPromptNarrationEnds();
  await testCancelDuringPromptNarrationExitsImmediately();
  await testEmptyPromptNotSpoken();
  await testPromptGetsItsOwnPanel();
}

runTests().then(
  () => {
    console.log('\n5 passed, 0 failed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    process.exit(1);
  },
);
