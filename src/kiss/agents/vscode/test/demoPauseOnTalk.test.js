// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the "pause the demo while talking" behavior in
// ``media/demo.js`` (fix for the demo-mode infinite loop where queued
// speech lagged ever further behind the visual replay):
//
//   1. The replay must NOT advance past a ``talk`` tool-call panel
//      (no collapse, no result streaming) until the speech promise
//      returned by ``playTalkEvent`` resolves.
//   2. The replay must NOT proceed past the prompt display (no
//      ``resumeSession`` request) until the "User said ..." narration
//      promise returned by ``speakText`` resolves — even well after
//      the 2-second display pause.
//   3. Cancelling the demo during an in-flight talk resolves the
//      pending speech promise via ``stopSpeech`` so the paused replay
//      coroutine exits immediately instead of hanging forever.
//   4. A legacy host api whose speech hooks return undefined (older
//      main.js) still replays to completion without hanging.
//
// Drives the real ``media/demo.js`` inside jsdom (no mocks of project
// code; the ``window._demoApi`` host shim that main.js normally
// provides is stubbed, exactly like demoTalkRunParallel.test.js).
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/demoPauseOnTalk.test.js

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
 *   - ``speakText`` / ``playTalkEvent`` return pending promises and
 *     stash their resolvers in ``pending`` (unless ``opts.legacy``,
 *     in which case they return undefined like an old main.js);
 *   - ``stopSpeech`` resolves every pending promise, mirroring the
 *     real discard behavior in main.js.
 *
 * ``calls`` records every interesting host-api invocation in order.
 */
function makeDemoWindow(events, opts) {
  const legacy = !!(opts && opts.legacy);
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
      if (legacy) return undefined;
      return new Promise(resolve => {
        pending.push({kind: 'speakText', text, resolve});
      });
    },
    playTalkEvent(ev) {
      calls.push({fn: 'playTalkEvent', ev});
      if (legacy) return undefined;
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

/** One session: a talk tool call followed by a result. */
function talkEvents() {
  return [
    {
      type: 'tool_call',
      name: 'talk',
      extras: {text: 'Long spoken narration here.', language: 'en-US'},
    },
    {type: 'tool_result', content: 'Playing audio', tool_name: 'talk'},
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

async function testReplayPausesUntilTalkEnds() {
  const {win, api, calls, pending} = makeDemoWindow(talkEvents());
  const replay = startReplay(win, 'pause on talk');

  // Release the prompt narration so the replay reaches the talk panel.
  await waitFor(() => pending.length >= 1, 5000, 'prompt narration');
  assert.strictEqual(pending[0].kind, 'speakText');
  pending.shift().resolve();

  // The talk starts playing...
  await waitFor(
    () => calls.some(c => c.fn === 'playTalkEvent'),
    5000,
    'talk playback to start',
  );
  // ...and while its promise is unresolved the replay must stay
  // paused: no panel collapse, no result streaming — even long after
  // the usual 500ms panel pause would have elapsed.
  await sleep(1200);
  assert.strictEqual(
    calls.filter(c => c.fn === 'collapsePanels').length,
    0,
    'replay must not collapse the talk panel while speech is playing',
  );
  assert.ok(
    !resultRendered(win),
    'replay must not stream the result while speech is playing',
  );

  // End the speech — the replay resumes and runs to completion.
  await waitFor(() => pending.length >= 1, 1000, 'talk promise registration');
  assert.strictEqual(pending[0].kind, 'playTalkEvent');
  pending.shift().resolve();
  await replay;
  assert.ok(resultRendered(win), 'replay completes after the talk ends');
  assert.ok(
    calls.some(c => c.fn === 'collapsePanels'),
    'talk panel collapsed after the speech finished',
  );
  assert.strictEqual(api.active, false);
  win.close();
  console.log('  ok - replay pauses at a talk panel until the speech ends');
}

async function testReplayWaitsForPromptNarration() {
  const {win, api, calls, pending} = makeDemoWindow(talkEvents());
  const replay = startReplay(win, 'slow narration');

  await waitFor(() => pending.length >= 1, 5000, 'prompt narration');
  // Well past the 2-second prompt display the narration is still
  // playing — the replay must NOT have requested the session events.
  await sleep(2600);
  assert.strictEqual(
    calls.filter(c => c.fn === 'sendMessage' && c.msg.type === 'resumeSession')
      .length,
    0,
    'replay must not fetch events while the prompt narration is playing',
  );

  // Narration ends — the replay proceeds (talk resolved immediately
  // here; this test only pins the prompt-narration pause).
  pending.shift().resolve();
  await waitFor(() => pending.length >= 1, 5000, 'talk playback');
  pending.shift().resolve();
  await replay;
  assert.ok(resultRendered(win), 'replay completes after narration ends');
  assert.strictEqual(api.active, false);
  win.close();
  console.log('  ok - replay waits for the "User said ..." narration');
}

async function testCancelDuringTalkExitsImmediately() {
  const {win, api, calls, pending} = makeDemoWindow(talkEvents());
  const replay = startReplay(win, 'cancel mid talk');

  await waitFor(() => pending.length >= 1, 5000, 'prompt narration');
  pending.shift().resolve();
  await waitFor(
    () => calls.some(c => c.fn === 'playTalkEvent'),
    5000,
    'talk playback to start',
  );

  // Cancel while the replay is paused awaiting the in-flight talk:
  // _cancelDemoReplay -> stopSpeech resolves the pending promise, so
  // the paused coroutine wakes up, sees cancelRequested and exits.
  win._cancelDemoReplay();
  await replay;
  assert.ok(
    calls.some(c => c.fn === 'stopSpeech'),
    'cancel must stop in-flight demo speech',
  );
  assert.ok(
    !resultRendered(win),
    'cancelled replay must not stream the result',
  );
  assert.strictEqual(api.active, false, 'cancelled replay clears active');
  assert.strictEqual(pending.length, 0, 'no speech promise left dangling');

  // The demo can be restarted after a cancel (nothing deadlocked).
  const again = startReplay(win, 'restart after cancel');
  await waitFor(() => pending.length >= 1, 5000, 'restart narration');
  pending.shift().resolve();
  await waitFor(() => pending.length >= 1, 5000, 'restart talk');
  pending.shift().resolve();
  await again;
  assert.ok(resultRendered(win), 'demo replays fully after a cancel');
  win.close();
  console.log('  ok - cancel during an in-flight talk exits immediately');
}

async function testLegacyHooksReturningUndefinedStillComplete() {
  const {win, api} = makeDemoWindow(talkEvents(), {legacy: true});
  await startReplay(win, 'legacy host');
  assert.ok(
    resultRendered(win),
    'replay must complete when speech hooks return undefined',
  );
  assert.strictEqual(api.active, false);
  win.close();
  console.log('  ok - legacy hooks returning undefined never hang the demo');
}

async function runTests() {
  await testReplayPausesUntilTalkEnds();
  await testReplayWaitsForPromptNarration();
  await testCancelDuringTalkExitsImmediately();
  await testLegacyHooksReturningUndefinedStillComplete();
}

runTests().then(
  () => {
    console.log('\n4 passed, 0 failed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    process.exit(1);
  },
);
