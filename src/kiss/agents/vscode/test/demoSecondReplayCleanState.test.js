// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests (real chat.html + main.js + demo.js in jsdom, a
// recording acquireVsCodeApi stub standing in for the extension host)
// for the demo-mode invariant that a finished OR stopped demo replay
// MUST NOT leave any state behind that interferes with the next demo:
//
//   * ZOMBIE REPLAY: stopping demo A while its replay coroutine is
//     suspended at an ``await`` (panel-show sleep) and then starting
//     demo B must NOT let A's coroutine resume when B resets the
//     shared cancel flag — pre-fix, A's remaining panels rendered
//     interleaved into B's output and A's epilogue then tore down the
//     running-demo UI (spinner, demo-playing class, _demoActive)
//     while B was still playing, so B's own task_events fell into the
//     instant replayTaskEvents path and B hung forever.
//
//   * STALE EVENT RESOLVER: stopping demo A while its task_events
//     request is still in flight must clear ``_demoApi.resolveEvents``
//     (and wake the suspended fetch) so A's LATE reply can never be
//     mistaken for demo B's events.
//
//   * CLEAN COMPLETION: a demo that finishes naturally leaves no
//     active/paused/resolver state and the next demo replays its own
//     events only.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/demoSecondReplayCleanState.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview: the real
 * ``chat.html`` body, ``panelCopy.js``, ``main.js`` AND ``demo.js``
 * evaluated in the window, plus a recording ``acquireVsCodeApi`` stub.
 * ``win._onPosted`` (settable per test) observes every posted message
 * so tests can answer like the extension host / daemon would.
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
      postMessage: msg => {
        posted.push(msg);
        if (win._onPosted) win._onPosted(msg);
      },
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'demo.js'), 'utf8'));

  return {win, posted};
}

/** Install a no-op recording Audio implementation (clips end in 5ms). */
function installAudio(win) {
  const players = [];
  win.Audio = function Audio(src) {
    this.src = src;
    players.push(this);
    this.play = () => {
      setTimeout(() => {
        if (typeof this.onended === 'function') this.onended();
      }, 5);
      return Promise.resolve();
    };
    this.pause = () => {};
  };
  return players;
}

/** Install a no-op Web Speech API so nothing can hang on speech. */
function installSpeech(win) {
  win.SpeechSynthesisUtterance = function (text) {
    this.text = text;
  };
  win.speechSynthesis = {
    getVoices: () => [],
    speak: u => {
      setTimeout(() => {
        if (typeof u.onend === 'function') u.onend();
      }, 5);
    },
    cancel: () => {},
    pause: () => {},
    resume: () => {},
  };
}

/** Deliver a daemon/extension-host message to the webview. */
function dispatch(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

async function waitUntil(pred, timeoutMs, label) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    if (pred()) return;
    await sleep(25);
  }
  throw new Error('timed out waiting for ' + label);
}

// Two independent history tasks, each in its OWN chat.
const SESSIONS = [
  {
    id: 'chat-B',
    task_id: 'b1',
    preview: 'Demo task B',
    title: 'Demo task B',
    has_events: true,
    ts: 2000,
  },
  {
    id: 'chat-A',
    task_id: 'a1',
    preview: 'Demo task A',
    title: 'Demo task A',
    has_events: true,
    ts: 1000,
  },
];

/**
 * Task A's recorded events: FOUR panel groups (thought, tool call,
 * thought, result) so that stopping the replay during the first
 * group's post-render pause leaves three distinctly-markered groups
 * that a zombie coroutine would go on to render.
 */
function eventsA() {
  return [
    {type: 'thinking_start'},
    {type: 'thinking_delta', text: 'A-THOUGHT-ONE'},
    {type: 'tool_call', name: 'Bash', extras: {command: 'echo A-TOOL-TWO'}},
    {type: 'tool_result', content: 'A-TOOL-TWO-OUT', tool_name: 'Bash'},
    {type: 'thinking_start'},
    {type: 'thinking_delta', text: 'A-THOUGHT-THREE'},
    {
      type: 'result',
      summary: 'A-RESULT-FINAL. ' + 'Alpha outcome words stream here. '.repeat(20),
      total_tokens: 1,
      cost: '$0',
    },
  ];
}

/**
 * Task B's recorded events: a markered thought and a LONG result so B
 * keeps streaming for several seconds — the window in which a zombie
 * replay of task A would (pre-fix) interleave its panels and tear
 * down the running-demo UI.
 */
function eventsB() {
  return [
    {type: 'thinking_start'},
    {type: 'thinking_delta', text: 'B-THOUGHT-ONE'},
    {
      type: 'result',
      summary: 'B-RESULT-FINAL. ' + 'Bravo outcome words stream here. '.repeat(60),
      total_tokens: 2,
      cost: '$0',
    },
  ];
}

const EVENTS_BY_TASK = {a1: eventsA, b1: eventsB};

/**
 * Enable demo mode, deliver SESSIONS and install a resumeSession
 * responder that answers each request with that task's own events
 * after 10ms.  Tasks listed in *skipTaskIds* are NOT answered — their
 * requests are recorded in the returned ``pending`` array so a test
 * can deliver a LATE reply by hand.
 */
function setupDemo(win, skipTaskIds) {
  const skip = skipTaskIds || [];
  dispatch(win, {type: 'configData', config: {demo_mode: true}, apiKeys: {}});
  dispatch(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: SESSIONS,
  });
  const pending = [];
  win._onPosted = msg => {
    if (msg.type !== 'resumeSession') return;
    if (skip.indexOf(msg.taskId) !== -1) {
      pending.push(msg);
      return;
    }
    setTimeout(() => {
      dispatch(win, {
        type: 'task_events',
        tabId: msg.tabId,
        events: EVENTS_BY_TASK[msg.taskId](),
        task: 'task ' + msg.taskId,
        task_id: msg.taskId,
        chat_id: msg.id,
        extra: '',
      });
    }, 10);
  };
  return pending;
}

/** Click the history row whose text contains *label*. */
function clickHistoryRow(win, label) {
  const rows = Array.from(
    win.document.querySelectorAll('#history-list > div'),
  );
  const row = rows.find(r => r.textContent.indexOf(label) !== -1);
  assert.ok(row, 'history row for "' + label + '" rendered');
  row.click();
}

/** Text content of the live (active-tab) output area. */
function outputText(win) {
  const O = win.document.getElementById('output');
  return O ? O.textContent : '';
}

async function waitForDemoEnd(win, timeoutMs) {
  await waitUntil(
    () => !win._demoApi.active,
    timeoutMs,
    'demo replay to finish',
  );
}

async function testStoppedDemoDoesNotZombieIntoNextDemo() {
  const {win} = makeWebview();
  installAudio(win);
  installSpeech(win);
  setupDemo(win);

  // Start demo A and let its first panel render; the replay coroutine
  // is now suspended in the post-render show pause.
  clickHistoryRow(win, 'Demo task A');
  await waitUntil(
    () => outputText(win).indexOf('A-THOUGHT-ONE') !== -1,
    3000,
    "demo A's first panel",
  );

  // Stop demo A mid-pause, then immediately start demo B.
  win.document.getElementById('stop-btn').click();
  assert.strictEqual(win._demoApi.active, false, 'stop cancels demo A');
  clickHistoryRow(win, 'Demo task B');
  await waitUntil(
    () => outputText(win).indexOf('B-THOUGHT-ONE') !== -1,
    3000,
    "demo B's first panel",
  );

  // Demo A's coroutine wakes from its ~500ms pause while B plays.  It
  // must observe the cancel and exit: none of A's remaining panels may
  // render into B's output, and A's epilogue must NOT tear down the
  // running-demo state under B.
  await sleep(1800);
  const text = outputText(win);
  assert.strictEqual(
    text.indexOf('A-TOOL-TWO'),
    -1,
    "stopped demo A's tool panel must not render into demo B's output",
  );
  assert.strictEqual(
    text.indexOf('A-THOUGHT-THREE'),
    -1,
    "stopped demo A's later thought must not render into demo B's output",
  );
  assert.strictEqual(
    text.indexOf('A-RESULT-FINAL'),
    -1,
    "stopped demo A's result must not stream into demo B's output",
  );
  assert.strictEqual(
    win._demoApi.active,
    true,
    "demo B must still be ACTIVE — stopped demo A's epilogue must not " +
      'flip _demoActive off under the running demo B',
  );
  assert.ok(
    win.document.body.classList.contains('demo-playing'),
    "demo B's demo-playing UI must survive stopped demo A's epilogue",
  );

  // Demo B runs to its natural end with only its own content.
  await waitForDemoEnd(win, 30000);
  const finalText = outputText(win);
  assert.ok(
    finalText.indexOf('B-RESULT-FINAL') !== -1,
    "demo B's result streamed to completion",
  );
  assert.strictEqual(
    finalText.indexOf('A-RESULT-FINAL'),
    -1,
    "demo A's result never leaked into demo B's tab",
  );
  assert.ok(
    !win.document.body.classList.contains('demo-playing'),
    'demo UI restored after demo B finishes',
  );
  console.log('PASS: stopped demo cannot zombie into the next demo');
}

async function testStopDuringEventFetchLeavesNoStaleResolver() {
  const {win} = makeWebview();
  installAudio(win);
  installSpeech(win);
  // Never answer task A's resumeSession — the stop happens while the
  // event fetch is still in flight.
  const pendingA = setupDemo(win, ['a1']);

  clickHistoryRow(win, 'Demo task A');
  await waitUntil(
    () => pendingA.length === 1,
    3000,
    "demo A's resumeSession request",
  );

  win.document.getElementById('stop-btn').click();
  assert.strictEqual(
    win._demoApi.resolveEvents,
    null,
    'stopping a demo whose event fetch is in flight must clear the ' +
      'resolveEvents hook — a stale hook would swallow the next ' +
      "demo's (or a live task's) task_events",
  );

  // Start demo B; it must replay normally.
  clickHistoryRow(win, 'Demo task B');
  await waitUntil(
    () => outputText(win).indexOf('B-THOUGHT-ONE') !== -1,
    3000,
    "demo B's first panel",
  );

  // Task A's LATE reply arrives now (addressed to A's old tab).  It
  // must not corrupt demo B's output.
  dispatch(win, {
    type: 'task_events',
    tabId: pendingA[0].tabId,
    events: eventsA(),
    task: 'task a1',
    task_id: 'a1',
    chat_id: pendingA[0].id,
    extra: '',
  });
  await sleep(300);
  assert.strictEqual(
    outputText(win).indexOf('A-THOUGHT-ONE'),
    -1,
    "task A's late task_events reply must not render into demo B's output",
  );

  await waitForDemoEnd(win, 30000);
  assert.ok(
    outputText(win).indexOf('B-RESULT-FINAL') !== -1,
    "demo B's result streamed to completion despite A's late reply",
  );
  console.log('PASS: stop during event fetch leaves no stale resolver');
}

async function testMismatchedActiveTabReplyDoesNotSettleDemoFetch() {
  const {win} = makeWebview();
  installAudio(win);
  installSpeech(win);
  // Never auto-answer task B — its reply is delivered by hand.
  const pendingB = setupDemo(win, ['b1']);

  clickHistoryRow(win, 'Demo task B');
  await waitUntil(
    () => pendingB.length === 1,
    3000,
    "demo B's resumeSession request",
  );

  // A DIFFERENT task's late reply arrives addressed to B's own
  // (active) tab — e.g. a stopped demo's reply after a legacy
  // same-tab restart.  It must NOT settle B's fetch or render.
  dispatch(win, {
    type: 'task_events',
    tabId: pendingB[0].tabId,
    events: eventsA(),
    task: 'task a1',
    task_id: 'a1',
    chat_id: 'chat-A',
    extra: '',
  });
  await sleep(300);
  assert.strictEqual(
    outputText(win).indexOf('A-THOUGHT-ONE'),
    -1,
    "another task's reply on the same tab must not render into the demo",
  );
  assert.ok(
    win._demoApi.resolveEvents,
    "demo B's fetch must still be pending after the mismatched reply",
  );

  // The RIGHT reply settles the fetch and B replays to completion.
  dispatch(win, {
    type: 'task_events',
    tabId: pendingB[0].tabId,
    events: eventsB(),
    task: 'task b1',
    task_id: 'b1',
    chat_id: 'chat-B',
    extra: '',
  });
  await waitUntil(
    () => outputText(win).indexOf('B-THOUGHT-ONE') !== -1,
    3000,
    "demo B's first panel",
  );
  await waitForDemoEnd(win, 30000);
  const text = outputText(win);
  assert.ok(
    text.indexOf('B-RESULT-FINAL') !== -1,
    "demo B's result streamed to completion after the mismatched reply",
  );
  assert.strictEqual(
    text.indexOf('A-THOUGHT-ONE'),
    -1,
    "the mismatched task's events never leaked into demo B's output",
  );
  console.log('PASS: mismatched active-tab reply cannot settle demo fetch');
}

async function testCompletedDemoLeavesCleanStateForNextDemo() {
  const {win} = makeWebview();
  installAudio(win);
  installSpeech(win);
  setupDemo(win);

  // Demo A runs to its natural end.
  clickHistoryRow(win, 'Demo task A');
  await waitUntil(
    () => outputText(win).indexOf('A-RESULT-FINAL') !== -1,
    30000,
    "demo A's result",
  );
  await waitForDemoEnd(win, 30000);

  // Every piece of demo state must be back to idle.
  assert.strictEqual(win._demoApi.active, false, 'demo A inactive');
  assert.strictEqual(
    win._demoApi.resolveEvents,
    null,
    'no stale resolveEvents after natural completion',
  );
  assert.strictEqual(win._isDemoPaused(), false, 'not paused after A');
  assert.ok(
    !win.document.body.classList.contains('demo-playing'),
    'demo UI restored after A',
  );

  // Demo B replays cleanly: only its own events, and ends idle.
  clickHistoryRow(win, 'Demo task B');
  await waitUntil(
    () => outputText(win).indexOf('B-THOUGHT-ONE') !== -1,
    3000,
    "demo B's first panel",
  );
  await waitForDemoEnd(win, 30000);
  const text = outputText(win);
  assert.ok(
    text.indexOf('B-RESULT-FINAL') !== -1,
    "demo B's result streamed to completion",
  );
  assert.strictEqual(
    text.indexOf('A-THOUGHT-ONE'),
    -1,
    "demo A's content must not appear in demo B's fresh tab",
  );
  assert.strictEqual(win._demoApi.active, false, 'demo B inactive at end');
  assert.strictEqual(
    win._demoApi.resolveEvents,
    null,
    'no stale resolveEvents after demo B',
  );
  console.log('PASS: completed demo leaves clean state for the next demo');
}

/**
 * Build a jsdom window with ONLY demo.js and a stub host api (like
 * demoPauseOnTalk.test.js) so tests can inject faults — the
 * full-webview harness cannot make main.js hooks throw.  The stub
 * answers each resumeSession with ``api.eventsToDeliver`` after 10ms
 * unless ``api.autoRespond`` is false; ``api.throwOnProcess`` makes
 * ``processEvent`` throw.  ``calls`` records processed events.
 */
function makeStubDemoWindow() {
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
    eventsToDeliver: [],
    autoRespond: true,
    throwOnProcess: false,
    clearForReplay() {},
    resetOutputState() {},
    processEvent(ev) {
      calls.push(ev);
      if (api.throwOnProcess) throw new Error('injected processEvent fault');
    },
    setTaskText() {},
    updateTabTitle() {},
    hideWelcome() {},
    scrollToBottom() {},
    getActiveTabId() {
      return 'demo-tab';
    },
    sendMessage(msg) {
      if (msg && msg.type === 'resumeSession' && api.autoRespond) {
        setTimeout(() => {
          if (api.resolveEvents) api.resolveEvents(api.eventsToDeliver);
        }, 10);
      }
    },
    collapsePanels() {},
    setRunningState() {},
    showSpinner() {},
    removeSpinner() {},
    setDemoUi() {},
    stopSpeech() {},
  };
  win._demoApi = api;
  win.eval(fs.readFileSync(path.join(MEDIA, 'demo.js'), 'utf8'));
  return {win, api, calls};
}

async function testThrowingHostHookStillRestoresIdleState() {
  const {win, api} = makeStubDemoWindow();
  api.eventsToDeliver = [
    {type: 'thinking_start'},
    {type: 'thinking_delta', text: 'faulty panel'},
    {type: 'result', summary: 'STUB-RESULT-OK', total_tokens: 1, cost: '$0'},
  ];

  // Demo #1: the host's processEvent THROWS.  The replay must still
  // end with every piece of demo state back to idle — a stuck
  // api.active would block every later demo forever.
  api.throwOnProcess = true;
  await win._startDemoReplay([
    {id: 7, task_id: 't7', has_events: true, preview: 'faulty', ts: 1},
  ]);
  assert.strictEqual(
    api.active,
    false,
    'a throwing processEvent must not leave the demo marked active',
  );
  assert.strictEqual(
    api.resolveEvents,
    null,
    'a throwing replay must not leave a stale resolveEvents hook',
  );
  assert.strictEqual(win._isDemoPaused(), false, 'not paused after fault');

  // Demo #2 (fault cleared) must replay to completion.
  api.throwOnProcess = false;
  await win._startDemoReplay([
    {id: 7, task_id: 't7', has_events: true, preview: 'healthy', ts: 1},
  ]);
  assert.ok(
    win.document
      .getElementById('output')
      .textContent.includes('STUB-RESULT-OK'),
    'the next demo replays fine after a faulted one',
  );
  assert.strictEqual(api.active, false, 'demo #2 ends inactive');
  win.close();
  console.log('PASS: throwing host hook still restores idle demo state');
}

async function testStaleCapturedResolverCannotClobberNextFetch() {
  const {win, api, calls} = makeStubDemoWindow();
  api.autoRespond = false;

  // Demo A suspends on its event fetch; capture its deliver hook the
  // way a late caller would.
  const replayA = win._startDemoReplay([
    {id: 1, task_id: 'a1', has_events: true, preview: 'task A', ts: 1},
  ]);
  await waitUntil(
    () => api.resolveEvents !== null,
    2000,
    "demo A's event fetch",
  );
  const staleDeliver = api.resolveEvents;

  // Stop A (clears the hook), then start demo B up to ITS fetch.
  win._cancelDemoReplay();
  assert.strictEqual(api.resolveEvents, null, 'cancel cleared the hook');
  await replayA;
  const replayB = win._startDemoReplay([
    {id: 2, task_id: 'b1', has_events: true, preview: 'task B', ts: 2},
  ]);
  await waitUntil(
    () => api.resolveEvents !== null,
    2000,
    "demo B's event fetch",
  );
  const bDeliver = api.resolveEvents;

  // A's LATE captured deliver fires now.  It must settle only its own
  // abandoned promise — clobbering B's hook would hang B forever, and
  // rendering A's events would corrupt B's output.
  staleDeliver([
    {type: 'thinking_start'},
    {type: 'thinking_delta', text: 'A-STALE'},
  ]);
  assert.strictEqual(
    api.resolveEvents,
    bDeliver,
    "a stale deliver hook must not clear the NEXT demo's pending fetch",
  );
  await sleep(50);
  assert.ok(
    !calls.some(ev => ev.text === 'A-STALE'),
    "the cancelled demo's late events must not be rendered",
  );

  // B's own events arrive and B completes normally.
  bDeliver([
    {type: 'thinking_start'},
    {type: 'thinking_delta', text: 'B-LIVE'},
    {type: 'result', summary: 'B-STUB-DONE', total_tokens: 1, cost: '$0'},
  ]);
  await replayB;
  assert.ok(
    calls.some(ev => ev.text === 'B-LIVE'),
    "demo B rendered its own events after A's stale delivery",
  );
  assert.strictEqual(api.active, false, 'demo B ends inactive');
  win.close();
  console.log('PASS: stale captured resolver cannot clobber the next fetch');
}

(async () => {
  await testStoppedDemoDoesNotZombieIntoNextDemo();
  await testStopDuringEventFetchLeavesNoStaleResolver();
  await testMismatchedActiveTabReplyDoesNotSettleDemoFetch();
  await testCompletedDemoLeavesCleanStateForNextDemo();
  await testThrowingHostHookStillRestoresIdleState();
  await testStaleCapturedResolverCannotClobberNextFetch();
  console.log('demoSecondReplayCleanState.test.js: all tests passed');
  // Webview timers keep the node event loop alive; match the other
  // jsdom e2e demo tests and exit explicitly once assertions pass.
  process.exit(0);
})().catch(err => {
  console.error(err);
  process.exit(1);
});
