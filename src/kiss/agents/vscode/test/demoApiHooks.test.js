// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the demo-mode host hooks in ``media/main.js``
// (``window._demoApi``): demo replay must be able to
//
//   * read a user prompt aloud (``speakText``) with the NATURAL
//     synthesized voice — the webview posts a 'demoSpeak' request and
//     plays the 'demoSpeakAudio' clip the daemon answers with (the
//     robotic Web Speech voice is never used; failed synthesis is
//     skipped silently — see demoExtensionReplay.test.js),
//   * actually play a replayed ``talk`` tool call (``playTalkEvent``),
//     using the recorded GPT audio directly when the event carries it,
//   * actually materialise sub-agent tabs for a replayed
//     ``run_parallel`` fan-out (``openSubagentTab``),
//   * cancel queued speech and pending synthesis (``stopSpeech``)
//     when the demo is stopped, and
//   * PAUSE the replay while talking: ``speakText`` / ``playTalkEvent``
//     return promises that resolve when the playback ends, and
//     ``stopSpeech`` resolves the promises of both queued and
//     in-flight jobs so a paused replay can never hang after cancel.
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

  return {win, posted};
}

/**
 * Install a recording Audio implementation on *win*.  With
 * ``opts.manual`` clips do NOT auto-complete — tests fire
 * ``player.onended()`` by hand; otherwise clips end after 5ms.
 * Returns the created players.
 */
function installAudio(win, opts) {
  const manual = !!(opts && opts.manual);
  const players = [];
  win.Audio = function Audio(src) {
    this.src = src;
    players.push(this);
    this.play = () => {
      if (!manual) {
        setTimeout(() => {
          if (typeof this.onended === 'function') this.onended();
        }, 5);
      }
      return Promise.resolve();
    };
  };
  return players;
}

/**
 * Install a recording Web Speech API on *win* (jsdom has none) whose
 * utterances end after 5ms.  Demo speech must NOT reach it in the
 * webview — tests assert the recording stays empty.
 */
function installSpeech(win) {
  const spoken = [];
  win.SpeechSynthesisUtterance = function SpeechSynthesisUtterance(text) {
    this.text = text;
    this.lang = '';
  };
  win.speechSynthesis = {
    getVoices: () => [],
    speak: u => {
      spoken.push(u);
      setTimeout(() => {
        if (typeof u.onend === 'function') u.onend();
      }, 5);
    },
    cancel: () => {},
    resume: () => {},
  };
  return spoken;
}

/** Deliver a daemon message to the webview. */
function dispatch(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Answer every posted 'demoSpeak' with a clip after 5ms. */
function autoAnswerDemoSpeak(win) {
  const prev = win._onPosted;
  win._onPosted = msg => {
    if (prev) prev(msg);
    if (msg.type !== 'demoSpeak') return;
    setTimeout(() => {
      dispatch(win, {
        type: 'demoSpeakAudio',
        reqId: msg.reqId,
        audioB64: 'QUJD',
        audioMime: 'audio/mpeg',
        tabId: msg.tabId,
      });
    }, 5);
  };
}

function sleep(ms) {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

/** Await *promise* but fail fast after *ms* — a hung promise means the
 * paused demo replay would hang too. */
function withTimeout(promise, ms, what) {
  return Promise.race([
    promise,
    new Promise((_resolve, reject) => {
      setTimeout(() => reject(new Error('timed out: ' + what)), ms);
    }),
  ]);
}

async function testSpeakTextSynthesizesAndPlaysPrompt() {
  const {win, posted} = makeWebview();
  const players = installAudio(win);
  const spoken = installSpeech(win);
  autoAnswerDemoSpeak(win);

  await withTimeout(
    win._demoApi.speakText('User said plan my trip', 'en-US'),
    1000,
    'speakText',
  );

  const req = posted.find(m => m.type === 'demoSpeak');
  assert.ok(req, 'speakText must request natural-voice synthesis');
  assert.strictEqual(req.text, 'User said plan my trip');
  assert.strictEqual(req.language, 'en-US');
  assert.strictEqual(players.length, 1, 'the synthesized clip plays');
  assert.strictEqual(
    spoken.length,
    0,
    'the robotic Web Speech voice must NOT narrate the prompt',
  );
  console.log('PASS: speakText synthesizes and plays the prompt clip');
}

async function testPlayTalkEventUsesRecordedAudioDirectly() {
  const {win, posted} = makeWebview();
  const players = installAudio(win);
  installSpeech(win);

  await withTimeout(
    win._demoApi.playTalkEvent({
      text: 'Hello there, I am on it.',
      language: 'en-GB',
      audioB64: 'UkVD',
      audioMime: 'audio/mpeg',
    }),
    1000,
    'playTalkEvent with recorded audio',
  );

  assert.ok(
    !posted.find(m => m.type === 'demoSpeak'),
    'a talk event that already carries audio must not be re-synthesized',
  );
  assert.strictEqual(players.length, 1);
  assert.ok(
    players[0].src.indexOf('base64,UkVD') !== -1,
    'the RECORDED clip is what plays',
  );
  console.log('PASS: playTalkEvent plays recorded GPT audio directly');
}

async function testPlayTalkEventSynthesizesWhenNoAudio() {
  const {win, posted} = makeWebview();
  const players = installAudio(win);
  const spoken = installSpeech(win);
  autoAnswerDemoSpeak(win);

  await withTimeout(
    win._demoApi.playTalkEvent({
      text: 'Hello there, I am on it.',
      language: 'en-GB',
      emotion: 'cheerful',
    }),
    1000,
    'playTalkEvent without audio',
  );

  const req = posted.find(m => m.type === 'demoSpeak');
  assert.ok(req, 'an audio-less talk event is synthesized');
  assert.strictEqual(req.text, 'Hello there, I am on it.');
  assert.strictEqual(req.language, 'en-GB');
  assert.strictEqual(req.emotion, 'cheerful');
  assert.strictEqual(players.length, 1, 'the synthesized clip plays');
  assert.strictEqual(spoken.length, 0, 'never the robotic voice');
  console.log('PASS: playTalkEvent synthesizes a replayed talk call');
}

async function testQueuedSpeechIsSerialized() {
  const {win, posted} = makeWebview();
  const players = installAudio(win, {manual: true});
  installSpeech(win);
  autoAnswerDemoSpeak(win);

  const first = win._demoApi.speakText('User said first task');
  const second = win._demoApi.playTalkEvent({text: 'Second utterance here.'});

  await sleep(30);
  assert.strictEqual(players.length, 1, 'first clip playing');
  assert.strictEqual(
    posted.filter(m => m.type === 'demoSpeak').length,
    1,
    'the queued job must not synthesize until the first one finished',
  );

  players[0].onended();
  await withTimeout(first, 1000, 'first speech promise');
  await sleep(30);
  assert.strictEqual(players.length, 2, 'second clip plays after the first');

  players[1].onended();
  await withTimeout(second, 1000, 'second speech promise');
  console.log('PASS: demo speech is serialized through the talk queue');
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

async function testSpeakTextPromiseResolvesOnClipEnd() {
  const {win} = makeWebview();
  const players = installAudio(win, {manual: true});
  installSpeech(win);
  autoAnswerDemoSpeak(win);

  const p = win._demoApi.speakText('User said pause until I finish', 'en-US');
  assert.ok(p && typeof p.then === 'function', 'speakText returns a promise');

  let resolved = false;
  p.then(() => {
    resolved = true;
  });
  await sleep(50);
  assert.strictEqual(players.length, 1, 'clip playing');
  assert.strictEqual(
    resolved,
    false,
    'the narration promise must stay pending while the clip is playing',
  );

  players[0].onended();
  await withTimeout(p, 1000, 'speakText promise after clip end');
  console.log('PASS: speakText promise resolves when the clip ends');
}

async function testStopSpeechResolvesQueuedAndInFlightPromises() {
  const {win, posted} = makeWebview();
  const players = installAudio(win, {manual: true});
  const spoken = installSpeech(win);
  // No auto-answer: the in-flight job hangs on a pending synthesis
  // request — the worst case for a cancelled paused replay.

  const inFlight = win._demoApi.speakText('User said in-flight narration');
  const queued = win._demoApi.playTalkEvent({text: 'Still queued talk.'});
  await sleep(10);
  const req = posted.find(m => m.type === 'demoSpeak');
  assert.ok(req, 'in-flight job requested synthesis');

  win._demoApi.stopSpeech();

  // Both promises must resolve — the paused demo replay awaits them,
  // and a dangling promise would hang the cancelled replay forever.
  await withTimeout(inFlight, 1000, 'in-flight promise after stopSpeech');
  await withTimeout(queued, 1000, 'queued promise after stopSpeech');

  // A LATE synthesis reply for the cancelled request must not play.
  dispatch(win, {
    type: 'demoSpeakAudio',
    reqId: req.reqId,
    audioB64: 'QUJD',
    audioMime: 'audio/mpeg',
  });
  await sleep(30);
  assert.strictEqual(players.length, 0, 'late clip must not play');
  assert.strictEqual(spoken.length, 0, 'and must not speak');
  console.log('PASS: stopSpeech resolves queued and in-flight promises');
}

async function testLateClipEndAfterStopSpeechDoesNotBreakQueue() {
  const {win} = makeWebview();
  const players = installAudio(win, {manual: true});
  installSpeech(win);
  autoAnswerDemoSpeak(win);

  const a = win._demoApi.speakText('User said job A');
  await sleep(30);
  assert.strictEqual(players.length, 1, 'job A clip playing');

  win._demoApi.stopSpeech();
  await withTimeout(a, 1000, 'job A promise after stopSpeech');

  const b = win._demoApi.speakText('User said job B');
  await sleep(30);
  assert.strictEqual(players.length, 2, 'job B starts after the cancel');

  // The cancelled clip completes LATE — its stale finish must be a
  // no-op: it must NOT release the queue under job B.
  players[0].onended();

  const c = win._demoApi.speakText('User said job C');
  await sleep(30);
  assert.strictEqual(
    players.length,
    2,
    'job C must stay queued behind B — a late finish of the ' +
      'cancelled job must not pump the queue (overlapping speech)',
  );

  // The late finish must not clobber B's discard hook either: a
  // second stopSpeech must still resolve BOTH the in-flight B and
  // the queued C, or a cancelled paused replay would hang forever.
  win._demoApi.stopSpeech();
  await withTimeout(b, 1000, 'in-flight job B promise after 2nd stopSpeech');
  await withTimeout(c, 1000, 'queued job C promise after 2nd stopSpeech');
  console.log('PASS: late clip end after stopSpeech cannot break the queue');
}

async function runTests() {
  await testSpeakTextSynthesizesAndPlaysPrompt();
  await testPlayTalkEventUsesRecordedAudioDirectly();
  await testPlayTalkEventSynthesizesWhenNoAudio();
  await testQueuedSpeechIsSerialized();
  testOpenSubagentTabMaterialisesTab();
  testOpenSubagentTabUnknownParentIgnored();
  await testSpeakTextPromiseResolvesOnClipEnd();
  await testStopSpeechResolvesQueuedAndInFlightPromises();
  await testLateClipEndAfterStopSpeechDoesNotBreakQueue();
  console.log('All demoApiHooks tests passed.');
  // setRunningState(true) starts webview timers that keep the node
  // event loop alive — exit explicitly once every assertion passed.
  process.exit(0);
}

runTests().catch(err => {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
});
