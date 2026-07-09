// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the demo-replay input controls (real chat.html
// + main.js + demo.js in jsdom, a recording acquireVsCodeApi stub
// standing in for the extension host / daemon).  While a demo replay
// is playing:
//
//   * the input textbox, burger menu, model picker, attach,
//     inject-promptlet, mic, and send buttons must be HIDDEN;
//
//   * a STOP button is shown in place of the send button — pressing
//     it stops the demo animations and speech;
//
//   * a pause/play button is shown to the LEFT of the stop button —
//     pressing pause freezes the demo animations AND speech and flips
//     the icon to play; pressing play resumes.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/demoControlsStopPause.test.js

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

/**
 * Install a recording Audio implementation whose clips play for
 * *durationMs* and then fire ``onended``.  ``pause()`` records the
 * call and suppresses the pending ``onended`` (a paused clip never
 * ends on its own); a later ``play()`` re-arms it.  Returns the
 * created players.
 */
function installAudio(win, durationMs) {
  const players = [];
  win.Audio = function Audio(src) {
    this.src = src;
    this.pauseCalls = 0;
    let timer = null;
    players.push(this);
    this.play = () => {
      timer = setTimeout(() => {
        timer = null;
        if (typeof this.onended === 'function') this.onended();
      }, durationMs || 15);
      return Promise.resolve();
    };
    this.pause = () => {
      this.pauseCalls++;
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
    };
  };
  return players;
}

/** Install a no-op Web Speech API so fallbacks never hang. */
function installSpeech(win) {
  const spoken = [];
  win.SpeechSynthesisUtterance = function (text) {
    this.text = text;
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
    pause: () => {},
    resume: () => {},
  };
  return spoken;
}

/** Deliver a daemon/extension-host message to the webview. */
function dispatch(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Answer every posted 'demoSpeak' with a synthesized clip after *delayMs*. */
function autoAnswerDemoSpeak(win, delayMs) {
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
    }, delayMs == null ? 10 : delayMs);
  };
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

// A long recorded result so the result panel streams for several
// seconds — the window in which pause/resume is exercised.
const LONG_RESULT = (
  'The agent audited every module and confirmed the invariants hold ' +
  'across restarts, replays, cancellations, and reconnects. '
).repeat(25);

const SESSIONS = [
  {
    id: 'chat-A',
    task_id: 'a1',
    preview: 'Investigate the flaky pipeline',
    title: 'Investigate the flaky pipeline',
    has_events: true,
    ts: 1000,
  },
];

/** Replayed event stream: a thought panel then a long result. */
function eventsFor(label) {
  return [
    {type: 'thinking_start'},
    {type: 'thinking_delta', text: 'working on ' + label},
    {type: 'result', summary: LONG_RESULT, total_tokens: 1, cost: '$0'},
  ];
}

/**
 * Enable demo mode, deliver SESSIONS, auto-answer resumeSession with
 * per-task events, click the (single) history row.  Returns a promise
 * resolving when the replay ends.
 */
function startDemoFlow(win) {
  dispatch(win, {type: 'updateSetting', key: 'demo_mode', value: true});
  dispatch(win, {type: 'history', offset: 0, generation: 0, sessions: SESSIONS});

  const prev = win._onPosted;
  win._onPosted = msg => {
    if (prev) prev(msg);
    if (msg.type !== 'resumeSession') return;
    setTimeout(() => {
      dispatch(win, {
        type: 'task_events',
        tabId: msg.tabId,
        events: eventsFor(String(msg.taskId || msg.id)),
        task: 'task ' + String(msg.taskId || msg.id),
        chat_id: msg.id,
        extra: '',
      });
    }, 10);
  };

  const rows = Array.from(win.document.querySelectorAll('#history-list > div'));
  const row = rows.find(
    r => r.textContent.indexOf('Investigate the flaky pipeline') !== -1,
  );
  assert.ok(row, 'history row rendered');
  row.click();

  return (async () => {
    const t0 = Date.now();
    while (Date.now() - t0 < 30000) {
      await sleep(50);
      if (!win._demoApi.active && Date.now() - t0 > 500) return;
    }
    throw new Error('demo replay did not finish within 30s');
  })();
}

/** The 7 controls that must be hidden while a demo replay plays. */
const HIDDEN_SELECTORS = [
  '#input-text-wrap',
  '#menu-btn',
  '#model-btn',
  '#upload-btn',
  '#tricks-btn',
  '#voice-btn',
  '#send-btn',
];

/** Read the pause/play icon visibility of #demo-pause-btn. */
function iconState(btn) {
  const pauseIcon = btn.querySelector('.icon-pause');
  const playIcon = btn.querySelector('.icon-play');
  assert.ok(pauseIcon, 'pause icon svg present in #demo-pause-btn');
  assert.ok(playIcon, 'play icon svg present in #demo-pause-btn');
  return {
    pauseVisible: pauseIcon.style.display !== 'none',
    playVisible: playIcon.style.display !== 'none',
  };
}

async function testControlsHiddenWhileDemoPlays() {
  const {win} = makeWebview();
  installAudio(win, 40);
  installSpeech(win);
  autoAnswerDemoSpeak(win);

  const done = startDemoFlow(win);
  await sleep(300);
  assert.ok(win._demoApi.active, 'demo replay is running');

  // The demo-playing body class drives the hiding (jsdom does not
  // load <link> stylesheets, so the CSS rules are asserted on the
  // real stylesheet text below).
  assert.ok(
    win.document.body.classList.contains('demo-playing'),
    'body carries the demo-playing class while a demo replay plays',
  );
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  for (const sel of HIDDEN_SELECTORS) {
    assert.ok(
      css.indexOf('body.demo-playing ' + sel) !== -1,
      'main.css hides ' + sel + ' under body.demo-playing',
    );
  }
  assert.ok(
    /body\.demo-playing[^{}]*\{[^}]*display:\s*none\s*!important/.test(css),
    'the demo-playing hiding rules use display: none !important ' +
      '(setRunningState writes inline display styles)',
  );
  assert.ok(
    /body\.demo-playing #stop-btn\s*\{[^}]*display:\s*flex\s*!important/.test(
      css,
    ),
    'main.css shows #stop-btn under body.demo-playing with !important',
  );
  assert.ok(
    /body\.demo-playing #demo-pause-btn\s*\{[^}]*display:\s*flex\s*!important/.test(
      css,
    ),
    'main.css shows #demo-pause-btn under body.demo-playing',
  );

  // The pause/play button exists to the LEFT of the stop button and
  // starts in the "pause" state.
  const btn = win.document.getElementById('demo-pause-btn');
  assert.ok(btn, '#demo-pause-btn exists');
  const stopBtn = win.document.getElementById('stop-btn');
  assert.strictEqual(
    btn.parentElement,
    stopBtn.parentElement,
    '#demo-pause-btn sits in #input-actions with the stop button',
  );
  assert.ok(
    btn.compareDocumentPosition(stopBtn) &
      win.Node.DOCUMENT_POSITION_FOLLOWING,
    '#demo-pause-btn is to the LEFT of #stop-btn',
  );
  assert.notStrictEqual(
    btn.style.display,
    'none',
    '#demo-pause-btn is displayed while the demo plays',
  );
  const icons = iconState(btn);
  assert.ok(icons.pauseVisible, 'pause icon shown at the beginning');
  assert.ok(!icons.playVisible, 'play icon hidden at the beginning');

  await done;
  assert.ok(
    !win.document.body.classList.contains('demo-playing'),
    'demo-playing class removed after the replay finishes naturally',
  );
  assert.strictEqual(
    win.document.getElementById('demo-pause-btn').style.display,
    'none',
    '#demo-pause-btn hidden again after the replay finishes',
  );
  console.log('PASS: controls hidden and pause/stop shown while demo plays');
}

async function testPauseBeforeClipArrivesDefersSpeechUntilResume() {
  const {win} = makeWebview();
  const players = installAudio(win, 40);
  installSpeech(win);
  // Delay the natural-voice clip long enough to pause while the daemon
  // synthesis request is still pending.  The pre-fix bug started the
  // Audio clip as soon as this late reply arrived, even though the demo
  // was paused.
  autoAnswerDemoSpeak(win, 300);

  const done = startDemoFlow(win);
  const btn = win.document.getElementById('demo-pause-btn');
  await sleep(50);
  btn.click();
  assert.strictEqual(win._isDemoPaused(), true, 'demo paused before clip reply');

  await sleep(450);
  assert.strictEqual(
    players.length,
    0,
    'a demo speech clip whose synthesis reply arrives while paused ' +
      'must not start playing until the demo is resumed',
  );

  // Rapid pause/resume/pause must not leak a resolved waiter that starts
  // the clip in the microtask after the demo has already been paused
  // again.
  btn.click();
  assert.strictEqual(win._isDemoPaused(), false, 'demo briefly resumed');
  btn.click();
  assert.strictEqual(win._isDemoPaused(), true, 'demo rapidly paused again');
  await sleep(100);
  assert.strictEqual(
    players.length,
    0,
    'rapid pause/resume/pause must still keep the pending clip silent',
  );

  btn.click();
  assert.strictEqual(win._isDemoPaused(), false, 'demo finally resumed');
  await waitUntil(
    () => players.length === 1,
    1000,
    'deferred narration clip to start after final resume',
  );
  await done;
  console.log('PASS: pause before clip arrival defers speech until resume');
}

async function testPauseFreezesSpeechAndStreamingThenResumes() {
  const {win} = makeWebview();
  const players = installAudio(win, 600);
  installSpeech(win);
  autoAnswerDemoSpeak(win);

  const done = startDemoFlow(win);
  const btn = win.document.getElementById('demo-pause-btn');
  assert.ok(btn, '#demo-pause-btn exists');

  // Pause while the narration clip is playing.
  await sleep(150);
  assert.ok(players.length > 0, 'narration clip started');
  btn.click();
  assert.strictEqual(
    typeof win._isDemoPaused,
    'function',
    'window._isDemoPaused is exposed',
  );
  assert.strictEqual(win._isDemoPaused(), true, 'demo is paused');
  assert.ok(
    players[players.length - 1].pauseCalls > 0,
    'pausing the demo pauses the playing speech clip',
  );
  let icons = iconState(btn);
  assert.ok(icons.playVisible, 'icon flips to play while paused');
  assert.ok(!icons.pauseVisible, 'pause icon hidden while paused');

  // Resume — the clip plays out and the replay proceeds to streaming.
  btn.click();
  assert.strictEqual(win._isDemoPaused(), false, 'demo resumed');
  icons = iconState(btn);
  assert.ok(icons.pauseVisible, 'icon back to pause after resume');
  assert.ok(!icons.playVisible, 'play icon hidden after resume');

  // Wait for the result panel to start streaming.
  const body = await (async () => {
    const t0 = Date.now();
    while (Date.now() - t0 < 15000) {
      const el = win.document.querySelector('.rc-body');
      if (el && el.textContent.length > 20) return el;
      await sleep(50);
    }
    throw new Error('result panel never started streaming');
  })();

  // Pause: word streaming must stop growing across ~300ms.
  btn.click();
  assert.strictEqual(win._isDemoPaused(), true, 'demo paused mid-stream');
  await sleep(120); // let an already-scheduled tick settle
  const frozenLen = body.textContent.length;
  await sleep(300);
  assert.strictEqual(
    body.textContent.length,
    frozenLen,
    'result streaming must not grow while the demo is paused',
  );
  icons = iconState(btn);
  assert.ok(icons.playVisible, 'play icon shown while paused mid-stream');

  // Resume: streaming grows again.
  btn.click();
  assert.strictEqual(win._isDemoPaused(), false, 'demo resumed mid-stream');
  await sleep(400);
  assert.ok(
    body.textContent.length > frozenLen,
    'result streaming grows again after resume',
  );
  icons = iconState(btn);
  assert.ok(icons.pauseVisible, 'pause icon restored after resume');

  await done;
  console.log('PASS: pause freezes speech and streaming; play resumes');
}

async function testDemoToggleOffCancelsPausedReplayAndRestoresUi() {
  const {win} = makeWebview();
  installAudio(win, 600);
  installSpeech(win);
  autoAnswerDemoSpeak(win);

  const done = startDemoFlow(win);
  await sleep(150);
  win.document.getElementById('demo-pause-btn').click();
  assert.strictEqual(win._isDemoPaused(), true, 'demo paused before toggle off');

  const toggle = win.document.getElementById('cfg-demo-mode');
  assert.ok(toggle, 'demo-mode checkbox exists');
  toggle.checked = false;
  toggle.dispatchEvent(new win.Event('change', {bubbles: true}));

  assert.strictEqual(win._demoApi.active, false, 'toggle off cancels demo');
  assert.strictEqual(
    win._isDemoPaused(),
    false,
    'toggle-off cancel clears paused state',
  );
  assert.ok(
    !win.document.body.classList.contains('demo-playing'),
    'demo UI restored when checkbox is turned off mid-replay',
  );
  assert.strictEqual(
    win.document.getElementById('demo-pause-btn').style.display,
    'none',
    'pause/play button hidden after checkbox cancel',
  );
  await done;
  console.log('PASS: demo-mode checkbox off cancels paused replay cleanly');
}

async function testStopWhilePausedRestoresUiAndNextDemoResetsPause() {
  const {win} = makeWebview();
  installAudio(win, 600);
  installSpeech(win);
  autoAnswerDemoSpeak(win);

  const done = startDemoFlow(win);
  const pauseBtn = win.document.getElementById('demo-pause-btn');
  await sleep(150);
  pauseBtn.click();
  assert.strictEqual(win._isDemoPaused(), true, 'demo paused before stop');

  win.document.getElementById('stop-btn').click();
  assert.strictEqual(
    win._isDemoPaused(),
    false,
    'stop while paused clears the paused state so pauseGate cannot hang',
  );
  assert.strictEqual(win._demoApi.active, false, 'stop cancels paused demo');
  assert.ok(
    !win.document.body.classList.contains('demo-playing'),
    'demo UI restored after stopping while paused',
  );
  await done;

  // Start another demo in the SAME webview.  Reset the posted-message
  // hook so the first startDemoFlow resumeSession responder is not
  // chained and cannot answer the second run twice.
  win._onPosted = null;
  autoAnswerDemoSpeak(win);
  const done2 = startDemoFlow(win);
  await sleep(150);
  assert.ok(win._demoApi.active, 'second demo starts after a paused stop');
  assert.strictEqual(
    win._isDemoPaused(),
    false,
    'second demo starts unpaused',
  );
  const icons = iconState(pauseBtn);
  assert.ok(icons.pauseVisible, 'second demo starts with pause icon shown');
  assert.ok(!icons.playVisible, 'second demo does not inherit play icon');

  win.document.getElementById('stop-btn').click();
  await done2;
  console.log('PASS: stop while paused restores UI; next demo resets pause');
}

async function testStopButtonCancelsDemoAndRestoresControls() {
  const {win} = makeWebview();
  const players = installAudio(win, 600);
  installSpeech(win);
  autoAnswerDemoSpeak(win);

  const done = startDemoFlow(win);
  await sleep(150); // narration clip is playing now
  assert.ok(win._demoApi.active, 'demo replay is running');
  assert.ok(players.length > 0, 'narration clip started');

  win.document.getElementById('stop-btn').click();
  assert.ok(
    players[players.length - 1].pauseCalls > 0,
    'stopping the demo must silence the playing speech clip',
  );
  assert.strictEqual(
    win._demoApi.active,
    false,
    'stop button cancels the demo replay',
  );
  assert.ok(
    !win.document.body.classList.contains('demo-playing'),
    'demo-playing class removed on stop — hidden controls restored',
  );
  assert.strictEqual(
    win.document.getElementById('demo-pause-btn').style.display,
    'none',
    '#demo-pause-btn hidden after stop',
  );
  await done;
  console.log('PASS: stop button cancels demo and restores the controls');
}

(async () => {
  await testControlsHiddenWhileDemoPlays();
  await testPauseBeforeClipArrivesDefersSpeechUntilResume();
  await testPauseFreezesSpeechAndStreamingThenResumes();
  await testStopButtonCancelsDemoAndRestoresControls();
  await testDemoToggleOffCancelsPausedReplayAndRestoresUi();
  await testStopWhilePausedRestoresUiAndNextDemoResetsPause();
  console.log('demoControlsStopPause.test.js: all tests passed');
  // Demo/webview timers can keep the node event loop alive; match the
  // other jsdom e2e demo tests and exit explicitly once assertions pass.
  process.exit(0);
})().catch(err => {
  console.error(err);
  process.exit(1);
});
