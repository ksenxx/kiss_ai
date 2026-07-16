// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the demo-replay ENDED state (real chat.html +
// main.js + demo.js in jsdom, a recording acquireVsCodeApi stub
// standing in for the extension host / daemon).  When a demo replay
// ends naturally OR is stopped with the stop button:
//
//   * ONLY the play button is shown — the input textbox, burger menu,
//     model picker, attach, inject-promptlet, mic, send AND stop
//     buttons all stay hidden;
//
//   * pressing the play button RESTARTS the demo from the beginning
//     (the events are re-fetched and replayed in the same tab);
//
//   * turning demo mode off, or switching to another tab, dismisses
//     the play button and restores the normal input controls.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/demoEndedRestart.test.js

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
 * *durationMs* and then fire ``onended``.
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

/** A short replayed event stream so replays end quickly. */
function eventsFor(label) {
  return [
    {type: 'thinking_start'},
    {type: 'thinking_delta', text: 'working on ' + label},
    {type: 'result', summary: 'All done for ' + label, total_tokens: 1, cost: '$0'},
  ];
}

/**
 * Enable demo mode, deliver SESSIONS, auto-answer resumeSession with
 * events, click the (single) history row.  Returns a promise resolving
 * when the replay ends.
 */
function startDemoFlow(win) {
  dispatch(win, {type: 'configData', config: {demo_mode: true}, apiKeys: {}});
  dispatch(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: SESSIONS,
  });

  installResumeResponder(win);

  const rows = Array.from(
    win.document.querySelectorAll('#history-list > div'),
  );
  const row = rows.find(
    r => r.textContent.indexOf('Investigate the flaky pipeline') !== -1,
  );
  assert.ok(row, 'history row rendered');
  row.click();

  return waitForDemoEnd(win);
}

/** Answer every posted resumeSession with the recorded events. */
function installResumeResponder(win) {
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
}

/** Wait until the running demo replay finishes. */
async function waitForDemoEnd(win) {
  const t0 = Date.now();
  while (Date.now() - t0 < 30000) {
    await sleep(50);
    if (!win._demoApi.active && Date.now() - t0 > 500) return;
  }
  throw new Error('demo replay did not finish within 30s');
}

/** The 7 input controls that must stay hidden in the ended state. */
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

/**
 * Assert the ended-state UI: ONLY the play button is visible — the
 * demo-ended body class hides the input controls AND the stop button
 * (CSS rules asserted on the real stylesheet text since jsdom does
 * not load <link> stylesheets) while #demo-pause-btn shows the PLAY
 * icon.
 */
function assertEndedUi(win, label) {
  assert.ok(
    !win.document.body.classList.contains('demo-playing'),
    label + ': demo-playing class removed',
  );
  assert.ok(
    win.document.body.classList.contains('demo-ended'),
    label +
      ': body carries the demo-ended class so the input controls and ' +
      'stop button stay hidden',
  );
  const btn = win.document.getElementById('demo-pause-btn');
  assert.notStrictEqual(
    btn.style.display,
    'none',
    label + ': #demo-pause-btn (the play button) stays visible',
  );
  const icons = iconState(btn);
  assert.ok(icons.playVisible, label + ': play icon shown');
  assert.ok(!icons.pauseVisible, label + ': pause icon hidden');
  assert.strictEqual(
    btn.getAttribute('data-tooltip'),
    'Restart demo',
    label + ': tooltip announces the restart',
  );

  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  for (const sel of HIDDEN_SELECTORS) {
    assert.ok(
      css.indexOf('body.demo-ended ' + sel) !== -1,
      label + ': main.css hides ' + sel + ' under body.demo-ended',
    );
  }
  assert.ok(
    /body\.demo-ended #stop-btn\s*\{[^}]*display:\s*none\s*!important/.test(
      css,
    ) ||
      /body\.demo-ended[^{}]*#stop-btn[^{}]*\{[^}]*display:\s*none\s*!important/.test(
        css,
      ),
    label + ': main.css hides #stop-btn under body.demo-ended',
  );
  assert.ok(
    /body\.demo-ended #demo-pause-btn\s*\{[^}]*display:\s*flex\s*!important/.test(
      css,
    ),
    label + ': main.css shows #demo-pause-btn under body.demo-ended',
  );
}

async function testEndedNaturallyShowsOnlyPlayButton() {
  const {win} = makeWebview();
  installAudio(win);
  installSpeech(win);

  await startDemoFlow(win);
  assert.strictEqual(win._demoApi.active, false, 'demo finished');
  assertEndedUi(win, 'natural end');
  console.log('PASS: natural demo end shows only the play button');
}

async function testStoppedShowsOnlyPlayButton() {
  const {win} = makeWebview();
  installAudio(win);
  installSpeech(win);

  const done = startDemoFlow(win);
  await waitUntil(() => win._demoApi.active, 3000, 'demo to start');
  win.document.getElementById('stop-btn').click();
  assert.strictEqual(win._demoApi.active, false, 'stop cancels the demo');
  assertEndedUi(win, 'stopped');
  await done;
  console.log('PASS: stopping the demo shows only the play button');
}

async function testPlayButtonRestartsDemo() {
  const {win, posted} = makeWebview();
  installAudio(win);
  installSpeech(win);

  await startDemoFlow(win);
  assertEndedUi(win, 'before restart');
  const resumesBefore = posted.filter(m => m.type === 'resumeSession').length;
  const tabsBefore = win.document.querySelectorAll(
    '#tab-list .chat-tab',
  ).length;

  // Press play: the demo restarts from the beginning.
  win.document.getElementById('demo-pause-btn').click();
  await waitUntil(
    () => win._demoApi.active,
    3000,
    'demo to restart after pressing play',
  );
  assert.ok(
    win.document.body.classList.contains('demo-playing'),
    'restarted demo shows the playing UI again',
  );
  assert.ok(
    !win.document.body.classList.contains('demo-ended'),
    'demo-ended class cleared while the restarted demo plays',
  );
  const icons = iconState(win.document.getElementById('demo-pause-btn'));
  assert.ok(icons.pauseVisible, 'restarted demo shows the pause icon');
  assert.ok(!icons.playVisible, 'play icon hidden while restarted demo plays');

  await waitForDemoEnd(win);
  const resumesAfter = posted.filter(m => m.type === 'resumeSession').length;
  assert.strictEqual(
    resumesAfter,
    resumesBefore + 1,
    'the restart re-fetched the SAME task events (one more resumeSession)',
  );
  const lastResume = posted.filter(m => m.type === 'resumeSession').pop();
  assert.strictEqual(lastResume.id, 'chat-A', 'restart replays the same chat');
  assert.strictEqual(
    String(lastResume.taskId),
    'a1',
    'restart replays the same task',
  );
  const tabsAfter = win.document.querySelectorAll(
    '#tab-list .chat-tab',
  ).length;
  assert.strictEqual(
    tabsAfter,
    tabsBefore,
    'restart replays in the SAME tab — no new tab is opened',
  );
  const out = win.document.getElementById('output').textContent;
  assert.ok(
    out.indexOf('All done for a1') !== -1,
    'restarted replay streamed the result again',
  );
  assertEndedUi(win, 'after restart finishes');

  // A second restart works too (the ended state is re-armed).
  win.document.getElementById('demo-pause-btn').click();
  await waitUntil(
    () => win._demoApi.active,
    3000,
    'demo to restart a second time',
  );
  win.document.getElementById('stop-btn').click();
  assertEndedUi(win, 'after stopping the second restart');
  console.log('PASS: play button restarts the demo (natural end and stop)');
}

async function testToggleOffFromEndedRestoresFullUi() {
  const {win, posted} = makeWebview();
  installAudio(win);
  installSpeech(win);

  await startDemoFlow(win);
  assertEndedUi(win, 'before toggle off');

  const toggle = win.document.getElementById('cfg-demo-mode');
  toggle.checked = false;
  toggle.dispatchEvent(new win.Event('change', {bubbles: true}));

  assert.ok(
    !win.document.body.classList.contains('demo-ended'),
    'demo-ended class removed when demo mode is turned off',
  );
  assert.ok(
    !win.document.body.classList.contains('demo-playing'),
    'demo-playing class stays off after toggle off',
  );
  assert.strictEqual(
    win.document.getElementById('demo-pause-btn').style.display,
    'none',
    'play button hidden after demo mode is turned off',
  );

  // The play button no longer restarts anything.
  const resumesBefore = posted.filter(m => m.type === 'resumeSession').length;
  win.document.getElementById('demo-pause-btn').click();
  await sleep(200);
  assert.strictEqual(win._demoApi.active, false, 'no replay started');
  assert.strictEqual(
    posted.filter(m => m.type === 'resumeSession').length,
    resumesBefore,
    'no resumeSession posted after demo mode was turned off',
  );
  console.log('PASS: toggling demo mode off dismisses the play button');
}

async function testSwitchingTabClearsEndedUi() {
  const {win} = makeWebview();
  installAudio(win);
  installSpeech(win);

  await startDemoFlow(win);
  assertEndedUi(win, 'before tab switch');

  // The history click opened a NEW tab for the demo, so the original
  // tab is still there — switch back to it.
  const tabEls = Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab'),
  );
  assert.ok(tabEls.length >= 2, 'demo replay opened a second tab');
  const inactive = tabEls.find(el => !el.classList.contains('active'));
  assert.ok(inactive, 'a non-active tab exists');
  inactive.click();

  assert.ok(
    !win.document.body.classList.contains('demo-ended'),
    'demo-ended class removed after switching to another tab',
  );
  assert.strictEqual(
    win.document.getElementById('demo-pause-btn').style.display,
    'none',
    'play button hidden after switching to another tab',
  );
  console.log('PASS: switching tabs dismisses the play button');
}

async function testClosingDemoTabClearsEndedUi() {
  const {win, posted} = makeWebview();
  installAudio(win);
  installSpeech(win);

  await startDemoFlow(win);
  assertEndedUi(win, 'before closing the demo tab');

  // Close the demo's own (active) tab with the tab-bar × — the
  // adjacent tab becomes active and the ended play-button UI must be
  // dismissed: a restart there would wipe THAT tab's chat.
  const activeTab = win.document.querySelector('#tab-list .chat-tab.active');
  assert.ok(activeTab, 'demo tab is the active tab');
  activeTab.querySelector('.chat-tab-close').click();

  assert.ok(
    !win.document.body.classList.contains('demo-ended'),
    'demo-ended class removed after closing the demo tab',
  );
  assert.strictEqual(
    win.document.getElementById('demo-pause-btn').style.display,
    'none',
    'play button hidden after closing the demo tab',
  );

  // The forgotten replay can no longer be restarted into the
  // surviving tab.
  const resumesBefore = posted.filter(m => m.type === 'resumeSession').length;
  win.document.getElementById('demo-pause-btn').click();
  await sleep(200);
  assert.strictEqual(win._demoApi.active, false, 'no replay started');
  assert.strictEqual(
    posted.filter(m => m.type === 'resumeSession').length,
    resumesBefore,
    'no resumeSession posted after the demo tab was closed',
  );
  console.log('PASS: closing the demo tab dismisses the play button');
}

async function testNewHistoryClickFromEndedStartsFreshDemo() {
  const {win} = makeWebview();
  installAudio(win);
  installSpeech(win);

  await startDemoFlow(win);
  assertEndedUi(win, 'before second history click');

  // Clicking a history row while the ended UI shows starts a fresh
  // demo (a new tab, playing UI) — the ended state must not block it.
  const rows = Array.from(
    win.document.querySelectorAll('#history-list > div'),
  );
  const row = rows.find(
    r => r.textContent.indexOf('Investigate the flaky pipeline') !== -1,
  );
  assert.ok(row, 'history row still rendered');
  row.click();
  await waitUntil(
    () => win._demoApi.active,
    3000,
    'fresh demo to start from the ended state',
  );
  assert.ok(
    win.document.body.classList.contains('demo-playing'),
    'fresh demo shows the playing UI',
  );
  assert.ok(
    !win.document.body.classList.contains('demo-ended'),
    'demo-ended cleared when the fresh demo starts',
  );
  await waitForDemoEnd(win);
  assertEndedUi(win, 'after the fresh demo ends');
  console.log('PASS: history click from the ended state starts a fresh demo');
}

(async () => {
  await testEndedNaturallyShowsOnlyPlayButton();
  await testStoppedShowsOnlyPlayButton();
  await testPlayButtonRestartsDemo();
  await testToggleOffFromEndedRestoresFullUi();
  await testSwitchingTabClearsEndedUi();
  await testClosingDemoTabClearsEndedUi();
  await testNewHistoryClickFromEndedStartsFreshDemo();
  console.log('demoEndedRestart.test.js: all tests passed');
  // Demo/webview timers can keep the node event loop alive; match the
  // other jsdom e2e demo tests and exit explicitly once assertions pass.
  process.exit(0);
})().catch(err => {
  console.error(err);
  process.exit(1);
});
