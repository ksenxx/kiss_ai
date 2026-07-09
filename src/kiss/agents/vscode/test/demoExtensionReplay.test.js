// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression tests for demo mode running inside the VS Code
// EXTENSION webview (real chat.html + main.js + demo.js in jsdom, a
// recording acquireVsCodeApi stub standing in for the extension host /
// daemon).  Reproduces two field-reported bugs:
//
//   * "I do not see any tab": processOutputEvent's automatic
//     collapseOlderPanels() pass fired on the very next replayed event
//     — milliseconds later in a demo — collapsing every demo panel
//     instantly and closing a replayed run_parallel fan-out's
//     sub-agent tabs right after they opened, so no tab was ever
//     visible.  collapseOlderPanels must be a no-op while the demo is
//     active (demo.js owns collapsing), and fan-out groups pause
//     longer so the tabs are actually watchable.
//
//   * "robotic voice in the extension": demo narration and replayed
//     ``talk`` calls were read with the Web Speech API (the robotic
//     system voice).  The webview now asks the daemon to synthesize a
//     natural GPT-voice clip ('demoSpeak' -> 'demoSpeakAudio') and
//     plays it in-page; the Web Speech fallback survives ONLY on the
//     remote browser chat page — the VS Code webview degrades to
//     silence, never to the robotic voice.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/demoExtensionReplay.test.js

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
 * Install a recording Audio implementation.  ``opts.reject`` makes
 * play() reject like a Chromium autoplay block; otherwise clips play
 * and fire ``onended`` after 30ms.  Returns the created players.
 */
function installAudio(win, opts) {
  const reject = !!(opts && opts.reject);
  const players = [];
  win.Audio = function Audio(src) {
    this.src = src;
    players.push(this);
    this.play = () => {
      if (reject) return Promise.reject(new Error('NotAllowedError'));
      this.playedNaturally = true;
      setTimeout(() => {
        if (typeof this.onended === 'function') this.onended();
      }, 30);
      return Promise.resolve();
    };
  };
  return players;
}

/**
 * Install a recording Web Speech API whose utterances complete
 * asynchronously.  Returns the spoken utterances.
 */
function installSpeech(win) {
  const spoken = [];
  win.SpeechSynthesisUtterance = function (text) {
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

/** Deliver a daemon/extension-host message to the webview. */
function dispatch(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Answer every posted 'demoSpeak' with a synthesized clip (or an
 * empty audioB64 when ``audioB64`` is passed as '') after 20ms. */
function autoAnswerDemoSpeak(win, audioB64) {
  const clip = audioB64 === undefined ? 'QUJD' : audioB64;
  const prev = win._onPosted;
  win._onPosted = msg => {
    if (prev) prev(msg);
    if (msg.type !== 'demoSpeak') return;
    setTimeout(() => {
      dispatch(win, {
        type: 'demoSpeakAudio',
        reqId: msg.reqId,
        audioB64: clip,
        audioMime: 'audio/mpeg',
        tabId: msg.tabId,
      });
    }, 20);
  };
}

function sleep(ms) {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

/** Utterances actually SPOKEN via the Web Speech API, ignoring the
 * empty click-unlock utterance main.js speaks on the first click. */
function realSpoken(spoken) {
  return spoken.filter(u => (u.text || '').trim() !== '');
}

const REPLAY_EVENTS = [
  {type: 'thinking_start'},
  {type: 'thinking_delta', text: 'planning the demo'},
  {
    type: 'tool_call',
    name: 'talk',
    extras: {text: 'Hello, I am the demo.', language: 'en-US'},
  },
  {type: 'tool_result', content: 'ok'},
  {
    type: 'tool_call',
    name: 'run_parallel',
    extras: {tasks: '["research topic A", "summarize topic B"]'},
  },
  {type: 'tool_result', content: 'done'},
  {type: 'result', summary: 'All done.', total_tokens: 10, cost: '$0.01'},
];

/**
 * Drive the full extension demo flow: enable demo mode, deliver one
 * history session, answer resumeSession with REPLAY_EVENTS, click the
 * history row.  Returns a promise resolving when the replay ends.
 */
function startDemoFlow(win) {
  dispatch(win, {type: 'updateSetting', key: 'demo_mode', value: true});
  dispatch(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      {
        id: 'chat-1',
        preview: 'Do the demo task',
        title: 'Do the demo task',
        has_events: true,
        ts: Date.now() / 1000,
      },
    ],
  });
  const prev = win._onPosted;
  win._onPosted = msg => {
    if (prev) prev(msg);
    if (msg.type !== 'resumeSession') return;
    setTimeout(() => {
      dispatch(win, {
        type: 'task_events',
        tabId: msg.tabId,
        events: REPLAY_EVENTS,
        task: 'Do the demo task',
        chat_id: 'chat-1',
        extra: '',
      });
    }, 10);
  };
  const row = win.document.querySelector('#history-list > div');
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

async function testFanoutTabsVisibleAndPanelsExpanded() {
  const {win, posted} = makeWebview();
  installAudio(win);
  const spoken = installSpeech(win);
  autoAnswerDemoSpeak(win);

  const doc = win.document;
  let maxSubTabs = 0;
  let subTabsVisibleMs = 0;
  let rpExpandedWhileTabsOpen = false;
  let lastSaw = 0;
  const poll = setInterval(() => {
    const n = doc.querySelectorAll('.subagent-tab').length;
    if (n > maxSubTabs) maxSubTabs = n;
    const now = Date.now();
    if (n > 0) {
      if (lastSaw) subTabsVisibleMs += now - lastSaw;
      lastSaw = now;
      const rp = doc.querySelector('.tc-run-parallel');
      if (rp && !rp.classList.contains('collapsed')) {
        rpExpandedWhileTabsOpen = true;
      }
    } else {
      lastSaw = 0;
    }
  }, 25);

  await startDemoFlow(win);
  clearInterval(poll);

  assert.strictEqual(
    maxSubTabs,
    2,
    'both replayed fan-out sub-agent tabs must appear in the tab bar',
  );
  assert.ok(
    subTabsVisibleMs >= 1500,
    'sub-agent tabs must stay visible for the long fan-out pause ' +
      '(saw ' +
      subTabsVisibleMs +
      'ms)',
  );
  assert.ok(
    rpExpandedWhileTabsOpen,
    'the run_parallel panel must stay EXPANDED while its tabs are open ' +
      '(the automatic collapse pass must not fire during demo replay)',
  );
  assert.strictEqual(
    doc.querySelectorAll('.subagent-tab').length,
    0,
    'sub-agent tabs close when the demo collapses the fan-out panel',
  );
  assert.ok(doc.querySelector('#output .rc'), 'result panel rendered');
  assert.strictEqual(
    realSpoken(spoken).length,
    0,
    'no demo utterance may use the robotic Web Speech voice in the ' +
      'extension webview',
  );
  const closes = posted.filter(m => m.type === 'closeTab');
  assert.strictEqual(closes.length, 2, 'both sub tabs closed via backend');
  console.log('PASS: fan-out tabs visible >=1.5s, panels stay expanded');
}

async function testDemoSpeechUsesNaturalClip() {
  const {win} = makeWebview();
  const players = installAudio(win);
  const spoken = installSpeech(win);
  autoAnswerDemoSpeak(win);

  await win._demoApi.speakText('User said hello world', 'en-US');
  assert.strictEqual(players.length, 1, 'narration played as an Audio clip');
  assert.ok(
    players[0].src.indexOf('data:audio/mpeg;base64,QUJD') === 0,
    'the synthesized clip from demoSpeakAudio is what plays',
  );
  assert.strictEqual(
    realSpoken(spoken).length,
    0,
    'Web Speech must not be used when the natural clip is available',
  );

  await win._demoApi.playTalkEvent({text: 'Replay me.', language: 'en-GB'});
  assert.strictEqual(players.length, 2, 'talk replay also plays a clip');
  assert.strictEqual(realSpoken(spoken).length, 0, 'still no robotic voice');
  console.log('PASS: demo speech plays the natural synthesized clip');
}

async function testWebviewFailedSynthesisIsSilent() {
  const {win} = makeWebview();
  const players = installAudio(win);
  const spoken = installSpeech(win);
  autoAnswerDemoSpeak(win, ''); // synthesis failed: empty audioB64

  await win._demoApi.speakText('User said no key available');
  assert.strictEqual(players.length, 0, 'no clip to play');
  assert.strictEqual(
    realSpoken(spoken).length,
    0,
    'failed synthesis must degrade to SILENCE in the webview, ' +
      'never to the robotic Web Speech voice',
  );
  console.log('PASS: failed synthesis is silent in the webview');
}

async function testWebviewBlockedPlaybackIsSilent() {
  const {win} = makeWebview();
  installAudio(win, {reject: true}); // autoplay policy blocks play()
  const spoken = installSpeech(win);
  autoAnswerDemoSpeak(win);

  await win._demoApi.speakText('User said blocked playback');
  await sleep(20); // let the rejected play() settle its microtask
  assert.strictEqual(
    realSpoken(spoken).length,
    0,
    'a blocked play() must degrade to SILENCE in the webview',
  );
  console.log('PASS: blocked clip playback is silent in the webview');
}

async function testRemotePageKeepsWebSpeechFallback() {
  const {win} = makeWebview();
  installAudio(win);
  const spoken = installSpeech(win);
  win.document.body.classList.add('remote-chat');
  autoAnswerDemoSpeak(win, ''); // synthesis failed

  await win._demoApi.speakText('User said browser fallback', 'en-US');
  const texts = realSpoken(spoken)
    .map(u => u.text)
    .join(' ');
  assert.ok(
    texts.indexOf('browser fallback') !== -1,
    'the remote browser page keeps the Web Speech fallback',
  );
  console.log('PASS: remote chat page falls back to Web Speech');
}

async function testStopSpeechCancelsPendingSynthesis() {
  const {win, posted} = makeWebview();
  const players = installAudio(win);
  const spoken = installSpeech(win);
  // No auto-answer: the synthesis request stays pending.

  const p = win._demoApi.speakText('User said cancel me');
  await sleep(10);
  const req = posted.find(m => m.type === 'demoSpeak');
  assert.ok(req, 'synthesis was requested');

  win._demoApi.stopSpeech();
  await Promise.race([
    p,
    sleep(1000).then(() => {
      throw new Error('speech promise did not resolve after stopSpeech');
    }),
  ]);

  // A LATE synthesis reply after the cancel must not start playback.
  dispatch(win, {
    type: 'demoSpeakAudio',
    reqId: req.reqId,
    audioB64: 'QUJD',
    audioMime: 'audio/mpeg',
  });
  await sleep(50);
  assert.strictEqual(players.length, 0, 'late reply must not play audio');
  assert.strictEqual(realSpoken(spoken).length, 0, 'and must not speak');

  // The queue must be free for the next demo run.
  autoAnswerDemoSpeak(win);
  await win._demoApi.speakText('User said next run');
  assert.strictEqual(players.length, 1, 'queue serves speech after cancel');
  console.log('PASS: stopSpeech cancels pending synthesis cleanly');
}

async function runTests() {
  await testFanoutTabsVisibleAndPanelsExpanded();
  await testDemoSpeechUsesNaturalClip();
  await testWebviewFailedSynthesisIsSilent();
  await testWebviewBlockedPlaybackIsSilent();
  await testRemotePageKeepsWebSpeechFallback();
  await testStopSpeechCancelsPendingSynthesis();
  console.log('All demoExtensionReplay tests passed.');
  // The demo replay's running state starts webview timers that keep
  // the node event loop alive — exit explicitly.
  process.exit(0);
}

runTests().catch(err => {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
});
