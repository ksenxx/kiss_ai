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
//   * replayed ``talk`` calls play their RECORDED audio clip in-page
//     (demo mode never synthesizes speech).  Events without recorded
//     audio — and prompt narration — are skipped SILENTLY and the
//     replay advances; if Audio.play() fails, both VS Code and remote
//     webviews degrade to SILENCE too.  The robotic Web-Speech
//     fallback is gone for good, so the speech stub below is a CANARY
//     that must never record an utterance.
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
  const rejectDelay = (opts && opts.rejectDelay) || 0;
  const players = [];
  win.Audio = function Audio(src) {
    this.src = src;
    players.push(this);
    this.play = () => {
      if (reject) {
        return new Promise((resolve, rejectPlay) => {
          setTimeout(() => rejectPlay(new Error('NotAllowedError')), rejectDelay);
        });
      }
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

function sleep(ms) {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

/** Utterances actually SPOKEN via the Web Speech API — the webview
 * never uses Web Speech anymore, so this must always be empty. */
function realSpoken(spoken) {
  return spoken.filter(u => (u.text || '').trim() !== '');
}

const REPLAY_EVENTS = [
  {type: 'thinking_start'},
  {type: 'thinking_delta', text: 'planning the demo'},
  {
    type: 'tool_call',
    name: 'talk',
    extras: {
      text: 'Hello, I am the demo.',
      language: 'en-US',
      audioB64: 'QUJD',
      audioMime: 'audio/mpeg',
    },
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
  dispatch(win, {type: 'configData', config: {demo_mode: true}, apiKeys: {}});
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

async function testDemoSpeechUsesRecordedClip() {
  const {win} = makeWebview();
  const players = installAudio(win);
  const spoken = installSpeech(win);

  await win._demoApi.playTalkEvent({
    text: 'Replay me.',
    language: 'en-GB',
    audioB64: 'QUJD',
    audioMime: 'audio/mpeg',
  });
  assert.strictEqual(players.length, 1, 'talk replay played an Audio clip');
  assert.ok(
    players[0].src.indexOf('data:audio/mpeg;base64,QUJD') === 0,
    'the RECORDED clip is what plays',
  );
  assert.strictEqual(
    realSpoken(spoken).length,
    0,
    'Web Speech must not be used when the recorded clip is available',
  );

  // Prompt narration never carries audio — it resolves silently.
  await win._demoApi.speakText('User said hello world', 'en-US');
  assert.strictEqual(players.length, 1, 'narration plays no clip');
  assert.strictEqual(realSpoken(spoken).length, 0, 'still no robotic voice');
  console.log('PASS: demo speech plays the recorded clip; narration silent');
}

async function testWebviewNoRecordedAudioIsSkippedSilently() {
  const {win} = makeWebview();
  const players = installAudio(win);
  const spoken = installSpeech(win);

  await Promise.race([
    win._demoApi.playTalkEvent({text: 'No audio was recorded for me.'}),
    sleep(2000).then(() => {
      throw new Error('an audio-less talk event must not stall the replay');
    }),
  ]);
  assert.strictEqual(players.length, 0, 'no clip to play');
  assert.strictEqual(
    realSpoken(spoken).length,
    0,
    'a talk event without recorded audio must be skipped SILENTLY, ' +
      'never spoken by the robotic Web Speech voice',
  );
  console.log('PASS: talk event without recorded audio is skipped silently');
}

async function testWebviewBlockedPlaybackDegradesSilently() {
  const {win} = makeWebview();
  const players = installAudio(win, {reject: true}); // autoplay block
  const spoken = installSpeech(win);

  await Promise.race([
    win._demoApi.playTalkEvent({
      text: 'Blocked playback.',
      audioB64: 'QUJD',
      audioMime: 'audio/mpeg',
    }),
    sleep(2000).then(() => {
      throw new Error('a blocked play() must not stall the replay');
    }),
  ]);
  assert.strictEqual(players.length, 1, 'the clip playback was attempted');
  assert.strictEqual(
    realSpoken(spoken).length,
    0,
    'a blocked play() must degrade to SILENCE, never to Web Speech',
  );
  console.log('PASS: blocked playback degrades to silence');
}

async function testPausedDemoDefersClipPlayback() {
  // A recorded clip whose queue turn comes while the demo pause
  // button is engaged must not start sounding until the user resumes.
  const {win} = makeWebview();
  const players = installAudio(win);
  const spoken = installSpeech(win);

  win._setDemoPaused(true);
  const speech = win._demoApi.playTalkEvent({
    text: 'play only after resume',
    audioB64: 'QUJD',
    audioMime: 'audio/mpeg',
  });
  await sleep(60);
  assert.strictEqual(
    players.length,
    0,
    'the clip must not start playing while the demo is paused',
  );
  win._setDemoPaused(false);
  await speech;
  assert.strictEqual(players.length, 1, 'the clip plays on resume');
  assert.ok(players[0].playedNaturally, 'the recorded clip actually played');
  assert.strictEqual(realSpoken(spoken).length, 0, 'no robotic voice');
  console.log('PASS: pause defers the clip playback until resume');
}

async function testStopBeforePlayRejectionLeavesQueueUsable() {
  // A play() rejection that lands AFTER stopSpeech must neither make
  // any sound nor wedge the serialized talk queue.
  const {win} = makeWebview();
  installAudio(win, {reject: true, rejectDelay: 50});
  const spoken = installSpeech(win);

  const speech = win._demoApi.playTalkEvent({
    text: 'must stay silent after stop',
    audioB64: 'QUJD',
    audioMime: 'audio/mpeg',
  });
  await sleep(30); // the clip started; Audio.play() is still pending
  win._demoApi.stopSpeech();
  await speech;
  await sleep(60); // the delayed rejection settles after the stop
  assert.strictEqual(
    realSpoken(spoken).length,
    0,
    'a late play() rejection after stop must not create zombie speech',
  );
  // The queue must be free for the next demo run.
  await Promise.race([
    win._demoApi.speakText('User said next run'),
    sleep(2000).then(() => {
      throw new Error('talk queue wedged by the late play() rejection');
    }),
  ]);
  console.log('PASS: stop absorbs a late rejected play(), queue stays free');
}

async function testRemotePageAlsoDegradesSilently() {
  // The remote browser page shares the exact same silent-degradation
  // path — no Web Speech fallback anywhere.
  const {win} = makeWebview();
  const players = installAudio(win);
  const spoken = installSpeech(win);
  win.document.body.classList.add('remote-chat');

  await Promise.race([
    win._demoApi.speakText('User said browser fallback', 'en-US'),
    sleep(2000).then(() => {
      throw new Error('silent narration must not stall the remote page');
    }),
  ]);
  assert.strictEqual(players.length, 0, 'no clip to play');
  assert.strictEqual(
    realSpoken(spoken).length,
    0,
    'the remote page must degrade to silence, never Web Speech',
  );
  console.log('PASS: remote chat page degrades to silence too');
}

async function testStopSpeechCancelsQueuedClips() {
  // Stopping the demo must resolve every queued clip promise without
  // sound and leave the talk queue usable for the next run.
  const {win} = makeWebview();
  const players = installAudio(win);
  const spoken = installSpeech(win);

  win._setDemoPaused(true); // park the clip before it starts
  const p = win._demoApi.playTalkEvent({
    text: 'User said cancel me',
    audioB64: 'QUJD',
    audioMime: 'audio/mpeg',
  });
  await sleep(10);
  assert.strictEqual(players.length, 0, 'clip parked behind the pause');

  win._demoApi.stopSpeech();
  win._setDemoPaused(false);
  await Promise.race([
    p,
    sleep(1000).then(() => {
      throw new Error('speech promise did not resolve after stopSpeech');
    }),
  ]);
  await sleep(50);
  assert.strictEqual(players.length, 0, 'a stopped clip must not play');
  assert.strictEqual(realSpoken(spoken).length, 0, 'and must not speak');

  // The queue must be free for the next demo run.
  await win._demoApi.playTalkEvent({
    text: 'User said next run',
    audioB64: 'REVG',
    audioMime: 'audio/mpeg',
  });
  assert.strictEqual(players.length, 1, 'queue serves speech after cancel');
  console.log('PASS: stopSpeech cancels queued clips cleanly');
}

async function runTests() {
  await testFanoutTabsVisibleAndPanelsExpanded();
  await testDemoSpeechUsesRecordedClip();
  await testWebviewNoRecordedAudioIsSkippedSilently();
  await testWebviewBlockedPlaybackDegradesSilently();
  await testPausedDemoDefersClipPlayback();
  await testStopBeforePlayRejectionLeavesQueueUsable();
  await testRemotePageAlsoDegradesSilently();
  await testStopSpeechCancelsQueuedClips();
  console.log('All demoExtensionReplay tests passed.');
  // The demo replay's running state starts webview timers that keep
  // the node event loop alive — exit explicitly.
  process.exit(0);
}

runTests().catch(err => {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
});
