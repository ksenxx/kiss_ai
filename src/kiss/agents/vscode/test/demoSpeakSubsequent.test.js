// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for the field-reported demo-mode speech
// bug: "after you speak once, you fail to speak for all subsequent
// speeches in the demo or when you load another task in the demo
// mode."  Every replayed ``talk`` tool call must actually SOUND:
//
//   1. a single demo task containing several ``talk`` tool calls must
//      play every one of them (not just the first);
//   2. after one demo replay finishes, loading ANOTHER task in demo
//      mode must play that task's ``talk`` calls too;
//   3. after a demo is STOPPED mid-speech, the next demo replay must
//      still play its ``talk`` calls.
//
// Drives the production webview (real chat.html + main.js + demo.js
// in jsdom, recording acquireVsCodeApi stub as the extension host /
// daemon), exactly like demoExtensionReplay.test.js.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/demoSpeakSubsequent.test.js

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
 * Install a recording Audio implementation: clips play and fire
 * ``onended`` after 30ms.  Returns the created players (each gets
 * ``playedNaturally = true`` once play() ran).
 */
function installAudio(win) {
  const players = [];
  win.Audio = function Audio(src) {
    this.src = src;
    players.push(this);
    this.play = () => {
      this.playedNaturally = true;
      setTimeout(() => {
        if (typeof this.onended === 'function') this.onended();
      }, 30);
      return Promise.resolve();
    };
    this.pause = () => {};
  };
  return players;
}

/** Deliver a daemon/extension-host message to the webview. */
function dispatch(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Answer every posted 'demoSpeak' with a synthesized clip after
 * 20ms, exactly like the daemon's _cmd_demo_speak reply. */
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
    }, 20);
  };
}

/** Answer every posted 'demoSpeak' the way the daemon answers a LOCAL
 * VS Code webview: the daemon plays the clip natively on its own
 * machine's speakers and stamps the reply ``muted: true`` (see
 * WebPrinter._arbitrate_demo_speak in web_server.py — an unmuted
 * in-page play() would be autoplay-rejected without a fresh user
 * gesture, microsoft/vscode#197937). */
function autoAnswerDemoSpeakMuted(win) {
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
        muted: true,
        tabId: msg.tabId,
      });
    }, 20);
  };
}

/** Answer every posted 'resumeSession' with that chat's events. */
function autoAnswerResumeSession(win, eventsByChatId) {
  const prev = win._onPosted;
  win._onPosted = msg => {
    if (prev) prev(msg);
    if (msg.type !== 'resumeSession') return;
    const events = eventsByChatId[msg.id] || [];
    setTimeout(() => {
      dispatch(win, {
        type: 'task_events',
        tabId: msg.tabId,
        task_id: msg.taskId,
        events: events,
        task: 'demo task',
        chat_id: msg.id,
        extra: '',
      });
    }, 10);
  };
}

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
    await sleep(25);
  }
  throw new Error('timed out waiting for ' + what);
}

/** Events of a demo task speaking twice via the ``talk`` tool. */
function twoTalkEvents() {
  return [
    {type: 'thinking_start'},
    {type: 'thinking_delta', text: 'planning the demo'},
    {
      type: 'tool_call',
      name: 'talk',
      extras: {text: 'First utterance.', language: 'en-US'},
    },
    {type: 'tool_result', content: 'ok'},
    {type: 'thinking_start'},
    {type: 'thinking_delta', text: 'more work'},
    {
      type: 'tool_call',
      name: 'talk',
      extras: {text: 'Second utterance.', language: 'en-US'},
    },
    {type: 'tool_result', content: 'ok'},
    {type: 'result', summary: 'All done.', total_tokens: 10, cost: '$0.01'},
  ];
}

/** Events of a demo task speaking once via the ``talk`` tool. */
function oneTalkEvents(text) {
  return [
    {type: 'thinking_start'},
    {type: 'thinking_delta', text: 'thinking'},
    {
      type: 'tool_call',
      name: 'talk',
      extras: {text: text, language: 'en-US'},
    },
    {type: 'tool_result', content: 'ok'},
    {type: 'result', summary: 'Done.', total_tokens: 5, cost: '$0.01'},
  ];
}

/** One history session row as the backend's 'history' message lists it. */
function sessionRow(chatId, taskId, preview) {
  return {
    id: chatId,
    task_id: taskId,
    preview: preview,
    title: preview,
    has_events: true,
    ts: Date.now() / 1000,
  };
}

/** Enable demo mode and deliver *sessions* to the history sidebar. */
function setupDemoHistory(win, sessions) {
  dispatch(win, {type: 'updateSetting', key: 'demo_mode', value: true});
  dispatch(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: sessions,
  });
}

/** Click the *index*-th rendered history row. */
function clickHistoryRow(win, index) {
  const rows = win.document.querySelectorAll('#history-list > div');
  assert.ok(rows[index], 'history row ' + index + ' rendered');
  rows[index].click();
}

/** Wait until the running demo replay ends. */
async function waitForDemoEnd(win) {
  await sleep(300);
  await waitFor(() => !win._demoApi.active, 30000, 'demo replay end');
}

/** The talk texts actually PLAYED through Audio elements (decoded
 * from the demoSpeak requests that produced each clip). */
function playedCount(players) {
  return players.filter(p => p.playedNaturally).length;
}

async function testAllTalksInOneDemoAreSpoken() {
  const {win, posted} = makeWebview();
  const players = installAudio(win);
  autoAnswerDemoSpeak(win);
  autoAnswerResumeSession(win, {'chat-1': twoTalkEvents()});

  setupDemoHistory(win, [sessionRow('chat-1', 'task-1', 'Two talks demo')]);
  clickHistoryRow(win, 0);
  await waitForDemoEnd(win);

  const speaks = posted.filter(m => m.type === 'demoSpeak');
  assert.strictEqual(
    speaks.length,
    2,
    'both replayed talk calls must request synthesis (got ' +
      speaks.length +
      ')',
  );
  assert.strictEqual(
    playedCount(players),
    2,
    'BOTH talk utterances of the demo must actually play (got ' +
      playedCount(players) +
      ')',
  );
  console.log('PASS: every talk in a single demo replay is spoken');
}

async function testSecondDemoTaskStillSpeaks() {
  const {win, posted} = makeWebview();
  const players = installAudio(win);
  autoAnswerDemoSpeak(win);
  autoAnswerResumeSession(win, {
    'chat-1': oneTalkEvents('Hello from task one.'),
    'chat-2': oneTalkEvents('Hello from task two.'),
  });

  setupDemoHistory(win, [
    sessionRow('chat-1', 'task-1', 'First demo task'),
    sessionRow('chat-2', 'task-2', 'Second demo task'),
  ]);

  clickHistoryRow(win, 0);
  await waitForDemoEnd(win);
  assert.strictEqual(
    playedCount(players),
    1,
    'first demo task spoke once (got ' + playedCount(players) + ')',
  );

  clickHistoryRow(win, 1);
  await waitForDemoEnd(win);

  const speaks = posted.filter(m => m.type === 'demoSpeak');
  assert.strictEqual(
    speaks.length,
    2,
    'the second demo task must request synthesis too (got ' +
      speaks.length +
      ')',
  );
  assert.strictEqual(
    playedCount(players),
    2,
    'loading another task in demo mode must speak again (got ' +
      playedCount(players) +
      ')',
  );
  console.log('PASS: a second demo-mode task load speaks again');
}

async function testDemoStoppedMidSpeechThenNextDemoSpeaks() {
  const {win} = makeWebview();
  const players = installAudio(win);
  autoAnswerDemoSpeak(win);
  autoAnswerResumeSession(win, {
    'chat-1': oneTalkEvents('Hello from task one.'),
    'chat-2': oneTalkEvents('Hello from task two.'),
  });

  setupDemoHistory(win, [
    sessionRow('chat-1', 'task-1', 'First demo task'),
    sessionRow('chat-2', 'task-2', 'Second demo task'),
  ]);

  clickHistoryRow(win, 0);
  // Stop the demo while its first speech is queued/in flight.
  await sleep(60);
  win._cancelDemoReplay();
  await waitFor(() => !win._demoApi.active, 5000, 'stopped demo teardown');

  clickHistoryRow(win, 1);
  await waitForDemoEnd(win);

  assert.ok(
    players.some(
      p =>
        p.playedNaturally &&
        p.src.indexOf('data:audio/mpeg;base64,QUJD') === 0,
    ),
    'the demo replay started after a mid-speech stop must still speak',
  );
  console.log('PASS: demo stopped mid-speech does not silence the next demo');
}

async function testDaemonMutedClipsStayMutedAndKeepDemoPacing() {
  const {win, posted} = makeWebview();
  const players = installAudio(win);
  autoAnswerDemoSpeakMuted(win);
  autoAnswerResumeSession(win, {'chat-1': twoTalkEvents()});

  setupDemoHistory(win, [sessionRow('chat-1', 'task-1', 'Muted demo')]);
  clickHistoryRow(win, 0);
  await waitForDemoEnd(win);

  const speaks = posted.filter(m => m.type === 'demoSpeak');
  assert.strictEqual(
    speaks.length,
    2,
    'both replayed talk calls must request synthesis (got ' +
      speaks.length +
      ')',
  );
  const played = players.filter(p => p.playedNaturally);
  assert.strictEqual(
    played.length,
    2,
    'muted daemon-played clips must still play in-page so their ' +
      "'ended' events pace the demo (got " +
      played.length +
      ')',
  );
  assert.ok(
    played.every(p => p.muted === true),
    'every in-page copy of a daemon-played clip must have muted=true ' +
      '(the daemon supplies the sound; an unmuted copy would double-play ' +
      'or get autoplay-rejected)',
  );
  console.log(
    'PASS: daemon-muted demo clips play muted in-page and keep pacing',
  );
}

async function testUnmutedClipsPlayAudiblyInPage() {
  const {win} = makeWebview();
  const players = installAudio(win);
  autoAnswerDemoSpeak(win);
  autoAnswerResumeSession(win, {
    'chat-1': oneTalkEvents('Hello from a remote browser tab.'),
  });

  setupDemoHistory(win, [sessionRow('chat-1', 'task-1', 'Remote demo')]);
  clickHistoryRow(win, 0);
  await waitForDemoEnd(win);

  const played = players.filter(p => p.playedNaturally);
  assert.strictEqual(played.length, 1, 'the clip must play');
  assert.ok(
    played.every(p => !p.muted),
    'a reply WITHOUT the muted stamp (remote WSS browser tab — another ' +
      'device) must keep its in-page playback audible',
  );
  console.log('PASS: unmuted demo clips stay audible in-page');
}

async function main() {
  await testAllTalksInOneDemoAreSpoken();
  await testSecondDemoTaskStillSpeaks();
  await testDemoStoppedMidSpeechThenNextDemoSpeaks();
  await testDaemonMutedClipsStayMutedAndKeepDemoPacing();
  await testUnmutedClipsPlayAudiblyInPage();
  console.log('ALL TESTS PASSED');
  // main.js leaves webview keep-alive timers running; exit explicitly.
  process.exit(0);
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
