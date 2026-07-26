// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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

async function waitFor(pred, timeoutMs, what) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (pred()) return;
    await sleep(20);
  }
  throw new Error('timed out waiting for ' + what);
}

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
      while (pending.length) pending.shift().resolve();
    },
  };
  win._demoApi = api;
  win.eval(fs.readFileSync(DEMO_PATH, 'utf8'));
  return {win, api, calls, pending};
}

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

  await waitFor(
    () => calls.some(c => c.fn === 'playTalkEvent'),
    5000,
    'talk playback to start',
  );
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

async function testCancelDuringTalkExitsImmediately() {
  const {win, api, calls, pending} = makeDemoWindow(talkEvents());
  const replay = startReplay(win, 'cancel mid talk');

  await waitFor(
    () => calls.some(c => c.fn === 'playTalkEvent'),
    5000,
    'talk playback to start',
  );

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

  const again = startReplay(win, 'restart after cancel');
  await waitFor(() => pending.length >= 1, 5000, 'restart talk');
  assert.strictEqual(pending[0].kind, 'playTalkEvent');
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
  await testCancelDuringTalkExitsImmediately();
  await testLegacyHooksReturningUndefinedStillComplete();
}

runTests().then(
  () => {
    console.log('\n3 passed, 0 failed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    process.exit(1);
  },
);
