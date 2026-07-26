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
      while (pending.length) pending.shift().resolve();
    },
  };
  win._demoApi = api;
  win.eval(fs.readFileSync(DEMO_PATH, 'utf8'));
  return {win, api, calls, pending};
}

const PROMPT_TEXT = 'please also add documentation';

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

  await waitFor(
    () => calls.some(c => c.fn === 'playTalkEvent'),
    5000,
    'prompt narration to start',
  );
  await sleep(1200);
  assert.ok(
    !resultRendered(win),
    'replay must not stream the result while the prompt narration plays',
  );

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
