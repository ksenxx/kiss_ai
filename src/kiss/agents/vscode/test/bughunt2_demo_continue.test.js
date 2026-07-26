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
  return new Promise(r => setTimeout(r, ms));
}

function makeDemoWindow(events) {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body><div id="output"></div></body></html>',
    {runScripts: 'dangerously', pretendToBeVisual: true},
  );
  const win = dom.window;
  let active = false;
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
    processEvent() {},
    setTaskText() {},
    updateTabTitle() {},
    hideWelcome() {},
    scrollToBottom() {},
    getActiveTabId() {
      return 'demo-tab';
    },
    sendMessage(msg) {
      if (msg && msg.type === 'resumeSession') {
        setTimeout(() => {
          if (api.resolveEvents) api.resolveEvents(events);
        }, 10);
      }
    },
    collapsePanels() {},
    setRunningState() {},
    showSpinner() {},
    removeSpinner() {},
  };
  win._demoApi = api;
  win.eval(fs.readFileSync(DEMO_PATH, 'utf8'));
  return {win, api};
}

async function runReplay(win, api) {
  const replay = win._startDemoReplay([
    {id: 1, has_events: true, preview: 'continue this big task', timestamp: 1},
  ]);
  await replay;
  assert.strictEqual(api.active, false, 'replay must clear active flag');
}

async function testContinueResultIsNotLabelledFailed() {
  const {win, api} = makeDemoWindow([
    {
      type: 'result',
      success: false,
      is_continue: true,
      summary: 'Pausing here; will continue in a fresh session.',
      total_tokens: 1234,
      cost: '$0.10',
    },
  ]);
  await runReplay(win, api);

  const text = win.document.getElementById('output').textContent;
  assert.ok(
    !text.includes('Status: FAILED'),
    'BUG: demo replay labels a paused-to-continue result as ' +
      '"Status: FAILED" (main.js renders "Status: Continue" for ' +
      'is_continue results)',
  );
  assert.ok(
    text.includes('Status: Continue'),
    'demo replay must render the "Status: Continue" banner like main.js',
  );
  win.close();
  console.log('  ok - is_continue result renders "Status: Continue"');
}

async function testGenuineFailureStillLabelledFailed() {
  const {win, api} = makeDemoWindow([
    {
      type: 'result',
      success: false,
      summary: 'Could not finish.',
      total_tokens: 99,
      cost: '$0.01',
    },
  ]);
  await runReplay(win, api);

  const text = win.document.getElementById('output').textContent;
  assert.ok(
    text.includes('Status: FAILED'),
    'a genuinely failed result must keep its FAILED banner',
  );
  assert.ok(
    !text.includes('Status: Continue'),
    'a genuinely failed result must not show Continue',
  );
  win.close();
  console.log('  ok - genuinely failed result still renders "Status: FAILED"');
}

async function runTests() {
  await testContinueResultIsNotLabelledFailed();
  await testGenuineFailureStillLabelledFailed();
}

runTests().then(
  () => {
    console.log('\n2 passed, 0 failed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    process.exit(1);
  },
);
