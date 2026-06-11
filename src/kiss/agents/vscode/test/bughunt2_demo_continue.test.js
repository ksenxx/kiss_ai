// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Bug-hunt integration test for demo-mode result rendering in
// ``media/demo.js``.
//
// Bug locked in:
//
//   ``streamResultEvent`` renders the status banner for a replayed
//   ``result`` event with ONLY a ``success === false`` branch
//   ("Status: FAILED").  The canonical renderer in ``media/main.js``
//   (``handleOutputEvent``, case 'result') checks ``is_continue``
//   FIRST and renders "Status: Continue" for results of agents that
//   paused to continue in a new session — ``types.ts`` documents
//   ``is_continue`` as exactly that ("main.js renders a 'Status:
//   Continue' banner for it").  Continue-results are emitted with
//   ``success: false`` by the backend, so demo replay mislabels every
//   paused-to-continue task as FAILED.
//
// This test drives the real ``media/demo.js`` inside jsdom (no mocks
// of project code; the ``window._demoApi`` host shim that main.js
// normally provides is stubbed, exactly like the ``vscode`` host stub
// in bughunt_isNewFile.test.js).
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/bughunt2_demo_continue.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const DEMO_PATH = path.join(__dirname, '..', 'media', 'demo.js');

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

/**
 * Build a jsdom window with ``demo.js`` evaluated and a minimal
 * ``window._demoApi`` host shim that hands the supplied *events* back
 * when the replay requests them via ``resumeSession``.
 */
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
        // Deliver the stored events asynchronously, like the backend.
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
  // The replay pauses 2 s showing the task text, then streams the
  // result panel word-by-word; wait for it to finish.
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
