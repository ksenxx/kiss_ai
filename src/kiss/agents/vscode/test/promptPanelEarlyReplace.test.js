// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Early prompt-panel test: the server broadcasts optimistic
// ``early``-flagged ``system_prompt`` / ``prompt`` events the moment a
// task is submitted (``_broadcast_early_prompts`` in task_runner.py),
// so the panels appear immediately instead of seconds later when the
// inner agent emits the authoritative events.  When the authoritative
// (non-early) events arrive, the webview must REPLACE the pending
// early panels in place — never append duplicates.  Later sub-session
// prompt events (no pending early panel) must still append.
//
// Drives the real ``media/main.js`` inside jsdom, exactly like
// ``bughunt3_warning_event.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/promptPanelEarlyReplace.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/** Build a jsdom window running the production chat webview. */
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

  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function panels(win, cls) {
  return win.document.querySelectorAll('#output .ev.' + cls);
}

function testEarlyPanelsRenderImmediately() {
  const {win} = makeWebview();

  send(win, {type: 'system_prompt', text: 'early SYSTEM text', early: true});
  send(win, {type: 'prompt', text: 'early USER text', early: true});

  const sys = panels(win, 'system-prompt');
  const pr = panels(win, 'prompt');
  assert.strictEqual(sys.length, 1, 'one early system-prompt panel');
  assert.strictEqual(pr.length, 1, 'one early prompt panel');
  assert.ok(sys[0].textContent.includes('early SYSTEM text'));
  assert.ok(pr[0].textContent.includes('early USER text'));
  assert.strictEqual(sys[0].dataset.early, '1');
  assert.strictEqual(pr[0].dataset.early, '1');
  win.close();
  console.log('  ok - early panels render immediately at submit');
}

function testAuthoritativeEventReplacesEarlyPanel() {
  const {win} = makeWebview();

  send(win, {type: 'system_prompt', text: 'early SYSTEM text', early: true});
  send(win, {type: 'prompt', text: 'early USER text', early: true});
  send(win, {type: 'system_prompt', text: 'authoritative SYSTEM text'});
  send(win, {type: 'prompt', text: 'authoritative FULL prompt'});

  const sys = panels(win, 'system-prompt');
  const pr = panels(win, 'prompt');
  assert.strictEqual(
    sys.length,
    1,
    'BUG: authoritative system_prompt appended a duplicate panel ' +
      'instead of replacing the pending early one',
  );
  assert.strictEqual(
    pr.length,
    1,
    'BUG: authoritative prompt appended a duplicate panel ' +
      'instead of replacing the pending early one',
  );
  assert.ok(sys[0].textContent.includes('authoritative SYSTEM text'));
  assert.ok(!sys[0].textContent.includes('early SYSTEM text'));
  assert.ok(pr[0].textContent.includes('authoritative FULL prompt'));
  assert.ok(!pr[0].textContent.includes('early USER text'));
  assert.strictEqual(sys[0].dataset.early, undefined);
  assert.strictEqual(pr[0].dataset.early, undefined);
  // Copy button must reproduce the authoritative markdown.
  assert.strictEqual(pr[0].dataset.rawText, 'authoritative FULL prompt');
  win.close();
  console.log('  ok - authoritative events replace early panels in place');
}

function testLaterSessionsStillAppend() {
  const {win} = makeWebview();

  send(win, {type: 'prompt', text: 'early USER text', early: true});
  send(win, {type: 'prompt', text: 'session-0 prompt'});
  send(win, {type: 'prompt', text: 'session-1 prompt'});

  const pr = panels(win, 'prompt');
  assert.strictEqual(
    pr.length,
    2,
    'a second authoritative prompt (relentless sub-session 1) must ' +
      'append a new panel — only the pending early panel is replaced',
  );
  assert.ok(pr[0].textContent.includes('session-0 prompt'));
  assert.ok(pr[1].textContent.includes('session-1 prompt'));
  win.close();
  console.log('  ok - later sub-session prompts still append panels');
}

function testReplayWithoutEarlyEventsUnchanged() {
  const {win} = makeWebview();

  // Persisted replay streams never contain early events: plain
  // authoritative events must append exactly as before the fix.
  send(win, {type: 'system_prompt', text: 'replayed SYSTEM'});
  send(win, {type: 'prompt', text: 'replayed PROMPT'});

  assert.strictEqual(panels(win, 'system-prompt').length, 1);
  assert.strictEqual(panels(win, 'prompt').length, 1);
  assert.ok(
    panels(win, 'prompt')[0].textContent.includes('replayed PROMPT'),
  );
  win.close();
  console.log('  ok - replay (no early events) renders unchanged');
}

function runTests() {
  testEarlyPanelsRenderImmediately();
  testAuthoritativeEventReplacesEarlyPanel();
  testLaterSessionsStillAppend();
  testReplayWithoutEarlyEventsUnchanged();
}

try {
  runTests();
  console.log('\n4 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
