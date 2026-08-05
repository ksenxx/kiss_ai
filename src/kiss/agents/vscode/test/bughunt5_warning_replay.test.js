// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function persistedEvents(warningMsg) {
  return [
    {type: 'text_delta', text: 'Working on it.'},
    {type: 'text_end'},
    {type: 'tool_call', name: 'Bash', command: 'make test'},
    {type: 'tool_result', content: 'ok', is_error: false},
    {type: 'warning', message: warningMsg},
    {type: 'result', text: 'summary: done', total_tokens: 10, cost: '$0.01'},
  ];
}

function testReplayRendersPersistedWarning() {
  const {win} = makeWebview();

  const msg =
    'Failed to restore your uncommitted changes: `git stash pop` ' +
    'conflicted. Your changes are preserved in the git stash.';
  send(win, {
    type: 'task_events',
    task: 'fix the bug',
    events: persistedEvents(msg),
  });

  const output = win.document.getElementById('output');
  const text = output ? output.textContent : '';
  assert.ok(
    text.includes('git stash pop'),
    'BUG: a PERSISTED warning event was silently dropped on chat ' +
      'reopen (replayEventsInto/handleOutputEvent has no warning ' +
      'case) — the user never learns their uncommitted changes are ' +
      'stuck in the stash',
  );
  win.close();
  console.log('  ok - persisted warning renders on chat reopen (replay)');
}

function testReplayWarningMatchesLiveRendering() {
  const liveView = makeWebview();
  send(liveView.win, {
    type: 'warning',
    message: 'conflict in <img src=x onerror=alert(1)> & "branch"',
  });
  const liveEl = liveView.win.document.querySelector('#output .warn');
  assert.ok(liveEl, 'live warning banner must exist (bughunt3 fix)');
  const liveHtml = liveEl.outerHTML;
  liveView.win.close();

  const replayView = makeWebview();
  send(replayView.win, {
    type: 'task_events',
    task: 'fix the bug',
    events: persistedEvents(
      'conflict in <img src=x onerror=alert(1)> & "branch"',
    ),
  });
  const replayEl = replayView.win.document.querySelector('#output .warn');
  assert.ok(
    replayEl,
    'BUG: replayed warning banner missing (.warn element not created)',
  );
  assert.strictEqual(
    replayEl.querySelector('img'),
    null,
    'BUG: replayed warning message HTML was not escaped (XSS)',
  );
  assert.strictEqual(
    replayEl.outerHTML,
    liveHtml,
    'replayed warning must render identically to the live banner',
  );
  replayView.win.close();
  console.log('  ok - replayed warning renders identically to live');
}

function testProcessEventRendersWarning() {
  const {win} = makeWebview();
  const api = win._testApi;
  assert.ok(
    api && typeof api.processEvent === 'function',
    '_testApi.processEvent must be exposed by main.js',
  );

  api.processEvent({type: 'text_delta', text: 'thinking...'});
  api.processEvent({
    type: 'warning',
    message: 'replayed stash warning XYZZY',
  });

  const output = win.document.getElementById('output');
  const text = output ? output.textContent : '';
  assert.ok(
    text.includes('replayed stash warning XYZZY'),
    'BUG: a persisted warning event is dropped when replayed ' +
      '(processOutputEvent → handleOutputEvent has no warning case)',
  );
  win.close();
  console.log('  ok - replaying a persisted warning renders it');
}

function testLiveWarningNotDoubleRendered() {
  const {win} = makeWebview();
  send(win, {type: 'warning', message: 'live once QWERTY'});
  const banners = win.document.querySelectorAll('#output .warn');
  assert.strictEqual(
    banners.length,
    1,
    'live warning must render exactly once, got ' + banners.length,
  );
  win.close();
  console.log('  ok - live warning still renders exactly once');
}

function runTests() {
  testReplayRendersPersistedWarning();
  testReplayWarningMatchesLiveRendering();
  testProcessEventRendersWarning();
  testLiveWarningNotDoubleRendered();
}

try {
  runTests();
  console.log('\n4 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
