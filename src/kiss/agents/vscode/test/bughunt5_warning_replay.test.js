// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Bug-hunt 5 integration test: persisted ``warning`` events must be
// rendered on REPLAY (chat reopen / demo replay), not only live.
//
// Bug locked in:
//
//   Iteration 4 added ``"warning"`` to ``_DISPLAY_EVENT_TYPES`` in
//   ``json_printer.py`` so backend warnings (e.g. the worktree agent's
//   stash-pop failure) are recorded and persisted with the task's
//   event stream.  The LIVE path renders them via the top-level
//   message switch in ``media/main.js`` (``case 'warning'`` →
//   ``addWarning``).  But the REPLAY paths — ``replayEventsInto``
//   (used by ``task_events`` when a chat is reopened, and for
//   background/sub-agent tabs) and ``processOutputEvent`` (used by
//   ``demo.js`` panel replay) — both route every event through
//   ``handleOutputEvent``, whose switch has NO ``case 'warning'``.
//   A persisted warning therefore silently vanishes when the task is
//   reopened or demo-replayed, even though it was persisted precisely
//   so it would survive replay.
//
// This test drives the real ``media/main.js`` (plus the real
// ``media/chat.html`` markup and ``media/panelCopy.js``) inside jsdom,
// exactly like ``bughunt3_warning_event.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/bughunt5_warning_replay.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview: the real
 * ``chat.html`` body (placeholders blanked), ``panelCopy.js`` and
 * ``main.js`` evaluated in the window, and a recording
 * ``acquireVsCodeApi`` stub (the only host API the webview has).
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
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

// The exact persisted event stream of a worktree task whose stash-pop
// failed: the warning is interleaved between display events, exactly
// as json_printer records it.
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
  // Chat reopen: the server replays the persisted event stream via a
  // single ``task_events`` message.
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
  // The replayed banner must be rendered identically to the live one
  // (same ``ev tr warn`` element, same escaped "Warning: ..." text).
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

function testDemoProcessEventRendersWarning() {
  // demo.js replays each grouped panel via ``api.processEvent`` (which
  // main.js wires to ``processOutputEvent``).  A persisted warning in
  // a panel group must render during demo replay too.
  const {win} = makeWebview();
  const api = win._demoApi;
  assert.ok(
    api && typeof api.processEvent === 'function',
    '_demoApi.processEvent must be exposed by main.js',
  );

  api.processEvent({type: 'text_delta', text: 'thinking...'});
  api.processEvent({
    type: 'warning',
    message: 'demo-replay stash warning XYZZY',
  });

  const output = win.document.getElementById('output');
  const text = output ? output.textContent : '';
  assert.ok(
    text.includes('demo-replay stash warning XYZZY'),
    'BUG: a persisted warning event is dropped during demo replay ' +
      '(processOutputEvent → handleOutputEvent has no warning case)',
  );
  win.close();
  console.log('  ok - demo replay renders persisted warning');
}

function testLiveWarningNotDoubleRendered() {
  // Regression guard for the fix: the LIVE path must still render the
  // warning exactly once (the top-level switch handles it and breaks
  // before the display-event default route).
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
  testDemoProcessEventRendersWarning();
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
