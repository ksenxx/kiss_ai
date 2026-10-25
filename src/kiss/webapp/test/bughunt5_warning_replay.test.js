// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send, stubRaf} = require('./simplify2_harness');

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

// A persisted warning event must survive a chat reopen (task_events replay
// through replayEventsInto → handleOutputEvent) and render exactly as the
// live banner does.
function testReplayWarningMatchesLiveRendering() {
  const liveView = makePage({beforeScripts: stubRaf});
  send(liveView.win, {
    type: 'warning',
    message: 'conflict in <img src=x onerror=alert(1)> & "branch"',
  });
  const liveEl = liveView.win.document.querySelector('#output .warn');
  assert.ok(liveEl, 'live warning banner must exist (bughunt3 fix)');
  const liveHtml = liveEl.outerHTML;
  liveView.win.close();

  const replayView = makePage({beforeScripts: stubRaf});
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

try {
  testReplayWarningMatchesLiveRendering();
  console.log('\n1 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
