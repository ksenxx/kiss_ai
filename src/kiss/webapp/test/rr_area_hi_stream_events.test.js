// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Stream-path regressions:
// - I-R5: handleEvent's default branch must not funnel non-transcript
//   messages (the voice replies, unknown types) into
//   processOutputEvent — each such call costs an O(transcript) DOM
//   sweep and re-arms the wait spinner. Transcript-bearing types must
//   keep rendering.
// - I-R6: a pending bash-output flush frame (tState.bashRaf) must be
//   CANCELLED when its panel closes (tool_call / tool_result), not just
//   zeroed — an orphaned callback fires against the NEXT tool's panel.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  recordTimers,
  fireTimers,
  queueRaf,
} = require('./simplify2_harness');

// main.js showSpinner() arms the wait spinner 250 ms after a transcript
// event; the test records that timer (cancellation aware) and fires it
// by hand instead of outwaiting it, and a negative "the spinner was not
// re-armed" check looks for a new timer.
const SPINNER_DELAY_MS = 250;

function startRunningTask(win, posted, taskId) {
  const tabId = posted.find(m => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  send(win, {type: 'setTaskText', text: 'live task', tabId});
  send(win, {type: 'status', running: true, tabId});
  send(win, {type: 'prompt', text: 'live task', tabId, taskId});
  send(win, {type: 'system_output', text: 'started\n', tabId, taskId});
  return tabId;
}

function testVoiceMessagesDoNotDisturbTranscript() {
  const {win, posted} = makePage();
  const timers = recordTimers(win, SPINNER_DELAY_MS);
  const tabId = startRunningTask(win, posted, '123');
  const O = win.document.getElementById('output');
  const spinner = win.document.getElementById('wait-spinner');
  assert.ok(spinner, 'wait spinner missing from the page');
  // Fire the live spinner timer the real transcript armed (the spinner
  // comes on), then force the spinner off, as if the user were reading
  // quietly.
  assert.ok(
    timers.some(t => !t.cleared),
    'the running task never armed the wait spinner',
  );
  assert.ok(!spinner.classList.contains('active'), 'not active before its delay');
  fireTimers(timers);
  assert.ok(spinner.classList.contains('active'), 'the spinner comes on after 250 ms');
  spinner.classList.remove('active');
  const armed = timers.length;
  const transcriptBefore = O.innerHTML;

  for (const msg of [
    {type: 'voiceWake', roundId: 1},
    {type: 'voiceTranscribing'},
    {type: 'voiceSpeech', roundId: 1, text: 'hello'},
    {type: 'voiceState', listening: true},
    {type: 'someFutureUnknownType', tabId, taskId: '123'},
  ]) {
    send(win, msg);
  }
  assert.strictEqual(
    O.innerHTML,
    transcriptBefore,
    'non-transcript host messages must not touch the transcript',
  );
  assert.strictEqual(
    timers.length,
    armed,
    'non-transcript host messages must not re-arm the wait spinner',
  );
  assert.ok(!spinner.classList.contains('active'));

  // Transcript-bearing types still render through the same branch, and
  // do arm the spinner (one new live timer).
  send(win, {
    type: 'system_output',
    text: 'more output\n',
    tabId,
    taskId: '123',
  });
  assert.ok(
    O.textContent.includes('more output'),
    'whitelisted transcript types must keep rendering',
  );
  const rearmed = timers.slice(armed).filter(t => !t.cleared);
  assert.strictEqual(rearmed.length, 1, 'transcript output re-arms it');
  fireTimers(rearmed);
  assert.ok(spinner.classList.contains('active'), 'and it comes on again');
  win.close();
  console.log('  ok - voice/unknown messages bypass the transcript pipeline');
}

function testBashFlushFrameCancelled() {
  const {win, posted} = makePage();
  const tabId = startRunningTask(win, posted, '123');

  // Deterministic rAF: callbacks run only when the test says so.
  const {frames, run: runRafs} = queueRaf(win);
  const pending = () => frames.map(f => f.id);

  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'echo one',
    tabId,
    taskId: '123',
  });
  runRafs();

  // A buffered chunk leaves a flush frame pending...
  const before = new Set(pending());
  send(win, {type: 'system_output', text: 'two', tabId, taskId: '123'});
  const chunkFrames = pending().filter(id => !before.has(id));
  assert.ok(chunkFrames.length > 0, 'chunk must schedule a flush frame');

  // ...which the panel-closing tool_result must cancel, not orphan.
  send(win, {type: 'tool_result', content: 'done', tabId, taskId: '123'});
  const leaked = chunkFrames.filter(id => pending().includes(id));
  assert.deepStrictEqual(
    leaked,
    [],
    'tool_result left the bash flush frame pending against the next panel',
  );

  // The buffered text was flushed synchronously exactly once.
  const panels = win.document.querySelectorAll('.bash-panel-content');
  const withTwo = Array.from(panels).filter(p => p.textContent.includes('two'));
  assert.strictEqual(withTwo.length, 1, 'buffered chunk must render once');
  assert.ok(
    withTwo[0].textContent.split('two').length === 2,
    'buffered chunk must not be duplicated',
  );

  // Same requirement on the tool_call path (flush-on-next-tool).
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'echo three',
    tabId,
    taskId: '123',
  });
  runRafs();
  const before2 = new Set(pending());
  send(win, {type: 'system_output', text: 'four', tabId, taskId: '123'});
  const chunkFrames2 = pending().filter(id => !before2.has(id));
  assert.ok(chunkFrames2.length > 0, 'second chunk must schedule a frame');
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'echo five',
    tabId,
    taskId: '123',
  });
  const leaked2 = chunkFrames2.filter(id => pending().includes(id));
  assert.deepStrictEqual(
    leaked2,
    [],
    'tool_call left the bash flush frame pending against the next panel',
  );
  runRafs();
  win.close();
  console.log('  ok - closing a bash panel cancels its pending flush frame');
}

testVoiceMessagesDoNotDisturbTranscript();
testBashFlushFrameCancelled();
console.log('rr_area_hi_stream_events: all tests passed');
process.exit(0);
