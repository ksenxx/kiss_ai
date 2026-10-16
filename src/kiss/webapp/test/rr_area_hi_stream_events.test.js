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

/* global require, process, console, __dirname, global, setTimeout, clearTimeout */

'use strict';

const assert = require('assert');
const {makePage, send, sleep, waitFor} = require('./simplify2_harness');

// main.js showSpinner() arms the wait spinner 250 ms after a transcript
// event; a negative "the spinner was not re-armed" check has to outwait
// that delay.
const SPINNER_DELAY_MS = 250;
const SPINNER_SETTLE_MS = SPINNER_DELAY_MS + 150;

function startRunningTask(win, posted, taskId) {
  const tabId = posted.find(m => m.type === 'ready').tabId;
  win._testApi.hideWelcome();
  send(win, {type: 'setTaskText', text: 'live task', tabId});
  send(win, {type: 'status', running: true, tabId});
  send(win, {type: 'prompt', text: 'live task', tabId, taskId});
  send(win, {type: 'system_output', text: 'started\n', tabId, taskId});
  return tabId;
}

async function testVoiceMessagesDoNotDisturbTranscript() {
  const {win, posted} = makePage();
  const tabId = startRunningTask(win, posted, '123');
  const O = win.document.getElementById('output');
  const spinner = win.document.getElementById('wait-spinner');
  assert.ok(spinner, 'wait spinner missing from the page');

  // Let the pending spinner timer from the real transcript fire (the
  // spinner comes on), then force the spinner off, as if the user were
  // reading quietly.
  await waitFor(
    () => spinner.classList.contains('active'),
    'the running task never armed the wait spinner',
  );
  spinner.classList.remove('active');
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
  // Negative check: outwait the spinner delay so a wrongly re-armed
  // spinner would have come on by now.
  await sleep(SPINNER_SETTLE_MS);

  assert.strictEqual(
    O.innerHTML,
    transcriptBefore,
    'non-transcript host messages must not touch the transcript',
  );
  assert.strictEqual(
    spinner.classList.contains('active'),
    false,
    'non-transcript host messages must not re-arm the wait spinner',
  );

  // Transcript-bearing types still render through the same branch.
  send(win, {type: 'system_output', text: 'more output\n', tabId, taskId: '123'});
  assert.ok(
    O.textContent.includes('more output'),
    'whitelisted transcript types must keep rendering',
  );
  win.close();
  console.log('  ok - voice/unknown messages bypass the transcript pipeline');
}

function testBashFlushFrameCancelled() {
  const {win, posted} = makePage();
  const tabId = startRunningTask(win, posted, '123');

  // Deterministic rAF: callbacks run only when the test says so.
  const scheduled = new Map();
  let nextRaf = 1;
  win.requestAnimationFrame = cb => {
    const id = nextRaf++;
    scheduled.set(id, cb);
    return id;
  };
  win.cancelAnimationFrame = id => {
    scheduled.delete(id);
  };
  const runRafs = () => {
    for (const [id, cb] of Array.from(scheduled)) {
      scheduled.delete(id);
      cb();
    }
  };

  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'echo one',
    tabId,
    taskId: '123',
  });
  runRafs();

  // A buffered chunk leaves a flush frame pending...
  const before = new Set(scheduled.keys());
  send(win, {type: 'system_output', text: 'two', tabId, taskId: '123'});
  const chunkFrames = Array.from(scheduled.keys()).filter(
    id => !before.has(id),
  );
  assert.ok(chunkFrames.length > 0, 'chunk must schedule a flush frame');

  // ...which the panel-closing tool_result must cancel, not orphan.
  send(win, {type: 'tool_result', content: 'done', tabId, taskId: '123'});
  const leaked = chunkFrames.filter(id => scheduled.has(id));
  assert.deepStrictEqual(
    leaked,
    [],
    'tool_result left the bash flush frame pending against the next panel',
  );

  // The buffered text was flushed synchronously exactly once.
  const panels = win.document.querySelectorAll('.bash-panel-content');
  const withTwo = Array.from(panels).filter(p =>
    p.textContent.includes('two'),
  );
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
  const before2 = new Set(scheduled.keys());
  send(win, {type: 'system_output', text: 'four', tabId, taskId: '123'});
  const chunkFrames2 = Array.from(scheduled.keys()).filter(
    id => !before2.has(id),
  );
  assert.ok(chunkFrames2.length > 0, 'second chunk must schedule a frame');
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'echo five',
    tabId,
    taskId: '123',
  });
  const leaked2 = chunkFrames2.filter(id => scheduled.has(id));
  assert.deepStrictEqual(
    leaked2,
    [],
    'tool_call left the bash flush frame pending against the next panel',
  );
  runRafs();
  win.close();
  console.log('  ok - closing a bash panel cancels its pending flush frame');
}

async function main() {
  await testVoiceMessagesDoNotDisturbTranscript();
  testBashFlushFrameCancelled();
  console.log('rr_area_hi_stream_events: all tests passed');
  process.exit(0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
