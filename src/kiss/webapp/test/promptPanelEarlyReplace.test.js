// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

function panels(win, cls) {
  return win.document.querySelectorAll('#output .ev.' + cls);
}

// The early panels render at once; the authoritative events then
// replace them in place: still one panel each, showing the
// authoritative text instead of the early one, without the `early`
// mark, and carrying the authoritative raw text.
function testEarlyPanelsRenderThenShedTheirMark() {
  const {win} = makePage();

  send(win, {type: 'system_prompt', text: 'early SYSTEM text', early: true});
  send(win, {type: 'prompt', text: 'early USER text', early: true});

  let sys = panels(win, 'system-prompt');
  let pr = panels(win, 'prompt');
  assert.strictEqual(sys.length, 1, 'one early system-prompt panel');
  assert.strictEqual(pr.length, 1, 'one early prompt panel');
  assert.ok(sys[0].textContent.includes('early SYSTEM text'));
  assert.ok(pr[0].textContent.includes('early USER text'));
  assert.strictEqual(sys[0].dataset.early, '1');
  assert.strictEqual(pr[0].dataset.early, '1');
  console.log('  ok - early panels render immediately at submit');

  send(win, {type: 'system_prompt', text: 'authoritative SYSTEM text'});
  send(win, {type: 'prompt', text: 'authoritative FULL prompt'});

  sys = panels(win, 'system-prompt');
  pr = panels(win, 'prompt');
  assert.strictEqual(sys.length, 1, 'the system prompt replaced its panel');
  assert.strictEqual(pr.length, 1, 'the prompt replaced its panel');
  assert.ok(sys[0].textContent.includes('authoritative SYSTEM text'));
  assert.ok(!sys[0].textContent.includes('early SYSTEM text'));
  assert.ok(pr[0].textContent.includes('authoritative FULL prompt'));
  assert.ok(!pr[0].textContent.includes('early USER text'));
  assert.strictEqual(sys[0].dataset.early, undefined);
  assert.strictEqual(pr[0].dataset.early, undefined);
  assert.strictEqual(pr[0].dataset.rawText, 'authoritative FULL prompt');
  win.close();
  console.log('  ok - a replaced early panel sheds its early mark');
}

function testLaterSessionsStillAppend() {
  const {win} = makePage();

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
  const {win} = makePage();

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

const TESTS = [
  testEarlyPanelsRenderThenShedTheirMark,
  testLaterSessionsStillAppend,
  testReplayWithoutEarlyEventsUnchanged,
];

try {
  for (const test of TESTS) test();
  console.log(`\n${TESTS.length} passed, 0 failed`);
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.message ? err.message : err);
  process.exit(1);
}
