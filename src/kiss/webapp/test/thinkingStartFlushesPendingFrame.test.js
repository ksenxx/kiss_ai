// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// End-to-end jsdom test: a thinking_start that arrives while the
// previous block's flush frame is still pending (the stream cut that
// block without a thinking_end, e.g. an LLM error mid-thought followed
// by the retry's block) must settle the buffered text into the OLD
// block and cancel its frame.  Otherwise the orphaned frame runs against
// the new block, writes the new block's buffer one frame early and
// leaves the old block empty.

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

/** A page whose animation frames are queued and run by the test. */
function makeQueuedRafPage() {
  const frames = [];
  let nextId = 0;
  const page = makePage({
    beforeScripts(win) {
      win.requestAnimationFrame = function (cb) {
        const id = ++nextId;
        frames.push({id, cb});
        return id;
      };
      win.cancelAnimationFrame = function (id) {
        const i = frames.findIndex(f => f.id === id);
        if (i >= 0) frames.splice(i, 1);
      };
    },
  });
  return Object.assign(page, {frames});
}

function runFrames(frames) {
  let guard = 0;
  while (frames.length) {
    const batch = frames.splice(0, frames.length);
    for (const f of batch) f.cb();
    assert.ok(++guard < 50, 'frames must not re-arm forever');
  }
}

function testThinkingStartFlushesPendingBlock() {
  const {win, posted, frames} = makeQueuedRafPage();
  const ready = posted.find(m => m.type === 'ready');
  win._testApi.hideWelcome();
  send(win, {type: 'status', running: true, tabId: ready.tabId, startTs: 1});
  const O = win.document.getElementById('output');

  send(win, {type: 'thinking_start'});
  send(win, {type: 'thinking_delta', text: 'A'});
  send(win, {type: 'thinking_start'});
  const thinks = O.querySelectorAll('.think');
  assert.strictEqual(thinks.length, 2, 'two thinking blocks expected');
  assert.strictEqual(
    thinks[0].textContent,
    'A',
    'the cut block must receive its buffered text when the next starts',
  );
  send(win, {type: 'thinking_delta', text: 'B'});
  assert.strictEqual(thinks[1].textContent, '', 'B is still buffered');
  runFrames(frames);
  assert.strictEqual(thinks[0].textContent, 'A', 'A stays in its block');
  assert.strictEqual(thinks[1].textContent, 'B', 'B lands in the new one');
  assert.strictEqual(frames.length, 0, 'no frame may stay queued');
  win.close();
  console.log('  ok - thinking_start flushes the pending block');
}

testThinkingStartFlushesPendingBlock();
console.log('thinkingStartFlushesPendingFrame: all passed');
