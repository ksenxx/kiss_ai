// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A thoughts panel shows the USD cost of the model call(s) it holds next
// to its time label, from the `llm_call` event, live and on replay
// (chat.html + main.js + panelCopy.js in JSDOM).

'use strict';

/* global require, console, process, setImmediate */

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  readyTabId,
} = require('./simplify2_harness');

function makePage() {
  return makeChatPage({beforeScripts: stubRaf});
}

function flush() {
  return new Promise(resolve => setImmediate(resolve));
}

async function testThoughtsPanelCostLive() {
  const wv = makePage();
  const win = wv.win;
  const TAB = readyTabId(wv.posted);
  const output = win.document.getElementById('output');
  const now = Date.now();
  send(win, {type: 'clear', chat_id: 'chat-cost', tabId: TAB});
  send(win, {type: 'status', running: true, tabId: TAB, startTs: now});
  send(win, {type: 'prompt', text: 'do the thing', tabId: TAB, ts: now});
  send(win, {type: 'thinking_start', tabId: TAB, ts: now});
  send(win, {type: 'thinking_delta', text: 'Plan: run ls.', tabId: TAB});
  send(win, {type: 'thinking_end', tabId: TAB});
  await flush();
  const thoughts = output.querySelector('.llm-panel');
  assert.ok(thoughts, 'Thoughts panel exists');
  assert.strictEqual(
    thoughts.querySelector('.panel-cost'),
    null,
    'no cost yet',
  );
  send(win, {
    type: 'llm_call',
    model: 'test-model',
    cost: 0.01234,
    input_tokens: 1000,
    output_tokens: 20,
    duration_ms: 1500,
    step: 1,
    tabId: TAB,
    ts: now + 1500,
  });
  send(win, {
    type: 'usage_info',
    text: 'Steps: 1/10',
    total_tokens: 1020,
    cost: '$0.0123',
    total_steps: 1,
    tabId: TAB,
  });
  await flush();
  const cost = thoughts.querySelector(':scope > .panel-time > .panel-cost');
  assert.ok(cost, 'the cost sits in the panel footer');
  assert.strictEqual(cost.textContent, '$0.0123');
  assert.strictEqual(cost.title, 'Cost of this model call (test-model)');
  const ts = thoughts.querySelector(':scope > .panel-time > .panel-ts');
  assert.ok(ts, 'the time label is there');
  assert.strictEqual(ts.nextElementSibling, cost, 'cost follows the time text');

  // The tool call seals the panel; the next call's cost lands in the
  // next thoughts panel, not this one.
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'ls',
    tabId: TAB,
    ts: now + 2000,
  });
  send(win, {
    type: 'tool_result',
    content: 'ok',
    is_error: false,
    tabId: TAB,
    ts: now + 2500,
  });
  send(win, {type: 'text_delta', text: 'Done.', tabId: TAB, ts: now + 3000});
  send(win, {
    type: 'llm_call',
    model: 'test-model',
    cost: 0.5,
    tabId: TAB,
    ts: now + 3100,
  });
  await flush();
  const panels = output.querySelectorAll('.llm-panel');
  assert.strictEqual(panels.length, 2, 'two thoughts panels');
  assert.strictEqual(
    panels[0].querySelector('.panel-cost').textContent,
    '$0.0123',
  );
  assert.strictEqual(
    panels[1].querySelector('.panel-cost').textContent,
    '$0.5000',
  );

  // A cost with no open thoughts panel (a tool-only response after the
  // tool_call sealed the panel) is not rendered anywhere.
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'pwd',
    tabId: TAB,
    ts: now + 4000,
  });
  send(win, {
    type: 'llm_call',
    model: 'test-model',
    cost: 0.25,
    tabId: TAB,
    ts: now + 4100,
  });
  await flush();
  assert.strictEqual(output.querySelectorAll('.panel-cost').length, 2);

  // A text-only reply gets a retry call whose words land in the same
  // panel: the panel shows the SUM of its calls, never the last one.
  send(win, {
    type: 'tool_result',
    content: '/',
    is_error: false,
    tabId: TAB,
    ts: now + 5000,
  });
  send(win, {
    type: 'text_delta',
    text: 'I am done.',
    tabId: TAB,
    ts: now + 5100,
  });
  send(win, {type: 'text_end', tabId: TAB});
  send(win, {
    type: 'llm_call',
    model: 'call-one',
    cost: 0.25,
    step: 3,
    tabId: TAB,
  });
  await flush();
  const third = output.querySelectorAll('.llm-panel')[2];
  assert.strictEqual(third.querySelector('.panel-cost').textContent, '$0.2500');
  send(win, {
    type: 'text_delta',
    text: 'Calling finish now.',
    tabId: TAB,
    ts: now + 6000,
  });
  send(win, {
    type: 'llm_call',
    model: 'call-two',
    cost: 0.01,
    step: 4,
    tabId: TAB,
  });
  send(win, {
    type: 'llm_call',
    model: 'call-two',
    cost: 'garbage',
    step: 4,
    tabId: TAB,
  });
  await flush();
  const summed = third.querySelector('.panel-cost');
  assert.strictEqual(summed.textContent, '$0.2600', 'both calls are summed');
  assert.strictEqual(
    summed.title,
    'Cost of the 2 model calls in this panel (call-two)',
  );
  assert.strictEqual(third.querySelectorAll('.panel-cost').length, 1);
  win.close();
  console.log('  ok - live llm_call cost is shown under the thoughts panel');
}

async function testThoughtsPanelCostReplay() {
  const wv = makePage();
  const win = wv.win;
  const TAB = readyTabId(wv.posted);
  const output = win.document.getElementById('output');
  const oldTs = new Date(2021, 2, 5, 14, 7).getTime();
  send(win, {
    type: 'task_events',
    tabId: TAB,
    chat_id: 'chat-replay',
    task: 'replayed task',
    events: [
      {type: 'prompt', text: 'old prompt', ts: oldTs},
      {type: 'thinking_start', ts: oldTs},
      {type: 'thinking_delta', text: 'old thought'},
      {type: 'thinking_end'},
      {type: 'llm_call', model: 'm', cost: 0.00004, ts: oldTs + 100},
      {type: 'tool_call', name: 'Bash', command: 'ls', ts: oldTs + 200},
      {type: 'tool_result', content: 'ok', is_error: false, ts: oldTs + 2500},
      {type: 'text_delta', text: 'final words', ts: oldTs + 3000},
      {type: 'llm_call', model: 'm', cost: 0, ts: oldTs + 3100},
      {type: 'result', text: 'done', ts: oldTs + 3200},
    ],
  });
  await flush();
  const costs = Array.from(
    output.querySelectorAll('.llm-panel .panel-cost'),
  ).map(el => el.textContent);
  assert.deepStrictEqual(costs, ['<$0.0001', '$0'], 'replayed costs render');
  win.close();
  console.log('  ok - replayed llm_call costs render under their panels');
}

function testFormatCallCost() {
  const {win} = makePage();
  const fmt = win.PanelCopy.formatCallCost;
  assert.strictEqual(fmt(0.5), '$0.5000');
  assert.strictEqual(fmt(0), '$0');
  assert.strictEqual(fmt(0.00001), '<$0.0001');
  assert.strictEqual(fmt(-1), '');
  assert.strictEqual(fmt('abc'), '');
  assert.strictEqual(fmt(undefined), '');
  const doc = win.document;
  const panel = doc.createElement('div');
  assert.strictEqual(win.PanelCopy.setPanelCost(null, 1), null);
  assert.strictEqual(win.PanelCopy.setPanelCost(panel, NaN), null);
  // No time label yet: the cost leads the footer; a later timestamp
  // still goes first (addPanelTimestamp inserts at the front).
  const span = win.PanelCopy.setPanelCost(panel, 0.2);
  assert.strictEqual(span.parentElement.className, 'panel-time');
  assert.strictEqual(span.title, 'Cost of this model call');
  win.PanelCopy.addPanelTimestamp(panel, Date.now());
  assert.strictEqual(panel.querySelector('.panel-time').children[1], span);
  // Idempotent: a second cost replaces the text in place.
  const again = win.PanelCopy.setPanelCost(panel, 0.3, 'm');
  assert.strictEqual(again, span);
  assert.strictEqual(span.textContent, '$0.3000');
  assert.strictEqual(panel.querySelectorAll('.panel-cost').length, 1);
  // A footer whose time label is the last child: the cost is appended.
  const other = doc.createElement('div');
  win.PanelCopy.addPanelTimestamp(other, Date.now());
  const s2 = win.PanelCopy.setPanelCost(other, 1);
  assert.strictEqual(s2.previousElementSibling.className, 'panel-ts');
  assert.strictEqual(s2.nextElementSibling, null);
  win.close();
  console.log('  ok - formatCallCost / setPanelCost edge cases');
}

(async () => {
  await testThoughtsPanelCostLive();
  await testThoughtsPanelCostReplay();
  testFormatCallCost();
  console.log('All panelCost tests passed.');
})().catch(e => {
  console.error(e);
  process.exit(1);
});
