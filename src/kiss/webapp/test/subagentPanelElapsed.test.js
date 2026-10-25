// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Sub-agent tabs must show the per-panel time-spent label agent tabs
// show, in the bottom-right corner of every event panel.
//
// A sub-agent tab's transcript arrives in two ways:
//
//   1. live events streamed after the tab subscribed
//      (processOutputEventForBgTab), which stamp panels with the wall
//      clock exactly like the visible tab's stream, and
//   2. a `task_events` replay (the tab was opened or re-opened
//      mid-run, e.g. by expanding a collapsed run_parallel panel),
//      whose panels can NOT be stamped with the render-time clock --
//      that would measure the replay, not the tool.
//
// The replay path therefore derives every closed panel's duration from
// the persisted event timestamps (tool_result.ts - tool_call.ts), and
// hands the still-open panels of a still-RUNNING task back to the live
// ticker so they keep counting from their own event's wall-clock
// start. A finished task's unclosed panels (an interrupted tool call)
// keep no label: no duration exists for them. These tests pin all of
// that, plus the guard rails: closing events without a usable timestamp render
// nothing, and an adjacent-task replay never ticks even while its
// owner tab is running.

/* global require, console, process */

'use strict';

const assert = require('assert');
const {send, clickTab: activateTab} = require('./simplify2_harness');
const {makeWebappClient} = require('./transport_page_helpers');
const {bootParallelRun} = require('./subagent_tab_helpers');

/**
 * The `.panel-elapsed` label of *panel*'s own footer bar, or null.
 *
 * @param {Element} panel The event panel.
 * @returns {?Element} The label element.
 */
function elapsedLabel(panel) {
  return panel.querySelector(':scope > .panel-time > .panel-elapsed');
}

/**
 * Parse a `.panel-elapsed` label ("450ms", "3.5s", "1m 2.0s") to ms.
 *
 * @param {string} text The label text.
 * @returns {number} Milliseconds.
 */
function parseElapsedMs(text) {
  let m = /^(\d+)ms$/.exec(text);
  if (m) return Number(m[1]);
  m = /^(\d+(?:\.\d+)?)s$/.exec(text);
  if (m) return Number(m[1]) * 1000;
  m = /^(\d+)m (\d+(?:\.\d+)?)s$/.exec(text);
  if (m) return Number(m[1]) * 60000 + Number(m[2]) * 1000;
  assert.fail('unparsable elapsed label: ' + JSON.stringify(text));
  return 0;
}

/**
 * Boot a client, run a one-task fan-out and capture the sub-agent tab
 * id the client minted when it resumed the sub-agent's session.
 *
 * @returns {object} {win, all, parentId, taskId, subTabId}.
 */
function bootSubagent() {
  const client = makeWebappClient();
  const {win} = client;
  const all = () => client.drain();
  // The shim buffers frames until drained, so drain after every delivery
  // for the helper to see the page's resumeSession reply.
  function deliver(msg) {
    send(win, msg);
    all();
  }
  const {parentId, taskIds, subTabIds} = bootParallelRun(
    1,
    {win, posted: all()},
    {deliver},
  );
  const subTabId = subTabIds[0];
  send(win, {
    type: 'status',
    running: true,
    tabId: subTabId,
    startTs: Date.now(),
  });
  return {win, all, parentId, taskId: taskIds[0], subTabId};
}

// A sub-agent transcript replayed mid-run (`task_events` for a RUNNING
// sub-agent tab): closed panels show the timestamp-derived duration,
// and the still-open trailing tool_call resumes the live tick from its
// own event's start -- then freezes when its live tool_result lands.
function testReplayedSubagentPanelsShowDurations() {
  const {win, taskId, subTabId} = bootSubagent();
  const t0 = Date.now() - 60000;
  send(win, {
    type: 'task_events',
    tabId: subTabId,
    task: 'sub 1',
    task_id: taskId,
    chat_id: 'chat-elapsed',
    extra: '',
    events: [
      {type: 'prompt', text: 'sub task body', ts: t0},
      {type: 'thinking_start', ts: t0 + 1000},
      {type: 'thinking_delta', text: 'pondering'},
      {type: 'thinking_end'},
      {type: 'tool_call', name: 'Bash', command: 'echo one', ts: t0 + 4500},
      {type: 'tool_result', content: 'one', is_error: false, ts: t0 + 7000},
      {type: 'tool_call', name: 'Bash', command: 'echo two', ts: t0 + 9000},
    ],
  });

  activateTab(win, subTabId);
  const out = win.document.getElementById('output');

  const thoughts = out.querySelector('.llm-panel');
  assert.ok(thoughts, 'replayed Thoughts panel exists');
  const thLabel = elapsedLabel(thoughts);
  assert.ok(thLabel, 'replayed Thoughts panel has a duration');
  assert.strictEqual(
    thLabel.textContent,
    '3.5s',
    'Thoughts duration = tool_call.ts - thinking_start.ts',
  );

  const tcs = out.querySelectorAll('.ev.tc');
  assert.strictEqual(tcs.length, 2, 'both tool_call panels exist');
  const closedLabel = elapsedLabel(tcs[0]);
  assert.ok(closedLabel, 'closed tool_call has a duration');
  assert.strictEqual(
    closedLabel.textContent,
    '2.5s',
    'closed tool_call duration = tool_result.ts - tool_call.ts',
  );

  // The trailing tool_call has not reported back: it must be TICKING
  // from its own event's wall-clock start (~51s ago), not from the
  // moment the replay rendered it.
  const openTc = tcs[1];
  const openLabel = elapsedLabel(openTc);
  assert.ok(openLabel, 'the open tool_call shows a live elapsed');
  const openMs = parseElapsedMs(openLabel.textContent);
  assert.ok(
    openMs >= 50000 && openMs < 70000,
    'the open tool_call counts from its event start (~51s), got ' +
      openLabel.textContent,
  );
  assert.ok(openTc.dataset.startMs, 'the open tool_call joined the live tick');
  assert.ok(!openTc.dataset.timeDone, 'the open tool_call is not frozen yet');

  // Its live tool_result freezes the label at the true total duration:
  // the RESULT EVENT's own timestamp closes the panel, so a slow
  // delivery (here: the result stamped 2.5s after the call but
  // delivered ~51s later) must not inflate the duration.
  send(win, {
    type: 'tool_result',
    content: 'two',
    is_error: false,
    taskId: taskId,
    tabId: subTabId,
    ts: t0 + 9000 + 2500,
  });
  assert.strictEqual(
    openTc.dataset.timeDone,
    '1',
    'the live tool_result freezes the replayed panel',
  );
  assert.strictEqual(
    elapsedLabel(openTc).textContent,
    '2.5s',
    "the frozen duration is the events' own span, not delivery time",
  );

  const promptPanel = out.querySelector('.ev.prompt');
  assert.ok(promptPanel, 'replayed prompt panel exists');
  assert.strictEqual(
    elapsedLabel(promptPanel),
    null,
    'a prompt panel never carries a duration',
  );
}

// Live-streamed sub-agent panels (the pre-existing behavior, pinned):
// panels stamped while the events stream into the background tab carry
// the same elapsed label the visible tab's panels do.
function testLiveSubagentPanelsShowElapsed() {
  const {win, taskId, subTabId} = bootSubagent();
  send(win, {
    type: 'task_events',
    tabId: subTabId,
    task: 'sub 1',
    task_id: taskId,
    chat_id: 'chat-elapsed-live',
    extra: '',
    events: [],
  });
  const stamp = ev => Object.assign({taskId: taskId, tabId: subTabId}, ev);
  send(win, stamp({type: 'prompt', text: 'sub task body', ts: Date.now()}));
  send(win, stamp({type: 'thinking_start', ts: Date.now()}));
  send(win, stamp({type: 'thinking_delta', text: 'pondering'}));
  send(win, stamp({type: 'thinking_end'}));
  send(win, stamp({type: 'tool_call', name: 'Bash', command: 'echo hi'}));
  send(win, stamp({type: 'tool_result', content: 'hi', is_error: false}));

  activateTab(win, subTabId);
  const out = win.document.getElementById('output');
  const tc = out.querySelector('.ev.tc');
  assert.ok(tc, 'live tool_call panel exists');
  const tcLabel = elapsedLabel(tc);
  assert.ok(tcLabel, 'live sub-agent tool_call shows elapsed');
  assert.ok(
    parseElapsedMs(tcLabel.textContent) < 60000,
    'live elapsed is wall-clock small, got ' + tcLabel.textContent,
  );
  assert.strictEqual(tc.dataset.timeDone, '1', 'frozen on result');
  const thoughts = out.querySelector('.llm-panel');
  assert.ok(
    elapsedLabel(thoughts),
    'live sub-agent Thoughts panel shows elapsed',
  );
}

// A FINISHED task's replay: closed panels get their historical
// durations, but the interrupted trailing tool_call must NOT tick --
// no event ever closed it, so no duration exists.
function testFinishedReplayOpenPanelDoesNotTick() {
  const client = makeWebappClient();
  const {win} = client;
  const all = () => client.drain();
  const tabId = all().find(m => m.type === 'ready').tabId;
  const t0 = new Date(2021, 2, 5, 14, 7).getTime();
  send(win, {
    type: 'task_events',
    tabId: tabId,
    chat_id: 'chat-finished',
    task: 'old task',
    events: [
      {type: 'prompt', text: 'old prompt', ts: t0},
      {type: 'tool_call', name: 'Bash', command: 'ls', ts: t0 + 2000},
      {type: 'tool_result', content: 'ok', is_error: false, ts: t0 + 5500},
      {type: 'tool_call', name: 'Bash', command: 'oops', ts: t0 + 6000},
    ],
  });
  const out = win.document.getElementById('output');
  const tcs = out.querySelectorAll('.ev.tc');
  assert.strictEqual(tcs.length, 2, 'both tool_call panels exist');
  assert.strictEqual(
    elapsedLabel(tcs[0]).textContent,
    '3.5s',
    'the finished tool_call keeps its historical duration',
  );
  assert.strictEqual(
    elapsedLabel(tcs[1]),
    null,
    'an interrupted tool_call shows no duration',
  );
  assert.ok(
    !tcs[1].dataset.startMs,
    'a finished replay puts nothing on the live tick',
  );
}

// A closing event with no usable timestamp renders nothing, and the
// panel must still be sealed: a later promotion pass may not put a
// CLOSED panel on the live tick.
function testCloseWithoutTimestampSealsWithoutLabel() {
  const {win, taskId, subTabId} = bootSubagent();
  const t0 = Date.now() - 30000;
  send(win, {
    type: 'task_events',
    tabId: subTabId,
    task: 'sub 1',
    task_id: taskId,
    chat_id: 'chat-nots',
    extra: '',
    events: [
      {type: 'prompt', text: 'sub task body', ts: t0},
      {type: 'tool_call', name: 'Bash', command: 'ls', ts: t0 + 1000},
      {type: 'tool_result', content: 'ok', is_error: false},
      {type: 'tool_call', name: 'Read', path: '/tmp/x'},
    ],
  });
  activateTab(win, subTabId);
  const out = win.document.getElementById('output');
  const tcs = out.querySelectorAll('.ev.tc');
  assert.strictEqual(tcs.length, 2, 'both tool_call panels exist');
  assert.strictEqual(
    elapsedLabel(tcs[0]),
    null,
    'a tool_result without ts renders no duration',
  );
  assert.strictEqual(
    tcs[0].dataset.timeDone,
    '1',
    'the closed panel is sealed all the same',
  );
  assert.ok(
    !tcs[0].dataset.startMs,
    'the sealed panel must not join the live tick',
  );
  // The trailing tool_call carried no ts either: nothing to count from.
  assert.strictEqual(
    elapsedLabel(tcs[1]),
    null,
    'an open tool_call without ts cannot tick',
  );
  assert.ok(
    !tcs[1].dataset.startMs && !tcs[1].dataset.startTs,
    'no start was recorded for a ts-less replayed panel',
  );
}

// An adjacent-task replay is always a NEIGHBOURING task's finished
// transcript: its open panels never join the live tick, even though
// the owner tab is running its own live task.
function testAdjacentReplayNeverTicks() {
  const client = makeWebappClient();
  const {win} = client;
  const all = () => client.drain();
  const tabId = all().find(m => m.type === 'ready').tabId;
  send(win, {type: 'status', running: true, tabId: tabId, startTs: Date.now()});
  send(win, {type: 'prompt', text: 'live task', tabId: tabId, taskId: '77'});
  const t0 = Date.now() - 40000;
  send(win, {
    type: 'adjacent_task_events',
    tabId: tabId,
    direction: 'prev',
    task: 'Older task',
    task_id: '41',
    events: [
      {type: 'prompt', text: 'older prompt', ts: t0},
      {type: 'tool_call', name: 'Bash', command: 'ls', ts: t0 + 1000},
      {type: 'tool_result', content: 'ok', is_error: false, ts: t0 + 3000},
      {type: 'tool_call', name: 'Bash', command: 'oops', ts: t0 + 5000},
    ],
  });
  const out = win.document.getElementById('output');
  const adj = out.querySelector('.adjacent-task');
  assert.ok(adj, 'the adjacent task container rendered');
  const tcs = adj.querySelectorAll('.ev.tc');
  assert.strictEqual(tcs.length, 2, 'adjacent tool_calls exist');
  assert.strictEqual(
    elapsedLabel(tcs[0]).textContent,
    '2.0s',
    'the adjacent closed tool_call keeps its duration',
  );
  assert.strictEqual(
    adj.querySelectorAll('[data-start-ms]').length,
    0,
    'an adjacent replay must never join the live tick',
  );
  assert.strictEqual(
    elapsedLabel(tcs[1]),
    null,
    'the adjacent open tool_call shows no duration',
  );
}

// A terminal event (task_done here; task_error / task_stopped /
// task_interrupted share the same seal) freezes every panel the
// transcript still has open — a replay-promoted tool_call included —
// at the task's own end timestamp, and the sealed panel can never
// rejoin the live tick.
function testTerminalEventSealsOpenPanels() {
  const client = makeWebappClient();
  const {win} = client;
  const all = () => client.drain();
  const tabId = all().find(m => m.type === 'ready').tabId;
  const t0 = Date.now() - 60000;
  send(win, {type: 'status', running: true, tabId: tabId, startTs: t0});
  send(win, {
    type: 'task_events',
    tabId: tabId,
    chat_id: 'chat-terminal',
    task: 'running task',
    task_id: '91',
    events: [
      {type: 'prompt', text: 'go', ts: t0},
      {type: 'tool_call', name: 'Bash', command: 'sleep 999', ts: t0},
    ],
  });
  const out = win.document.getElementById('output');
  const tc = out.querySelector('.ev.tc');
  assert.ok(tc, 'the open tool_call panel exists');
  assert.ok(tc.dataset.startMs, 'the open panel was promoted');
  send(win, {
    type: 'task_done',
    success: true,
    tabId: tabId,
    startTs: t0,
    endTs: t0 + 12000,
  });
  assert.strictEqual(
    tc.dataset.timeDone,
    '1',
    'task_done seals the open panel',
  );
  assert.strictEqual(
    elapsedLabel(tc).textContent,
    '12.0s',
    "the seal uses the task's own end timestamp",
  );
}

const tests = [
  testReplayedSubagentPanelsShowDurations,
  testLiveSubagentPanelsShowElapsed,
  testFinishedReplayOpenPanelDoesNotTick,
  testCloseWithoutTimestampSealsWithoutLabel,
  testAdjacentReplayNeverTicks,
  testTerminalEventSealsOpenPanels,
];

let failures = 0;
for (const t of tests) {
  try {
    t();
    console.log('ok - ' + t.name);
  } catch (e) {
    failures += 1;
    console.error('FAIL - ' + t.name);
    console.error(e && e.stack ? e.stack : String(e));
  }
}
if (failures > 0) {
  console.error(failures + ' failure(s)');
  process.exit(1);
}
console.log('all subagentPanelElapsed tests passed');
// pretendToBeVisual keeps a rAF scheduler alive; exit explicitly.
process.exit(0);
