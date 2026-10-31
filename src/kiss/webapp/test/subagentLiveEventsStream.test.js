// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A freshly spawned sub-agent's tab must show the sub-agent's events.
//
// Live sequence under test (the daemon side is pinned by
// test_replay_session_subscribes_running_subagent_no_events.py):
//
//   1. ChatSorcarAgent.run broadcasts `new_tab` with the sub-agent's
//      task id; the client opens a background sub-agent tab and posts
//      `resumeSession {taskId, tabId}`.
//   2. The daemon replays the (possibly empty) transcript to that tab
//      (`openSubagentTab` + `status running` + `task_events`) and
//      subscribes the tab to the sub-agent's live stream.
//   3. Every later sub-agent event is fanned out stamped with the
//      tab's id and the sub-agent's task id.
//
// These tests pin step 3's client half: the streamed events must land
// in the sub-agent tab's transcript -- both while the tab is in the
// background (rendered into its detached fragment) and once the user
// activates it -- and must NOT leak into the parent tab's transcript.

/* global require, console */

'use strict';

const assert = require('assert');
const {makePage, send, clickTab} = require('./simplify2_harness');
const {bootParallelRun} = require('./subagent_tab_helpers');

// ---------------------------------------------------------------------------
// scenario helpers
// ---------------------------------------------------------------------------

/**
 * Boot a page, run a one-task fan-out and answer the page's
 * `resumeSession` the way `_replay_session` does for a sub-agent that
 * has not persisted any events yet.
 *
 * @returns {object} {win, parentId, taskId, subTabId}.
 */
function bootLiveSubagent() {
  const {win, parentId, taskIds, subTabIds} = bootParallelRun(1, makePage());
  const taskId = taskIds[0];
  const subTabId = subTabIds[0];

  // The daemon's reply for a just-started sub-agent continues after
  // openSubagentTab with status running, then the (empty) transcript.
  send(win, {
    type: 'status',
    running: true,
    tabId: subTabId,
    startTs: Date.now(),
  });
  send(win, {
    type: 'task_events',
    tabId: subTabId,
    task: 'sub 1',
    task_id: taskId,
    events: [],
    chat_id: 'chat-live',
    extra: '',
  });
  return {win, parentId, taskId, subTabId};
}

/**
 * Stream one recognizable run through the sub-agent's live fan-out,
 * exactly as WebPrinter._fanout_stamped emits it: every event carries
 * the sub-agent's taskId plus the subscribed tab's tabId.
 */
function streamSubagentRun(win, subTabId, taskId, marker) {
  const stamp = ev => Object.assign({taskId: taskId, tabId: subTabId}, ev);
  send(win, stamp({type: 'prompt', text: 'sub task body ' + marker}));
  send(win, stamp({type: 'thinking_start'}));
  send(win, stamp({type: 'thinking_delta', text: 'pondering ' + marker}));
  send(win, stamp({type: 'thinking_end'}));
  send(win, stamp({type: 'text_delta', text: 'live-words-' + marker}));
  send(win, stamp({type: 'text_end', text: 'live-words-' + marker}));
  send(
    win,
    stamp({
      type: 'tool_call',
      name: 'Bash',
      tool_input: {command: 'echo ' + marker},
    }),
  );
  send(win, stamp({type: 'tool_result', content: 'tool-output-' + marker}));
}

function outputText(win) {
  const out = win.document.getElementById('output');
  return out ? out.textContent || '' : '';
}

/** Text a streamSubagentRun(..., marker) run puts in its transcript. */
function runNeedles(marker) {
  return [
    'live-words-' + marker,
    'pondering ' + marker,
    'tool-output-' + marker,
  ];
}

/** Assert the on-screen (parent) transcript shows none of *marker*'s run. */
function assertNoLeak(win, marker) {
  const text = outputText(win);
  for (const needle of runNeedles(marker)) {
    assert.strictEqual(
      text.indexOf(needle),
      -1,
      'the parent transcript must not show sub-agent output: ' + needle,
    );
  }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

// The reported bug: an agent spawns a sub-task, the client opens the
// sub-agent tab -- and the tab never shows the sub-agent's events.
function testLiveEventsReachTheBackgroundSubagentTab() {
  const {win, taskId, subTabId} = bootLiveSubagent();
  const marker = 'bg';

  // The parent tab is on screen; the sub-agent streams in the background.
  streamSubagentRun(win, subTabId, taskId, marker);
  assertNoLeak(win, marker);

  clickTab(win, subTabId);
  const text = outputText(win);
  for (const needle of runNeedles(marker)) {
    assert.ok(
      text.indexOf(needle) !== -1,
      'the sub-agent tab must show its streamed events; missing "' +
        needle +
        '" in: ' +
        JSON.stringify(text.slice(0, 400)),
    );
  }
}

// Same stream, but the user is ALREADY standing on the sub-agent tab
// when the events arrive (they clicked it the moment it opened).
function testLiveEventsReachTheActiveSubagentTab() {
  const {win, parentId, taskId, subTabId} = bootLiveSubagent();
  const marker = 'fg';

  clickTab(win, subTabId);
  streamSubagentRun(win, subTabId, taskId, marker);

  const text = outputText(win);
  for (const needle of runNeedles(marker)) {
    assert.ok(
      text.indexOf(needle) !== -1,
      'the active sub-agent tab must show its streamed events; ' +
        'missing "' +
        needle +
        '" in: ' +
        JSON.stringify(text.slice(0, 400)),
    );
  }
  // Back on the parent: nothing of the run leaked into its transcript.
  clickTab(win, parentId);
  assertNoLeak(win, marker);
}

// A transcript head replayed by `task_events` and a live tail must
// BOTH be visible: the daemon replays whatever the sub-agent persisted
// before the tab subscribed and streams the rest live.
function testReplayedHeadAndLiveTailBothVisible() {
  const {win, taskId, subTabId} = bootLiveSubagent();

  // A late replay (e.g. the async writer flushed while the round-trip
  // was in flight) re-delivers the head...
  send(win, {
    type: 'task_events',
    tabId: subTabId,
    task: 'sub 1',
    task_id: taskId,
    events: [
      {type: 'prompt', text: 'sub task body head'},
      {type: 'text_delta', text: 'head-words'},
      {type: 'text_end', text: 'head-words'},
    ],
    chat_id: 'chat-live',
    extra: '',
  });
  // ... and the live tail keeps streaming afterwards.
  streamSubagentRun(win, subTabId, taskId, 'tail');

  clickTab(win, subTabId);
  const text = outputText(win);
  for (const needle of ['head-words', 'live-words-tail']) {
    assert.ok(
      text.indexOf(needle) !== -1,
      'replayed head and live tail must both render; missing "' +
        needle +
        '" in: ' +
        JSON.stringify(text.slice(0, 400)),
    );
  }
}

const tests = [
  testLiveEventsReachTheBackgroundSubagentTab,
  testLiveEventsReachTheActiveSubagentTab,
  testReplayedHeadAndLiveTailBothVisible,
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
console.log('all subagentLiveEventsStream tests passed');
// pretendToBeVisual keeps a rAF scheduler alive; exit explicitly.
process.exit(0);
