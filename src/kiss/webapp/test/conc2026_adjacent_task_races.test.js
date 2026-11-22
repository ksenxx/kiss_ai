// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests for adjacent-task / background-status races
// in media/main.js (audit candidates CAND-1, CAND-3..CAND-5). Each test asserts
// the CORRECT (post-fix) behavior:
//
//   CAND-1  After a neighbouring task W was spliced into tab A's
//           transcript, switching to tab B and back to A and then
//           delivering another adjacent_task_events for W must NOT
//           create a second .adjacent-task[data-task-id=W] container:
//           exactly one region per task id. (resetAdjacentState rewinds
//           the anchors on every tab switch while the spliced DOM
//           survives in the tab's fragment, and renderAdjacentTask
//           never dedups by task id.)
//
//   CAND-3  An #adjacent-loader that was detached into a tab's
//           fragment during a tab switch (its reply arrived addressed
//           to the switched-away tab and was dropped) is removed when
//           the tab is restored: no "Loading previous task…" row
//           survives its request.
//
//   CAND-4  A daemonStatus {connected:false} event clears
//           adjacentLoading and removes the loader, so a later
//           overscroll can fetch again instead of being blocked for
//           ever by a reply the outage swallowed.
//
//   CAND-5  A reply names the task it is adjacent to (`adjacent_to`).
//           One asked for by another surface showing the tab, whose
//           thread ends on another task, is NOT spliced here (the
//           tasks in between could never be loaded) and leaves this
//           page's own request in flight.
//
// The harness mirrors adjacentTaskScroll.test.js (overscroll + adjacent
// replies).

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');
const {historyRows, makeRow} = require('./history_helpers');

const {
  send,
  clickTab,
  wheel,
  getAdjacent,
  makeLaunchedPage: makePage,
  showHistoryTask,
} = harness;

// The daemon's reply to a 'prev' getAdjacentTask: task W = '41'.
function answerPrevTask41(win, tabId) {
  send(win, {
    type: 'adjacent_task_events',
    tabId,
    direction: 'prev',
    task: 'Older task',
    task_id: '41',
    events: [{type: 'system_output', text: 'older\n'}],
  });
}

// Overscroll at the top until a getAdjacentTask request leaves, then
// answer it with task W = '41'.
function loadPrevTask41(win, posted, tabId, O) {
  O.scrollTop = 0;
  wheel(win, O, -50, 10);
  const adj = getAdjacent(posted);
  assert.ok(adj.length >= 1, 'overscroll at top must request the prev task');
  answerPrevTask41(win, tabId);
  assert.strictEqual(
    O.querySelectorAll('.adjacent-task[data-task-id="41"]').length,
    1,
    'the first reply must splice exactly one container for task 41',
  );
}

// CAND-1: a tab switch must not forget which neighbours are already
// spliced into the transcript. The pagination anchors travel with the
// detached transcript (saveCurrentTab / restoreTab), so the overscroll
// after a switch-and-return asks for the task BEYOND the spliced
// neighbour 41, not for 41 again (A24-8); and should a reply for a
// task already on screen arrive anyway, renderAdjacentTask splices no
// second container for it.
function testNoDuplicateAdjacentRegionAfterTabSwitch() {
  const {win, posted} = makePage();
  try {
    const {tabId, O} = showHistoryTask(win, posted);
    loadPrevTask41(win, posted, tabId, O);

    // Away to tab B (transcript detaches into A's fragment) and back.
    win._testApi.createNewTab();
    const tabB = win._testApi.getActiveTabId();
    assert.notStrictEqual(tabB, tabId, 'createNewTab must open tab B');
    clickTab(win, tabId);
    assert.strictEqual(win._testApi.getActiveTabId(), tabId);
    assert.strictEqual(
      O.querySelectorAll('.adjacent-task[data-task-id="41"]').length,
      1,
      'the spliced container must come back with the restored transcript',
    );

    // The transcript still shows task 41: the next overscroll fetches
    // the task before it, and a reply naming 41 anyway must not splice
    // a second copy.
    O.scrollTop = 0;
    wheel(win, O, -50, 10);
    const adj = getAdjacent(posted);
    assert.ok(
      adj.length >= 2,
      'overscroll after the switch must still be able to request older ' +
        'tasks; got ' +
        JSON.stringify(adj),
    );
    assert.strictEqual(
      String(adj[adj.length - 1].taskId),
      '41',
      'the overscroll after a switch-and-return asks for the task beyond ' +
        'the spliced neighbour 41 (the anchors came back with the ' +
        'transcript), not for 41 again',
    );
    assert.strictEqual(adj[adj.length - 1].direction, 'prev');
    answerPrevTask41(win, tabId);
    assert.strictEqual(
      O.querySelectorAll('.adjacent-task[data-task-id="41"]').length,
      1,
      'a second adjacent_task_events for task 41 after a tab ' +
        'switch-and-return must NOT splice a second ' +
        '.adjacent-task[data-task-id="41"] container: one region per ' +
        'task id',
    );
    console.log('PASS no duplicate adjacent region after tab switch');
  } finally {
    win.close();
  }
}

// The anchors describe the thread they were saved with: a tab whose
// task changed while it was in the background (a new run started on it
// from another surface: its `clear`, then its task_settings naming the
// new task id) comes back with its anchors rewound to that task, as a
// fresh transcript's are, and the edge flags cleared.
function testAnchorsOfAnotherTaskAreNotRestored() {
  const {win, posted} = makePage();
  try {
    const {tabId, O} = showHistoryTask(win, posted);
    loadPrevTask41(win, posted, tabId, O);
    // The chat has nothing older than 41: the edge is remembered...
    O.scrollTop = 0;
    wheel(win, O, -50, 10);
    send(win, {type: 'adjacent_task_events', tabId, direction: 'prev'});
    const before = getAdjacent(posted).length;
    wheel(win, O, -50, 10);
    assert.strictEqual(getAdjacent(posted).length, before, 'no fetch past the edge');

    win._testApi.createNewTab();
    send(win, {type: 'clear', tabId});
    send(win, {type: 'task_settings', tabId, settings: {task_id: '43'}});
    clickTab(win, tabId);
    assert.strictEqual(win._testApi.getActiveTabId(), tabId);

    // ...but it was 42's edge, not 43's: the overscroll asks again,
    // from the tab's new task.
    O.scrollTop = 0;
    wheel(win, O, -50, 10);
    const adj = getAdjacent(posted);
    assert.strictEqual(adj.length, before + 1, 'the edge flag went with the old task');
    assert.strictEqual(String(adj[adj.length - 1].taskId), '43');
    console.log('PASS anchors of another task are not restored');
  } finally {
    win.close();
  }
}

// CAND-3: a loader detached into the switched-away tab's fragment
// (its reply was dropped by the isForActiveTab guard) must be removed
// when the tab is restored.
function testDetachedLoaderRemovedOnRestore() {
  const {win, posted} = makePage();
  try {
    const {tabId, O} = showHistoryTask(win, posted);
    O.scrollTop = 0;
    wheel(win, O, -50, 10);
    assert.ok(
      getAdjacent(posted).length >= 1,
      'overscroll at top must request the prev task',
    );
    assert.ok(
      O.querySelector('.adjacent-loader'),
      'the request must show the "Loading previous task…" row',
    );

    // Switch away while the request is in flight: the loader detaches
    // into tab A's fragment with the transcript.
    win._testApi.createNewTab();
    assert.notStrictEqual(win._testApi.getActiveTabId(), tabId);

    // The reply lands addressed to the switched-away tab and is dropped.
    answerPrevTask41(win, tabId);

    clickTab(win, tabId);
    assert.strictEqual(win._testApi.getActiveTabId(), tabId);
    assert.strictEqual(
      O.querySelector('.adjacent-loader'),
      null,
      'restoring the tab must remove the stale "Loading previous ' +
        'task…" row whose request was answered while the tab was hidden',
    );
    console.log('PASS detached adjacent loader removed on tab restore');
  } finally {
    win.close();
  }
}

// CAND-4: an outage that swallows the in-flight getAdjacentTask reply
// must not leave adjacent loading stuck: daemonStatus {connected:false}
// clears adjacentLoading and removes the loader, so a later overscroll
// fetches again.
function testDaemonDisconnectClearsAdjacentLoading() {
  const {win, posted} = makePage();
  try {
    const {O} = showHistoryTask(win, posted);
    O.scrollTop = 0;
    wheel(win, O, -50, 10);
    assert.strictEqual(
      getAdjacent(posted).length,
      1,
      'overscroll at top must post exactly one request',
    );
    assert.ok(
      O.querySelector('.adjacent-loader'),
      'the request must show the loader row',
    );

    // The daemon goes away; the reply never comes.
    send(win, {type: 'daemonStatus', connected: false});

    assert.strictEqual(
      O.querySelector('.adjacent-loader'),
      null,
      'a daemon disconnect must remove the adjacent loader: its ' +
        'request can never be answered',
    );
    O.scrollTop = 0;
    wheel(win, O, -50, 10);
    assert.strictEqual(
      getAdjacent(posted).length,
      2,
      'after the disconnect cleared the in-flight state, a later ' +
        'overscroll must be able to fetch the previous task again ' +
        '(adjacentLoading must not stay latched for ever)',
    );
    console.log('PASS daemon disconnect clears adjacent loading state');
  } finally {
    win.close();
  }
}

/**
 * What the daemon sends when it loses `machine`'s stream of tab `tabId`
 * (RemoteTabs.dropped): the terminal status naming the machine as
 * `lost`, the refusal's error, then the tab's unbinding.
 */
function loseMachine(win, tabId, machine) {
  send(win, {type: 'status', tabId, running: false, lost: machine});
  send(win, {
    type: 'error',
    tabId,
    code: 'prompt_refused',
    text: 'lost the connection to ' + machine,
  });
  send(win, {type: 'tabMachine', tabId, machine: ''});
}

// Audit 2026-11-21 (W10 F2): a tab bound to a registered machine sends
// its getAdjacentTask to that machine, whose stream can be lost while
// this daemon stays connected.  The answer is not coming then, as
// after a daemon outage: the loader row goes and a later overscroll
// asks again.  The loss of ANOTHER tab's machine leaves the request of
// the tab on screen in flight.
function testLostMachineCancelsItsTabsNeighbourRead() {
  const {win, posted} = makePage();
  try {
    const {tabId, O} = showHistoryTask(win, posted);
    send(win, {type: 'tabMachine', tabId, machine: 'box'});
    win._testApi.createNewTab();
    const otherTab = win._testApi.getActiveTabId();
    send(win, {type: 'tabMachine', tabId: otherTab, machine: 'other-box'});
    clickTab(win, tabId);

    O.scrollTop = 0;
    wheel(win, O, -50, 10);
    assert.strictEqual(getAdjacent(posted).length, 1, 'prev of 42 is asked');
    assert.ok(O.querySelector('.adjacent-loader'), 'the loader row is up');

    loseMachine(win, otherTab, 'other-box');
    // A tab this page does not have (closed here) can lose its machine too.
    loseMachine(win, 'tab-closed-here', 'other-box');
    assert.ok(
      O.querySelector('.adjacent-loader'),
      "the loss of a background tab's machine must leave the request " +
        'of the tab on screen in flight',
    );
    wheel(win, O, -50, 10);
    assert.strictEqual(getAdjacent(posted).length, 1, 'and its latch held');

    loseMachine(win, tabId, 'box');
    assert.strictEqual(
      O.querySelector('.adjacent-loader'),
      null,
      'BUG - "Loading previous task…" stays up for a neighbour asked ' +
        'of a machine whose stream was lost',
    );
    O.scrollTop = 0;
    wheel(win, O, -50, 10);
    assert.strictEqual(
      getAdjacent(posted).length,
      2,
      'BUG - the lost request keeps every later overscroll latched',
    );
    console.log('PASS a lost machine cancels its tab\u2019s neighbour read');
  } finally {
    win.close();
  }
}

// The same loss ends the reading back to a task that a history click
// asked for (seekSoughtTask): left waiting, it would start again with
// the next neighbour the reader scrolls to, long after the click.
function testLostMachineEndsTheReadBack() {
  const {win, posted} = makePage();
  try {
    const {tabId, O} = showHistoryTask(win, posted);
    send(win, {type: 'tabMachine', tabId, machine: 'box'});
    send(win, {type: 'status', tabId, running: true, startTs: Date.now()});
    send(win, {
      type: 'history',
      offset: 0,
      generation: 0,
      sessions: [
        makeRow({
          id: 'chat-abc',
          task_id: 39,
          title: 'Task 39',
          machine: 'box',
        }),
      ],
    });
    historyRows(win)[0].click();
    const asked = () =>
      getAdjacent(posted).map(m => String(m.taskId) + ' ' + m.direction);
    assert.deepStrictEqual(asked(), ['42 prev'], 'the thread is read back');

    loseMachine(win, tabId, 'box');
    O.scrollTop = 0;
    wheel(win, O, -50, 10);
    assert.deepStrictEqual(
      asked(),
      ['42 prev', '42 prev'],
      'the reader asks again',
    );
    answerPrevTask41(win, tabId);
    assert.deepStrictEqual(
      asked(),
      ['42 prev', '42 prev'],
      'BUG - the read-back that awaited the lost machine went on with ' +
        'the neighbour the reader scrolled to',
    );
    console.log('PASS a lost machine ends the read-back');
  } finally {
    win.close();
  }
}

// CAND-5: every surface showing the tab holds its own stretch of the
// chat's thread, so a neighbour belongs here only when this thread
// ends on the task the reply is adjacent to.
function testNeighbourOfAnotherEdgeIsNotSpliced() {
  const {win, posted} = makePage();
  try {
    const {tabId, O} = showHistoryTask(win, posted);
    const reply = (direction, adjacentTo, taskId) =>
      send(win, {
        type: 'adjacent_task_events',
        tabId,
        direction,
        adjacent_to: adjacentTo,
        task: 'Task ' + taskId,
        task_id: taskId,
        events: [{type: 'system_output', text: 'output ' + taskId + '\n'}],
      });
    const spliced = () =>
      Array.from(O.querySelectorAll('.adjacent-task')).map(
        el => el.dataset.taskId,
      );

    O.scrollTop = 0;
    wheel(win, O, -50, 10);
    assert.strictEqual(getAdjacent(posted).length, 1, 'prev of 42 is asked');

    // Another surface read three tasks further up and asked for more.
    reply('prev', '39', '38');
    reply('next', '50', '51');
    assert.deepStrictEqual(
      spliced(),
      [],
      'a neighbour of a task this thread does not end on must not be ' +
        'spliced: tasks 41..39 would be unreachable above it',
    );
    assert.ok(
      O.querySelector('.adjacent-loader'),
      "another surface's reply must leave this page's request in flight",
    );

    // The answers to this page's own edges are spliced.
    reply('prev', '42', '41');
    reply('next', '42', '43');
    assert.deepStrictEqual(spliced(), ['41', '43']);
    O.scrollTop = 0;
    wheel(win, O, -50, 10);
    const adj = getAdjacent(posted);
    assert.strictEqual(
      String(adj[adj.length - 1].taskId),
      '41',
      'the thread now ends on 41: the next overscroll asks beyond it',
    );
    console.log('PASS a neighbour of another edge is not spliced');
  } finally {
    win.close();
  }
}

// A run that has no history row yet (its setup) is replayed without a
// task id.  The task shown before it is not the thread's edge any
// more: the edges come with the run's next event that names its task,
// so the thread is read back from the run and reaches the task before
// it.  The run can get its row while the replay is on its way: an
// event that names the task then arrives before the replay as well.
function testReplayOfARunWithoutARowForgetsTheTaskBeforeIt() {
  for (const namedBeforeReplay of [false, true]) {
    const {win, posted} = makePage();
    try {
      const {tabId, O} = showHistoryTask(win, posted);
      const live = {type: 'text_delta', tabId, taskId: '43', text: 'live\n'};
      if (namedBeforeReplay) send(win, live);
      send(win, {
        type: 'task_events',
        tabId,
        chat_id: 'chat-abc',
        task_id: null,
        task: 'A run in its setup',
        events: [],
      });
      send(win, {type: 'status', tabId, running: true});
      send(win, live);
      O.scrollTop = 0;
      wheel(win, O, -50, 10);
      assert.deepStrictEqual(
        getAdjacent(posted).map(m => [String(m.taskId), m.direction]),
        [['43', 'prev']],
        'the thread is read back from the run, not from the task before it' +
          (namedBeforeReplay ? ' (task named before the replay)' : ''),
      );
    } finally {
      win.close();
    }
  }
  console.log('PASS a replay of a run without a row forgets the task before it');
}

function main() {
  const tests = [
    testNoDuplicateAdjacentRegionAfterTabSwitch,
    testAnchorsOfAnotherTaskAreNotRestored,
    testDetachedLoaderRemovedOnRestore,
    testDaemonDisconnectClearsAdjacentLoading,
    testLostMachineCancelsItsTabsNeighbourRead,
    testLostMachineEndsTheReadBack,
    testNeighbourOfAnotherEdgeIsNotSpliced,
    testReplayOfARunWithoutARowForgetsTheTaskBeforeIt,
  ];
  const failures = [];
  for (const t of tests) {
    try {
      t();
    } catch (e) {
      failures.push(t.name + ': ' + e.message);
      console.error('FAIL ' + t.name + '\n  ' + e.message);
    }
  }
  assert.strictEqual(
    failures.length,
    0,
    'conc2026_adjacent_task_races: ' +
      failures.length +
      ' test(s) failed:\n' +
      failures.join('\n'),
  );
  console.log('All conc2026_adjacent_task_races tests passed');
}

main();
