// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A run_agent dispatch's sub-agent tab must live and die exactly like a
// run_parallel sub-task's.
//
// The daemon gives a run_agent child the run_parallel sub-agent
// contract: a nested tab via `new_tab` when it starts, an
// `openSubagentTab` announcement while it runs (a client that connects
// mid-run gets it after the parent's replay) and a `subagentDone` when
// it ends, which is also how a replay of the parent announces a
// finished child.  The page ties no tab to a tool-call panel, so these
// hold for BOTH tools:
//
//  1. live: the child's tab opens when the child starts and
//     `subagentDone` closes it;
//  2. a client connecting while the parent runs shows the running
//     child's tab;
//  3. replay of the finished parent: a finished child gets no tab.
//
// The `run_agent` tool-call panel is a fan-out panel too -- an
// open-ended one, since a dispatch may spawn no child, one, or several
// in sequence.  The share export puts each child under the panel of
// the call that was running when it started: the child's start stamp
// (`task_settings.start_ts`) falls between the call's `tool_call.ts`
// and `tool_result.ts`.

/* global require, console, process */

'use strict';

const assert = require('assert');
const {JSDOM} = require('jsdom');
const harness = require('./simplify2_harness');
const {send} = harness;
const {
  announceFinished,
  subagentTabEls,
  spawnSubagent,
  toolPanel,
} = require('./subagent_tab_helpers');
const {shareWithTasks} = require('./share_helpers');

/** A fresh chat page plus the id of the tab it booted on. */
function makePage() {
  const {win, posted} = harness.makePage();
  return {win, posted, parentId: harness.readyTabId(posted)};
}

/** The registry snapshot: one chat tab, `parentId`, on PARENT_TASK. */
function parentTabsState(win, parentId) {
  send(win, {
    type: 'tabs_state',
    tabs: [
      {
        tabId: parentId,
        chatId: 'chat-1',
        title: 'parent',
        taskId: PARENT_TASK,
        workDir: '',
        scopeWorkDir: '',
      },
    ],
  });
}

/**
 * The daemon's sub-agent tab announcement for a child of `parentId`;
 * `fields` carries tab_id, task_id, description and the rest.
 */
function announceSubagent(win, parentId, fields) {
  send(
    win,
    Object.assign(
      {type: 'openSubagentTab', parent_tab_id: parentId, isSubagentTab: true},
      fields,
    ),
  );
}

/** The tool_call event a parent emits for *toolName* spawning one child. */
function spawnCall(toolName, parentId, parentTask) {
  const extras =
    toolName === 'run_parallel'
      ? {tasks: JSON.stringify(['child job'])}
      : {agent: 'gmail', task: 'child job'};
  return {
    type: 'tool_call',
    name: toolName,
    extras,
    taskId: parentTask,
    tabId: parentId,
  };
}

const CHILD = 'child-task-1';
const PARENT_TASK = 'parent-task-1';

// ---------------------------------------------------------------- live

function liveChildFinishes(toolName) {
  const {win, posted, parentId} = makePage();
  send(win, {
    type: 'status',
    running: true,
    tabId: parentId,
    startTs: Date.now(),
    taskId: PARENT_TASK,
  });
  send(win, spawnCall(toolName, parentId, PARENT_TASK));
  const panel = toolPanel(win, toolName);
  assert.ok(panel, toolName + ' tool_call must render a panel');

  const subTabId = spawnSubagent(
    {win, posted},
    parentId,
    CHILD,
    'child job',
    0,
    {
      extra: {isSubagentTab: true},
    },
  );
  send(win, {
    type: 'status',
    running: true,
    tabId: subTabId,
    startTs: Date.now(),
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'the child gets one tab while running',
  );

  // What the daemon sends when the child ends (captured from a live run).
  send(win, {type: 'result', text: 'done', taskId: CHILD, tabId: subTabId});
  send(win, {type: 'subagentDone', tab_id: subTabId, tabId: ''});
  send(win, {type: 'status', running: false, tabId: subTabId});
  send(win, {
    type: 'tool_result',
    tool_name: toolName,
    content: 'success: true',
    is_error: false,
    taskId: PARENT_TASK,
    tabId: parentId,
  });

  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    toolName + ': the finished child tab must close',
  );
  assert.ok(
    !posted.some(m => m.type === 'closeTab'),
    toolName + ': the daemon ended the tab; the page reports no close of it',
  );
  win.close();
  console.log('  ok - ' + toolName + ': live subagentDone closes the tab');
}

// -------------------------------------------------------------- replay

/**
 * Drive the daemon's `ready` replay of a FINISHED parent whose *toolName*
 * call spawned one (finished) child, then its `_open_persisted_subagent_tabs`
 * announcement of that child (`subagentDone`).  Returns the page for
 * further assertions.
 */
function replayFinishedParent(toolName) {
  const {win, parentId} = makePage();
  const subTabId = parentId + '__sub_' + CHILD;
  parentTabsState(win, parentId);
  send(win, {
    type: 'task_events',
    task: 'parent',
    task_id: PARENT_TASK,
    chat_id: 'chat-1',
    extra: '',
    tabId: parentId,
    events: [
      {type: 'prompt', text: 'parent'},
      spawnCall(toolName, parentId, PARENT_TASK),
      {
        type: 'tool_result',
        tool_name: toolName,
        content: 'success: true',
        taskId: PARENT_TASK,
      },
      {type: 'result', text: 'done', taskId: PARENT_TASK},
      {type: 'task_done'},
    ],
  });
  announceFinished(win, [CHILD], () => subTabId);
  return {win, subTabId};
}

function replayKeepsFinishedChildClosed(toolName) {
  const {win} = replayFinishedParent(toolName);
  const panel = toolPanel(win, toolName);
  assert.ok(panel, toolName + ' panel must be rendered by the replay');
  assert.ok(
    panel.classList.contains('collapsed'),
    toolName + ': a finished fan-out panel replays collapsed',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    toolName +
      ": the daemon's announcement of a finished child must not open its tab",
  );
  win.close();
  console.log(
    '  ok - ' + toolName + ': replay keeps the finished child closed',
  );
}

function replayRepeatedReadyStaysClosed(toolName) {
  // Every reconnect replays the parent again: the second announcement
  // must be as quiet as the first (still no tab).
  const {win, subTabId} = replayFinishedParent(toolName);
  announceFinished(win, [CHILD], () => subTabId);
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    toolName + ': a repeated announcement opens no tab either',
  );
  win.close();
  console.log('  ok - ' + toolName + ': repeated ready replays stay closed');
}

// ---------------------------------------------------------- run_agent
// A run_agent call declares no child count: it may spawn none (an
// argument error returns before any dispatch), one, or several in
// sequence (a multi-<task> prompt runs one child after another). Every
// child gets its tab when it starts, and the share export attributes
// each child to its call.

/** Every fan-out panel of the visible transcript, in transcript order. */
function fanoutPanels(win) {
  return Array.from(
    win.document.querySelectorAll('#output .ev.tc.tc-run-parallel'),
  );
}

/**
 * The task ids the share export puts under each fan-out panel of the
 * parent's transcript (`data-rp-subagents`), in transcript order.
 * *rows* are the parent's sub-agents as the daemon's `share_tasks`
 * reply lists them: [task id, task text, start stamp].
 */
function sharedChildrenOfPanels(win, posted, rows) {
  const msg = shareWithTasks({win, posted}, [
    {
      task: 'parent',
      task_id: PARENT_TASK,
      events: [],
      subagents: rows.map(([id, text, startTs]) => ({
        task: text,
        task_id: id,
        parent_task_id: PARENT_TASK,
        events: [{type: 'task_settings', settings: {start_ts: startTs}}],
      })),
    },
  ]);
  assert.ok(msg, 'the share_tasks reply must produce a shareChat command');
  const doc = new JSDOM('<div id="app">' + msg.html + '</div>').window.document;
  return Array.from(
    doc.querySelectorAll('.share-task:not(.share-subagent) .tc-run-parallel'),
    p => (p.getAttribute('data-rp-subagents') || '').split(' ').filter(Boolean),
  );
}

function runAgentNoChildDoesNotStealNextSpawn() {
  const {win, posted, parentId} = makePage();
  send(win, {
    type: 'status',
    running: true,
    tabId: parentId,
    startTs: Date.now(),
    taskId: PARENT_TASK,
  });
  // A run_agent call that never dispatched: the tool returns an error
  // string (is_error false -- the tool itself answered) with no child.
  send(win, spawnCall('run_agent', parentId, PARENT_TASK));
  send(win, {
    type: 'tool_result',
    tool_name: 'run_agent',
    is_error: false,
    content: "Error: unknown agent 'nope'",
    taskId: PARENT_TASK,
    tabId: parentId,
  });

  send(win, spawnCall('run_parallel', parentId, PARENT_TASK));
  const before = posted.length;
  send(win, {
    type: 'new_tab',
    task_id: 'parallel-child',
    parent_tab_id: parentId,
    taskId: '',
  });
  const panels = fanoutPanels(win);
  assert.strictEqual(panels.length, 2, 'both calls render a fan-out panel');
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'the run_parallel child gets its tab',
  );
  assert.ok(
    posted
      .slice(before)
      .some(m => m.type === 'resumeSession' && m.taskId === 'parallel-child'),
    'the run_parallel child is resumed into its tab',
  );
  send(win, {
    type: 'subagentDone',
    tab_id: subagentTabEls(win)[0].dataset.tabId,
  });
  send(win, {
    type: 'tool_result',
    tool_name: 'run_parallel',
    is_error: false,
    content: 'done',
    taskId: PARENT_TASK,
    tabId: parentId,
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'the finished child closes',
  );
  win.close();
  console.log(
    '  ok - run_agent: a dispatch that spawned nothing does not steal the next spawn',
  );
}

function runAgentMultiChildDispatchLive() {
  const {win, posted, parentId} = makePage();
  send(win, {
    type: 'status',
    running: true,
    tabId: parentId,
    startTs: Date.now(),
    taskId: PARENT_TASK,
  });
  send(win, spawnCall('run_agent', parentId, PARENT_TASK));

  // Child 1 runs and finishes: its tab closes while the dispatch is
  // still running.
  send(win, {
    type: 'new_tab',
    task_id: 'c1',
    parent_tab_id: parentId,
    taskId: '',
  });
  const sub1 = parentId + '__sub_c1';
  announceSubagent(win, parentId, {
    tab_id: sub1,
    description: 'child job',
    task_id: 'c1',
  });
  send(win, {type: 'subagentDone', tab_id: sub1, tabId: ''});
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'child 1 tab closed on subagentDone',
  );

  // Child 2 spawns while the call is still in flight: it gets a tab.
  const before = posted.length;
  send(win, {
    type: 'new_tab',
    task_id: 'c2',
    parent_tab_id: parentId,
    taskId: '',
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'the second child of the same dispatch gets a tab',
  );
  assert.ok(
    posted
      .slice(before)
      .some(m => m.type === 'resumeSession' && m.taskId === 'c2'),
    'the second child is resumed into its tab',
  );

  send(win, {type: 'subagentDone', tab_id: parentId + '__sub_c2', tabId: ''});
  assert.strictEqual(subagentTabEls(win).length, 0, 'child 2 tab closed');
  win.close();
  console.log(
    '  ok - run_agent: each child of a multi-child dispatch gets its own tab',
  );
}

function shareAttributesInterleavedCalls() {
  // Replay of a finished task: run_agent (two <task> children), then
  // run_parallel (one declared child), then another run_agent (one
  // child whose agent script rewrote the prompt, so its row text has
  // nothing in common with the call's `task`). The daemon announces
  // the four persisted rows finished, and its `share_tasks` reply lists
  // them in spawn order with their start stamps; the export must put
  // each under the call that was running when it started.
  const {win, posted, parentId} = makePage();
  const T0 = 1_700_000_000_000;
  parentTabsState(win, parentId);
  send(win, {
    type: 'task_events',
    task: 'parent',
    task_id: PARENT_TASK,
    chat_id: 'chat-1',
    extra: '',
    tabId: parentId,
    events: [
      {type: 'prompt', text: 'parent', ts: T0},
      {
        type: 'tool_call',
        name: 'run_agent',
        extras: {agent: 'gmail', task: 'review'},
        taskId: PARENT_TASK,
        tabId: parentId,
        ts: T0 + 1000,
      },
      {
        type: 'tool_result',
        tool_name: 'run_agent',
        content: 'success: true',
        taskId: PARENT_TASK,
        ts: T0 + 9000,
      },
      {
        type: 'tool_call',
        name: 'run_parallel',
        extras: {tasks: JSON.stringify(['review code'])},
        taskId: PARENT_TASK,
        tabId: parentId,
        ts: T0 + 10_000,
      },
      {
        type: 'tool_result',
        tool_name: 'run_parallel',
        content: 'success: true',
        taskId: PARENT_TASK,
        ts: T0 + 19_000,
      },
      {
        type: 'tool_call',
        name: 'run_agent',
        extras: {agent: './my_agent.py', task: 'original caller text'},
        taskId: PARENT_TASK,
        tabId: parentId,
        ts: T0 + 20_000,
      },
      {
        type: 'tool_result',
        tool_name: 'run_agent',
        content: 'success: true',
        taskId: PARENT_TASK,
        ts: T0 + 29_000,
      },
      {type: 'result', text: 'done', taskId: PARENT_TASK, ts: T0 + 30_000},
      {type: 'task_done', ts: T0 + 30_000},
    ],
  });
  const rows = [
    // [task id, row text, row start]
    ['a1', 'You are the gmail channel agent: review', T0 + 2000],
    ['a2', 'You are the gmail channel agent: review', T0 + 5000],
    ['p1', 'review code', T0 + 11_000],
    ['s1', 'fixed prompt from script', T0 + 21_000],
  ];
  announceFinished(
    win,
    rows.map(row => row[0]),
    id => parentId + '__sub_' + id,
  );
  const panels = fanoutPanels(win);
  assert.strictEqual(panels.length, 3, 'three fan-out panels replayed');
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'no finished child gets a tab on replay',
  );
  for (const p of panels)
    assert.ok(
      p.classList.contains('collapsed'),
      'every finished panel replays collapsed',
    );
  assert.deepStrictEqual(
    sharedChildrenOfPanels(win, posted, rows),
    [['a1', 'a2'], ['p1'], ['s1']],
    'each finished child is exported under the call that was running when it started',
  );
  win.close();
  console.log(
    '  ok - share export attributes interleaved run_agent / run_parallel children by start time',
  );
}

function shareBoundaryTieBelongsToNextCall() {
  // One model turn can issue two fan-out calls back to back, so the
  // first call's result, the second call's start and the second call's
  // first child may share one millisecond. A child cannot start on the
  // very millisecond its own call's result is stamped (it finished
  // before that), so the tie is the next call's. A row that started
  // on that millisecond with no call starting on it is no call's, and
  // a call that never returned (the task was stopped) takes every row
  // that started after it.
  const {win, posted, parentId} = makePage();
  const T0 = 1_700_000_000_000;
  parentTabsState(win, parentId);
  send(win, {
    type: 'task_events',
    task: 'parent',
    task_id: PARENT_TASK,
    chat_id: 'chat-1',
    extra: '',
    tabId: parentId,
    events: [
      {type: 'prompt', text: 'parent', ts: T0},
      {
        type: 'tool_call',
        name: 'run_agent',
        extras: {agent: 'gmail', task: 'a'},
        taskId: PARENT_TASK,
        tabId: parentId,
        ts: T0 + 1000,
      },
      {
        type: 'tool_result',
        tool_name: 'run_agent',
        content: 'ok',
        taskId: PARENT_TASK,
        ts: T0 + 5000,
      },
      {
        type: 'tool_call',
        name: 'run_agent',
        extras: {agent: 'gmail', task: 'b'},
        taskId: PARENT_TASK,
        tabId: parentId,
        ts: T0 + 5000,
      },
      {
        type: 'tool_result',
        tool_name: 'run_agent',
        content: 'ok',
        taskId: PARENT_TASK,
        ts: T0 + 9000,
      },
      {
        type: 'tool_call',
        name: 'run_agent',
        extras: {agent: 'gmail', task: 'c'},
        taskId: PARENT_TASK,
        tabId: parentId,
        ts: T0 + 9200,
      },
      {type: 'task_stopped', ts: T0 + 9900},
    ],
  });
  const rows = [
    ['first-child', 'first-child', T0 + 1000],
    ['second-child', 'second-child', T0 + 5000],
    ['between-calls', 'between-calls', T0 + 9000],
    ['third-child', 'third-child', T0 + 9500],
  ];
  assert.deepStrictEqual(
    sharedChildrenOfPanels(win, posted, rows),
    [['first-child'], ['second-child'], ['third-child']],
    "a start on a call's own start millisecond is that call's; on a call's result millisecond it is not that call's",
  );
  win.close();
  console.log(
    '  ok - share export: a start on the millisecond a call returned belongs to the next call',
  );
}

function doneUnderfilledRunParallelDoesNotStealLiveRunAgentChild() {
  // A run_parallel that declared a worker but spawned none (the fan-out
  // was refused) has returned: the next call's live child gets its tab
  // all the same.
  const {win, posted, parentId} = makePage();
  send(win, {
    type: 'status',
    running: true,
    tabId: parentId,
    startTs: Date.now(),
    taskId: PARENT_TASK,
  });
  send(win, spawnCall('run_parallel', parentId, PARENT_TASK));
  send(win, {
    type: 'tool_result',
    tool_name: 'run_parallel',
    is_error: false,
    content: 'Error: budget too small',
    taskId: PARENT_TASK,
    tabId: parentId,
  });
  send(win, spawnCall('run_agent', parentId, PARENT_TASK));
  const before = posted.length;
  send(win, {
    type: 'new_tab',
    task_id: 'real-child',
    parent_tab_id: parentId,
    taskId: '',
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'the run_agent child gets its tab',
  );
  assert.ok(
    posted
      .slice(before)
      .some(m => m.type === 'resumeSession' && m.taskId === 'real-child'),
    'the run_agent child is resumed into its tab',
  );
  send(win, {
    type: 'subagentDone',
    tab_id: subagentTabEls(win)[0].dataset.tabId,
  });
  send(win, {
    type: 'tool_result',
    tool_name: 'run_agent',
    is_error: false,
    content: 'success: true',
    taskId: PARENT_TASK,
    tabId: parentId,
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'the finished child closes',
  );
  win.close();
  console.log(
    '  ok - live: a returned, underfilled run_parallel does not steal the next spawn',
  );
}

function freshReconnectShowsRunningChild(toolName) {
  // A client connecting while the parent runs replays the parent's
  // transcript (its fan-out call still without a result) and is then
  // told about the running child: that child gets a tab on this client
  // too.
  const {win, posted, parentId} = makePage();
  const T0 = Date.now() - 60_000;
  parentTabsState(win, parentId);
  send(win, {
    type: 'status',
    running: true,
    tabId: parentId,
    startTs: T0,
    taskId: PARENT_TASK,
  });
  const call = spawnCall(toolName, parentId, PARENT_TASK);
  call.ts = T0 + 1000;
  send(win, {
    type: 'task_events',
    task: 'parent',
    task_id: PARENT_TASK,
    chat_id: 'chat-1',
    extra: '',
    tabId: parentId,
    events: [
      {type: 'prompt', text: 'parent', ts: T0},
      {
        type: 'tool_call',
        name: 'Read',
        path: 'x.txt',
        taskId: PARENT_TASK,
        ts: T0 + 100,
      },
      {
        type: 'tool_result',
        tool_name: 'Read',
        content: 'x',
        taskId: PARENT_TASK,
        ts: T0 + 200,
      },
      call,
    ],
  });
  const panel = toolPanel(win, toolName);
  assert.ok(panel, toolName + ' panel replayed');
  const before = posted.length;
  const subTabId = parentId + '__sub_' + CHILD;
  announceSubagent(win, parentId, {
    tab_id: subTabId,
    description: 'child job',
    task_id: CHILD,
    taskIndex: 0,
    startTs: T0 + 2000,
  });
  assert.deepStrictEqual(
    subagentTabEls(win).map(el => el.dataset.tabId),
    [subTabId],
    toolName + ': the running child gets a tab on the reconnected client',
  );
  assert.ok(
    !posted.slice(before).some(m => m.type === 'closeTab'),
    'nothing was closed behind the announcement',
  );
  // The child finishing later closes its tab like on any client.
  send(win, {type: 'subagentDone', tab_id: subTabId, tabId: ''});
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    toolName + ': subagentDone closes it',
  );
  win.close();
  console.log(
    '  ok - ' +
      toolName +
      ': a fresh reconnect mid-run shows the running child',
  );
}

for (const tool of ['run_agent', 'run_parallel']) {
  liveChildFinishes(tool);
  replayKeepsFinishedChildClosed(tool);
  replayRepeatedReadyStaysClosed(tool);
}
runAgentNoChildDoesNotStealNextSpawn();
runAgentMultiChildDispatchLive();
shareAttributesInterleavedCalls();
shareBoundaryTieBelongsToNextCall();
doneUnderfilledRunParallelDoesNotStealLiveRunAgentChild();
for (const tool of ['run_agent', 'run_parallel'])
  freshReconnectShowsRunningChild(tool);
console.log('runAgentSubagentTabParity: all tests passed');
process.exit(0);
