// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A run_agent dispatch's sub-agent tab must live and die exactly like a
// run_parallel sub-task's.
//
// The daemon already gives a run_agent child the run_parallel sub-agent
// contract (a nested tab via `new_tab`, a `subagentDone` when it ends,
// and a re-announce of every finished child whenever the parent task is
// replayed -- which happens on every webview `ready`: page load,
// reconnect, VS Code re-creating a hidden webview).  The webview closed
// the tab on `subagentDone` but had nothing to absorb the re-announce:
// a finished run_parallel child lands in its collapsed fan-out panel as
// a panel entry, while a run_agent child had no fan-out panel and got a
// fresh tab every time.  The user saw a sub-agent tab that "never
// closed".
//
// The fix renders the `run_agent` tool-call panel as a fan-out panel
// too -- an open-ended one, since a dispatch may spawn no child, one,
// or several in sequence -- so these hold for BOTH tools:
//
//  1. live: `subagentDone` closes the child's tab, and the panel is
//     collapsed once the fan-out is complete (run_parallel: its last
//     child closed; run_agent: its `tool_result` arrived, because the
//     dispatch may still spawn another child until then);
//  2. replay of the finished parent: the daemon's `openSubagentTab
//     {isDone: true}` re-announce opens no tab;
//  3. the transcript is still reachable: expanding the panel reopens
//     the child's tab (and resumes its session), collapsing closes it.
//
// Replayed children are attributed to their call by time: the
// announcement's `startTs` (the row's start) falls between the call's
// `tool_call.ts` and `tool_result.ts`.

/* global require, __dirname, console, process */

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  for (const f of ['panelCopy.js', 'api.js', 'main.js']) {
    win.eval(fs.readFileSync(path.join(MEDIA, f), 'utf8'));
  }
  const ready = posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  return {win, posted, parentId: ready.tabId};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function subagentTabEls(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab.subagent-tab'),
  );
}

/** The tool-call panel whose header names *toolName*. */
function toolPanel(win, toolName) {
  const headers = win.document.querySelectorAll('#output .ev.tc .tc-h');
  for (const h of headers) {
    const txt = (h.textContent || '').replace(/^[^A-Za-z]+/, '').trim();
    if (txt.startsWith(toolName)) return h.closest('.ev.tc');
  }
  return null;
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
  const {win, posted, parentId} = makeWebview();
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

  const before = posted.length;
  send(win, {
    type: 'new_tab',
    task_id: CHILD,
    parent_tab_id: parentId,
    taskId: '',
  });
  const resume = posted
    .slice(before)
    .find(m => m.type === 'resumeSession' && m.taskId === CHILD);
  assert.ok(resume, 'new_tab must make the webview resume the child');
  const subTabId = resume.tabId;
  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabId,
    parent_tab_id: parentId,
    description: 'child job',
    task_id: CHILD,
    isSubagentTab: true,
    isDone: false,
  });
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
  assert.ok(
    !panel.classList.contains('collapsed'),
    'the panel stays open while the child runs',
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
    posted.some(m => m.type === 'closeTab' && m.tabId === subTabId),
    toolName + ': the daemon must be told about the close',
  );
  assert.ok(
    panel.classList.contains('collapsed'),
    toolName + ': the panel collapses once its only child is done',
  );
  win.close();
  console.log(
    '  ok - ' +
      toolName +
      ': live subagentDone closes the tab and collapses the panel',
  );
}

// -------------------------------------------------------------- replay

/**
 * Drive the daemon's `ready` replay of a FINISHED parent whose *toolName*
 * call spawned one (finished) child, then its `_open_persisted_subagent_tabs`
 * re-announce of that child.  Returns the webview for further assertions.
 */
function replayFinishedParent(toolName) {
  const {win, posted, parentId} = makeWebview();
  const subTabId = parentId + '__sub_' + CHILD;
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
  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabId,
    parent_tab_id: parentId,
    description: 'child job',
    task_id: CHILD,
    taskIndex: 0,
    isSubagentTab: true,
    isDone: true,
  });
  send(win, {
    type: 'task_events',
    task: 'child job',
    task_id: CHILD,
    chat_id: 'chat-1',
    extra: '',
    tabId: subTabId,
    events: [
      {type: 'prompt', text: 'child job'},
      {type: 'result', text: 'ok'},
    ],
  });
  return {win, posted, parentId, subTabId};
}

function replayKeepsFinishedChildClosed(toolName) {
  const {win, posted, parentId} = replayFinishedParent(toolName);
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
      ": the daemon's re-announce of a finished child must not reopen its tab",
  );

  // The transcript stays reachable: expanding the panel reopens the tab.
  const before = posted.length;
  panel
    .querySelector('.tc-h')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    !panel.classList.contains('collapsed'),
    'header click expands the panel',
  );
  const reopened = subagentTabEls(win);
  assert.strictEqual(
    reopened.length,
    1,
    toolName + ': expanding the panel reopens the child tab',
  );
  assert.strictEqual(
    reopened[0].dataset.tabId,
    parentId + '__sub_' + CHILD,
    'the reopened tab keeps the deterministic sub-agent id',
  );
  assert.ok(
    posted
      .slice(before)
      .some(m => m.type === 'resumeSession' && m.taskId === CHILD),
    'the reopened tab resumes the child session',
  );

  // Collapsing closes it again.
  panel
    .querySelector('.tc-h')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    toolName + ': collapsing the panel closes the child tab',
  );
  win.close();
  console.log(
    '  ok - ' +
      toolName +
      ': replay keeps the finished child closed; the panel reopens it',
  );
}

function replayRepeatedReadyStaysClosed(toolName) {
  // Every reconnect replays the parent again: the second re-announce
  // must be as quiet as the first (one panel entry, still no tab).
  const {win, parentId, subTabId} = replayFinishedParent(toolName);
  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabId,
    parent_tab_id: parentId,
    description: 'child job',
    task_id: CHILD,
    taskIndex: 0,
    isSubagentTab: true,
    isDone: true,
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    toolName + ': a repeated re-announce opens no tab either',
  );
  const panel = toolPanel(win, toolName);
  assert.strictEqual(
    (panel._rpSubagents || []).filter(en => String(en.taskId) === CHILD).length,
    1,
    toolName + ': the child is recorded once on its panel',
  );
  win.close();
  console.log('  ok - ' + toolName + ': repeated ready replays stay closed');
}

// ---------------------------------------------------------- run_agent
// A run_agent call declares no child count: it may spawn none (an
// argument error returns before any dispatch), one, or several in
// sequence (a multi-<task> prompt runs one child after another). The
// panel is open-ended, so each of those must attribute and collapse
// correctly, live and on replay.

/** Every fan-out panel of the visible transcript, in transcript order. */
function fanoutPanels(win) {
  return Array.from(
    win.document.querySelectorAll('#output .ev.tc.tc-run-parallel'),
  );
}

/** The task ids each panel recorded, as JSON (jsdom-realm arrays). */
function panelChildren(panels) {
  return JSON.stringify(
    panels.map(p => (p._rpSubagents || []).map(en => String(en.taskId))),
  );
}

function runAgentNoChildDoesNotStealNextSpawn() {
  const {win, posted, parentId} = makeWebview();
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
  const agentPanel = fanoutPanels(win)[0];
  assert.strictEqual(
    (agentPanel._rpSubagents || []).length,
    0,
    'the failed dispatch recorded no child',
  );

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
    panelChildren(panels),
    JSON.stringify([[], ['parallel-child']]),
    'the run_parallel spawn belongs to the run_parallel call, not to the empty run_agent panel',
  );
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
  win.close();
  console.log(
    '  ok - run_agent: a dispatch that spawned nothing does not steal the next spawn',
  );
}

function runAgentMultiChildDispatchLive() {
  const {win, posted, parentId} = makeWebview();
  send(win, {
    type: 'status',
    running: true,
    tabId: parentId,
    startTs: Date.now(),
    taskId: PARENT_TASK,
  });
  send(win, spawnCall('run_agent', parentId, PARENT_TASK));
  const panel = fanoutPanels(win)[0];

  // Child 1 runs and finishes; its tab closes but the dispatch is
  // still running, so the panel must not collapse yet.
  send(win, {
    type: 'new_tab',
    task_id: 'c1',
    parent_tab_id: parentId,
    taskId: '',
  });
  const sub1 = parentId + '__sub_c1';
  send(win, {
    type: 'openSubagentTab',
    tab_id: sub1,
    parent_tab_id: parentId,
    description: 'child job',
    task_id: 'c1',
    isSubagentTab: true,
    isDone: false,
  });
  send(win, {type: 'subagentDone', tab_id: sub1, tabId: ''});
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'child 1 tab closed on subagentDone',
  );
  assert.ok(
    !panel.classList.contains('collapsed'),
    'the panel stays open: the dispatch may spawn again',
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
  assert.strictEqual(
    panelChildren([panel]),
    JSON.stringify([['c1', 'c2']]),
    'both children belong to the one run_agent call',
  );

  send(win, {type: 'subagentDone', tab_id: parentId + '__sub_c2', tabId: ''});
  assert.strictEqual(subagentTabEls(win).length, 0, 'child 2 tab closed');
  assert.ok(
    !panel.classList.contains('collapsed'),
    'still open: the result has not arrived',
  );
  send(win, {
    type: 'tool_result',
    tool_name: 'run_agent',
    is_error: false,
    content: 'success: true',
    taskId: PARENT_TASK,
    tabId: parentId,
  });
  assert.ok(
    panel.classList.contains('collapsed'),
    'the result completes the fan-out: the panel collapses',
  );

  // Expanding brings both children back, collapsing closes both.
  panel
    .querySelector('.tc-h')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'expanding reopens every child of the dispatch',
  );
  panel
    .querySelector('.tc-h')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'collapsing closes them again',
  );
  win.close();
  console.log(
    '  ok - run_agent: a multi-child dispatch keeps spawning into an open panel',
  );
}

function replayAttributesInterleavedCalls() {
  // Replay of a finished task: run_agent (two <task> children), then
  // run_parallel (one declared child), then another run_agent (one
  // child whose agent script rewrote the prompt, so its row text has
  // nothing in common with the call's `task`). The daemon re-announces
  // the four persisted rows in spawn order with their start stamps;
  // each must land on the call that was running when it started.
  const {win, parentId} = makeWebview();
  const T0 = 1_700_000_000_000;
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
  rows.forEach(([id, text, startTs], idx) => {
    send(win, {
      type: 'openSubagentTab',
      tab_id: parentId + '__sub_' + id,
      parent_tab_id: parentId,
      description: text,
      task_id: id,
      taskIndex: idx,
      isSubagentTab: true,
      isDone: true,
      startTs,
    });
  });
  const panels = fanoutPanels(win);
  assert.strictEqual(panels.length, 3, 'three fan-out panels replayed');
  assert.strictEqual(
    panelChildren(panels),
    JSON.stringify([['a1', 'a2'], ['p1'], ['s1']]),
    'each finished child is recorded on the call that was running when it started',
  );
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

  // Expanding the first run_agent panel reopens exactly its two children.
  panels[0]
    .querySelector('.tc-h')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    JSON.stringify(subagentTabEls(win).map(el => el.dataset.tabId)),
    JSON.stringify([parentId + '__sub_a1', parentId + '__sub_a2']),
    "expanding the run_agent panel reopens the dispatch's children only",
  );
  win.close();
  console.log(
    '  ok - replay attributes interleaved run_agent / run_parallel children by start time',
  );
}

function replayWithoutStampsFallsBackToCounting() {
  // Legacy transcripts carry no `ts` and legacy rows no `startTs`: the
  // run_parallel children are still dealt by declared count, in spawn
  // order, and whatever is left goes to the newest panel.
  const {win, parentId} = makeWebview();
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
  send(win, {
    type: 'task_events',
    task: 'parent',
    task_id: PARENT_TASK,
    chat_id: 'chat-1',
    extra: '',
    tabId: parentId,
    events: [
      {type: 'prompt', text: 'parent'},
      {
        type: 'tool_call',
        name: 'run_parallel',
        extras: {tasks: JSON.stringify(['w1', 'w2'])},
        taskId: PARENT_TASK,
        tabId: parentId,
      },
      {
        type: 'tool_result',
        tool_name: 'run_parallel',
        content: 'success: true',
        taskId: PARENT_TASK,
      },
      {
        type: 'tool_call',
        name: 'run_agent',
        extras: {agent: 'gmail', task: 'mail'},
        taskId: PARENT_TASK,
        tabId: parentId,
      },
      {
        type: 'tool_result',
        tool_name: 'run_agent',
        content: 'success: true',
        taskId: PARENT_TASK,
      },
      {type: 'result', text: 'done', taskId: PARENT_TASK},
      {type: 'task_done'},
    ],
  });
  ['p1', 'p2', 'a1'].forEach((id, idx) => {
    send(win, {
      type: 'openSubagentTab',
      tab_id: parentId + '__sub_' + id,
      parent_tab_id: parentId,
      description: id,
      task_id: id,
      taskIndex: idx,
      isSubagentTab: true,
      isDone: true,
    });
  });
  assert.strictEqual(
    panelChildren(fanoutPanels(win)),
    JSON.stringify([['p1', 'p2'], ['a1']]),
    'unstamped rows fill the declared run_parallel count first, the rest go to the newest panel',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'no finished child gets a tab on replay',
  );
  win.close();
  console.log('  ok - replay without stamps falls back to declared counts');
}

function replayBoundaryTieBelongsToNextCall() {
  // One model turn can issue two fan-out calls back to back, so the
  // first call's result, the second call's start and the second call's
  // first child may share one millisecond. A child cannot start on the
  // very millisecond its own call's result is stamped (it finished
  // before that), so the tie is the next call's.
  const {win, parentId} = makeWebview();
  const T0 = 1_700_000_000_000;
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
      {type: 'result', text: 'done', taskId: PARENT_TASK, ts: T0 + 9000},
      {type: 'task_done', ts: T0 + 9000},
    ],
  });
  const rows = [
    ['first-child', T0 + 1000],
    ['second-child', T0 + 5000],
  ];
  rows.forEach(([id, startTs], idx) => {
    send(win, {
      type: 'openSubagentTab',
      tab_id: parentId + '__sub_' + id,
      parent_tab_id: parentId,
      description: id,
      task_id: id,
      taskIndex: idx,
      isSubagentTab: true,
      isDone: true,
      startTs,
    });
  });
  assert.strictEqual(
    panelChildren(fanoutPanels(win)),
    JSON.stringify([['first-child'], ['second-child']]),
    "a start on a call's own start millisecond is that call's; on the previous call's result millisecond it is not the previous call's",
  );
  win.close();
  console.log(
    '  ok - replay: a start on the millisecond a call returned belongs to the next call',
  );
}

function doneUnderfilledRunParallelDoesNotStealLiveRunAgentChild() {
  // A run_parallel that declared a worker but spawned none (the fan-out
  // was refused) has returned: it cannot spawn any more, so the next
  // call's live child is not dealt to it by count.
  const {win, posted, parentId} = makeWebview();
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
    panelChildren(fanoutPanels(win)),
    JSON.stringify([[], ['real-child']]),
    'the live spawn belongs to the call still running, not to the returned, underfilled run_parallel',
  );
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
  win.close();
  console.log(
    '  ok - live: a returned, underfilled run_parallel does not steal the next spawn',
  );
}

function freshReconnectShowsRunningChild(toolName) {
  // A client connecting while the parent runs replays the parent's
  // transcript (its fan-out call still without a result) and is then
  // told about the running child. The in-flight panel must stay
  // expanded so that child gets a tab on this client too.
  const {win, posted, parentId} = makeWebview();
  const T0 = Date.now() - 60_000;
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
  assert.ok(
    !panel.classList.contains('collapsed'),
    toolName + ': the in-flight fan-out replays expanded on a running task',
  );
  const before = posted.length;
  const subTabId = parentId + '__sub_' + CHILD;
  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabId,
    parent_tab_id: parentId,
    description: 'child job',
    task_id: CHILD,
    taskIndex: 0,
    isSubagentTab: true,
    isDone: false,
    startTs: T0 + 2000,
  });
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    toolName + ': the running child gets a tab on the reconnected client',
  );
  assert.strictEqual(
    panelChildren([panel]),
    JSON.stringify([[CHILD]]),
    'the child is recorded on the in-flight panel',
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
replayAttributesInterleavedCalls();
replayWithoutStampsFallsBackToCounting();
replayBoundaryTieBelongsToNextCall();
doneUnderfilledRunParallelDoesNotStealLiveRunAgentChild();
for (const tool of ['run_agent', 'run_parallel'])
  freshReconnectShowsRunningChild(tool);
console.log('runAgentSubagentTabParity: all tests passed');
process.exit(0);
