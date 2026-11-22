// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Sub-agent tabs of repeated and nested run_parallel calls (chat page,
// media/main.js):
//
//  - every sub-agent gets its own tab when it starts, however many
//    calls its agent already made and however deep the fan-out nests,
//    also under a parent tab that has no DOM yet and under one that
//    has already closed (the nearest open ancestor then takes the tab);
//  - a tab closes when its own sub-agent finishes (`subagentDone`) and
//    only then: a finished ancestor leaves the tabs of the sub-agents
//    still running below it, and folding or unfolding a fan-out panel,
//    live or in adjacent history, opens and closes nothing;
//  - a sub-agent announced as finished gets no tab.

'use strict';

const assert = require('assert');
const {makePage, send, readyTabId, click} = require('./simplify2_harness');
const {
  announceFinished,
  subagentTabEls,
  togglePanel,
  openSubTabIds,
  tabEl,
  spawnSubagent,
} = require('./subagent_tab_helpers');

function runParallelPanels(win) {
  return Array.from(win.document.querySelectorAll('#output .tc-run-parallel'));
}

function switchToTabEl(win, tabId) {
  const el = tabEl(win, tabId);
  assert.ok(el, 'tab ' + tabId + ' must be rendered in the tab bar');
  click(win, el);
}

function runParallelCall(win, posted, agentTabId, taskIds, descPrefix) {
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: agentTabId,
    extras: {tasks: JSON.stringify(taskIds)},
  });
  // A tab must open for EVERY sub-agent, irrespective of how many
  // run_parallel calls the agent already made.
  return taskIds.map((taskId, i) =>
    spawnSubagent({win, posted}, agentTabId, taskId, descPrefix + (i + 1), i),
  );
}

function bootRunningRoot() {
  const {win, posted} = makePage();
  const rootId = readyTabId(posted);
  send(win, {
    type: 'status',
    running: true,
    tabId: rootId,
    startTs: Date.now(),
  });
  return {win, posted, rootId};
}

function testThreeSequentialRunParallelCallsOpenTabs() {
  const {win, posted, rootId} = bootRunningRoot();

  for (let k = 1; k <= 3; k++) {
    const taskIds = ['call' + k + '-sub-1', 'call' + k + '-sub-2'];
    const subTabIds = runParallelCall(
      win,
      posted,
      rootId,
      taskIds,
      'c' + k + ' sub ',
    );
    assert.strictEqual(
      subagentTabEls(win).length,
      2,
      'run_parallel call #' +
        k +
        ' must open one tab per sub-agent (got ' +
        subagentTabEls(win).length +
        ')',
    );
    const panels = runParallelPanels(win);
    assert.strictEqual(panels.length, k, 'call #' + k + ' renders panel #' + k);

    send(win, {type: 'subagentDone', tab_id: subTabIds[0]});
    assert.deepStrictEqual(
      openSubTabIds(win).sort(),
      [subTabIds[1]],
      'subagentDone for ' +
        taskIds[0] +
        ' must close ONLY the corresponding tab',
    );
    send(win, {type: 'subagentDone', tab_id: subTabIds[1]});
    assert.strictEqual(
      subagentTabEls(win).length,
      0,
      'all sub-agent tabs of call #' + k + ' must be closed when done',
    );

    send(win, {
      type: 'tool_result',
      tabId: rootId,
      content: 'call ' + k + ' done',
    });
    send(win, {type: 'thinking_start', tabId: rootId});
    send(win, {type: 'thinking_delta', tabId: rootId, text: 'next'});
    send(win, {type: 'thinking_end', tabId: rootId});
  }
  assert.strictEqual(
    runParallelPanels(win).length,
    3,
    'three run_parallel calls render three panels',
  );
  win.close();
  console.log('  ok - three sequential run_parallel calls each open tabs');
}

function testThreeLevelNestedRunParallel() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(
    win,
    posted,
    rootId,
    ['l1-a', 'l1-b', 'l1-c'],
    'L1 sub ',
  );
  assert.strictEqual(subagentTabEls(win).length, 3, 'level-1 tabs open');

  send(win, {type: 'thinking_start', tabId: l1[0]});
  send(win, {type: 'thinking_delta', tabId: l1[0], text: 'fanning out'});
  send(win, {type: 'thinking_end', tabId: l1[0]});
  const l2 = runParallelCall(
    win,
    posted,
    l1[0],
    ['l2-a', 'l2-b', 'l2-c'],
    'L2 sub ',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    6,
    "a sub-agent's run_parallel must open tabs for ITS sub-agents too",
  );

  send(win, {type: 'thinking_start', tabId: l2[0]});
  send(win, {type: 'thinking_delta', tabId: l2[0], text: 'fanning out'});
  send(win, {type: 'thinking_end', tabId: l2[0]});
  const l3 = runParallelCall(
    win,
    posted,
    l2[0],
    ['l3-a', 'l3-b', 'l3-c'],
    'L3 sub ',
  );
  assert.strictEqual(
    subagentTabEls(win).length,
    9,
    'a 3rd-level run_parallel must open tabs for its sub-agents too',
  );

  send(win, {type: 'subagentDone', tab_id: l3[0]});
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    [...l1, ...l2, l3[1], l3[2]].sort(),
    "finishing l3-a must close ONLY l3-a's tab",
  );
  send(win, {type: 'subagentDone', tab_id: l3[1]});
  send(win, {type: 'subagentDone', tab_id: l3[2]});
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    [...l1, ...l2].sort(),
    'all level-3 tabs closed, levels 1–2 untouched',
  );

  send(win, {type: 'tool_result', tabId: l2[0], content: 'l3 done'});
  send(win, {type: 'result', tabId: l2[0], summary: 'done', success: true});
  send(win, {type: 'subagentDone', tab_id: l2[0]});
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    [...l1, l2[1], l2[2]].sort(),
    "finishing l2-a must close ONLY l2-a's tab",
  );
  send(win, {type: 'subagentDone', tab_id: l2[1]});
  send(win, {type: 'subagentDone', tab_id: l2[2]});
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    [...l1].sort(),
    'all level-2 tabs closed, level 1 untouched',
  );

  send(win, {type: 'tool_result', tabId: l1[0], content: 'l2 done'});
  send(win, {type: 'result', tabId: l1[0], summary: 'done', success: true});
  send(win, {type: 'subagentDone', tab_id: l1[0]});
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    [l1[1], l1[2]].sort(),
    "finishing l1-a must close ONLY l1-a's tab",
  );
  send(win, {type: 'subagentDone', tab_id: l1[1]});
  send(win, {type: 'subagentDone', tab_id: l1[2]});
  assert.strictEqual(subagentTabEls(win).length, 0, 'all fan-outs closed');
  win.close();
  console.log('  ok - 3-level nested run_parallel opens/closes per level');
}

function testNestedPanelCollapseExpand() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(win, posted, rootId, ['l1-a', 'l1-b'], 'L1 ');
  const l2 = runParallelCall(win, posted, l1[0], ['l2-a', 'l2-b'], 'L2 ');
  const running = [...l1, ...l2].sort();
  assert.strictEqual(subagentTabEls(win).length, 4, 'both levels open');

  switchToTabEl(win, l1[0]);
  const nestedPanels = runParallelPanels(win);
  assert.strictEqual(
    nestedPanels.length,
    1,
    'the sub-agent tab shows its own run_parallel panel',
  );
  const nestedPanel = nestedPanels[0];
  assert.ok(
    nestedPanel.classList.contains('collapsed'),
    'nested panel starts collapsed',
  );

  const before = posted.length;
  togglePanel(win, nestedPanel);
  assert.ok(
    !nestedPanel.classList.contains('collapsed'),
    'clicking the nested header expands the nested panel',
  );
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    running,
    'expansion retains the same tabs',
  );
  assert.ok(
    !posted.slice(before).some(m => m.type === 'resumeSession'),
    'expansion must not resubscribe tabs that remained open',
  );

  togglePanel(win, nestedPanel);
  assert.ok(
    nestedPanel.classList.contains('collapsed'),
    'clicking the nested header collapses the nested panel',
  );
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    running,
    'collapsing the nested panel must keep every running descendant visible',
  );

  switchToTabEl(win, rootId);
  const rootPanel = runParallelPanels(win)[0];
  assert.ok(rootPanel, 'root panel present in the root chat DOM');
  togglePanel(win, rootPanel);
  assert.ok(!rootPanel.classList.contains('collapsed'), 'root expanded');
  togglePanel(win, rootPanel);
  assert.ok(rootPanel.classList.contains('collapsed'), 'root collapsed');
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    running,
    'collapsing the root must preserve its running children and descendants',
  );
  for (const id of [...l2, ...l1])
    send(win, {type: 'subagentDone', tab_id: id});
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'completion closes every child',
  );
  win.close();
  console.log('  ok - nested collapse preserves running tabs until completion');
}

function testSubagentMakesMultipleRunParallelCalls() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(win, posted, rootId, ['l1-a'], 'L1 ');

  const c1 = runParallelCall(win, posted, l1[0], ['n1-a', 'n1-b'], 'N1 ');
  send(win, {type: 'subagentDone', tab_id: c1[0]});
  send(win, {type: 'subagentDone', tab_id: c1[1]});
  send(win, {type: 'tool_result', tabId: l1[0], content: 'call 1 done'});
  send(win, {type: 'thinking_start', tabId: l1[0]});
  send(win, {type: 'thinking_delta', tabId: l1[0], text: 'next call'});
  send(win, {type: 'thinking_end', tabId: l1[0]});
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    [...l1].sort(),
    'nested call #1 fan-out fully closed after both subagentDone',
  );

  const c2 = runParallelCall(win, posted, l1[0], ['n2-a', 'n2-b'], 'N2 ');
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    [...l1, ...c2].sort(),
    "the sub-agent's SECOND run_parallel call must open one tab per " +
      'sub-agent, exactly like the first call',
  );

  send(win, {type: 'subagentDone', tab_id: c2[0]});
  send(win, {type: 'subagentDone', tab_id: c2[1]});
  send(win, {type: 'tool_result', tabId: l1[0], content: 'call 2 done'});
  send(win, {type: 'thinking_start', tabId: l1[0]});
  send(win, {type: 'thinking_delta', tabId: l1[0], text: 'next call'});
  send(win, {type: 'thinking_end', tabId: l1[0]});
  const c3 = runParallelCall(win, posted, l1[0], ['n3-a', 'n3-b'], 'N3 ');
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    [...l1, ...c3].sort(),
    "the sub-agent's THIRD run_parallel call must open one tab per " +
      'sub-agent',
  );

  switchToTabEl(win, l1[0]);
  const panels = runParallelPanels(win);
  assert.strictEqual(panels.length, 3, 'three nested panels rendered');
  win.close();
  console.log("  ok - a sub-agent's repeated run_parallel calls open tabs");
}

function testAdjacentHistoryRunParallelPanelIsInert() {
  const {win, posted, rootId} = bootRunningRoot();
  send(win, {type: 'status', running: false, tabId: rootId});

  send(win, {
    type: 'adjacent_task_events',
    tabId: rootId,
    direction: 'prev',
    task: 'Older parallel task',
    task_id: '41',
    events: [
      {
        type: 'tool_call',
        name: 'run_parallel',
        tabId: 'ghost-tab-from-old-session',
        extras: {tasks: JSON.stringify(['old sub 1'])},
      },
      {type: 'tool_result', content: 'done'},
      {type: 'result', summary: 'done', success: true},
    ],
  });
  const adj = win.document.querySelector('#output .adjacent-task');
  assert.ok(adj, 'adjacent task container must render');
  const panel = adj.querySelector('.tc-run-parallel');
  assert.ok(panel, 'the history block renders its run_parallel panel');
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'rendering a history run_parallel panel must not open tabs',
  );

  const before = posted.length;
  togglePanel(win, panel);
  assert.ok(!panel.classList.contains('collapsed'), 'panel expanded');
  togglePanel(win, panel);
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'toggling a history run_parallel panel must not open tabs',
  );
  assert.ok(
    !posted.slice(before).some(m => m.type === 'resumeSession'),
    'toggling a history run_parallel panel must not resume anything',
  );
  win.close();
  console.log('  ok - adjacent-history run_parallel panel is inert');
}

function testSpawnUnderFragmentlessParentStillOpensTab() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(win, posted, rootId, ['l1-a'], 'L1 ');
  // A sub-agent spawned under a DOM-less parent tab must still open a
  // tab (resumeSession posted).
  const gId = spawnSubagent({win, posted}, l1[0], 'g-task', 'grandchild', 0);
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    [l1[0], gId].sort(),
    'the grandchild tab must be open next to its parent sub-agent tab',
  );

  win.close();
  console.log('  ok - spawn under a DOM-less parent still opens a tab');
}

function testAdjacentHistoryPanelDoesNotStealLiveFanout() {
  const {win, posted, rootId} = bootRunningRoot();

  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: rootId,
    extras: {tasks: JSON.stringify(['live sub'])},
  });
  const livePanel = runParallelPanels(win)[0];
  assert.ok(livePanel, 'live panel rendered');

  send(win, {
    type: 'adjacent_task_events',
    tabId: rootId,
    direction: 'next',
    task: 'Newer old task',
    task_id: '77',
    events: [
      {
        type: 'tool_call',
        name: 'run_parallel',
        tabId: 'stale-session-tab',
        extras: {tasks: '"not-a-list"'},
      },
      {type: 'tool_result', content: 'done'},
      {type: 'result', summary: 'done', success: true},
    ],
  });
  const adjPanel = win.document.querySelector(
    '#output .adjacent-task .tc-run-parallel',
  );
  assert.ok(adjPanel, 'adjacent history panel rendered');

  const before = posted.length;
  send(win, {
    type: 'new_tab',
    task_id: 'live-sub-task',
    parent_tab_id: rootId,
    taskId: '',
  });
  const resume = posted
    .slice(before)
    .find(m => m.type === 'resumeSession' && m.taskId === 'live-sub-task');
  assert.ok(
    resume,
    'BUG: the live sub-agent must open its tab — the collapsed ' +
      'adjacent-history run_parallel panel must not defer/own it',
  );
  assert.strictEqual(subagentTabEls(win).length, 1, 'live sub tab open');

  togglePanel(win, livePanel);
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'toggling the live panel must preserve its running sub-agent tab',
  );
  win.close();
  console.log('  ok - adjacent history panel cannot steal a live fan-out');
}

// A grandchild spawned before its parent sub-agent's own fan-out calls
// (their `tasks` argument unparsable or missing) keeps its tab through
// those calls, the parent's result and a click on the newest panel.
function testGrandchildKeepsTabThroughParentsFanoutCalls() {
  const {win, posted, rootId} = bootRunningRoot();

  const l1 = runParallelCall(win, posted, rootId, ['l1-a'], 'L1 ');
  send(win, {
    type: 'new_tab',
    task_id: 'g-task',
    parent_tab_id: l1[0],
    taskId: '',
  });
  const g = posted.find(
    m => m.type === 'resumeSession' && m.taskId === 'g-task',
  );
  assert.ok(g, 'grandchild tab opened');

  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: l1[0],
    extras: {tasks: 'not json {'},
  });
  send(win, {type: 'tool_result', tabId: l1[0], content: 'c1 done'});
  send(win, {type: 'tool_call', name: 'run_parallel', tabId: l1[0]});
  send(win, {type: 'tool_call', name: 'Bash', tabId: l1[0]});
  send(win, {type: 'result', tabId: l1[0], summary: 'done', success: true});
  assert.ok(
    subagentTabEls(win).some(el => el.dataset.tabId === g.tabId),
    "the grandchild must keep its tab through its parent's calls and result",
  );

  switchToTabEl(win, l1[0]);
  const nested = runParallelPanels(win);
  assert.strictEqual(nested.length, 2, 'two nested panels rendered');
  assert.ok(
    nested[0].classList.contains('collapsed'),
    'finished call #1 stays collapsed',
  );
  togglePanel(win, nested[1]);
  assert.ok(
    subagentTabEls(win).some(el => el.dataset.tabId === g.tabId),
    'toggling the newest panel must preserve the running grandchild',
  );
  win.close();
  console.log("  ok - a grandchild keeps its tab through its parent's calls");
}

// R -> A -> B -> C, all live. A finishes first (its completion helper
// pattern): A's tab closes, B survives under R, C stays under B.
function testFinishedAncestorKeepsLiveDescendants() {
  const {win, posted, rootId} = bootRunningRoot();
  const [a] = runParallelCall(win, posted, rootId, ['a'], 'A ');
  const [b] = runParallelCall(win, posted, a, ['b'], 'B ');
  const [c] = runParallelCall(win, posted, b, ['c'], 'C ');

  send(win, {type: 'subagentDone', tab_id: a});
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    [b, c].sort(),
    'A closed, B and C kept',
  );

  // B finished: closes alone; C (never reparented) stays open.
  send(win, {type: 'subagentDone', tab_id: b});
  assert.deepStrictEqual(openSubTabIds(win).sort(), [c], 'only C remains');
  send(win, {type: 'subagentDone', tab_id: c});
  assert.strictEqual(subagentTabEls(win).length, 0, 'all closed');
  win.close();
  console.log('  ok - finished ancestor keeps live descendants');
}

// A helper announced after its parent sub-agent finished (parent tab
// already closed) opens under the nearest open ancestor; a finished
// sub-agent of the gone parent, announced as such, opens nothing.
function testLateSpawnUnderFinishedParentOpensUnderAncestor() {
  const {win, posted, rootId} = bootRunningRoot();
  const [a] = runParallelCall(win, posted, rootId, ['a'], 'A ');
  send(win, {type: 'subagentDone', tab_id: a});
  assert.strictEqual(subagentTabEls(win).length, 0, 'A closed');

  const before = posted.length;
  send(win, {type: 'new_tab', task_id: 'helper', parent_tab_id: a, taskId: ''});
  const resume = posted
    .slice(before)
    .find(m => m.type === 'resumeSession' && m.taskId === 'helper');
  assert.ok(resume, 'the late helper must still get a tab');
  assert.strictEqual(resume.tabId, a + '__sub_helper', 'stable daemon id kept');
  assert.deepStrictEqual(openSubTabIds(win).sort(), [resume.tabId]);

  announceFinished(win, ['old'], taskId => a + '__sub_' + taskId);
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    [resume.tabId],
    'a finished sub-agent opens no tab',
  );

  send(win, {type: 'subagentDone', tab_id: resume.tabId});
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'helper closed on completion',
  );
  win.close();
  console.log('  ok - late spawn under finished parent opens under ancestor');
}

function main() {
  const tests = [
    testThreeSequentialRunParallelCallsOpenTabs,
    testThreeLevelNestedRunParallel,
    testNestedPanelCollapseExpand,
    testSubagentMakesMultipleRunParallelCalls,
    testAdjacentHistoryRunParallelPanelIsInert,
    testSpawnUnderFragmentlessParentStillOpensTab,
    testAdjacentHistoryPanelDoesNotStealLiveFanout,
    testGrandchildKeepsTabThroughParentsFanoutCalls,
    testFinishedAncestorKeepsLiveDescendants,
    testLateSpawnUnderFinishedParentOpensUnderAncestor,
  ];
  for (const t of tests) t();
  console.log('runParallelMultiCallNested.test.js: all tests passed');
}

try {
  main();
} catch (err) {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
}
