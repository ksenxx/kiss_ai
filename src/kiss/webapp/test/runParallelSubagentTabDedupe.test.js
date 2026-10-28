// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// One sub-agent => at most ONE tab on a client.
//
// The daemon addresses a sub-agent's tab by three different ids over
// the life of a fan-out:
//
//   * a synthetic live id of the form "task-<parentTaskId>__sub_<idx>"
//     (the form the in-agent fan-out minted before run_parallel moved
//     to agent_dispatch.make_run_parallel_tool; kept here as "an id
//     that differs from the tab's current one"),
//   * the deterministic replay id minted by
//     AgentServer._open_persisted_subagent_tabs
//     ("<parentTabId>__sub_<subTaskId>"), and
//   * whatever tab id the page itself asked to resume the sub-agent
//     on (media/main.js mints a fresh one when a collapsed
//     run_parallel panel is expanded again).
//
// All three name the SAME sub-agent -- the sub-agent's task id is the
// only stable identity.  A client that keys tabs on the tab id alone
// therefore stacks a second (third, ...) tab for one sub-agent.  These
// tests pin the invariant: whatever id the daemon uses, a sub-agent
// task never occupies more than one open tab, and the run_parallel
// panel's collapsed state still decides whether that one tab exists.
//
// The page runs on the REAL webapp transport (media/transport.js
// driving a fake WebSocket), so the frames the daemon would see are
// exactly the ones the assertions inspect.

/* global require, console */

'use strict';

const assert = require('assert');
const {send} = require('./simplify2_harness');
const {makeWebappClient} = require('./transport_page_helpers');
const {
  subagentTabEls,
  togglePanel,
  clickClose,
  finishFanOut,
  announceFinished,
  openSubTabIds,
  bootParallelRun,
} = require('./subagent_tab_helpers');

// ---------------------------------------------------------------------------
// scenario helpers
// ---------------------------------------------------------------------------

// Every scenario boots a transport client and runs a live fan-out on it:
// `bootParallelRun(n, client, {replay: true})` (the daemon always follows
// openSubagentTab with the sub-agent's transcript, which is what binds
// the tab to its task id); `client.drain()` is every frame posted so far.

function deterministicId(parentId) {
  return taskId => parentId + '__sub_' + taskId;
}

function assertOneTabPerSubagent(win, taskIds, where) {
  assert.strictEqual(
    subagentTabEls(win).length,
    taskIds.length,
    'DUPLICATE SUB-AGENT TABS (' +
      where +
      '): ' +
      taskIds.length +
      ' sub-agents but ' +
      subagentTabEls(win).length +
      ' open sub-agent tabs: ' +
      JSON.stringify(openSubTabIds(win)),
  );
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

// The parent tab is replayed (history click / reconnect) while the
// fan-out's tabs are open.  _open_persisted_subagent_tabs addresses
// each sub-agent by its deterministic replay id — which, since the
// deterministic-id fix (subagentTabIdFor), is the SAME id the client
// minted live: the announcement must land on the existing tab with no
// retag and no closeTab.
function testPersistedReplayIdsDoNotDuplicate() {
  const client = makeWebappClient();
  const {
    win,
    parentId,
    taskIds,
    subTabIds: liveTabIds,
  } = bootParallelRun(2, client, {replay: true});
  const closesBefore = client.drain().filter(m => m.type === 'closeTab').length;

  taskIds.forEach((taskId, i) => {
    const replayId = parentId + '__sub_' + taskId;
    assert.strictEqual(
      replayId,
      liveTabIds[i],
      'live and replay ids must converge on the deterministic id',
    );
    send(win, {
      type: 'openSubagentTab',
      tab_id: replayId,
      parent_tab_id: parentId,
      description: 'sub ' + (i + 1),
      task_id: taskId,
      taskIndex: i,
      isSubagentTab: true,
      isDone: false,
    });
    send(win, {
      type: 'task_events',
      tabId: replayId,
      task: 'sub ' + (i + 1),
      task_id: taskId,
      events: [],
    });
  });

  assertOneTabPerSubagent(win, taskIds, 'persisted replay ids');
  const closesAfter = client.drain().filter(m => m.type === 'closeTab').length;
  assert.strictEqual(
    closesAfter,
    closesBefore,
    'a converged replay announcement must not close any tab id — a ' +
      'closeTab here unsubscribes the shared id for every client',
  );
  win.close();
  console.log('  ok - persisted replay ids reuse one tab');
}

// The fan-out finishes (tabs close, the panel folds), the user expands
// it (the page re-mints the deterministic tab ids and resumes) and
// only then does the daemon's replay burst for the finished sub-agents
// arrive under those same deterministic ids.
function testReplayAfterExpandDoesNotDuplicate() {
  const {
    win,
    panel,
    parentId,
    taskIds,
    subTabIds: liveTabIds,
  } = bootParallelRun(2, makeWebappClient(), {replay: true});

  finishFanOut(win, liveTabIds);
  assert.strictEqual(subagentTabEls(win).length, 0, 'done closes tabs');
  assert.ok(panel.classList.contains('collapsed'), 'finished panel folds');
  togglePanel(win, panel);
  assert.strictEqual(subagentTabEls(win).length, 2, 'expand reopens tabs');

  const reopenedIds = openSubTabIds(win);
  taskIds.forEach((taskId, i) => {
    const replayId = parentId + '__sub_' + taskId;
    assert.ok(
      reopenedIds.includes(replayId),
      'the reopened tab must reuse the deterministic id, so the ' +
        'replay burst below lands on it instead of forcing a retag',
    );
    send(win, {
      type: 'openSubagentTab',
      tab_id: replayId,
      parent_tab_id: parentId,
      description: 'sub ' + (i + 1),
      task_id: taskId,
      taskIndex: i,
      isSubagentTab: true,
      isDone: true,
    });
    send(win, {
      type: 'task_events',
      tabId: replayId,
      task: 'sub ' + (i + 1),
      task_id: taskId,
      events: [],
    });
  });

  assertOneTabPerSubagent(win, taskIds, 'replay after expand');
  assert.ok(
    !panel.classList.contains('collapsed'),
    'the panel must stay uncollapsed while its sub-agent tabs are open',
  );
  win.close();
  console.log('  ok - replay after expand reuses one tab');
}

// A sub-agent spawned while the panel is collapsed is remembered, and
// a later duplicate `new_tab` for a sub-agent that already has a tab
// must not open a second one.
function testRepeatedNewTabDoesNotDuplicate() {
  const {win, parentId, taskIds} = bootParallelRun(2, makeWebappClient(), {
    replay: true,
  });

  for (const taskId of taskIds) {
    send(win, {
      type: 'new_tab',
      task_id: taskId,
      parent_tab_id: parentId,
      taskId: '',
    });
  }
  assertOneTabPerSubagent(win, taskIds, 'repeated new_tab');
  win.close();
  console.log('  ok - a repeated new_tab reuses one tab');
}

// Expanding a panel that learned about the same sub-agent twice (once
// as a live tab, once while collapsed) must still open one tab each.
function testExpandAfterMixedRegistrationDoesNotDuplicate() {
  const {
    win,
    panel,
    parentId,
    taskIds,
    subTabIds: liveTabIds,
  } = bootParallelRun(2, makeWebappClient(), {replay: true});

  togglePanel(win, panel);
  assert.ok(panel.classList.contains('collapsed'), 'panel collapses');
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'collapsing keeps the RUNNING sub-agent tabs open',
  );
  // While collapsed the daemon keeps addressing the running sub-agents,
  // under both id schemes.
  taskIds.forEach((taskId, i) => {
    send(win, {
      type: 'new_tab',
      task_id: taskId,
      parent_tab_id: parentId,
      taskId: '',
    });
    send(win, {
      type: 'openSubagentTab',
      tab_id: parentId + '__sub_' + taskId,
      parent_tab_id: parentId,
      description: 'sub ' + (i + 1),
      task_id: taskId,
      taskIndex: i,
      isSubagentTab: true,
    });
  });
  assertOneTabPerSubagent(win, taskIds, 'mixed regs, collapsed');
  togglePanel(win, panel);
  assertOneTabPerSubagent(win, taskIds, 'expand after mixed regs');

  // Once finished, the sub-agents are history: the collapsed panel keeps
  // them closed however the daemon re-announces them.
  togglePanel(win, panel);
  finishFanOut(win, liveTabIds);
  assert.strictEqual(subagentTabEls(win).length, 0, 'done closes tabs');
  announceFinished(win, parentId, taskIds, deterministicId(parentId));
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'nothing may open a finished sub-agent tab while the run_parallel ' +
      'panel is collapsed',
  );

  togglePanel(win, panel);
  assertOneTabPerSubagent(win, taskIds, 'expand after finish');
  win.close();
  console.log('  ok - mixed registrations expand to one tab');
}

// The daemon may re-announce a sub-agent tab under a THIRD id (e.g. the
// parent tab is reopened in a new window and replays) while the panel
// is collapsed: a running sub-agent moves onto that id (one tab each),
// a finished one stays closed, and expanding yields one each.
function testCollapsedStaysClosedAcrossIdSchemes() {
  const {win, panel, parentId, taskIds} = bootParallelRun(
    2,
    makeWebappClient(),
    {replay: true},
  );
  const thirdId = (taskId, i) => 'task-parent__sub_' + i;

  togglePanel(win, panel);
  taskIds.forEach((taskId, i) => {
    send(win, {
      type: 'openSubagentTab',
      tab_id: thirdId(taskId, i),
      parent_tab_id: parentId,
      description: 'sub ' + (i + 1),
      task_id: taskId,
      taskIndex: i,
      isSubagentTab: true,
    });
  });
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    taskIds.map(thirdId).sort(),
    'a running sub-agent follows the daemon onto its new id: one tab each',
  );

  finishFanOut(win, taskIds.map(thirdId));
  assert.strictEqual(subagentTabEls(win).length, 0, 'done closes tabs');
  announceFinished(win, parentId, taskIds, thirdId);
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'a collapsed run_parallel panel must keep every finished sub-agent ' +
      'tab closed',
  );

  togglePanel(win, panel);
  assertOneTabPerSubagent(win, taskIds, 'expand after 3rd id');
  win.close();
  console.log('  ok - collapsed panel ignores every id form');
}

// The daemon re-delivering a spawn (`new_tab`, always a live child) must
// land on the child's one tab, and a finished sub-agent tab the user
// closed by hand must stay closed when the daemon re-announces it (nor
// may a sibling be duplicated).  The collapse/expand contract itself is
// pinned by runParallelPanelTabsSync.test.js.
function testNewTabDoesNotRevivHandClosedSubagent() {
  const {
    win,
    panel,
    parentId,
    taskIds,
    subTabIds: liveTabIds,
  } = bootParallelRun(2, makeWebappClient(), {replay: true});

  clickClose(win, liveTabIds[0]);
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'closing a RUNNING sub-agent tab by hand closes that tab alone',
  );
  send(win, {
    type: 'new_tab',
    task_id: taskIds[0],
    parent_tab_id: parentId,
    taskId: '',
  });
  assertOneTabPerSubagent(win, taskIds, 're-delivered new_tab');

  finishFanOut(win, liveTabIds);
  togglePanel(win, panel);
  assertOneTabPerSubagent(win, taskIds, 'reopened history');
  announceFinished(win, parentId, taskIds, deterministicId(parentId));
  clickClose(win, liveTabIds[0]);
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'closing one of two finished sub-agent tabs must leave the sibling open',
  );

  announceFinished(win, parentId, taskIds, deterministicId(parentId));
  assert.strictEqual(
    subagentTabEls(win).length,
    1,
    'RESURRECTED SUB-AGENT TAB: a re-announcement reopened the finished sub-agent tab the ' +
      'user closed by hand: ' +
      JSON.stringify(openSubTabIds(win)),
  );
  assert.ok(
    !panel.classList.contains('collapsed'),
    'the panel must stay uncollapsed while a sibling tab is open',
  );

  // Collapsing and expanding is the way back: it reopens the whole
  // fan-out, hand-closed sub-agents included.
  togglePanel(win, panel);
  togglePanel(win, panel);
  assertOneTabPerSubagent(win, taskIds, 'expand after hand close');
  win.close();
  console.log('  ok - re-announce cannot revive a closed sub tab');
}

// Two run_parallel calls in one task: a sub-agent spawned late belongs
// to the call that requested it. A live spawn always gets its tab, so
// the ownership shows once the FIRST fan-out finishes: expanding the
// first panel reopens the late child, the second panel never touches it.
function testLateSpawnHonoursItsOwnPanel() {
  // The first call fanned out to three tasks but only two have started.
  const client = makeWebappClient();
  const {
    win,
    panel,
    parentId,
    subTabIds: liveTabIds,
  } = bootParallelRun(2, client, {replay: true, declared: 3});
  const all = client.drain;

  togglePanel(win, panel);
  assert.ok(panel.classList.contains('collapsed'), 'first fan-out folds');
  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'collapsing the first fan-out keeps its running sub-agent tabs open',
  );

  // A second run_parallel call renders a second, uncollapsed panel.
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(['sub 3'])},
  });
  const panels = Array.from(
    win.document.querySelectorAll('#output .ev.tc.tc-run-parallel'),
  );
  assert.strictEqual(panels.length, 2, 'the task must show two fan-outs');
  const second = panels[1];
  assert.ok(
    !second.classList.contains('collapsed'),
    'the second fan-out starts uncollapsed',
  );

  let before = all().length;
  // The first fan-out's third sub-agent starts only now.
  send(win, {
    type: 'new_tab',
    task_id: 'sub-task-late',
    parent_tab_id: parentId,
    taskId: '',
  });
  const lateId = parentId + '__sub_sub-task-late';
  assert.deepStrictEqual(
    openSubTabIds(win).sort(),
    [...liveTabIds, lateId].sort(),
    'INVARIANT VIOLATED: a live sub-agent spawned into a fan-out whose panel is ' +
      'collapsed must still get its tab: ' +
      JSON.stringify(openSubTabIds(win)),
  );
  assert.ok(
    all()
      .slice(before)
      .some(m => m.type === 'resumeSession' && m.taskId === 'sub-task-late'),
    'the late sub-agent must be resumed right away',
  );
  assert.ok(
    panel.classList.contains('collapsed'),
    'the late spawn does not unfold the first panel the user collapsed',
  );

  // The second fan-out's own sub-agent gets a tab too.
  send(win, {
    type: 'new_tab',
    task_id: 'sub-task-3',
    parent_tab_id: parentId,
    taskId: '',
  });
  const thirdId = parentId + '__sub_sub-task-3';
  assert.strictEqual(
    subagentTabEls(win).length,
    4,
    'the uncollapsed second fan-out must open its own sub-agent tab',
  );

  // The first fan-out finishes: only ITS three tabs close, and expanding
  // its (already collapsed) panel resumes exactly those three.
  finishFanOut(win, [...liveTabIds, lateId]);
  assert.deepStrictEqual(
    openSubTabIds(win),
    [thirdId],
    "finishing the first fan-out must leave the second fan-out's " +
      'running child open',
  );
  assert.ok(
    !second.classList.contains('collapsed'),
    'the second fan-out stays uncollapsed: it owns a running child',
  );
  before = all().length;
  togglePanel(win, panel);
  assert.deepStrictEqual(
    all()
      .slice(before)
      .filter(m => m.type === 'resumeSession')
      .map(m => m.taskId)
      .sort(),
    ['sub-task-1', 'sub-task-2', 'sub-task-late'],
    'expanding the first fan-out must reopen its late child with its ' +
      'own two, and nothing of the second fan-out',
  );
  assert.strictEqual(subagentTabEls(win).length, 4, 'one tab per sub-agent');
  win.close();
  console.log('  ok - a late spawn honours its own panel');
}

// Renaming a sub-agent's tab must tell the host to release the retired
// id (the daemon opened the new id as an extra viewer of the same
// sub-agent, so the old one would otherwise keep host state alive).
function testRetagReleasesTheOldTabId() {
  const client = makeWebappClient();
  const {
    win,
    parentId,
    taskIds,
    subTabIds: liveTabIds,
  } = bootParallelRun(1, client, {replay: true});
  const all = client.drain;

  const before = all().length;
  // A daemon id that DIFFERS from the tab's current id.  Since the
  // deterministic-id fix (subagentTabIdFor) the client already sits on
  // "<parentId>__sub_<taskId>", so the rename is exercised with the
  // other id form: the synthetic "task-<parentTaskId>__sub_<idx>" the
  // in-agent fan-out minted before run_parallel moved to
  // agent_dispatch.make_run_parallel_tool (any differing id works).
  const replayId = 'task-parent-task__sub_0';
  assert.notStrictEqual(replayId, liveTabIds[0], 'sanity: ids must differ');
  send(win, {
    type: 'openSubagentTab',
    tab_id: replayId,
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: taskIds[0],
    taskIndex: 0,
    isSubagentTab: true,
  });
  assert.deepStrictEqual(
    openSubTabIds(win),
    [replayId],
    'the sub-agent must move onto the id the daemon now uses',
  );
  assert.ok(
    all()
      .slice(before)
      .some(m => m.type === 'closeTab' && m.tabId === liveTabIds[0]),
    'STALE HOST STATE: the host was never told to release the retired sub-agent tab ' +
      'id ' +
      liveTabIds[0],
  );
  assert.ok(
    !all()
      .slice(before)
      .some(m => m.type === 'closeTab' && m.tabId === replayId),
    'the surviving sub-agent tab must not be closed',
  );
  win.close();
  console.log('  ok - a retag releases the retired tab id');
}

// Sub-agent broadcasts (new_tab with taskId '', openSubagentTab) reach
// every connected client.  A client that does not own the announced
// parent tab (another chat, another window) must neither open a tab for
// the child nor resume its session, or every window shows the phantoms
// of every other window's fan-outs.
function testUnknownParentSpawnsNoPhantomTab() {
  const client = makeWebappClient();
  const {win} = bootParallelRun(1, client, {replay: true});
  const all = client.drain;
  const foreignParent = 'tab-of-another-chat';
  const tabIds = () => win._testApi.openTabs().map(t => t.id);
  assert.ok(
    !tabIds().includes(foreignParent),
    'sanity: the foreign parent is not a tab of this client',
  );
  const tabsBefore = tabIds();
  const before = all().length;

  send(win, {
    type: 'new_tab',
    task_id: 'foreign-task',
    parent_tab_id: foreignParent,
    taskId: '',
  });
  assert.ok(
    !all()
      .slice(before)
      .some(m => m.type === 'resumeSession' && m.taskId === 'foreign-task'),
    'PHANTOM RESUME: new_tab for a parent this client does not own must not resume ' +
      'the foreign sub-agent',
  );
  for (const isDone of [false, true]) {
    send(win, {
      type: 'openSubagentTab',
      tab_id: foreignParent + '__sub_foreign-task',
      parent_tab_id: foreignParent,
      description: 'foreign sub',
      task_id: 'foreign-task',
      taskIndex: 0,
      isSubagentTab: true,
      isDone,
    });
  }
  // The tab strip shows only the active group, so a foreign tab would
  // never appear there: the registry is the only place it could hide.
  assert.deepStrictEqual(
    tabIds(),
    tabsBefore,
    'PHANTOM TAB: a sub-agent of a parent this client does not own must not get ' +
      'a tab here',
  );
  win.close();
  console.log('  ok - an unknown parent spawns no phantom tab');
}

const SCENARIOS = [
  testUnknownParentSpawnsNoPhantomTab,
  testPersistedReplayIdsDoNotDuplicate,
  testReplayAfterExpandDoesNotDuplicate,
  testRepeatedNewTabDoesNotDuplicate,
  testExpandAfterMixedRegistrationDoesNotDuplicate,
  testCollapsedStaysClosedAcrossIdSchemes,
  testNewTabDoesNotRevivHandClosedSubagent,
  testLateSpawnHonoursItsOwnPanel,
  testRetagReleasesTheOldTabId,
];

function main() {
  for (const scenario of SCENARIOS) scenario();
  console.log('runParallelSubagentTabDedupe.test.js: all tests passed');
}

main();
