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
//     on (media/main.js mints the deterministic one when the spawn is
//     announced: subagentTabIdFor).
//
// All three name the SAME sub-agent -- the sub-agent's task id is the
// only stable identity.  A client that keys tabs on the tab id alone
// therefore stacks a second (third, ...) tab for one sub-agent.  These
// tests pin the invariant: whatever id the daemon uses, a sub-agent
// task never occupies more than one open tab.  An announcement under
// another id renames the tab (the retired id is released, the tabs and
// files below follow the rename), a repeated spawn opens no second
// one, a finished sub-agent has none, and neither has the sub-agent of
// a parent this page does not show.
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
  finishFanOut,
  announceFinished,
  openSubTabIds,
  bootParallelRun,
  spawnSubagent,
  contentTabFor,
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
  // Live fan-out events for the shared ids keep each sub-agent on its
  // one tab (where they render is subagentLiveEventsStream.test.js's).
  for (const tabId of liveTabIds) {
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      tabId,
      extras: {command: 'echo sub-live'},
    });
  }
  assertOneTabPerSubagent(win, taskIds, 'live events after the replay');
  win.close();
  console.log('  ok - persisted replay ids reuse one tab');
}

// A duplicate `new_tab` for a sub-agent that already has a tab must not
// open a second one, whatever id that tab has by now (the daemon
// announced the first sub-agent under another one).  (Mixed with the
// daemon's own announcement, it is
// testMixedRegistrationDoesNotDuplicate's.)
function testRepeatedNewTabDoesNotDuplicate() {
  const {win, parentId, taskIds} = bootParallelRun(2, makeWebappClient(), {
    replay: true,
  });
  send(win, {
    type: 'openSubagentTab',
    tab_id: 'task-parent__sub_0',
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: taskIds[0],
    taskIndex: 0,
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

// A page that hears of each running sub-agent again (its spawn
// re-delivered, then the daemon's own announcement of it) must still
// show one tab each, and none once the sub-agents have finished.
function testMixedRegistrationDoesNotDuplicate() {
  const {
    win,
    parentId,
    taskIds,
    subTabIds: liveTabIds,
  } = bootParallelRun(2, makeWebappClient(), {replay: true});

  // The daemon keeps addressing the running sub-agents, as spawns and
  // as open tabs.
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
  assertOneTabPerSubagent(win, taskIds, 'mixed regs');

  // Once finished, the sub-agents are history: they stay closed
  // however the daemon re-announces their end.
  finishFanOut(win, liveTabIds);
  assert.strictEqual(subagentTabEls(win).length, 0, 'done closes tabs');
  announceFinished(win, taskIds, deterministicId(parentId));
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'nothing may open a finished sub-agent tab',
  );
  win.close();
  console.log('  ok - mixed registrations keep one tab');
}

// The daemon may re-announce a sub-agent tab under a THIRD id (e.g. the
// parent tab is reopened in a new window and replays): a running
// sub-agent moves onto that id (one tab each), a finished one stays
// closed.
function testFinishedStaysClosedAcrossIdSchemes() {
  const {win, parentId, taskIds} = bootParallelRun(2, makeWebappClient(), {
    replay: true,
  });
  const thirdId = (taskId, i) => 'task-parent__sub_' + i;

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
  announceFinished(win, taskIds, thirdId);
  assert.strictEqual(
    subagentTabEls(win).length,
    0,
    'every finished sub-agent tab must stay closed',
  );
  win.close();
  console.log('  ok - a finished sub-agent stays closed in every id form');
}

// A sub-agent's tab is the daemon's view of the sub-agent: renaming it
// tells the daemon nothing (`closeTab` is for a chat's own tab).  The
// retired id stays the daemon's until the sub-agent ends, and the
// daemon names it as the parent of what the sub-agent starts next: the
// page opens that tab below the renamed one.
function testRetagTellsTheDaemonNothing() {
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
  assert.deepStrictEqual(
    all()
      .slice(before)
      .filter(m => m.type === 'closeTab'),
    [],
    "the rename of a sub-agent's tab is not reported to the daemon",
  );
  const nested = spawnSubagent(client, liveTabIds[0], 'nested-task', 'nested', 0);
  send(win, {type: 'status', running: false, tabId: nested});
  send(win, {type: 'subagentDone', tab_id: replayId});
  assert.deepStrictEqual(
    openSubTabIds(win),
    [],
    'a sub-agent started under the retired id hangs below the renamed tab',
  );
  win.close();
  console.log('  ok - a retag tells the daemon nothing');
}

// What hangs below a sub-agent's tab names that tab as its parent: the
// tab of a sub-agent it started (a fan-out of its own), a file it
// opened.  Renaming the sub-agent's tab must take them along: left on
// the retired id, each hangs off a tab that no longer exists and lands
// as a root tab of its own instead of in the chat's group.
function testRetagMovesTabsAndFilesBelow() {
  const client = makeWebappClient();
  const {win, parentId, taskIds, subTabIds} = bootParallelRun(1, client, {
    replay: true,
  });
  const sub = subTabIds[0];
  send(win, {type: 'status', running: true, tabId: sub, taskId: 2});
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: sub,
    extras: {tasks: JSON.stringify(['nested'])},
  });
  const nested = spawnSubagent(client, sub, 'nested-task', 'nested', 0);
  send(win, {
    type: 'fileContent',
    tabId: sub,
    path: '/w/notes.md',
    name: 'notes.md',
    content: 'notes',
  });
  assert.ok(contentTabFor(win, 'notes.md'), 'sanity: the file opened');
  const newId = 'task-parent-task__sub_0';
  send(win, {
    type: 'openSubagentTab',
    tab_id: newId,
    parent_tab_id: parentId,
    description: 'sub 1',
    task_id: taskIds[0],
    taskIndex: 0,
    isSubagentTab: true,
  });
  assert.deepStrictEqual(openSubTabIds(win), [newId, nested], 'sanity');
  const tabs = () => win._testApi.openTabs();
  assert.strictEqual(
    tabs().find(t => t.id === nested).rootId,
    parentId,
    "the nested child stays in the chat's group",
  );
  assert.strictEqual(
    contentTabFor(win, 'notes.md').rootId,
    parentId,
    "the file stays in the chat's group",
  );
  send(win, {type: 'subagentDone', tab_id: nested});
  assert.ok(!tabs().some(t => t.id === nested), 'the finished child closed');
  win.close();
  console.log('  ok - a retag moves the tabs and files below to the new id');
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
  send(win, {
    type: 'openSubagentTab',
    tab_id: foreignParent + '__sub_foreign-task',
    parent_tab_id: foreignParent,
    description: 'foreign sub',
    task_id: 'foreign-task',
    taskIndex: 0,
    isSubagentTab: true,
  });
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
  testRepeatedNewTabDoesNotDuplicate,
  testMixedRegistrationDoesNotDuplicate,
  testFinishedStaysClosedAcrossIdSchemes,
  testRetagTellsTheDaemonNothing,
  testRetagMovesTabsAndFilesBelow,
];

function main() {
  for (const scenario of SCENARIOS) scenario();
  console.log('runParallelSubagentTabDedupe.test.js: all tests passed');
}

main();
