// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A sub-agent's tab is open exactly while the sub-agent runs
// (media/main.js).  It opens when the daemon says the sub-agent started
// (`new_tab`, `openSubagentTab`) and closes when the daemon says it
// finished (`subagentDone`, naming the tab or the sub-agent's task).
// The daemon ended the view itself: the page tells it nothing
// (`closeTab` is for a chat's own tab).
// Nothing else opens or closes it: the run_parallel / run_agent panel
// that fanned it out is an ordinary tool-call panel with no tie to
// tabs, whether it is folded, unfolded, adopted by a `summary`,
// replayed, or its task ends.

'use strict';

const assert = require('assert');
const {makePage, send, snapshotEntry, tabIds} = require('./simplify2_harness');
const {
  announceFinished,
  bootParallelRun,
  finishFanOut,
  openSubTabIds,
  spawnSubagent,
  subagentTabEls,
  togglePanel,
} = require('./subagent_tab_helpers');

/**
 * The tab ids of the `closeTab` frames posted after frame *from*: none
 * for a sub-agent's tab.  Every page holds it under the same id, and a
 * close arriving after another page resumed a chat into that id would
 * remove the chat.
 */
function closedSince(posted, from) {
  return posted
    .slice(from)
    .filter(m => m.type === 'closeTab')
    .map(m => m.tabId);
}

function testDoneClosesOnlyFinishedTab() {
  const {win, posted, subTabIds} = bootParallelRun(2, makePage());
  assert.ok(
    subagentTabEls(win).every(el =>
      el.querySelector('.subagent-indicator.status-spinner'),
    ),
    "a running sub-agent's tab shows the spinner",
  );

  send(win, {type: 'subagentDone', tab_id: subTabIds[0]});

  const remaining = subagentTabEls(win);
  assert.strictEqual(
    remaining.length,
    1,
    'the finished sub-agent tab must close as soon as it finishes, ' +
      'leaving only the still-running sibling open',
  );
  assert.strictEqual(
    remaining[0].dataset.tabId,
    subTabIds[1],
    'the surviving tab must be the still-running sibling',
  );
  assert.deepStrictEqual(
    closedSince(posted, 0),
    [],
    'the daemon ended the view: it is told nothing',
  );
  win.close();
  console.log('  ok - subagentDone closes only the finished tab');
}

function testDoneUnknownTabIsNoop() {
  const {win, posted, subTabIds} = bootParallelRun(2, makePage());

  const before = posted.length;
  send(win, {type: 'subagentDone', tab_id: 'no-such-tab'});
  send(win, {type: 'subagentDone', tab_id: 'no-such-tab', task_id: 'no-task'});

  assert.strictEqual(
    subagentTabEls(win).length,
    2,
    'a subagentDone for an unknown tab must not touch open tabs',
  );
  assert.deepStrictEqual(closedSince(posted, before), []);
  assert.deepStrictEqual(openSubTabIds(win), subTabIds);
  win.close();
  console.log('  ok - subagentDone for an unknown tab id is a no-op');
}

// The daemon names a sub-agent's tab as it addresses it (a replay
// derives the id from the parent's tab and the task); a page that
// shows the sub-agent under another id finds the tab by the task.
function testDoneFindsTheTabByItsTask() {
  const {win, posted, subTabIds, taskIds} = bootParallelRun(2, makePage(), {
    replay: true,
  });
  const before = posted.length;
  announceFinished(win, [taskIds[0]], taskId => 'elsewhere__sub_' + taskId);
  assert.deepStrictEqual(openSubTabIds(win), [subTabIds[1]]);
  assert.deepStrictEqual(closedSince(posted, before), []);
  win.close();
  console.log('  ok - subagentDone finds the tab by the sub-agent task');
}

// A sub-agent's end is not the end of a tab that shows something else
// under the id the daemon names: the chat tab itself, in the registry
// or not yet, or a tab the registry lists (another client resumed a
// chat into that tab id).
function testDoneLeavesTabsThatAreNotTheSubagents() {
  const {win, posted, parentId, subTabIds} = bootParallelRun(2, makePage());
  const [child1, child2] = subTabIds;
  finishFanOut(win, [parentId]);
  assert.deepStrictEqual(
    tabIds(win),
    [parentId, child1, child2],
    'the chat tab the registry has not listed yet stays',
  );
  send(win, {
    type: 'tabs_state',
    tabs: [
      snapshotEntry(parentId, 'parent', 'chat-parent'),
      snapshotEntry(child1, 'other chat', 'chat-other'),
    ],
  });
  const before = posted.length;
  finishFanOut(win, [parentId, child1, child2]);
  assert.deepStrictEqual(closedSince(posted, before), []);
  assert.deepStrictEqual(tabIds(win), [parentId, child1]);
  win.close();
  console.log('  ok - subagentDone leaves a tab that is not the sub-agent');
}

// A tab the registry lists shows a chat (another surface resumed one
// into the sub-agent's tab id).  It is not taken along when the tab
// above it closes, and its own close is a chat tab's: the daemon is
// told.
function testATabTheRegistryListsIsAChatsTab() {
  const page = makePage();
  const {win, posted, parentId, subTabIds} = bootParallelRun(1, page);
  const below = spawnSubagent(page, subTabIds[0], 'below-task', 'below', 0);
  send(win, {type: 'status', running: false, tabId: below});
  send(win, {
    type: 'tabs_state',
    tabs: [
      snapshotEntry(parentId, 'parent', 'chat-parent'),
      snapshotEntry(below, 'other chat', 'chat-other'),
    ],
  });
  finishFanOut(win, subTabIds);
  assert.deepStrictEqual(
    tabIds(win),
    [parentId, below],
    'the chat below the finished sub-agent keeps its tab',
  );
  send(win, {type: 'status', running: false, tabId: parentId});
  for (const left of [parentId, below]) {
    win._testApi.switchToTab(left);
    win.document.getElementById('new-chat-btn').click();
  }
  assert.deepStrictEqual(
    closedSince(posted, 0),
    [parentId, below],
    'each chat left idle is retired on the daemon, one by one',
  );
  win.close();
  console.log('  ok - a tab the registry lists is a chat tab');
}

function testRunningSubagentBelowAFinishedOneKeepsItsTab() {
  const page = makePage();
  const {win, posted, parentId, subTabIds} = bootParallelRun(1, page);
  const grand = spawnSubagent(page, subTabIds[0], 'grand-task', 'grand', 0);
  // A second grandchild whose own stream has ended runs no more: its
  // tab goes with its parent's.
  const ended = spawnSubagent(page, subTabIds[0], 'ended-task', 'ended', 1);
  send(win, {type: 'status', running: false, tabId: ended});
  finishFanOut(win, subTabIds);
  assert.deepStrictEqual(
    tabIds(win),
    [parentId, grand],
    'the running grandchild keeps its tab when its parent finishes',
  );
  assert.ok(win._testApi.openTabs()[1].isRunning, 'and it still runs');
  assert.deepStrictEqual(closedSince(posted, 0), []);
  finishFanOut(win, [grand]);
  assert.deepStrictEqual(tabIds(win), [parentId]);
  win.close();
  console.log('  ok - a running sub-agent below a finished one keeps its tab');
}

/** The `resumeSession` frames posted after frame *from*. */
function askedSince(posted, from) {
  return posted.slice(from).filter(m => m.type === 'resumeSession');
}

function testFoldingThePanelNeitherClosesNorOpensATab() {
  const page = makePage();
  const {win, posted, parentId, panel, subTabIds} = bootParallelRun(2, page);
  const collapsed = () => panel.classList.contains('collapsed');
  assert.ok(collapsed(), 'the fan-out panel starts folded');

  // Unfolded, then folded again: the running sub-agents keep their tabs.
  let from = posted.length;
  togglePanel(win, panel);
  assert.ok(!collapsed(), 'precondition: the click unfolds the panel');
  togglePanel(win, panel);
  assert.deepStrictEqual(openSubTabIds(win), subTabIds);
  assert.deepStrictEqual(closedSince(posted, from), []);

  // Under the folded panel a sub-agent that starts gets its tab at
  // once, and one that finishes loses it at once.
  const third = spawnSubagent(page, parentId, 'sub-task-3', 'sub 3', 2);
  finishFanOut(win, [subTabIds[0]]);
  assert.deepStrictEqual(openSubTabIds(win), [subTabIds[1], third]);
  assert.ok(collapsed(), 'no tab opening or closing unfolds the panel');

  // Under the unfolded panel too; unfolding it asked the daemon for no
  // finished sub-agent and opened none.
  from = posted.length;
  togglePanel(win, panel);
  assert.deepStrictEqual(openSubTabIds(win), [subTabIds[1], third]);
  assert.deepStrictEqual(askedSince(posted, from), []);
  finishFanOut(win, [subTabIds[1], third]);
  assert.deepStrictEqual(openSubTabIds(win), []);
  assert.ok(!collapsed(), 'the last tab closing folds no panel');
  togglePanel(win, panel);
  togglePanel(win, panel);
  assert.deepStrictEqual(tabIds(win), [parentId]);
  assert.deepStrictEqual(askedSince(posted, from), []);
  win.close();
  console.log('  ok - folding the fan-out panel neither closes nor opens a tab');
}

// What folds or hides the fan-out panel as the parent's task goes on
// leaves the running sub-agents' tabs alone: the call's result, a
// `summary` adopting the panel, the end of the task.
function testSummaryResultAndTaskEndLeaveRunningTabs() {
  const {win, posted, parentId, panel, subTabIds} = bootParallelRun(
    2,
    makePage(),
  );
  const from = posted.length;
  const steps = [
    {type: 'tool_result', content: 'fanned out', is_error: false},
    {type: 'tool_call', name: 'summary', description: 'progress so far'},
    {type: 'result', text: 'done', summary: 'Done.', success: true},
    {type: 'status', running: false},
    {type: 'task_done', success: true},
  ];
  for (const step of steps) {
    send(win, {...step, tabId: parentId});
    assert.deepStrictEqual(
      openSubTabIds(win),
      subTabIds,
      'the running sub-agents keep their tabs after ' + step.type,
    );
  }
  assert.ok(panel.closest('.summary-sub'), 'the summary adopted the panel');
  assert.ok(panel.closest('.trajectory'), 'the task end folded it away');
  assert.ok(win._testApi.openTabs().every(t => !t.isSubagentTab || t.isRunning));
  assert.deepStrictEqual(closedSince(posted, from), []);
  finishFanOut(win, subTabIds);
  assert.deepStrictEqual(tabIds(win), [parentId]);
  win.close();
  console.log('  ok - a summary, the result and the task end leave running tabs');
}

// A replay of the parent's transcript renders its fan-out panel anew:
// the running sub-agent keeps its tab, the finished one gets none, and
// the daemon's words after the replay (finished, running) change
// neither.
function testParentReplayNeitherOpensNorClosesTabs() {
  const {win, posted, parentId, subTabIds, taskIds} = bootParallelRun(
    2,
    makePage(),
    {replay: true},
  );
  finishFanOut(win, [subTabIds[0]]);
  const from = posted.length;
  send(win, {
    type: 'task_events',
    tabId: parentId,
    task: 'fan out',
    task_id: 'parent-task',
    events: [
      {
        type: 'tool_call',
        name: 'run_parallel',
        extras: {tasks: JSON.stringify(['sub 1', 'sub 2'])},
      },
    ],
  });
  assert.deepStrictEqual(openSubTabIds(win), [subTabIds[1]]);
  announceFinished(win, [taskIds[0]], taskId => parentId + '__sub_' + taskId);
  send(win, {
    type: 'openSubagentTab',
    tab_id: subTabIds[1],
    parent_tab_id: parentId,
    description: 'sub 2',
    task_id: taskIds[1],
    taskIndex: 1,
  });
  assert.deepStrictEqual(openSubTabIds(win), [subTabIds[1]]);
  assert.deepStrictEqual(closedSince(posted, from), []);
  win.close();
  console.log('  ok - a replay of the parent neither opens nor closes a tab');
}

function main() {
  testDoneClosesOnlyFinishedTab();
  testDoneUnknownTabIsNoop();
  testDoneFindsTheTabByItsTask();
  testDoneLeavesTabsThatAreNotTheSubagents();
  testATabTheRegistryListsIsAChatsTab();
  testRunningSubagentBelowAFinishedOneKeepsItsTab();
  testFoldingThePanelNeitherClosesNorOpensATab();
  testSummaryResultAndTaskEndLeaveRunningTabs();
  testParentReplayNeitherOpensNorClosesTabs();
  console.log('subagentTabAutoCloseOnDone.test.js: all tests passed');
}

main();
