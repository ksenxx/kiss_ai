// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Expanding a fan-out panel on one surface reopens the sub-agents the
// user closed there; every surface must then show the same tabs.  The
// page asks for each such sub-agent with `resumeSession` marked
// `reopen: true`, and the daemon's announcement of it says `reopened`
// (see server.py `_replay_session`).  A surface opens a `reopened`
// sub-agent whatever closed it there: a hand-close mirrored from
// another surface (`closeSubagentTab`), its finish, or a folded panel,
// which then unfolds the way the reopening surface's did (a folded
// panel's sync would close the finished tab again, on every surface).
// A surface connecting later gets the reopened finished sub-agent
// announced `reopened` (not `closedHistory`) and opens it too.
//
// Each surface is a page on the same persisted chat tab `parent-sync`,
// so the daemon frames name the same tab ids on all of them.

const assert = require('assert');
const {makePreloadedPage, send} = require('./simplify2_harness');
const {
  togglePanel,
  clickClose,
  finishFanOut,
  openSubTabIds,
  runParallelPanel,
  spawnSubagent,
} = require('./subagent_tab_helpers');
const {test, report} = require('./ui_antipattern_harness').makeRunner();

const PARENT = 'parent-sync';

/**
 * The parent task on *page* fans out to sub-task-1 and sub-task-2, as
 * the daemon drives a page: `status running`, the run_parallel
 * `tool_call`, then per child its spawn, announcement and transcript.
 * Returns the fan-out panel and the children's tab ids.
 */
function bootFanOut(page) {
  send(page.win, {type: 'status', running: true, tabId: PARENT});
  send(page.win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: PARENT,
    extras: {tasks: JSON.stringify(['sub 1', 'sub 2'])},
  });
  const panel = runParallelPanel(page.win);
  assert.ok(panel, 'the run_parallel call renders its panel');
  const subTabIds = ['sub-task-1', 'sub-task-2'].map((taskId, i) => {
    const tabId = spawnSubagent(page, PARENT, taskId, 'sub ' + (i + 1), i);
    send(page.win, {
      type: 'task_events',
      tabId,
      task: 'sub ' + (i + 1),
      task_id: taskId,
      events: [],
    });
    return tabId;
  });
  return {panel, subTabIds};
}

/** The open sub-agent tab ids of *page*, sorted. */
function shown(page) {
  return openSubTabIds(page.win).slice().sort();
}

/**
 * Fold and expand *panel* on *page*, and return the `resumeSession` the
 * expansion posted for sub-agent *taskId*.
 */
function reopenByPanel(page, panel, taskId) {
  togglePanel(page.win, panel);
  const from = page.posted.length;
  togglePanel(page.win, panel);
  const resume = page.posted
    .slice(from)
    .find(m => m.type === 'resumeSession' && m.taskId === taskId);
  assert.ok(resume, 'expanding the panel asks for ' + taskId + ' again');
  return resume;
}

/** The daemon's announcement of sub-agent *taskId* (index *idx*). */
function announcement(tabId, taskId, idx, extra) {
  return Object.assign(
    {
      type: 'openSubagentTab',
      tab_id: tabId,
      parent_tab_id: PARENT,
      description: 'sub ' + (idx + 1),
      task_id: taskId,
      taskIndex: idx,
      isSubagentTab: true,
    },
    extra,
  );
}

/** The tab ids of the `closeTab` frames *page* posted after frame *from*. */
function closedSince(page, from) {
  return page.posted
    .slice(from)
    .filter(m => m.type === 'closeTab')
    .map(m => m.tabId);
}

test('a sub-agent reopened on one surface opens where its close was mirrored', () => {
  const a = makePreloadedPage(PARENT);
  const b = makePreloadedPage(PARENT);
  const runA = bootFanOut(a);
  const runB = bootFanOut(b);
  assert.deepStrictEqual(runA.subTabIds, runB.subTabIds);
  const [child1, child2] = runA.subTabIds;

  // The user closes running child 1 on A; the daemon mirrors it to B.
  clickClose(a.win, child1);
  send(b.win, {type: 'closeSubagentTab', tab_id: child1});
  assert.deepStrictEqual(shown(a), [child2]);
  assert.deepStrictEqual(shown(b), [child2]);

  const resume = reopenByPanel(a, runA.panel, 'sub-task-1');
  const reopened = announcement(resume.tabId, 'sub-task-1', 0, {
    isDone: false,
    reopened: true,
  });
  send(a.win, reopened);
  send(b.win, reopened);

  assert.deepStrictEqual(shown(a), [child1, child2].sort());
  assert.deepStrictEqual(shown(b), shown(a), 'both surfaces show child 1');
  assert.strictEqual(resume.reopen, true, 'the request is a reopening');
  a.close();
  b.close();
});

test('a finished sub-agent reopened on one surface opens on every surface, a new one too', () => {
  const a = makePreloadedPage(PARENT);
  const b = makePreloadedPage(PARENT);
  const runA = bootFanOut(a);
  bootFanOut(b);
  const [child1, child2] = runA.subTabIds;
  finishFanOut(a.win, [child1]);
  finishFanOut(b.win, [child1]);
  assert.deepStrictEqual(shown(a), [child2], 'a finished tab closes');
  assert.deepStrictEqual(shown(b), [child2], 'a finished tab closes');

  const resume = reopenByPanel(a, runA.panel, 'sub-task-1');
  const reopened = announcement(resume.tabId, 'sub-task-1', 0, {
    isDone: true,
    reopened: true,
  });
  send(a.win, reopened);
  send(b.win, reopened);
  assert.deepStrictEqual(shown(a), [child1, child2].sort());
  assert.deepStrictEqual(shown(b), shown(a), 'the open surface shows it');
  assert.strictEqual(resume.reopen, true, 'the request is a reopening');

  // A surface connecting now: the running parent's replay, then the
  // daemon's announcements, the reopened finished child marked so.
  const c = makePreloadedPage(PARENT);
  send(c.win, {type: 'status', running: true, tabId: PARENT});
  send(c.win, {type: 'tool_call', name: 'run_parallel', tabId: PARENT});
  send(
    c.win,
    announcement(child1, 'sub-task-1', 0, {isDone: true, reopened: true}),
  );
  send(c.win, announcement(child2, 'sub-task-2', 1, {isDone: false}));
  assert.deepStrictEqual(shown(c), shown(a), 'the new surface shows it');
  for (const page of [a, b, c]) page.close();
});

test('a reopened finished sub-agent unfolds the panel it opens under', () => {
  const a = makePreloadedPage(PARENT);
  const b = makePreloadedPage(PARENT);
  const runA = bootFanOut(a);
  const runB = bootFanOut(b);
  const [child1, child2] = runA.subTabIds;
  finishFanOut(a.win, [child1]);
  finishFanOut(b.win, [child1]);
  // B's user folds B's panel; running child 2 keeps its tab.
  togglePanel(b.win, runB.panel);
  assert.ok(runB.panel.classList.contains('collapsed'));
  assert.deepStrictEqual(shown(b), [child2]);

  const resume = reopenByPanel(a, runA.panel, 'sub-task-1');
  const from = b.posted.length;
  send(
    b.win,
    announcement(resume.tabId, 'sub-task-1', 0, {isDone: true, reopened: true}),
  );
  assert.deepStrictEqual(shown(b), [child1, child2].sort(), 'it opens on B');
  assert.strictEqual(runParallelPanel(b.win), runB.panel);
  assert.ok(
    !runB.panel.classList.contains('collapsed'),
    "B's panel unfolds, as A's did",
  );
  // A later finished announcement under that panel closes nothing.
  send(b.win, announcement(child2, 'sub-task-2', 1, {isDone: true}));
  assert.deepStrictEqual(
    closedSince(b, from),
    [],
    'B closes no tab, so no close reaches A',
  );
  assert.ok(shown(b).includes(child1), 'the reopened tab stays open');
  a.close();
  b.close();
});

test('a reopened sub-agent opens on a surface that has not rendered its fan-out', () => {
  // The parent tab is open but its transcript is not loaded yet (a
  // deferred tab): there is no panel to unfold, the tab still opens.
  const d = makePreloadedPage(PARENT);
  send(d.win, {type: 'status', running: true, tabId: PARENT});
  assert.strictEqual(runParallelPanel(d.win), null);
  const child1 = PARENT + '__sub_sub-task-1';
  send(
    d.win,
    announcement(child1, 'sub-task-1', 0, {isDone: true, reopened: true}),
  );
  assert.deepStrictEqual(shown(d), [child1]);
  d.close();
});

report('subagentReopenShared_wave4');
