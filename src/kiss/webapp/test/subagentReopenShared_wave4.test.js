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
// The announcement of a reopened finished sub-agent counts the
// reopening (`incarnation`), and a close the page makes by itself says
// which one it closed: the daemon drops one that a later reopening has
// overtaken (server.py `_close_tab`).
//
// A `subagentDone` is not always the sub-agent's finish: a daemon ends
// the sub-agent tabs of a remote machine's stream when that stream
// ends, while the sub-agents run on (remote_tabs.py `_end_stream`).
// When the machine announces them running again, a surface that saw
// those ends opens the tabs as a new surface does.
//
// Each surface is a page on the same persisted chat tab `parent-sync`,
// so the daemon frames name the same tab ids on all of them.

const assert = require('assert');
const {makePreloadedPage, send, snapshotEntry} = require('./simplify2_harness');
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

test('a close the page makes by itself names the tab as the daemon last announced it', () => {
  const a = makePreloadedPage(PARENT);
  const run = bootFanOut(a);
  const [child1] = run.subTabIds;
  /** The `closeTab` frames page a posted for child 1 after frame *from*. */
  const closes = from =>
    a.posted
      .slice(from)
      .filter(m => m.type === 'closeTab' && m.tabId === child1);

  // Child 1 finishes: the tab the fan-out opened was never reopened.
  let from = a.posted.length;
  finishFanOut(a.win, [child1]);
  assert.deepStrictEqual(
    closes(from).map(m => m.incarnation),
    [0],
  );

  // The user expands the panel and folds it before the daemon answers:
  // that close follows the page's own request and names nothing.
  const resume = reopenByPanel(a, run.panel, 'sub-task-1');
  from = a.posted.length;
  togglePanel(a.win, run.panel);
  assert.deepStrictEqual(
    closes(from).map(m => 'incarnation' in m),
    [false],
  );

  // Expanded again, the daemon's answer and another surface's later
  // reopening arrive; a fold then closes the tab as last announced.
  togglePanel(a.win, run.panel);
  for (const incarnation of [7, 9]) {
    send(
      a.win,
      announcement(resume.tabId, 'sub-task-1', 0, {
        isDone: true,
        reopened: true,
        incarnation,
      }),
    );
  }
  from = a.posted.length;
  togglePanel(a.win, run.panel);
  assert.deepStrictEqual(
    closes(from).map(m => m.incarnation),
    [9],
  );
  a.close();
});

test("a sub-agent's end leaves a tab the registry lists for another chat", () => {
  const a = makePreloadedPage(PARENT);
  const [child1, child2] = bootFanOut(a).subTabIds;
  // Another client resumed a chat into child 1's tab id.
  send(a.win, {
    type: 'tabs_state',
    tabs: [
      snapshotEntry(PARENT, 'parent', 'chat-parent'),
      snapshotEntry(child1, 'other chat', 'chat-other'),
    ],
  });
  const from = a.posted.length;
  finishFanOut(a.win, [child1, child2]);
  assert.deepStrictEqual(closedSince(a, from), [child2]);
  assert.deepStrictEqual(shown(a), [child1]);
  a.close();
});

test("a sub-agent announced running after its stream's end opens where that end closed its tab", () => {
  const online = makePreloadedPage(PARENT);
  const fresh = makePreloadedPage(PARENT);
  const [child1, child2] = bootFanOut(online).subTabIds;
  const grand = spawnSubagent(online, child1, 'grand-task', 'grand', 0);
  const live = [
    announcement(child1, 'sub-task-1', 0, {isDone: false}),
    announcement(child2, 'sub-task-2', 1, {isDone: false}),
    announcement(grand, 'grand-task', 0, {
      isDone: false,
      parent_tab_id: child1,
    }),
  ];
  // The machine's stream is lost: its sub-agent tabs end.  An
  // announcement that was in flight then is a late echo.
  finishFanOut(online.win, [grand, child1, child2]);
  send(online.win, live[1]);
  assert.deepStrictEqual(shown(online), [], 'a late echo opens nothing');

  // The chat is opened in the tab again: the machine replays it to
  // every surface and announces what still runs below it.
  const back = [
    {type: 'status', running: true, tabId: PARENT},
    {
      type: 'task_events',
      tabId: PARENT,
      task: 'fan out',
      task_id: 'parent-task',
      events: [{type: 'tool_call', name: 'run_parallel', tabId: PARENT}],
    },
    ...live,
  ];
  for (const page of [online, fresh])
    back.forEach(frame => send(page.win, frame));
  assert.deepStrictEqual(shown(fresh), [child1, child2, grand].sort());
  assert.deepStrictEqual(
    shown(online),
    shown(fresh),
    'the surface online through the loss shows the same tabs',
  );

  // The user closes child 2 and the grandchild finishes: neither
  // announcement, repeated, opens its tab again.
  clickClose(online.win, child2);
  send(online.win, live[1]);
  finishFanOut(online.win, [grand]);
  send(online.win, live[2]);
  assert.deepStrictEqual(shown(online), [child1]);
  online.close();
  fresh.close();
});

test('a chat opened in another tab opens its running sub-agent on every surface', () => {
  // The machine's chat is opened in another tab while it runs: the
  // daemon ends the sub-agent tabs under the tab that showed it, and
  // the machine replays the chat into the other tab.
  const OTHER = 'parent-other';
  const moved = OTHER + '__sub_sub-task-1';
  const online = makePreloadedPage(PARENT);
  const fresh = makePreloadedPage(PARENT);
  const frames = [
    {type: 'tabs_state', tabs: [snapshotEntry(PARENT), snapshotEntry(OTHER)]},
    ...bootFanOut(online).subTabIds.map(tab_id => ({
      type: 'subagentDone',
      tab_id,
    })),
    {type: 'task_events', tabId: OTHER, task: 'fan out', events: []},
    announcement(moved, 'sub-task-1', 0, {
      isDone: false,
      parent_tab_id: OTHER,
    }),
  ];
  // Any group's running sub-agent tabs: the strip shows one group's.
  const running = page =>
    Array.from(page.win._testApi.openTabs())
      .filter(t => t.isSubagentTab && t.isRunning)
      .map(t => t.id);
  for (const page of [online, fresh])
    frames.forEach(frame => send(page.win, frame));
  assert.deepStrictEqual(running(fresh), [moved]);
  assert.deepStrictEqual(running(online), running(fresh));
  online.close();
  fresh.close();
});

test('a reopening, or a reconnect, opens a sub-agent whose stream ended', () => {
  const page = makePreloadedPage(PARENT);
  send(page.win, {type: 'daemonStatus', connected: true});
  const [child1, child2] = bootFanOut(page).subTabIds;
  finishFanOut(page.win, [child1, child2]);
  // A user asks for running child 2 again on another surface.
  send(
    page.win,
    announcement(child2, 'sub-task-2', 1, {isDone: false, reopened: true}),
  );
  assert.deepStrictEqual(shown(page), [child2]);

  // The page drops off and comes back: the daemon answers its `ready`
  // with running child 1, ahead of the parent's transcript.
  send(page.win, {type: 'daemonStatus', connected: false, reconnecting: true});
  send(page.win, {type: 'daemonStatus', connected: true});
  send(page.win, announcement(child1, 'sub-task-1', 0, {isDone: false}));
  assert.deepStrictEqual(shown(page), [child1, child2].sort());
  page.close();
});

report('subagentReopenShared_wave4');
