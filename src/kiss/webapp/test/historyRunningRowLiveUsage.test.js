// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The History row of a RUNNING task must show the same cost, tokens and
// steps as the task's own chat header.
//
// The daemon overlays a running task's live totals on its row whenever
// the history is fetched, but the page only refetches the history on
// task news (`tasks_updated`: a task or sub-task starting or ending).
// Between two refetches the header moved on with every `usage_info`
// while the row kept the totals of its last fetch: on 2026-11-14 the
// live web UI showed "58 steps • 6.34M tok • $5.02" in the History row
// of a task whose header read "Cost: $6.70" (69 steps, 8.53M tokens).
// The row now follows the task's `usage_info` events.
//
// Review round 1 (gpt-6-astra-xhigh) added two cases: a task id is a
// task's identity on its own machine only, so the totals are kept by
// machine and task id (the machine being the one the event's tab is
// bound to); and an event that arrives BEFORE the task's first row is
// kept, because that row can be a snapshot taken before the event.
//
// The 2026-11-15 review added two more: the totals a history reply
// brings for a running row are banked (an older snapshot accepted
// after a newer one lowered the row), and the usage_info events of a
// replayed transcript (`task_events`) reach the row as the live ones
// do (the replay painted the header and left the row behind it).

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  rowsByTitle,
} = require('./simplify2_harness');
const {freezeClock, makeRow} = require('./history_helpers');

const NOW_MS = 1_794_655_015_000;
const RUNNING_ID = 'b77e58aa17804afb8063c7726ab17da3';
const FINISHED_ID = 'd7948af2000000000000000000000001';

function runningRow(extra) {
  return makeRow(
    Object.assign(
      {
        id: 'chat-running',
        task_id: RUNNING_ID,
        title: 'cost audit',
        timestamp: Math.floor((NOW_MS - 65_000) / 1000),
        is_running: true,
        tokens: 6_336_077,
        cost: 5.018,
        steps: 58,
        startTs: NOW_MS - 65_000,
        endTs: 0,
      },
      extra,
    ),
  );
}

function finishedRow(extra) {
  return makeRow(
    Object.assign(
      {
        id: 'chat-finished',
        task_id: FINISHED_ID,
        title: 'autorouter',
        timestamp: 1_794_646_856,
        tokens: 1_400_694,
        cost: 2.9467,
        steps: 24,
        startTs: 1_794_646_856_000,
        endTs: 1_794_647_233_000,
      },
      extra,
    ),
  );
}

function usage(taskId, tokens, cost, steps, extra) {
  return Object.assign(
    {
      type: 'usage_info',
      text: `Steps: ${steps}/10000, Total tokens: ${tokens}, `,
      total_tokens: tokens,
      cost: '$' + cost.toFixed(4),
      cost_usd: cost,
      total_steps: steps,
      taskId,
    },
    extra,
  );
}

function makePage(sessions) {
  const page = makeChatPage({beforeScripts: win => freezeClock(win, NOW_MS)});
  send(page.win, {type: 'configData', config: {work_dir: ''}, apiKeys: {}});
  send(page.win, {type: 'history', sessions, offset: 0});
  return page;
}

/** The "steps • tokens • cost" head of the row titled *title*. */
function metrics(win, title) {
  const row = rowsByTitle(win)[title];
  assert.ok(row, `the "${title}" row must be listed`);
  const text = row.querySelector('.running-item-metrics').textContent;
  return text.split(' • ').slice(0, 3).join(' • ');
}

function testRunningRowFollowsUsageInfo() {
  const {win} = makePage([runningRow(), finishedRow()]);
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '58 steps • 6.34M tok • $5.02',
    'the fetched totals show first',
  );

  // The task's next steps and a banked task-update charge, as the
  // daemon published them (history.db, 2026-11-14 11:16:55 UTC).
  send(win, usage(RUNNING_ID, 8_530_789, 6.7026, 69));
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '69 steps • 8.53M tok • $6.70',
    "the running row must show the task's newest totals, as its header does",
  );
  assert.strictEqual(
    metrics(win, 'autorouter'),
    '24 steps • 1.40M tok • $2.95',
    "another task's row is untouched",
  );
  win.close();
  console.log('  ok - a running row follows its task usage_info events');
}

function testOlderEventsAndRefetchesLowerNothing() {
  const {win} = makePage([runningRow()]);
  send(win, usage(RUNNING_ID, 8_530_789, 6.7026, 69));

  // A replayed (older) event of the same task.
  send(win, usage(RUNNING_ID, 6_524_853, 5.1087, 58));
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '69 steps • 8.53M tok • $6.70',
  );

  // A history reply fetched before the newest event (identical rows:
  // the fast path) and one with a changed row (a rebuild).
  send(win, {type: 'history', sessions: [runningRow()], offset: 0});
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '69 steps • 8.53M tok • $6.70',
  );
  send(win, {
    type: 'history',
    sessions: [runningRow({tokens: 7_404_963, cost: 6.1763, steps: 65})],
    offset: 0,
  });
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '69 steps • 8.53M tok • $6.70',
  );

  // A refetch that is NEWER than the last event this page saw wins.
  send(win, {
    type: 'history',
    sessions: [runningRow({tokens: 9_000_000, cost: 7.5, steps: 75})],
    offset: 0,
  });
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '75 steps • 9.00M tok • $7.50',
  );
  win.close();
  console.log('  ok - older events and older refetches lower nothing');
}

function testFinishedRowShowsItsStoredTotals() {
  const {win} = makePage([runningRow(), finishedRow()]);
  send(win, usage(RUNNING_ID, 8_530_789, 6.7026, 69));

  // A finished task's late charge names the task too: its row waits
  // for the refetch the daemon's `tasks_updated` asks for.
  send(win, usage(FINISHED_ID, 1_500_000, 3.5, 26));
  assert.strictEqual(
    metrics(win, 'autorouter'),
    '24 steps • 1.40M tok • $2.95',
  );

  // The running task ends: the row shows what its row stores, whatever
  // the last live event said.
  const ended = runningRow({
    is_running: false,
    tokens: 8_600_000,
    cost: 6.65,
    steps: 70,
    endTs: NOW_MS - 1000,
  });
  send(win, {type: 'history', sessions: [ended, finishedRow()], offset: 0});
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '70 steps • 8.60M tok • $6.65',
  );

  // A straggling event of the ended task changes nothing, and neither
  // does a later run of the same row id as running (nothing was kept).
  send(win, usage(RUNNING_ID, 9_999_999, 9.99, 99));
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '70 steps • 8.60M tok • $6.65',
  );
  send(win, {
    type: 'history',
    sessions: [runningRow(), finishedRow()],
    offset: 0,
  });
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '58 steps • 6.34M tok • $5.02',
  );
  win.close();
  console.log('  ok - a finished row shows its stored totals');
}

function testEventsThatNameNoListedRunningTask() {
  const {win} = makePage([runningRow(), finishedRow()]);
  const before = win.document.getElementById('history-list').innerHTML;

  // A sub-agent's totals (its row is not listed), an event with no
  // task id, one with no totals (an older daemon's text-only event),
  // one with no cost, and one for a tab this page does not have (the
  // machine its task runs on is unknown).
  send(win, usage('5ub4gent000000000000000000000001', 42_688, 0.5348, 2));
  send(win, usage('', 8_530_789, 6.7026, 69));
  send(win, {
    type: 'usage_info',
    taskId: RUNNING_ID,
    text: 'Steps: 69/10000, ',
  });
  send(win, {
    type: 'usage_info',
    taskId: RUNNING_ID,
    total_tokens: 8_530_789,
    cost: 'N/A',
    total_steps: 69,
  });
  send(win, usage(RUNNING_ID, 8_530_789, 6.7026, 69, {tabId: 'no-such-tab'}));
  assert.strictEqual(
    win.document.getElementById('history-list').innerHTML,
    before,
    'none of these events may change the list',
  );

  // An older daemon's event carries the cost as a string only.
  const legacy = usage(RUNNING_ID, 8_530_789, 6.7026, 69);
  delete legacy.cost_usd;
  send(win, legacy);
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '69 steps • 8.53M tok • $6.70',
  );

  // An emptied list (a search with no match) loses nothing: the older
  // snapshot that brings the row back does not lower it.
  send(win, {type: 'history', sessions: [], offset: 0});
  send(win, {type: 'history', sessions: [runningRow()], offset: 0});
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '69 steps • 8.53M tok • $6.70',
  );
  win.close();
  console.log('  ok - events naming no listed running task change nothing');
}

function testUsageBeforeTheFirstRow() {
  // The daemon reads a new task's live totals, then waits for the
  // other machines' rows: the task's next `usage_info` can reach the
  // page before that (older) first row does.
  const {win} = makePage([finishedRow()]);
  send(win, usage(RUNNING_ID, 8_530_789, 6.7026, 69));
  send(win, {
    type: 'history',
    sessions: [runningRow(), finishedRow()],
    offset: 0,
  });
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '69 steps • 8.53M tok • $6.70',
    'the older first row must not hide the totals the task published',
  );

  // A task that is already finished when its first row arrives shows
  // what that row stores, and nothing of it is kept.
  const quick = 'q' + '0'.repeat(31);
  send(win, usage(quick, 50_000, 0.25, 3));
  const quickRow = extra =>
    finishedRow(
      Object.assign(
        {id: 'chat-quick', task_id: quick, title: 'quick', tokens: 61_000},
        extra,
      ),
    );
  send(win, {
    type: 'history',
    sessions: [quickRow({cost: 0.3, steps: 4}), runningRow()],
    offset: 0,
  });
  assert.strictEqual(metrics(win, 'quick'), '4 steps • 61.0K tok • $0.30');
  send(win, {
    type: 'history',
    sessions: [
      quickRow({is_running: true, endTs: 0, cost: 0.1, steps: 1}),
      runningRow(),
    ],
    offset: 0,
  });
  assert.strictEqual(metrics(win, 'quick'), '1 steps • 61.0K tok • $0.10');
  win.close();
  console.log('  ok - totals published before the first row are kept');
}

function testTotalsBelongToTheTaskOfOneMachine() {
  // The same task id on two machines (a history copied from one to
  // the other): two tasks, two rows.
  const local = runningRow();
  const remote = runningRow({
    id: 'chat-remote',
    title: 'remote audit',
    machine: 'remote-box',
    tokens: 1_000,
    cost: 1,
    steps: 1,
  });
  const {win} = makePage([local, remote]);

  // This daemon's own task: an event naming no tab, or one for a tab
  // bound to no machine.
  send(win, usage(RUNNING_ID, 8_530_789, 6.7026, 69));
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '69 steps • 8.53M tok • $6.70',
  );
  assert.strictEqual(
    metrics(win, 'remote audit'),
    '1 steps • 1.00K tok • $1.00',
    "this machine's task must not repaint another machine's row",
  );

  // The other machine's task: its events come through the tab the
  // daemon bound to that machine (`tabMachine`).
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'tabMachine', tabId, machine: 'remote-box'});
  send(win, usage(RUNNING_ID, 2_000, 2, 2, {tabId}));
  assert.strictEqual(
    metrics(win, 'remote audit'),
    '2 steps • 2.00K tok • $2.00',
  );
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '69 steps • 8.53M tok • $6.70',
    "another machine's task must not repaint this machine's row",
  );

  // The list filtered to the other machine (whose newer snapshot is
  // taken), then unfiltered again: each row still shows its own task's
  // totals, also the ones published while the row was not listed.
  const remoteNewer = runningRow({
    id: 'chat-remote',
    title: 'remote audit',
    machine: 'remote-box',
    tokens: 1_000,
    cost: 3,
    steps: 3,
  });
  send(win, {type: 'history', sessions: [remoteNewer], offset: 0});
  send(win, usage(RUNNING_ID, 8_600_000, 6.75, 70));
  assert.strictEqual(
    metrics(win, 'remote audit'),
    '3 steps • 2.00K tok • $3.00',
  );
  send(win, {type: 'history', sessions: [local, remoteNewer], offset: 0});
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '70 steps • 8.60M tok • $6.75',
  );
  assert.strictEqual(
    metrics(win, 'remote audit'),
    '3 steps • 2.00K tok • $3.00',
  );

  // Unbound again, the tab's events are this machine's.
  send(win, {type: 'tabMachine', tabId, machine: ''});
  send(win, usage(RUNNING_ID, 8_700_000, 6.8, 71, {tabId}));
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '71 steps • 8.70M tok • $6.80',
  );
  assert.strictEqual(
    metrics(win, 'remote audit'),
    '3 steps • 2.00K tok • $3.00',
  );
  win.close();
  console.log('  ok - totals are kept by machine and task id');
}

function testKeptTotalsAreBounded() {
  // Totals of tasks whose row never comes (sub-agents, tasks of
  // unloaded pages) must not pile up: 256 tasks are kept, the one that
  // has been silent the longest goes first.  Never a running row of
  // the list, however long it has been silent (a task waiting for its
  // sub-agents): the row would fall back to its older fetched totals
  // (review round 2).
  const {win} = makePage([runningRow(), finishedRow()]);
  send(win, usage(RUNNING_ID, 8_530_789, 6.7026, 69)); // the oldest entry
  const id = i => 'pending-' + i;
  for (let i = 0; i < 255; i++) send(win, usage(id(i), 9_000, 9, 9));
  send(win, usage(id(0), 9_500, 9.5, 10)); // task 0 speaks again
  send(win, usage(id(255), 9_000, 9, 9)); // one too many: task 1 is dropped
  const row = i =>
    runningRow({
      id: 'chat-' + i,
      task_id: id(i),
      title: 'task ' + i,
      tokens: 1_000,
      cost: 1,
      steps: 1,
    });
  // The view toggle rebuilds the rows from what the page holds.
  win.document.getElementById('history-view-toggle').click();
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '69 steps • 8.53M tok • $6.70',
    "a listed running row's totals must survive the bound",
  );
  send(win, {
    type: 'history',
    sessions: [runningRow(), row(0), row(1), row(2), row(255), finishedRow()],
    offset: 0,
  });
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '69 steps • 8.53M tok • $6.70',
  );
  assert.strictEqual(metrics(win, 'task 0'), '10 steps • 9.50K tok • $9.50');
  assert.strictEqual(metrics(win, 'task 1'), '1 steps • 1.00K tok • $1.00');
  assert.strictEqual(metrics(win, 'task 2'), '9 steps • 9.00K tok • $9.00');
  assert.strictEqual(metrics(win, 'task 255'), '9 steps • 9.00K tok • $9.00');

  // Rows 0, 2 and 255 are running rows of the list now: 300 more
  // silent-task totals drop none of theirs.
  for (let i = 300; i < 600; i++) send(win, usage(id(i), 9_000, 9, 9));
  win.document.getElementById('history-view-toggle').click();
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '69 steps • 8.53M tok • $6.70',
  );
  assert.strictEqual(metrics(win, 'task 0'), '10 steps • 9.50K tok • $9.50');
  assert.strictEqual(metrics(win, 'task 2'), '9 steps • 9.00K tok • $9.00');
  assert.strictEqual(metrics(win, 'task 255'), '9 steps • 9.00K tok • $9.00');

  // ... and the silent tasks' own totals stay bounded: the first of
  // those 300 is gone when its row comes, the last is there.
  send(win, {
    type: 'history',
    sessions: [runningRow(), row(300), row(599), finishedRow()],
    offset: 0,
  });
  assert.strictEqual(metrics(win, 'task 300'), '1 steps • 1.00K tok • $1.00');
  assert.strictEqual(metrics(win, 'task 599'), '9 steps • 9.00K tok • $9.00');
  win.close();
  console.log('  ok - the kept totals are bounded');
}

function testTheBoundHoldsAgainOnceRunningRowsLeave() {
  // More running rows than the bound keep all their totals; when the
  // rows leave the list the bound holds again at once, not at the
  // high-water mark (review round 3).
  const {win} = makePage([]);
  const expectCounts = (kept, running, when) => {
    const counts = win._testApi.historyLiveUsageCounts();
    assert.strictEqual(counts.kept, kept, `kept totals ${when}`);
    assert.strictEqual(counts.running, running, `running rows ${when}`);
  };
  const id = i => 'run-' + i;
  const row = i =>
    runningRow({
      id: 'chat-' + i,
      task_id: id(i),
      title: 'run ' + i,
      tokens: 1_000,
      cost: 1,
      steps: 1,
    });
  const rows = [];
  for (let i = 0; i < 300; i++) rows.push(row(i));
  const listAll = () => {
    for (let at = 0; at < rows.length; at += 50) {
      send(win, {
        type: 'history',
        sessions: rows.slice(at, at + 50),
        offset: at,
      });
    }
  };
  const live = '9 steps • 9.00K tok • $9.00';
  const fetched = '1 steps • 1.00K tok • $1.00';

  listAll();
  for (let i = 0; i < 300; i++) send(win, usage(id(i), 9_000, 9, 9));
  expectCounts(300, 300, 'with 300 running rows listed');
  assert.strictEqual(metrics(win, 'run 0'), live);
  assert.strictEqual(metrics(win, 'run 299'), live);

  // A silent task finds no room among them, and takes none of theirs.
  send(win, usage('silent', 9_000, 9, 9));
  expectCounts(300, 300, 'after a silent task spoke');

  // A list that keeps two of the rows (the two that have been silent
  // the longest): theirs stay, the others' are cut to the bound.
  send(win, {type: 'history', sessions: [row(0), row(1)], offset: 0});
  expectCounts(256, 2, 'with two of the rows left');
  assert.strictEqual(metrics(win, 'run 0'), live);
  assert.strictEqual(metrics(win, 'run 1'), live);

  // All 300 again (rows 2..45 lost theirs to the cut and publish
  // anew), then an emptied list: the bound holds at once, and under
  // 1000 more silent tasks.
  listAll();
  assert.strictEqual(metrics(win, 'run 2'), fetched);
  assert.strictEqual(metrics(win, 'run 46'), live);
  for (let i = 2; i < 46; i++) send(win, usage(id(i), 9_000, 9, 9));
  expectCounts(300, 300, 'with 300 running rows listed again');
  send(win, {type: 'history', sessions: [], offset: 0});
  expectCounts(256, 0, 'after the list was emptied');
  for (let i = 0; i < 1000; i++) send(win, usage('s-' + i, 9_000, 9, 9));
  expectCounts(256, 0, 'after 1000 silent tasks');

  // Its row gone from the list, a task is a silent task like any
  // other: run 0 was dropped, and its row shows what it is fetched with.
  send(win, {type: 'history', sessions: [row(0)], offset: 0});
  assert.strictEqual(metrics(win, 'run 0'), fetched);
  expectCounts(256, 1, 'with one row back');
  win.close();
  console.log('  ok - the bound holds again once running rows leave');
}

function testSecondPageRowFollowsToo() {
  // Rows of a later page are found by the same task id.
  const first = [];
  for (let i = 0; i < 3; i++) {
    first.push(
      finishedRow({id: 'chat-f' + i, task_id: 'f' + i, title: 'done ' + i}),
    );
  }
  const {win} = makePage(first);
  send(win, {type: 'history', sessions: [runningRow()], offset: first.length});
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '58 steps • 6.34M tok • $5.02',
  );
  send(win, usage(RUNNING_ID, 8_530_789, 6.7026, 69));
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '69 steps • 8.53M tok • $6.70',
  );
  assert.strictEqual(metrics(win, 'done 1'), '24 steps • 1.40M tok • $2.95');
  win.close();
  console.log('  ok - a running row of a later page follows too');
}

function testOlderSnapshotAfterANewerOneLowersNothing() {
  // Two history requests of one generation can overlap, so a reply
  // that was snapshotted earlier can be accepted after a newer one.
  // The totals a reply brought for a running row are banked, with or
  // without a live event in between.
  const at = n => runningRow({tokens: n * 1000, cost: n, steps: n});
  const {win} = makePage([at(5)]);
  send(win, usage(RUNNING_ID, 6_000, 6, 6));
  send(win, {type: 'history', sessions: [at(7)], offset: 0});
  assert.strictEqual(metrics(win, 'cost audit'), '7 steps • 7.00K tok • $7.00');
  send(win, {type: 'history', sessions: [at(5)], offset: 0});
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '7 steps • 7.00K tok • $7.00',
    'an older snapshot accepted after a newer one must not lower the row',
  );
  win.close();

  const fetchedOnly = makePage([at(7)]).win;
  send(fetchedOnly, {type: 'history', sessions: [at(5)], offset: 0});
  assert.strictEqual(
    metrics(fetchedOnly, 'cost audit'),
    '7 steps • 7.00K tok • $7.00',
    'also when no live event was seen in between',
  );
  // A later page that repeats the row with older totals: the same.
  send(fetchedOnly, {type: 'history', sessions: [at(8)], offset: 0});
  send(fetchedOnly, {type: 'history', sessions: [at(6)], offset: 0});
  assert.strictEqual(
    metrics(fetchedOnly, 'cost audit'),
    '8 steps • 8.00K tok • $8.00',
  );

  // The task ended: its row shows the totals it stored, whatever was
  // banked while it ran.
  send(fetchedOnly, {
    type: 'history',
    sessions: [
      runningRow({
        is_running: false,
        tokens: 7_500,
        cost: 7.5,
        steps: 7,
        endTs: NOW_MS,
      }),
    ],
    offset: 0,
  });
  assert.strictEqual(
    metrics(fetchedOnly, 'cost audit'),
    '7 steps • 7.50K tok • $7.50',
  );
  assert.deepStrictEqual(
    Object.assign({}, fetchedOnly._testApi.historyLiveUsageCounts()),
    {kept: 0, running: 0},
    'nothing is kept for a task that ended',
  );
  fetchedOnly.close();
  console.log('  ok - an older snapshot after a newer one lowers nothing');
}

/** A `task_events` replay of *taskId* for *tabId* carrying *events*. */
function replay(tabId, taskId, chatId, events) {
  return {
    type: 'task_events',
    tabId,
    task: 'cost audit',
    task_id: taskId,
    chat_id: chatId,
    events,
  };
}

function testReplayedTotalsReachTheRow() {
  // Opening a running task's tab (or a reconnect) replays the task's
  // transcript: its usage_info events paint the chat header without
  // passing the live event path.  The row must follow them too (it
  // stayed at the fetched totals until the task's next step).
  const {win} = makePage([runningRow(), finishedRow()]);
  const tabId = win._testApi.getActiveTabId();
  const header = () => win.document.getElementById('status-budget').textContent;
  send(win, {type: 'status', tabId, running: true});
  send(
    win,
    replay(tabId, RUNNING_ID, 'chat-running', [
      usage(RUNNING_ID, 6_524_853, 5.1087, 58),
      usage(RUNNING_ID, 8_530_789, 6.7026, 69),
    ]),
  );
  assert.strictEqual(header(), 'Cost: $6.70', 'the replay paints the header');
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '69 steps • 8.53M tok • $6.70',
    'the running row must read what the replayed header reads',
  );
  assert.strictEqual(
    metrics(win, 'autorouter'),
    '24 steps • 1.40M tok • $2.95',
  );

  // A replay snapshotted before the newest live event lowers nothing.
  send(win, usage(RUNNING_ID, 9_000_000, 7.5, 75));
  send(
    win,
    replay(tabId, RUNNING_ID, 'chat-running', [
      usage(RUNNING_ID, 8_530_789, 6.7026, 69),
    ]),
  );
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '75 steps • 9.00M tok • $7.50',
  );

  // Replayed events name their task themselves: the transcript of a
  // finished task changes neither its own (finished) row nor the
  // running one, and an event naming no task is not guessed onto the
  // task of the replay.
  send(
    win,
    replay(tabId, FINISHED_ID, 'chat-finished', [
      usage(FINISHED_ID, 99_000_000, 99, 99),
    ]),
  );
  send(
    win,
    replay(tabId, RUNNING_ID, 'chat-running', [
      usage(undefined, 98_000_000, 98, 98),
      {type: 'usage_info', text: 'Steps: 97/10000', taskId: RUNNING_ID},
    ]),
  );
  assert.strictEqual(
    metrics(win, 'cost audit'),
    '75 steps • 9.00M tok • $7.50',
  );
  assert.strictEqual(
    metrics(win, 'autorouter'),
    '24 steps • 1.40M tok • $2.95',
  );
  win.close();

  // A replay for a tab bound to another machine is that machine's
  // task, as for live events.
  const remote = runningRow({
    id: 'chat-remote',
    title: 'remote audit',
    machine: 'remote-box',
    tokens: 1_000,
    cost: 1,
    steps: 1,
  });
  const two = makePage([runningRow(), remote]).win;
  const boundTab = two._testApi.getActiveTabId();
  send(two, {type: 'tabMachine', tabId: boundTab, machine: 'remote-box'});
  send(
    two,
    replay(boundTab, RUNNING_ID, 'chat-remote', [
      usage(RUNNING_ID, 2_000, 2, 2),
    ]),
  );
  assert.strictEqual(
    metrics(two, 'remote audit'),
    '2 steps • 2.00K tok • $2.00',
  );
  assert.strictEqual(
    metrics(two, 'cost audit'),
    '58 steps • 6.34M tok • $5.02',
    "another machine's replay must not repaint this machine's row",
  );
  // A replay for a tab this page does not have names no machine.
  send(
    two,
    replay('no-such-tab', RUNNING_ID, 'chat-running', [
      usage(RUNNING_ID, 50_000_000, 50, 50),
    ]),
  );
  assert.strictEqual(
    metrics(two, 'cost audit'),
    '58 steps • 6.34M tok • $5.02',
  );
  assert.strictEqual(
    metrics(two, 'remote audit'),
    '2 steps • 2.00K tok • $2.00',
  );
  two.close();
  console.log('  ok - replayed totals reach the running row');
}

testRunningRowFollowsUsageInfo();
testOlderEventsAndRefetchesLowerNothing();
testOlderSnapshotAfterANewerOneLowersNothing();
testReplayedTotalsReachTheRow();
testFinishedRowShowsItsStoredTotals();
testEventsThatNameNoListedRunningTask();
testSecondPageRowFollowsToo();
testUsageBeforeTheFirstRow();
testTotalsBelongToTheTaskOfOneMachine();
testKeptTotalsAreBounded();
testTheBoundHoldsAgainOnceRunningRowsLeave();
console.log('historyRunningRowLiveUsage.test.js: all assertions passed.');
