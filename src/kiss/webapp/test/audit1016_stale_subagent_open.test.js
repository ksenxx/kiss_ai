// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A sub-agent tab must close as soon as the sub-agent finishes and stay
// closed.  The daemon's `openSubagentTab` is the reply to the page's
// `resumeSession` (sent on `new_tab`), and it decides `isDone` BEFORE
// it broadcasts; the child can finish in that window, so the page can
// receive `subagentDone` first and the stale announcement (isDone
// false) after it.  Before the fix the stale announcement re-opened the
// finished sub-agent as a running tab that nothing ever closed.
//
// The real media/chat.html + panelCopy.js + api.js + main.js are booted
// in jsdom; nothing under test is mocked.

/* global require, console, process */
'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  readyTabId,
  ofType,
} = require('./simplify2_harness');
const {bootParallelRun} = require('./subagent_tab_helpers');

function subagentTabs(win) {
  return win._testApi.openTabs().filter(t => t.isSubagentTab);
}

function announce(win, parentId, tabId, taskId, extra) {
  send(
    win,
    Object.assign(
      {
        type: 'openSubagentTab',
        tab_id: tabId,
        parent_tab_id: parentId,
        description: 'sub 1',
        task_id: taskId,
        taskIndex: 0,
      },
      extra || {},
    ),
  );
}

// A parent task spawns one sub-agent through run_parallel, exactly the
// way the daemon drives the page: tool_call, new_tab (the page answers
// with resumeSession), then the daemon's openSubagentTab reply.
function bootOneChild() {
  const {win, posted, parentId, subTabIds, taskIds} = bootParallelRun(1, makePage());
  return {win, posted, parentId, subTabId: subTabIds[0], taskId: taskIds[0]};
}

function testStaleAnnouncementAfterDoneStaysClosed() {
  const {win, posted, parentId, subTabId, taskId} = bootOneChild();

  send(win, {type: 'subagentDone', tab_id: subTabId});
  assert.strictEqual(
    subagentTabs(win).length,
    0,
    'subagentDone closes the sub-agent tab',
  );

  // The resumeSession reply that lost the race: isDone decided before
  // the child finished.
  const closesBefore = ofType(posted, 'closeTab').length;
  announce(win, parentId, subTabId, taskId);
  assert.strictEqual(
    subagentTabs(win).length,
    0,
    'BUG - a stale openSubagentTab (isDone false) arriving after ' +
      'subagentDone re-opened the finished sub-agent as a running tab',
  );
  assert.strictEqual(
    ofType(posted, 'closeTab').length,
    closesBefore,
    'nothing new to close: no extra closeTab is echoed to the daemon',
  );

  // Later news for the finished sub-agent (a second stale copy, then
  // the daemon's own finished-history announcement) changes nothing.
  announce(win, parentId, subTabId, taskId);
  announce(win, parentId, subTabId, taskId, {isDone: true});
  assert.strictEqual(subagentTabs(win).length, 0, 'still closed');

  // The parent ends with no sub-agent tab left behind.
  send(win, {type: 'status', running: false, tabId: parentId});
  send(win, {type: 'task_done', success: true, tabId: parentId});
  assert.strictEqual(
    win._testApi
      .openTabs()
      .map(t => t.id)
      .join(','),
    parentId,
    'only the parent tab remains once its task is done',
  );
  win.close();
  console.log('  ok - a stale openSubagentTab after subagentDone stays closed');
}

function testFreshAnnouncementStillOpens() {
  // Control: a sub-agent the page has NOT seen finish opens as before,
  // and a different sub-agent's completion does not taint it.
  const {win, parentId, subTabId} = bootOneChild();
  send(win, {type: 'subagentDone', tab_id: subTabId});
  const before = win._testApi.openTabs().length;
  send(win, {type: 'new_tab', task_id: 'sub-task-2', parent_tab_id: parentId});
  announce(win, parentId, parentId + '__sub_sub-task-2', 'sub-task-2', {
    description: 'sub 2',
    taskIndex: 1,
  });
  const subs = subagentTabs(win);
  assert.strictEqual(subs.length, 1, 'the second sub-agent opens its tab');
  assert.strictEqual(subs[0].id, parentId + '__sub_sub-task-2');
  assert.ok(subs[0].isRunning, 'and it is running');
  assert.strictEqual(win._testApi.openTabs().length, before + 1);
  win.close();
  console.log('  ok - a sub-agent not seen finishing still opens');
}

// A parent adopted from the registry snapshot and running in the
// background: its transcript is only offered (replay_deferred), never
// fetched, so there is no fan-out panel to remember a closed sub-agent
// tab (rpAfterTabsClosed only books a close against a panel).
function bootDeferredParent() {
  const {win, posted} = makePage();
  const readyId = readyTabId(posted);
  const parentId = 'bg-parent';
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: readyId, chatId: 'c-ready', title: 'ready', workDir: ''},
      {tabId: parentId, chatId: 'c-bg', title: 'background', workDir: ''},
    ],
  });
  send(win, {type: 'status', running: true, tabId: parentId, startTs: 1});
  send(win, {type: 'replay_deferred', tabId: parentId, chatId: 'c-bg'});
  assert.strictEqual(win._testApi.getActiveTabId(), readyId);
  return {win, posted, parentId};
}

function testStaleAnnouncementNoPanelStaysClosed() {
  const {win, parentId} = bootDeferredParent();
  const taskId = 'bg-sub-1';
  const subTabId = parentId + '__sub_' + taskId;
  announce(win, parentId, subTabId, taskId);
  assert.strictEqual(subagentTabs(win).length, 1, 'the child opens');
  send(win, {type: 'subagentDone', tab_id: subTabId});
  assert.strictEqual(subagentTabs(win).length, 0, 'subagentDone closes it');

  announce(win, parentId, subTabId, taskId);
  assert.strictEqual(
    subagentTabs(win).length,
    0,
    'BUG - with no fan-out panel the stale openSubagentTab re-created ' +
      'the finished child as a done tab',
  );
  // The finished-history announcement the parent's automatic replay
  // makes once the child has finished (isDone, not reopened) keeps the
  // child closed here as it does on a surface with a fan-out panel: a
  // finished sub-agent's tab closes when it finishes, on every surface.
  announce(win, parentId, subTabId, taskId, {isDone: true});
  assert.strictEqual(
    subagentTabs(win).length,
    0,
    'BUG - a late finished-history announcement re-created the closed ' +
      'child on the surface without a fan-out panel',
  );
  // The daemon's deliberate reopening (a panel expanded again, a
  // history click on the child) says `reopened` and opens it everywhere.
  announce(win, parentId, subTabId, taskId, {isDone: true, reopened: true});
  const subs = subagentTabs(win);
  assert.strictEqual(subs.length, 1, 'a deliberate reopening still works');
  assert.ok(subs[0].isDone && !subs[0].isRunning, 'as a finished tab');
  win.close();
  console.log('  ok - a stale openSubagentTab stays closed without a panel');
}

// A child open on this surface finishes while the surface is away (its
// subagentDone never arrives).  The reconnect replay announces the
// finished child as closed history (isDone, closedHistory: its tab
// closed when it finished): the surface closes its copy, as a fresh
// surface never opens one, so both show the same tabs (L3-1).  A
// running sibling announced by the same replay stays open.
function testMissedCompletionClosesOnReconnect() {
  const {win, posted, parentId} = bootDeferredParent();
  const taskId = 'bg-sub-away';
  const subTabId = parentId + '__sub_' + taskId;
  announce(win, parentId, subTabId, taskId);
  const sibling = parentId + '__sub_bg-sub-live';
  announce(win, parentId, sibling, 'bg-sub-live', {taskIndex: 1});
  assert.strictEqual(subagentTabs(win).length, 2, 'both children open');

  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  send(win, {type: 'daemonStatus', connected: true});
  const closesBefore = ofType(posted, 'closeTab').length;
  announce(win, parentId, subTabId, taskId, {
    isDone: true,
    closedHistory: true,
  });
  announce(win, parentId, sibling, 'bg-sub-live', {taskIndex: 1});
  send(win, {type: 'replay_deferred', tabId: parentId, chatId: 'c-bg'});
  assert.strictEqual(
    subagentTabs(win)
      .map(t => t.id)
      .join(','),
    sibling,
    'BUG - the child that finished while this surface was away stayed ' +
      'open after the reconnect replay said it is closed history',
  );
  assert.ok(subagentTabs(win)[0].isRunning, 'the live sibling runs on');
  assert.strictEqual(
    ofType(posted, 'closeTab').length,
    closesBefore + 1,
    'the close is booked with the daemon once, like a subagentDone close',
  );
  // The replay's later copies of the same history change nothing.
  announce(win, parentId, subTabId, taskId, {
    isDone: true,
    closedHistory: true,
  });
  assert.strictEqual(
    subagentTabs(win)
      .map(t => t.id)
      .join(','),
    sibling,
    'closed history stays closed',
  );
  win.close();
  console.log('  ok - a completion missed while away closes on reconnect');
}

/** The replays the page asked for since message *from*: [tab, task, scoped]. */
function asked(posted, from) {
  return ofType(posted, 'resumeSession')
    .slice(from)
    .map(m => [m.tabId, m.taskId, m.scoped]);
}

/** The daemon replays task *taskId* into tab *tabId*. */
function replay(win, tabId, taskId) {
  send(win, {
    type: 'task_events',
    tabId,
    task: 'task ' + taskId,
    task_id: taskId,
    events: [{type: 'text_delta', text: 'answer'}],
  });
}

/** The connection drops and comes back. */
function outage(win) {
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  send(win, {type: 'daemonStatus', connected: true});
}

// The child of a task its chat has replaced since: the reconnect's
// replay brings the newer task and announces that task's children
// only, so nothing would tell this surface that the child it still
// shows running has finished.  The page asks the daemon about it by
// name and the daemon's word decides: finished closes the tab, as on
// the surfaces that saw it finish; one a user reopened since
// (`reopened`) stays, finished; one still running (a side channel
// outlives its parent's task) runs on.  A replay of the task the tab
// showed asks nothing: it announces that task's children itself.
function testChildOfAReplacedTaskIsAskedOnReconnect() {
  const answers = {
    finished: {isDone: true},
    reopened: {isDone: true, reopened: true, incarnation: 5},
    running: {},
  };
  for (const answer of Object.keys(answers)) {
    const {win, posted, parentId, subTabId, taskId} = bootOneChild();
    send(win, {type: 'daemonStatus', connected: true});
    // Not asked about: another chat's running child, and a child that
    // knows no task id to be asked by.
    send(win, {
      type: 'tabs_state',
      tabs: [
        {tabId: parentId, chatId: 'c-1', title: 'parent', workDir: ''},
        {tabId: 'other', chatId: 'c-2', title: 'other', workDir: ''},
      ],
    });
    announce(win, 'other', 'other__sub_x', 'x');
    announce(win, parentId, parentId + '__sub_unnamed', null, {taskIndex: 1});
    replay(win, parentId, 'p1');
    outage(win);
    let from = ofType(posted, 'resumeSession').length;
    replay(win, parentId, 'p1');
    assert.deepStrictEqual(asked(posted, from), [], 'the same task: no question');
    replay(win, parentId, 'p2');
    assert.deepStrictEqual(
      asked(posted, from),
      [[subTabId, taskId, true]],
      'BUG - the child of the replaced task is not asked about',
    );
    announce(win, parentId, subTabId, taskId, answers[answer]);
    // The rest of the daemon's answer finds the tab as the
    // announcement left it.
    send(win, {type: 'status', running: answer === 'running', tabId: subTabId});
    replay(win, subTabId, taskId);
    const child = () => {
      const tab = subagentTabs(win).find(t => t.id === subTabId);
      return [!!tab, !!tab && tab.isRunning];
    };
    assert.deepStrictEqual(
      child(),
      [answer !== 'finished', answer === 'running'],
      'BUG - a child the daemon calls ' + answer + ' after the reconnect',
    );
    assert.ok(
      subagentTabs(win).some(t => t.id === 'other__sub_x' && t.isRunning),
      "the other chat's child runs on",
    );
    // Only the daemon's first word about the tab is news of a missed
    // end: finished history named later is shown as it always was.
    announce(win, parentId, subTabId, taskId, {isDone: true});
    assert.deepStrictEqual(
      child(),
      [answer !== 'finished', false],
      'finished history after the ' + answer + ' answer',
    );
    // What no longer runs is not asked about after the next outage.
    outage(win);
    from = ofType(posted, 'resumeSession').length;
    replay(win, parentId, 'p3');
    assert.deepStrictEqual(asked(posted, from), [], 'nothing left to ask');
    win.close();
  }
  console.log('  ok - a child of a replaced task is asked about on reconnect');
}

// What the daemon says of a chat after a reconnect names the
// sub-agents of one task, and what runs below them, only: the task
// whose transcript it replays into the chat's tab, or, that tab in the
// background, the one it offers (`replay_deferred`: the tab's pin,
// `taskId`, or else the chat's latest task, `latestTaskId`).  The page
// asks about the others by name.  The daemon tells which task started
// each child (`parent_task_id`): one of another task is asked about
// whatever its chat tab shows (a child that outlives its task, a side
// channel, runs on below the next task's transcript); one of that
// task is not, though the tab still shows an older transcript.  A
// child given no parent task counts as one of the task the tab shows.
// What hangs below a child shares its fate.  An offer that names no
// task (a machine's relay) asks nothing; one for a tab on screen is
// fetched at once, and the transcript it brings does the asking.  The
// offer is not fetched when another one is, left over below it (the
// grandchild on screen, its parent's offer of an earlier reconnect
// never taken): the chat's transcript stays withheld, so the page
// asks.  A sub-agent tab below no chat is named by its own replay.
function testSubagentsTheAnswerWillNotNameAreAsked() {
  const p2 = {latestTaskId: 'p2'};
  const rows = [
    {offer: {latestTaskId: 'p1'}, asks: false, what: 'offered the task it shows'},
    {offer: {}, asks: false, what: 'offered no task by name'},
    {offer: p2, asks: true, what: 'offered a newer task'},
    {offer: {latestTaskId: ''}, asks: true, what: 'offered no task: none left'},
    {offer: {taskId: 'p0'}, asks: true, what: 'offered a pin on another task'},
    {offer: p2, onScreen: true, asks: true, what: 'offered on screen'},
    {offer: p2, below: true, asks: true, what: 'offered below a pending offer'},
    {offer: p2, shows: 'p2', childOf: 'p1', asks: true, what: 'offered the next task'},
    {offer: p2, childOf: 'p2', asks: false, what: "offered the child's own task"},
    {shows: 'p2', childOf: 'p1', asks: true, what: 'replayed the next task'},
    {childOf: 'p2', asks: false, what: "replayed the child's own task"},
  ];
  for (const {offer, onScreen, below, shows, childOf, asks, what} of rows) {
    const {win, posted, parentId, subTabId, taskId} = bootOneChild();
    const grandchild = subTabId + '__sub_g';
    const child = [subTabId, taskId, true];
    // jsdom reports what a page handler throws here, not to the sender.
    const thrown = [];
    win.addEventListener('error', e => thrown.push(String(e.error)));
    send(win, {type: 'daemonStatus', connected: true});
    const snapshot = {
      type: 'tabs_state',
      tabs: [
        {tabId: parentId, chatId: 'c-1', title: 'parent', workDir: ''},
        {tabId: 'other', chatId: 'c-2', title: 'other', workDir: ''},
      ],
    };
    send(win, snapshot);
    announce(win, parentId, subTabId, taskId, childOf && {parent_task_id: childOf});
    announce(win, subTabId, grandchild, 'g', {parent_task_id: taskId});
    send(win, {type: 'new_tab', task_id: 'lone', parent_tab_id: ''});
    announce(win, '', 'task__sub_lone', 'lone');
    replay(win, parentId, shows || 'p1');
    if (below) {
      for (const [tabId, task] of [child, [grandchild, 'g']])
        send(win, {type: 'replay_deferred', tabId, chatId: '', taskId: task});
      win._testApi.switchToTab(grandchild);
    } else if (offer && !onScreen) win._testApi.switchToTab('other');
    outage(win);
    send(win, snapshot);
    const from = ofType(posted, 'resumeSession').length;
    if (offer)
      send(
        win,
        Object.assign(
          {type: 'replay_deferred', tabId: parentId, chatId: 'c-1', taskId: ''},
          offer,
        ),
      );
    if (onScreen)
      assert.deepStrictEqual(
        asked(posted, from),
        [[parentId, undefined, true]],
        'the transcript of the tab on screen is fetched at the offer',
      );
    if (!offer || onScreen) replay(win, parentId, 'p2');
    assert.deepStrictEqual(
      asked(posted, from).filter(q => q[0] !== parentId),
      // Below a pending offer, that offer is fetched first: the child's.
      (below ? [child] : []).concat(asks ? [child, [grandchild, 'g', true]] : []),
      'BUG - the sub-agents of a chat ' + what,
    );
    if (asks) {
      announce(win, parentId, subTabId, taskId, {isDone: true});
      assert.ok(
        !subagentTabs(win).some(t => t.id === subTabId),
        'BUG - the finished child keeps its tab, its chat ' + what,
      );
    }
    const questions = asked(posted, from).length;
    replay(win, 'task__sub_lone', 'lone');
    assert.deepStrictEqual(
      [asked(posted, from).length, thrown],
      [questions, []],
      'BUG - the replay of a sub-agent tab below no chat asks nothing',
    );
    win.close();
  }
  console.log('  ok - the sub-agents an answer will not name are asked about');
}

function testFinishedSetIsBounded() {
  const {win, parentId} = bootDeferredParent();
  const cap = 300;
  for (let i = 0; i < cap + 10; i++) {
    const taskId = 'bg-sub-' + i;
    const subTabId = parentId + '__sub_' + taskId;
    announce(win, parentId, subTabId, taskId, {taskIndex: i});
    send(win, {type: 'subagentDone', tab_id: subTabId});
  }
  assert.strictEqual(subagentTabs(win).length, 0, 'all children closed');
  assert.ok(
    win._testApi.finishedSubagentCount() <= cap,
    'BUG - _finishedSubagentTasks grew past ' +
      cap +
      ': ' +
      win._testApi.finishedSubagentCount(),
  );
  // The newest finishes are the ones a late announcement can still
  // name; they stay protected.
  const lastId = 'bg-sub-' + (cap + 9);
  announce(win, parentId, parentId + '__sub_' + lastId, lastId);
  assert.strictEqual(subagentTabs(win).length, 0, 'newest stays closed');
  win.close();
  console.log('  ok - the finished sub-agent set is bounded');
}

// Audit 2026-11-11 (W1): the record of a closed sub-agent tab keeps a
// repeated finished-history announcement from reopening it.  The
// records only grew, one id per sub-agent for the life of the page;
// they are bounded now, like the finished set, oldest forgotten first.
function testClosedTabRecordsAreBounded() {
  const {win, parentId} = bootDeferredParent();
  const cap = 1000;
  for (let i = 0; i < cap + 10; i++) {
    const taskId = 'bg-sub-' + i;
    const subTabId = parentId + '__sub_' + taskId;
    announce(win, parentId, subTabId, taskId, {taskIndex: i});
    send(win, {type: 'subagentDone', tab_id: subTabId});
  }
  assert.strictEqual(subagentTabs(win).length, 0, 'all children closed');
  const count = win._testApi.closedSubagentTabCount();
  assert.ok(count > 0, 'setup: closed sub-agent tabs are recorded');
  assert.ok(
    count <= cap,
    'BUG - the closed sub-agent tab records grew past ' + cap + ': ' + count,
  );
  // The newest records still do their job: finished history naming a
  // closed child opens nothing.
  const lastId = 'bg-sub-' + (cap + 9);
  announce(win, parentId, parentId + '__sub_' + lastId, lastId, {isDone: true});
  assert.strictEqual(subagentTabs(win).length, 0, 'newest stays closed');
  win.close();
  console.log('  ok - the closed sub-agent tab records are bounded');
}

function main() {
  testClosedTabRecordsAreBounded();
  testStaleAnnouncementAfterDoneStaysClosed();
  testFreshAnnouncementStillOpens();
  testStaleAnnouncementNoPanelStaysClosed();
  testMissedCompletionClosesOnReconnect();
  testChildOfAReplacedTaskIsAskedOnReconnect();
  testSubagentsTheAnswerWillNotNameAreAsked();
  testFinishedSetIsBounded();
}

try {
  main();
} catch (e) {
  console.error(e && e.stack ? e.stack : e);
  process.exit(1);
}
