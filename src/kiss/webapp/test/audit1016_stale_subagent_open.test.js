// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A sub-agent tab must close as soon as the sub-agent finishes and stay
// closed.  The daemon's `openSubagentTab` is the reply to the page's
// `resumeSession` (sent on `new_tab`), and it decides that the child
// runs BEFORE it broadcasts; the child can finish in that window, so
// the page can receive `subagentDone` first and the stale announcement
// after it, which must not re-open the finished sub-agent as a running
// tab that nothing ever closes.  A `subagentDone` is not always a
// finish (a remote machine's stream ended while its sub-agents run
// on): an announcement the daemon decided after it opens the tab.
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
const {bootParallelRun, spawnSubagent} = require('./subagent_tab_helpers');

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

/** The daemon's word that a sub-agent finished, live and in a replay. */
function finished(win, tabId, taskId) {
  send(win, {type: 'subagentDone', tab_id: tabId, task_id: taskId});
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

  // The resumeSession reply that lost the race: decided before the
  // child finished.
  const closesBefore = ofType(posted, 'closeTab').length;
  announce(win, parentId, subTabId, taskId);
  assert.strictEqual(
    subagentTabs(win).length,
    0,
    'BUG - a stale openSubagentTab arriving after ' +
      'subagentDone re-opened the finished sub-agent as a running tab',
  );
  assert.strictEqual(
    ofType(posted, 'closeTab').length,
    closesBefore,
    'nothing new to close: no extra closeTab is echoed to the daemon',
  );

  // Later news for the finished sub-agent (a second stale copy, then
  // a replay's word that it finished) changes nothing.
  announce(win, parentId, subTabId, taskId);
  finished(win, subTabId, taskId);
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
// fetched, so the page holds no fan-out panel for it.
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
      'the finished child',
  );
  // The parent's replay says again that the child finished: it stays
  // closed here as on a surface that renders the fan-out panel.
  finished(win, subTabId, taskId);
  assert.strictEqual(
    subagentTabs(win).length,
    0,
    "BUG - a replay's word that the child finished re-created it on " +
      'the surface without a fan-out panel',
  );
  win.close();
  console.log('  ok - a stale openSubagentTab stays closed without a panel');
}

// A child open on this surface finishes while the surface is away (its
// subagentDone never arrives).  The reconnect replay says that the
// child finished (`subagentDone`): the surface closes its copy, as a
// fresh surface never opens one.  A running sibling announced by the
// same replay stays open.
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
  finished(win, subTabId, taskId);
  announce(win, parentId, sibling, 'bg-sub-live', {taskIndex: 1});
  send(win, {type: 'replay_deferred', tabId: parentId, chatId: 'c-bg'});
  assert.strictEqual(
    subagentTabs(win)
      .map(t => t.id)
      .join(','),
    sibling,
    'BUG - the child that finished while this surface was away stayed ' +
      'open after the reconnect replay said it finished',
  );
  assert.ok(subagentTabs(win)[0].isRunning, 'the live sibling runs on');
  assert.strictEqual(
    ofType(posted, 'closeTab').length,
    closesBefore,
    'the daemon ended the view: it is told nothing, as at any subagentDone',
  );
  // The replay's later copies of the same word change nothing.
  finished(win, subTabId, taskId);
  assert.strictEqual(
    subagentTabs(win)
      .map(t => t.id)
      .join(','),
    sibling,
    'the finished child stays closed',
  );
  assert.strictEqual(ofType(posted, 'closeTab').length, closesBefore);
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

/**
 * What a daemon sends of a child from its replay to its end, less the
 * last frame (`subagentDone`): the connection drops before it.
 */
function endsAtTheDrop(win, subTabId, taskId) {
  send(win, {type: 'status', running: true, tabId: subTabId});
  replay(win, subTabId, taskId);
  send(win, {type: 'status', running: false, tabId: subTabId});
}

// The child of a task its chat has replaced since: the reconnect's
// replay brings the newer task and announces that task's children
// only, so nothing would tell this surface that the child it still
// shows running has finished.  The page asks the daemon about it by
// name and the daemon's word decides: finished closes the tab, as on
// the surfaces that saw it finish; one still running (a side channel
// outlives its parent's task) runs on.  A replay of the task the tab
// showed asks nothing: it announces that task's children itself.  The
// child is asked about all the same when its tab had stopped running
// before the connection dropped (`ended`): only its last frame was
// lost, and nothing but the daemon's word closes a sub-agent's tab.
function testChildOfAReplacedTaskIsAskedOnReconnect() {
  for (const [answer, ended] of [['finished'], ['running'], ['finished', true]]) {
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
    if (ended) endsAtTheDrop(win, subTabId, taskId);
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
    if (answer === 'finished') finished(win, subTabId, taskId);
    else announce(win, parentId, subTabId, taskId);
    // The rest of the daemon's answer finds the tab as that word left
    // it.
    send(win, {type: 'status', running: answer === 'running', tabId: subTabId});
    replay(win, subTabId, taskId);
    const child = () => {
      const tab = subagentTabs(win).find(t => t.id === subTabId);
      return [!!tab, !!tab && tab.isRunning];
    };
    assert.deepStrictEqual(
      child(),
      [answer === 'running', answer === 'running'],
      'BUG - a child the daemon calls ' + answer + ' after the reconnect',
    );
    assert.ok(
      subagentTabs(win).some(t => t.id === 'other__sub_x' && t.isRunning),
      "the other chat's child runs on",
    );
    // The daemon's word settled the child: the replay of yet another
    // task asks nothing about it.
    from = ofType(posted, 'resumeSession').length;
    replay(win, parentId, 'p3');
    assert.deepStrictEqual(asked(posted, from), [], 'a settled child');
    // The child's end, said (again) later, leaves no tab of it.
    finished(win, subTabId, taskId);
    assert.deepStrictEqual(
      child(),
      [false, false],
      'the end of the child after the ' + answer + ' answer',
    );
    // What no longer runs is not asked about after the next outage.
    outage(win);
    from = ofType(posted, 'resumeSession').length;
    replay(win, parentId, 'p4');
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
    {offer: p2, ended: true, asks: true, what: 'offered one, the child ended'},
    {offer: {latestTaskId: ''}, asks: true, what: 'offered no task: none left'},
    {offer: {taskId: 'p0'}, asks: true, what: 'offered a pin on another task'},
    {offer: p2, onScreen: true, asks: true, what: 'offered on screen'},
    {offer: p2, below: true, asks: true, what: 'offered below a pending offer'},
    {offer: p2, shows: 'p2', childOf: 'p1', asks: true, what: 'offered the next task'},
    {offer: p2, childOf: 'p2', asks: false, what: "offered the child's own task"},
    {shows: 'p2', childOf: 'p1', asks: true, what: 'replayed the next task'},
    {childOf: 'p2', asks: false, what: "replayed the child's own task"},
  ];
  for (const {offer, onScreen, below, shows, childOf, ended, asks, what} of rows) {
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
    if (ended) endsAtTheDrop(win, subTabId, taskId);
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
      finished(win, subTabId, taskId);
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

// A sub-agent tab below no chat (its `new_tab` names no parent) hangs
// below nothing the daemon replays or offers after a reconnect, so no
// answer would ever name it: the page asks about it by name once the
// new connection's registry snapshot is in, and the daemon's word
// decides.  Not asked about: one bound to a machine (the relay tells
// every surface that connects the ends it kept), one the snapshot
// lists (another surface resumed a chat into its id meanwhile) and one
// below a chat (that chat's answer names it).
function testSubagentBelowNoChatIsAskedAtTheReconnect() {
  for (const answer of ['finished', 'running']) {
    const {win, posted} = makePage();
    const chat = readyTabId(posted);
    const snapshot = {
      type: 'tabs_state',
      tabs: [
        {tabId: chat, chatId: '', title: 'new chat', workDir: ''},
        {tabId: 'task__sub_taken', chatId: 'c-2', title: 'other', workDir: ''},
      ],
    };
    send(win, {type: 'daemonStatus', connected: true});
    for (const [task, parent] of [['lone'], ['far'], ['taken'], ['below', chat]]) {
      send(win, {type: 'new_tab', task_id: task, parent_tab_id: parent || ''});
      announce(win, parent || '', (parent || 'task') + '__sub_' + task, task);
    }
    send(win, {type: 'tabMachine', tabId: 'task__sub_far', machine: 'box'});
    const from = ofType(posted, 'resumeSession').length;
    outage(win);
    assert.deepStrictEqual(asked(posted, from), [], 'asked before the snapshot');
    send(win, snapshot);
    send(win, snapshot);
    assert.deepStrictEqual(
      asked(posted, from),
      [['task__sub_lone', 'lone', true]],
      'BUG - a sub-agent tab below no chat is not asked about, once',
    );
    if (answer === 'finished') finished(win, 'task__sub_lone', 'lone');
    else announce(win, '', 'task__sub_lone', 'lone');
    assert.deepStrictEqual(
      Array.from(subagentTabs(win), t => t.id).sort(),
      [chat + '__sub_below', 'task__sub_far']
        .concat(answer === 'running' ? ['task__sub_lone'] : [])
        .concat('task__sub_taken')
        .sort(),
      'BUG - a sub-agent below no chat the daemon calls ' + answer,
    );
    assert.deepStrictEqual(ofType(posted, 'closeTab'), []);
    win.close();
  }
  console.log('  ok - a sub-agent tab below no chat is asked about at the reconnect');
}

/** The ids of the open sub-agent tabs that run, sorted. */
function runningSubagents(win) {
  const running = subagentTabs(win).filter(t => t.isRunning);
  return Array.from(running, t => t.id).sort();
}

// A `subagentDone` is not always a finish: a daemon ends the sub-agent
// tabs of a remote machine's stream when that stream ends, while the
// sub-agents run on.  An announcement in flight then is a late echo
// and opens nothing.  Once the chat is replayed into its tab again,
// the machine's announcements of what still runs below it open the
// tabs, as on a page that never saw those ends; a real finish after
// that stands against its own late echo.
function testAnnouncementAfterAReplayOpensAnEndedSubagent() {
  const page = makePage();
  const {win, parentId, subTabIds, taskIds} = bootParallelRun(2, page);
  const [child1, child2] = subTabIds;
  const grand = spawnSubagent(page, child1, 'grand-task', 'grand', 0);
  const announceAll = () => {
    announce(win, parentId, child1, taskIds[0]);
    announce(win, parentId, child2, taskIds[1], {taskIndex: 1});
    announce(win, child1, grand, 'grand-task');
  };
  for (const id of [grand, child1, child2])
    send(win, {type: 'subagentDone', tab_id: id});
  announce(win, parentId, child2, taskIds[1], {taskIndex: 1});
  assert.deepStrictEqual(runningSubagents(win), [], 'a late echo opens nothing');

  send(win, {type: 'status', running: true, tabId: parentId});
  send(win, {
    type: 'task_events',
    tabId: parentId,
    task: 'fan out',
    task_id: 'parent-task',
    events: [{type: 'tool_call', name: 'run_parallel', tabId: parentId}],
  });
  announceAll();
  assert.deepStrictEqual(
    runningSubagents(win),
    [child1, child2, grand].sort(),
    'BUG - what runs below a chat replayed after a stream end stays closed',
  );

  send(win, {type: 'subagentDone', tab_id: grand});
  announce(win, child1, grand, 'grand-task');
  assert.deepStrictEqual(runningSubagents(win), [child1, child2].sort());
  win.close();
  console.log('  ok - an announcement after a replay opens an ended sub-agent');
}

// The machine's chat is opened in another tab while it runs: the
// daemon ends the sub-agent tabs under the tab that showed it, and the
// machine replays the chat into the other tab and announces the
// running child below that one.
function testChatOpenedInAnotherTabOpensItsRunningSubagent() {
  const {win, parentId, subTabId, taskId} = bootOneChild();
  const other = 'parent-other';
  const moved = other + '__sub_' + taskId;
  send(win, {
    type: 'tabs_state',
    tabs: [
      {tabId: parentId, chatId: 'c-1', title: 'parent', workDir: ''},
      {tabId: other, chatId: 'c-2', title: 'other', workDir: ''},
    ],
  });
  send(win, {type: 'subagentDone', tab_id: subTabId});
  send(win, {type: 'task_events', tabId: other, task: 'fan out', events: []});
  announce(win, other, moved, taskId);
  assert.deepStrictEqual(runningSubagents(win), [moved]);
  win.close();
  console.log('  ok - a chat opened in another tab opens its running sub-agent');
}

// The page drops off and comes back: the daemon answers its `ready`
// with the running child, ahead of the parent's transcript.
function testReconnectOpensASubagentWhoseStreamEnded() {
  const {win, parentId, subTabId, taskId} = bootOneChild();
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'subagentDone', tab_id: subTabId});
  outage(win);
  announce(win, parentId, subTabId, taskId);
  assert.deepStrictEqual(runningSubagents(win), [subTabId]);
  win.close();
  console.log('  ok - a reconnect opens a sub-agent whose stream ended');
}

function main() {
  testStaleAnnouncementAfterDoneStaysClosed();
  testFreshAnnouncementStillOpens();
  testStaleAnnouncementNoPanelStaysClosed();
  testMissedCompletionClosesOnReconnect();
  testChildOfAReplacedTaskIsAskedOnReconnect();
  testSubagentsTheAnswerWillNotNameAreAsked();
  testFinishedSetIsBounded();
  testSubagentBelowNoChatIsAskedAtTheReconnect();
  testAnnouncementAfterAReplayOpensAnEndedSubagent();
  testChatOpenedInAnotherTabOpensItsRunningSubagent();
  testReconnectOpensASubagentWhoseStreamEnded();
}

try {
  main();
} catch (e) {
  console.error(e && e.stack ? e.stack : e);
  process.exit(1);
}
