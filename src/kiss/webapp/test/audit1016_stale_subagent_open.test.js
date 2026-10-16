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
const {makePage, send} = require('./simplify2_harness');

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
  const {win, posted} = makePage();
  const parentId = posted.find(m => m.type === 'ready').tabId;
  send(win, {type: 'status', running: true, tabId: parentId, startTs: 1});
  send(win, {
    type: 'tool_call',
    name: 'run_parallel',
    tabId: parentId,
    extras: {tasks: JSON.stringify(['sub 1'])},
  });
  const taskId = 'sub-task-1';
  const before = posted.length;
  send(win, {type: 'new_tab', task_id: taskId, parent_tab_id: parentId});
  const resume = posted
    .slice(before)
    .find(m => m.type === 'resumeSession' && m.taskId === taskId);
  assert.ok(resume, 'new_tab must make the page post resumeSession');
  const subTabId = resume.tabId;
  announce(win, parentId, subTabId, taskId);
  assert.strictEqual(subagentTabs(win).length, 1, 'precondition: one sub tab');
  return {win, posted, parentId, subTabId, taskId};
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
  const closesBefore = posted.filter(m => m.type === 'closeTab').length;
  announce(win, parentId, subTabId, taskId);
  assert.strictEqual(
    subagentTabs(win).length,
    0,
    'BUG - a stale openSubagentTab (isDone false) arriving after ' +
      'subagentDone re-opened the finished sub-agent as a running tab',
  );
  assert.strictEqual(
    posted.filter(m => m.type === 'closeTab').length,
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
  const readyId = posted.find(m => m.type === 'ready').tabId;
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
  // The daemon's deliberate finished-history announcement (a history
  // click, a panel expanded again) still reopens the child.
  announce(win, parentId, subTabId, taskId, {isDone: true});
  const subs = subagentTabs(win);
  assert.strictEqual(subs.length, 1, 'history reopening still works');
  assert.ok(subs[0].isDone && !subs[0].isRunning, 'as a finished tab');
  win.close();
  console.log('  ok - a stale openSubagentTab stays closed without a panel');
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

function main() {
  testStaleAnnouncementAfterDoneStaysClosed();
  testFreshAnnouncementStillOpens();
  testStaleAnnouncementNoPanelStaysClosed();
  testFinishedSetIsBounded();
}

try {
  main();
} catch (e) {
  console.error(e && e.stack ? e.stack : e);
  process.exit(1);
}
