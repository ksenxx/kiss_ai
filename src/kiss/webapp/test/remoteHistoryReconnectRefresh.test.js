// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  fakeMatchMedia,
} = require('./simplify2_harness');

function makePage(opts) {
  const {desktopMatches = false} = opts || {};
  return makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts(win) {
      fakeMatchMedia(win, desktopMatches);
    },
  });
}

function getHistoryMsgs(posted) {
  return posted.filter(m => m.type === 'getHistory');
}

function lastGetHistory(posted) {
  const msgs = getHistoryMsgs(posted);
  return msgs.length ? msgs[msgs.length - 1] : null;
}

function openDrawer(win) {
  const btn = win.document.getElementById('menu-btn');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    win.document.getElementById('sidebar').classList.contains('open'),
    'burger click must open the drawer',
  );
}

function rowByTitle(win, title) {
  const rows = win.document.querySelectorAll('#history-list .sidebar-item');
  for (const r of rows) {
    const t = r.querySelector('.sidebar-item-text');
    if (t && t.textContent === title) return r;
  }
  return null;
}

function testDesktopReconnectRefetchesHistory() {
  const {win, posted} = makePage({desktopMatches: true});
  const sidebar = win.document.getElementById('sidebar');
  assert.ok(
    sidebar.classList.contains('open'),
    'remote desktop docks the sidebar open on load',
  );
  assert.ok(
    getHistoryMsgs(posted).length >= 1,
    'docking on load posts the initial getHistory',
  );
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  assert.ok(
    lastGetHistory(posted),
    'REGRESSION: reconnect (daemonStatus connected:true) must refetch ' +
      'history for the open docked sidebar — a task started while the ' +
      'socket was down is otherwise invisible forever',
  );
  win.close();
  console.log('PASS remote-desktop reconnect refetches history');
}

function testDisconnectDoesNotFetch() {
  const {win, posted} = makePage({desktopMatches: true});
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: false});
  assert.strictEqual(
    getHistoryMsgs(posted).length,
    0,
    'daemonStatus connected:false must not post getHistory',
  );
  win.close();
  console.log('PASS disconnect does not fetch history');
}

function testClosedSidebarReconnectNoFetch() {
  const {win, posted} = makePage({desktopMatches: false});
  assert.ok(
    !win.document.getElementById('sidebar').classList.contains('open'),
    'mobile drawer starts closed',
  );
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  assert.strictEqual(
    getHistoryMsgs(posted).length,
    0,
    'reconnect with a closed sidebar must not post getHistory',
  );
  win.close();
  console.log('PASS closed-sidebar reconnect posts no getHistory');
}

function testMobileLifecycleShowsTaskStartedWhileDisconnected() {
  const {win, posted} = makePage({desktopMatches: false});
  send(win, {
    type: 'configData',
    config: {work_dir: '/var/repo/alpha'},
    apiKeys: {},
  });
  openDrawer(win);
  const first = lastGetHistory(posted);
  send(win, {
    type: 'history',
    generation: first.generation,
    offset: 0,
    sessions: [
      {
        id: 'chat-1',
        task_id: 1,
        title: 'old completed task',
        preview: 'old completed task',
        has_events: true,
        timestamp: 1_700_000_000,
        is_running: false,
        failed: false,
        work_dir: '/var/repo/alpha',
      },
    ],
  });
  assert.ok(
    rowByTitle(win, 'old completed task'),
    'pre-disconnect history renders',
  );
  send(win, {type: 'daemonStatus', connected: false});
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: true});
  const refetch = lastGetHistory(posted);
  assert.ok(refetch, 'reconnect must refetch history');
  send(win, {
    type: 'history',
    generation: refetch.generation,
    offset: 0,
    sessions: [
      {
        id: 'chat-2',
        task_id: 2,
        title: 'started while offline',
        preview: 'started while offline',
        has_events: true,
        timestamp: 1_700_000_100,
        is_running: true,
        failed: false,
        work_dir: '/private/var/repo/alpha',
      },
      {
        id: 'chat-1',
        task_id: 1,
        title: 'old completed task',
        preview: 'old completed task',
        has_events: true,
        timestamp: 1_700_000_000,
        is_running: false,
        failed: false,
        work_dir: '/var/repo/alpha',
      },
    ],
  });
  const row = rowByTitle(win, 'started while offline');
  assert.ok(row, 'the task started while offline must be rendered');
  assert.strictEqual(row.dataset.category, 'running');
  assert.notStrictEqual(
    row.style.display,
    'none',
    'the running task started while the phone was offline must be ' +
      'VISIBLE in the mobile History drawer after reconnect',
  );
  win.close();
  console.log('PASS mobile lifecycle shows task started while offline');
}

function testStaleGenerationReplyIsDropped() {
  const {win, posted} = makePage({desktopMatches: false});
  openDrawer(win);
  const stale = lastGetHistory(posted);
  assert.ok(stale, 'opening the drawer posts getHistory');
  send(win, {type: 'daemonStatus', connected: false});
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: true});
  const fresh = lastGetHistory(posted);
  assert.ok(fresh.generation > stale.generation, 'refetch bumps generation');
  send(win, {
    type: 'history',
    generation: fresh.generation,
    offset: 0,
    sessions: [
      {
        id: 'chat-9',
        task_id: 9,
        title: 'fresh running task',
        preview: 'fresh running task',
        has_events: true,
        timestamp: 1_700_000_200,
        is_running: true,
        failed: false,
        work_dir: '/repo/x',
      },
    ],
  });
  assert.ok(rowByTitle(win, 'fresh running task'), 'fresh reply renders');
  send(win, {
    type: 'history',
    generation: stale.generation,
    offset: 0,
    sessions: [
      {
        id: 'chat-8',
        task_id: 8,
        title: 'stale old task',
        preview: 'stale old task',
        has_events: true,
        timestamp: 1_600_000_000,
        is_running: false,
        failed: false,
        work_dir: '/repo/x',
      },
    ],
  });
  assert.ok(
    rowByTitle(win, 'fresh running task'),
    'the fresh running task must survive a straggling stale reply',
  );
  assert.strictEqual(
    rowByTitle(win, 'stale old task'),
    null,
    'a history reply carrying an outdated generation must be dropped',
  );
  win.close();
  console.log('PASS stale-generation history reply is dropped');
}

function testTasksUpdatedStillRefetches() {
  const {win, posted} = makePage({desktopMatches: true});
  posted.length = 0;
  send(win, {type: 'tasks_updated', taskId: ''});
  assert.ok(
    lastGetHistory(posted),
    'tasks_updated must still refetch history while the sidebar is open',
  );
  win.close();
  console.log('PASS tasks_updated still refetches while open');
}

testDesktopReconnectRefetchesHistory();
testDisconnectDoesNotFetch();
testClosedSidebarReconnectNoFetch();
testMobileLifecycleShowsTaskStartedWhileDisconnected();
testStaleGenerationReplyIsDropped();
testTasksUpdatedStillRefetches();
console.log('All remoteHistoryReconnectRefresh tests passed.');
