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
  ofType,
} = require('./simplify2_harness');
const {
  lastGetHistory,
  rowByTitle,
  makeRow,
  sendHistory,
} = require('./history_helpers');

const OLD_TASK = makeRow({
  id: 'chat-1',
  task_id: 1,
  title: 'old completed task',
  has_events: true,
  work_dir: '/var/repo/alpha',
});

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
  return ofType(posted, 'getHistory');
}

function openDrawer(win) {
  const btn = win.document.getElementById('menu-btn');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    win.document.getElementById('sidebar').classList.contains('open'),
    'burger click must open the drawer',
  );
}

// The reconnect refetch of an open sidebar and the idle tasks_updated
// refetch are pinned by historyRefreshSingleFlight.test.js; docking on
// load by remoteDesktopSidebar.test.js.

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
  sendHistory(win, posted, [OLD_TASK]);
  assert.ok(
    rowByTitle(win, 'old completed task'),
    'pre-disconnect history renders',
  );
  send(win, {type: 'daemonStatus', connected: false});
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: true});
  assert.ok(lastGetHistory(posted), 'reconnect must refetch history');
  sendHistory(win, posted, [
    makeRow({
      id: 'chat-2',
      task_id: 2,
      title: 'started while offline',
      has_events: true,
      timestamp: 1_700_000_100,
      is_running: true,
      work_dir: '/private/var/repo/alpha',
    }),
    OLD_TASK,
  ]);
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
  sendHistory(win, posted, [
    makeRow({
      id: 'chat-9',
      task_id: 9,
      title: 'fresh running task',
      has_events: true,
      timestamp: 1_700_000_200,
      is_running: true,
      work_dir: '/repo/x',
    }),
  ]);
  assert.ok(rowByTitle(win, 'fresh running task'), 'fresh reply renders');
  send(win, {
    type: 'history',
    generation: stale.generation,
    offset: 0,
    sessions: [
      makeRow({
        id: 'chat-8',
        task_id: 8,
        title: 'stale old task',
        has_events: true,
        timestamp: 1_600_000_000,
        work_dir: '/repo/x',
      }),
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

testDisconnectDoesNotFetch();
testClosedSidebarReconnectNoFetch();
testMobileLifecycleShowsTaskStartedWhileDisconnected();
testStaleGenerationReplyIsDropped();
console.log('All remoteHistoryReconnectRefresh tests passed.');
