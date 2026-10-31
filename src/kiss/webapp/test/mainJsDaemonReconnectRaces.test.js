// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

/* global require, console, process */
'use strict';

// End-to-end (jsdom, real media/main.js + api.js) regressions for the
// daemon-connection races found by the media JS audit:
//
// F1  A cold start (daemonStatus connected:false BEFORE the daemon was
//     ever seen up) must not re-send `ready` once the daemon comes up:
//     the boot ready is queued by the shim and flushed on auth, so a
//     second one replays every transcript twice.  A real reconnect
//     (up -> down -> up) still re-sends it.
// F2  The talk playback queue must not wedge when the user agent pauses
//     a clip without ever firing `ended` (tab backgrounded, incoming
//     call): `pause` releases the queue so later clips still play, and
//     the paused player is retired (source dropped) so a resume cannot
//     overlap the next clip.
// F3  requestTaskUpdate and the sidebar-panels poll tick must not post
//     while daemonStatus is disconnected (the shim would queue the
//     posts and burst them out on reconnect, ahead of `ready`).
// F4  A tab the user just closed must not be resurrected by a stale
//     tabs_state broadcast before the daemon processed the close -- but
//     that shield lapses after 5 s and on reconnect, so a tab another
//     surface re-opened under the same id is never hidden.

const assert = require('assert');
const {
  makePage: buildPage,
  send,
  stubRaf,
  fakeMatchMedia,
  fakeIntervals,
  ofType: msgsOf,
} = require('./simplify2_harness');
const {clickClose} = require('./subagent_tab_helpers');
const {installAudio} = require('./speech_helpers');

// `desktop` adds remote-desktop to the body, as the page does on a wide
// screen.  Intervals are recorded, not run, so a test fires a poll tick
// by hand.
function makePage(desktop) {
  let intervals;
  const {win, posted} = buildPage({
    bodyClass: 'remote-chat' + (desktop ? ' remote-desktop' : ''),
    beforeScripts: w => {
      stubRaf(w);
      intervals = fakeIntervals(w).intervals;
      fakeMatchMedia(w, true);
    },
  });
  return {win, posted, intervals};
}

function daemon(win, connected) {
  send(win, {type: 'daemonStatus', connected});
}

function entry(tabId, title) {
  return {tabId, title: title || tabId, chatId: '', workDir: ''};
}

// Every open tab, in tab order (copied into this realm's Array so
// deepStrictEqual compares values, not realms' prototypes).
function tabIds(win) {
  return Array.from(win._testApi.openTabs(), t => t.id);
}

// --- F1 ---------------------------------------------------------------------

function testColdStartDoesNotDoubleReady() {
  const {win, posted} = makePage();
  assert.strictEqual(msgsOf(posted, 'ready').length, 1, 'boot posts one ready');
  // The shim answers the boot ready with "daemon not up yet" (the
  // socket is not authenticated), then the daemon comes up: the
  // queued boot ready is flushed by the shim, so no second one.
  daemon(win, false);
  daemon(win, true);
  assert.strictEqual(
    msgsOf(posted, 'ready').length,
    1,
    'REGRESSION: a cold start (never connected -> connected) must not ' +
      're-send ready; the queued boot ready already went out',
  );
  // Still no re-ready on a repeated connected:true.
  daemon(win, true);
  assert.strictEqual(msgsOf(posted, 'ready').length, 1);
  win.close();
  console.log('PASS cold start posts ready exactly once');
}

function testRealReconnectStillResendsReady() {
  const {win, posted} = makePage();
  daemon(win, true);
  assert.strictEqual(msgsOf(posted, 'ready').length, 1);
  daemon(win, false);
  daemon(win, true);
  assert.strictEqual(
    msgsOf(posted, 'ready').length,
    2,
    'a daemon that was up and came back must get a fresh ready',
  );
  // A second outage re-sends again; a bare repeat does not.
  daemon(win, false);
  daemon(win, true);
  daemon(win, true);
  assert.strictEqual(msgsOf(posted, 'ready').length, 3);
  win.close();
  console.log('PASS real reconnect re-sends ready once per outage');
}

function testSubmitStillBlockedWhileDownOnColdStart() {
  // The "currently disconnected" guards moved off daemonWasDown: a
  // cold-start disconnect must still keep a submit out of the shim
  // queue, and the reconnect must let it through again.
  const {win, posted} = makePage();
  win._testApi.hideWelcome();
  const inp = win.document.getElementById('task-input');
  const sendBtn = win.document.getElementById('send-btn');
  daemon(win, false);
  inp.value = 'hello while down';
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  sendBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    msgsOf(posted, 'submit').length,
    0,
    'no submit is posted while the daemon is down',
  );
  assert.strictEqual(inp.value, 'hello while down', 'the draft stays put');
  daemon(win, true);
  sendBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.strictEqual(
    msgsOf(posted, 'submit').length,
    1,
    'the submit goes out once the daemon is back',
  );
  win.close();
  console.log('PASS submit is held while down and released on connect');
}

// --- F2 ---------------------------------------------------------------------

const B64 = 'SUQzBAAAAAAAAA==';

function talk(win, id) {
  send(win, {
    type: 'talk',
    language: 'en-US',
    text: 'clip ' + id,
    talkId: id,
    audioB64: B64,
  });
}

function testUaPauseReleasesTalkQueue() {
  const {win} = makePage();
  const players = installAudio(win);
  talk(win, 'a');
  talk(win, 'b');
  assert.strictEqual(players.length, 1, 'second clip waits for the first');
  assert.strictEqual(players[0].playCalls, 1);
  // The user agent pauses the clip (tab backgrounded, incoming call):
  // `pause` fires, `ended` never will.
  players[0].fire('pause');
  assert.strictEqual(
    players.length,
    2,
    'REGRESSION: a paused clip must release the queue so the next ' +
      'clip plays instead of every later talk being swallowed',
  );
  assert.strictEqual(players[1].playCalls, 1);
  // The paused player is retired, not merely abandoned: a user-agent
  // resume (audio focus regained) would otherwise play it on top of
  // the next clip. No source left to resume, no handlers left to
  // re-pump the queue.
  assert.strictEqual(
    players[0].src,
    '',
    'REGRESSION: the paused clip kept its source and could resume',
  );
  for (const h of ['onended', 'onerror', 'onabort', 'onpause']) {
    assert.strictEqual(
      players[0][h],
      null,
      h + ' cleared on the retired player',
    );
  }
  players[0].fire('ended');
  players[0].fire('pause');
  assert.strictEqual(players.length, 2, 'a retired player cannot re-pump');
  assert.notStrictEqual(players[1].src, '', 'the next clip keeps its source');
  // Natural completion fires pause THEN ended: the queue advances once.
  talk(win, 'c');
  assert.strictEqual(players.length, 2, 'third clip waits for the second');
  players[1].fire('pause');
  players[1].fire('ended');
  assert.strictEqual(players.length, 3, 'exactly one clip started');
  assert.strictEqual(players[2].playCalls, 1);
  talk(win, 'd');
  assert.strictEqual(players.length, 3, 'pause+ended did not double-pump');
  win.close();
  console.log('PASS a UA pause releases the talk queue');
}

// --- F3 ---------------------------------------------------------------------

function testPollersPauseWhileDisconnected() {
  // remote-desktop: the task-info panel is always docked, so the
  // task-update poll is wanted whenever the tab's task runs.
  const {win, posted, intervals} = makePage(true);
  send(win, {type: 'configData', config: {}, apiKeys: {}});
  daemon(win, true);
  send(win, {type: 'status', running: true});
  assert.ok(
    msgsOf(posted, 'getTaskUpdate').length >= 1,
    'a running task starts the task-update poll',
  );
  const metaTick = intervals.find(i => i.ms === 5000);
  assert.ok(metaTick, 'the 5 s task-update interval is armed');
  // The daemon's first sidebar reply arms the 30 s sidebar poll.
  send(win, {type: 'cronJobs', jobs: []});
  const sidebarTick = intervals.find(i => i.ms === 30000);
  assert.ok(sidebarTick, 'the 30 s sidebar-panels interval is armed');

  posted.length = 0;
  metaTick.cb();
  sidebarTick.cb();
  assert.strictEqual(msgsOf(posted, 'getTaskUpdate').length, 1);
  assert.strictEqual(msgsOf(posted, 'getCronJobs').length, 1);
  assert.strictEqual(msgsOf(posted, 'getAppsStatus').length, 1);
  assert.strictEqual(msgsOf(posted, 'getSpendReport').length, 1);

  // Outage: the ticks keep firing but must post nothing.
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  posted.length = 0;
  for (let i = 0; i < 5; i++) {
    metaTick.cb();
    sidebarTick.cb();
  }
  assert.deepStrictEqual(
    posted.map(m => m.type),
    [],
    'REGRESSION: pollers must not post while the daemon is down ' +
      '(the shim queues every post and bursts them out ahead of ready)',
  );

  // Reconnect: the connected handler refreshes the sidebar itself and
  // the next ticks post again.
  daemon(win, true);
  posted.length = 0;
  metaTick.cb();
  sidebarTick.cb();
  assert.strictEqual(msgsOf(posted, 'getTaskUpdate').length, 1);
  assert.strictEqual(msgsOf(posted, 'getCronJobs').length, 1);
  win.close();
  console.log('PASS pollers pause while disconnected and resume after');
}

// --- F4 ---------------------------------------------------------------------

// The user's close: only the chat on screen has an entry (with a close
// button) on the group strip, so the chat is brought on screen first.
function closeTabViaUi(win, tabId) {
  win._testApi.switchToTab(tabId);
  clickClose(win, tabId);
}

function testStaleSnapshotDoesNotResurrectClosedTab() {
  const {win, posted} = makePage();
  win._testApi.endLaunch();
  send(win, {
    type: 'tabs_state',
    tabs: [entry('t1'), entry('t2'), entry('t3')],
  });
  assert.deepStrictEqual(tabIds(win), ['t1', 't2', 't3']);
  posted.length = 0;
  closeTabViaUi(win, 't2');
  assert.deepStrictEqual(
    msgsOf(posted, 'closeTab').map(m => m.tabId),
    ['t2'],
  );
  assert.deepStrictEqual(tabIds(win), ['t1', 't3']);

  // A snapshot the daemon broadcast BEFORE it processed the close
  // (another client's openTab/title change) still lists t2.
  send(win, {
    type: 'tabs_state',
    tabs: [entry('t1'), entry('t2', 'renamed'), entry('t3')],
  });
  assert.deepStrictEqual(
    tabIds(win),
    ['t1', 't3'],
    'REGRESSION: a stale tabs_state in flight must not resurrect the ' +
      'tab the user just closed',
  );
  // The post-close snapshot omits it: the shield is lifted, and a
  // later snapshot listing t2 again (another client reopened that id)
  // shows it.
  send(win, {type: 'tabs_state', tabs: [entry('t1'), entry('t3')]});
  assert.deepStrictEqual(tabIds(win), ['t1', 't3']);
  send(win, {
    type: 'tabs_state',
    tabs: [entry('t1'), entry('t2'), entry('t3')],
  });
  assert.deepStrictEqual(
    tabIds(win),
    ['t1', 't2', 't3'],
    'once a snapshot confirmed the close, a listed t2 is a real tab',
  );
  win.close();
  console.log('PASS stale snapshot does not resurrect a closed tab');
}

function testCloseShieldExpiresWhenDaemonKeepsTab() {
  // The daemon kept the tab (it refused or lost the close): after a
  // few snapshots that still list it, the tab comes back rather than
  // staying hidden on this client for ever.
  const {win} = makePage();
  win._testApi.endLaunch();
  const snap = {type: 'tabs_state', tabs: [entry('t1'), entry('t2')]};
  send(win, snap);
  closeTabViaUi(win, 't2');
  assert.deepStrictEqual(tabIds(win), ['t1']);
  send(win, snap);
  send(win, snap);
  assert.deepStrictEqual(tabIds(win), ['t1'], 'still shielded after 2');
  send(win, snap);
  assert.deepStrictEqual(
    tabIds(win),
    ['t1', 't2'],
    'the third snapshot still listing the tab lifts the shield',
  );
  win.close();
  console.log('PASS close shield expires after repeated snapshots');
}

function testCloseShieldIsDroppedOnReconnect() {
  // The close went out, then the connection dropped before the
  // confirming snapshot arrived; meanwhile another surface re-opened
  // the same id. The reconnect's snapshot is authoritative and must
  // show that tab, not be skipped as a stale pre-close echo.
  const {win, posted} = makePage();
  win._testApi.endLaunch();
  daemon(win, true);
  send(win, {type: 'tabs_state', tabs: [entry('t1'), entry('t2')]});
  closeTabViaUi(win, 't2');
  assert.deepStrictEqual(tabIds(win), ['t1']);
  posted.length = 0;
  daemon(win, false);
  daemon(win, true);
  assert.strictEqual(msgsOf(posted, 'ready').length, 1, 'reconnect re-readies');
  send(win, {
    type: 'tabs_state',
    tabs: [entry('t1'), entry('t2', 'reopened')],
  });
  assert.deepStrictEqual(
    tabIds(win),
    ['t1', 't2'],
    'REGRESSION: the close shield survived the outage and hid a tab ' +
      'another surface re-opened under the same id',
  );
  win.close();
  console.log('PASS the close shield is dropped on reconnect');
}

function testCloseShieldExpiresAfterTtl() {
  // Same divergence without a visible outage (the confirming snapshot
  // was simply never delivered): the shield is a stale-echo defence
  // measured in milliseconds, so it lapses on its own after 5 s.
  const {win} = makePage();
  win._testApi.endLaunch();
  const snap = {type: 'tabs_state', tabs: [entry('t1'), entry('t2')]};
  send(win, snap);
  const realNow = win.Date.now;
  let now = realNow();
  win.Date.now = () => now;
  closeTabViaUi(win, 't2');
  now += 4000;
  send(win, snap);
  assert.deepStrictEqual(tabIds(win), ['t1'], 'still shielded at 4 s');
  now += 2000;
  send(win, snap);
  assert.deepStrictEqual(
    tabIds(win),
    ['t1', 't2'],
    'REGRESSION: a close shield older than 5 s must not hide a listed tab',
  );
  win.Date.now = realNow;
  win.close();
  console.log('PASS the close shield expires after 5 s');
}

function testServerCloseNeedsNoShield() {
  // A close mirrored FROM the daemon (closeSubagentTab / another
  // client's close arriving as a snapshot) posts no closeTab and arms
  // no shield: the next snapshot is authoritative as before.
  const {win, posted} = makePage();
  win._testApi.endLaunch();
  send(win, {type: 'tabs_state', tabs: [entry('t1'), entry('t2')]});
  posted.length = 0;
  send(win, {type: 'tabs_state', tabs: [entry('t1')]});
  assert.deepStrictEqual(tabIds(win), ['t1']);
  assert.strictEqual(msgsOf(posted, 'closeTab').length, 0);
  send(win, {type: 'tabs_state', tabs: [entry('t1'), entry('t2')]});
  assert.deepStrictEqual(tabIds(win), ['t1', 't2']);
  win.close();
  console.log('PASS a server-side removal is not shielded');
}

const tests = [
  testColdStartDoesNotDoubleReady,
  testRealReconnectStillResendsReady,
  testSubmitStillBlockedWhileDownOnColdStart,
  testUaPauseReleasesTalkQueue,
  testPollersPauseWhileDisconnected,
  testStaleSnapshotDoesNotResurrectClosedTab,
  testCloseShieldExpiresWhenDaemonKeepsTab,
  testCloseShieldIsDroppedOnReconnect,
  testCloseShieldExpiresAfterTtl,
  testServerCloseNeedsNoShield,
];

let failed = 0;
for (const t of tests) {
  try {
    t();
  } catch (e) {
    failed++;
    console.log('FAIL ' + t.name);
    console.log(e && e.stack ? e.stack : String(e));
  }
}
if (failed > 0) {
  console.log(failed + ' test(s) failed');
  process.exit(1);
}
console.log('all mainJsDaemonReconnectRaces tests passed');
