// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end (jsdom, real media/main.js + api.js) regressions for the
// daemon-connection races found by the media JS audit
// (tmp/audit-I-vscode-media.md, cluster 7):
//
// F1  A cold start (daemonStatus connected:false BEFORE the daemon was
//     ever seen up) must not re-send `ready` once the daemon comes up:
//     the boot ready is queued by the host/shim and flushed on auth, so
//     a second one replays every transcript twice.  A real reconnect
//     (up -> down -> up) still re-sends it, and a boot ready older than
//     the host's 10 s queue TTL (dropped as expired, silently) is
//     re-sent on the first connect.
// F2  The talk playback queue must not wedge when the user agent pauses
//     a clip without ever firing `ended` (tab backgrounded, incoming
//     call): `pause` releases the queue so later clips still play, and
//     the paused player is retired (source dropped) so a resume cannot
//     overlap the next clip.
// F3  requestTaskUpdate and the sidebar-panels poll tick must not post
//     while daemonStatus is disconnected (the host/shim would queue the
//     posts and burst them out on reconnect, ahead of `ready`).
// F4  A tab the user just closed must not be resurrected by a stale
//     tabs_state broadcast before the daemon processed the close -- but
//     that shield lapses after 5 s and on reconnect, so a tab another
//     surface re-opened under the same id is never hidden.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(bodyAttrs) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  if (bodyAttrs) html = html.replace('<body', '<body' + bodyAttrs);

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.cancelAnimationFrame = function () {};
  // Intervals are recorded, not run, so a test fires a poll tick by hand.
  const intervals = [];
  win.setInterval = function (fn, ms) {
    intervals.push({fn, ms});
    return intervals.length;
  };
  win.clearInterval = function () {};
  win.matchMedia = function (query) {
    return {
      matches: query === '(min-width: 900px)',
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
  };

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=mainjs-reconnect-races-main.js',
  );
  return {win, posted, intervals};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function msgsOf(posted, type) {
  return posted.filter(m => m && m.type === type);
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
  const {win, posted} = makeWebview();
  assert.strictEqual(msgsOf(posted, 'ready').length, 1, 'boot posts one ready');
  // The host answers the boot ready with "daemon not up yet" (the
  // extension is still launching it), then the daemon comes up: the
  // queued boot ready is flushed by the host, so no second one.
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
  const {win, posted} = makeWebview();
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

function testExpiredBootReadyIsResentOnFirstConnect() {
  const {win, posted} = makeWebview();
  assert.strictEqual(msgsOf(posted, 'ready').length, 1);
  daemon(win, false);
  // The VS Code host drops a queued command older than 10 s as
  // `expired` without telling the webview: the daemon took that long
  // to come up, so the boot ready never reached it.
  const realNow = win.Date.now;
  const t0 = realNow();
  win.Date.now = () => t0 + 10001;
  try {
    daemon(win, true);
  } finally {
    win.Date.now = realNow;
  }
  assert.strictEqual(
    msgsOf(posted, 'ready').length,
    2,
    'a boot ready older than the host queue TTL must be re-sent on the ' +
      'first connect, or the daemon never syncs this window',
  );
  // That first connect counts as "seen up": a later bare repeat is a
  // no-op, an outage re-sends.
  daemon(win, true);
  assert.strictEqual(msgsOf(posted, 'ready').length, 2);
  daemon(win, false);
  daemon(win, true);
  assert.strictEqual(msgsOf(posted, 'ready').length, 3);
  win.close();
  console.log('PASS expired boot ready is re-sent on the first connect');
}

function testRemoteShimQueueNeverExpiresSoNoLateResend() {
  // The remote webapp's shim keeps queued posts for ever (and even
  // re-sends unconfirmed batches), so a slow first connect there --
  // e.g. the user took a while at the password prompt -- must not
  // produce a second ready.
  const {win, posted} = makeWebview(' class="remote-chat"');
  assert.strictEqual(msgsOf(posted, 'ready').length, 1);
  daemon(win, false);
  const realNow = win.Date.now;
  const t0 = realNow();
  win.Date.now = () => t0 + 60000;
  try {
    daemon(win, true);
  } finally {
    win.Date.now = realNow;
  }
  assert.strictEqual(
    msgsOf(posted, 'ready').length,
    1,
    'the remote surface never re-sends the queued boot ready',
  );
  win.close();
  console.log('PASS remote cold start never re-sends ready');
}

function testSubmitStillBlockedWhileDownOnColdStart() {
  // The "currently disconnected" guards moved off daemonWasDown: a
  // cold-start disconnect must still keep a submit out of the host
  // queue, and the reconnect must let it through again.
  const {win, posted} = makeWebview();
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

function installAudio(win) {
  const players = [];
  win.Audio = function Audio(src) {
    this.src = src;
    this.playCalls = 0;
    this.play = () => {
      this.playCalls++;
      return Promise.resolve();
    };
    this.fire = type => {
      const handler = this['on' + type];
      if (typeof handler === 'function') handler({type});
    };
    players.push(this);
  };
  return players;
}

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
  const {win} = makeWebview();
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
    assert.strictEqual(players[0][h], null, h + ' cleared on the retired player');
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
  const {win, posted, intervals} = makeWebview(
    ' class="remote-chat remote-desktop"',
  );
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
  metaTick.fn();
  sidebarTick.fn();
  assert.strictEqual(msgsOf(posted, 'getTaskUpdate').length, 1);
  assert.strictEqual(msgsOf(posted, 'getCronJobs').length, 1);
  assert.strictEqual(msgsOf(posted, 'getAppsStatus').length, 1);
  assert.strictEqual(msgsOf(posted, 'getSpendReport').length, 1);

  // Outage: the ticks keep firing but must post nothing.
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  posted.length = 0;
  for (let i = 0; i < 5; i++) {
    metaTick.fn();
    sidebarTick.fn();
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
  metaTick.fn();
  sidebarTick.fn();
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
  const btn = win.document.querySelector(
    '#tab-list .chat-tab[data-tab-id="' + tabId + '"] .chat-tab-close',
  );
  assert.ok(btn, 'tab ' + tabId + ' has a close button');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function testStaleSnapshotDoesNotResurrectClosedTab() {
  const {win, posted} = makeWebview();
  win._testApi.endLaunch();
  send(win, {type: 'tabs_state', tabs: [entry('t1'), entry('t2'), entry('t3')]});
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
  send(win, {type: 'tabs_state', tabs: [entry('t1'), entry('t2'), entry('t3')]});
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
  const {win} = makeWebview();
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
  const {win, posted} = makeWebview();
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
  const {win} = makeWebview();
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
  const {win, posted} = makeWebview();
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
  testExpiredBootReadyIsResentOnFirstConnect,
  testRemoteShimQueueNeverExpiresSoNoLateResend,
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
