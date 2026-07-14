// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression tests: the REMOTE webapp's History panel must
// re-synchronise after a WebSocket reconnect so a task started while
// the socket was down (or whose start-time ``tasks_updated`` broadcast
// was lost in flight) still appears in the panel.
//
// User-reported bug: "The task history panel is not getting updated
// as soon as kiss-web starts a task ... the problem still remains when
// a task is sent from a remote mobile device from the webapp."
//
// Root cause: the start-time ``{"type": "tasks_updated"}`` broadcast
// by ``ChatSorcarAgent.run`` is a ONE-SHOT, never-persisted global
// system event (``WebPrinter.broadcast``, web_server.py — "Not
// recorded, not persisted").  Mobile Safari routinely kills the
// remote webapp's WebSocket when the phone backgrounds or switches
// apps; if the socket is down at the instant of the broadcast the
// event is LOST FOREVER.  On reconnect the shim synthesises
// ``daemonStatus connected:true`` — but ``media/main.js`` only called
// ``setServerLoading(!ev.connected)`` for that event, so an
// already-open History sidebar (ALWAYS open on the ≥900px remote
// desktop layout, and commonly open on the mobile drawer) kept
// showing a stale list without the just-started running task.
//
// Requirements driven by this test:
//
//   1. ``daemonStatus connected:true`` must call ``refreshHistory()``
//      so an OPEN sidebar refetches history after every reconnect
//      (remote desktop docked sidebar AND remote mobile drawer).
//
//   2. The refetch must use a FRESH ``generation`` so replies to
//      pre-disconnect ``getHistory`` requests are dropped.
//
//   3. ``daemonStatus connected:false`` must NOT post ``getHistory``.
//
//   4. When the sidebar is CLOSED, ``daemonStatus connected:true``
//      must NOT post ``getHistory`` (``refreshHistory`` is a no-op —
//      opening the drawer already posts a fresh request).
//
//   5. The full remote-mobile lifecycle converges: drawer open →
//      reconnect → refetch → running-row reply renders VISIBLE.
//
//   6. ``tasks_updated`` while the sidebar is open still refetches
//      (regression guard for the pre-existing live path).
//
//   7. The VS Code extension webview (no body.remote-chat) shares
//      main.js: ``daemonStatus connected:true`` with a closed sidebar
//      must stay a no-op there too (no ``getHistory`` spam on every
//      extension-host daemon reconnect).
//
// The tests run the real media/chat.html + panelCopy.js + main.js in
// jsdom (same harness as remoteDesktopSidebar.test.js) with a
// controllable window.matchMedia stub installed BEFORE main.js.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/remoteHistoryReconnectRefresh.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom webview running the real chat.html + panelCopy.js +
 * main.js.
 *
 * @param {object} [opts]
 * @param {boolean} [opts.remote=true]  add class="remote-chat" to body.
 * @param {boolean} [opts.desktopMatches=false]  initial
 *     matchMedia('(min-width: 900px)').matches.
 * @returns {{win: object, posted: Array}}
 */
function makeWebview(opts) {
  const {remote = true, desktopMatches = false} = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  if (remote) {
    html = html.replace('<body', '<body class="remote-chat"');
  }
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
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
  // Controllable matchMedia stub — installed BEFORE main.js runs so
  // the remote-desktop wiring sees it during initialisation.  jsdom
  // does not implement matchMedia natively.
  win.matchMedia = function (query) {
    return {
      matches: query === '(min-width: 900px)' && desktopMatches,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

/** Deliver a backend→webview event exactly like the shim/host does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function getHistoryMsgs(posted) {
  return posted.filter(m => m.type === 'getHistory');
}

/** Return the most recent posted ``getHistory`` command, or null. */
function lastGetHistory(posted) {
  const msgs = getHistoryMsgs(posted);
  return msgs.length ? msgs[msgs.length - 1] : null;
}

/** Open the mobile History drawer via the burger button. */
function openDrawer(win) {
  const btn = win.document.getElementById('menu-btn');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    win.document.getElementById('sidebar').classList.contains('open'),
    'burger click must open the drawer',
  );
}

/** Find a rendered history row (visible or hidden) by its title. */
function rowByTitle(win, title) {
  const rows = win.document.querySelectorAll('#history-list .sidebar-item');
  for (const r of rows) {
    const t = r.querySelector('.sidebar-item-text');
    if (t && t.textContent === title) return r;
  }
  return null;
}

// ---------------------------------------------------------------------------
// 1. REMOTE DESKTOP (docked, always-open sidebar): a WebSocket
//    reconnect (daemonStatus false → true) must refetch history.
//    This is THE regression: before the fix only setServerLoading ran.
// ---------------------------------------------------------------------------
function testDesktopReconnectRefetchesHistory() {
  const {win, posted} = makeWebview({remote: true, desktopMatches: true});
  const sidebar = win.document.getElementById('sidebar');
  assert.ok(
    sidebar.classList.contains('open'),
    'remote desktop docks the sidebar open on load',
  );
  assert.ok(
    getHistoryMsgs(posted).length >= 1,
    'docking on load posts the initial getHistory',
  );
  // Socket drops (phone sleeps / app switch / server restart) ...
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: false});
  // ... and comes back.
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

// ---------------------------------------------------------------------------
// 2. REMOTE MOBILE (drawer open): reconnect must refetch with a FRESH
//    generation so stale pre-disconnect replies are dropped.
// ---------------------------------------------------------------------------
function testMobileReconnectRefetchesWithNewGeneration() {
  const {win, posted} = makeWebview({remote: true, desktopMatches: false});
  openDrawer(win);
  const before = lastGetHistory(posted);
  assert.ok(before, 'opening the drawer posts getHistory');
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  const after = lastGetHistory(posted);
  assert.ok(
    after,
    'REGRESSION: reconnect must refetch history for the open mobile drawer',
  );
  assert.ok(
    after.generation > before.generation,
    'the reconnect refetch must bump the history generation so replies ' +
      'to pre-disconnect getHistory requests are dropped ' +
      `(got ${after.generation}, previous ${before.generation})`,
  );
  win.close();
  console.log('PASS remote-mobile reconnect refetches with new generation');
}

// ---------------------------------------------------------------------------
// 3. daemonStatus connected:false must NOT post getHistory (the socket
//    is down — the request could not be answered anyway).
// ---------------------------------------------------------------------------
function testDisconnectDoesNotFetch() {
  const {win, posted} = makeWebview({remote: true, desktopMatches: true});
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

// ---------------------------------------------------------------------------
// 4. Sidebar CLOSED (remote mobile default): reconnect must NOT post
//    getHistory — refreshHistory is a no-op, and opening the drawer
//    posts a fresh request anyway.
// ---------------------------------------------------------------------------
function testClosedSidebarReconnectNoFetch() {
  const {win, posted} = makeWebview({remote: true, desktopMatches: false});
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

// ---------------------------------------------------------------------------
// 5. FULL remote-mobile lifecycle: drawer open showing an old task →
//    socket drops → a task starts on the server (its tasks_updated is
//    lost) → socket reconnects → the refetch delivers the running row
//    → the row renders VISIBLE (running rows always pass the
//    Workspace filter per the b49ef08a fix).
// ---------------------------------------------------------------------------
function testMobileLifecycleShowsTaskStartedWhileDisconnected() {
  const {win, posted} = makeWebview({remote: true, desktopMatches: false});
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
  // Phone backgrounds; Safari kills the WebSocket.
  send(win, {type: 'daemonStatus', connected: false});
  // While the socket is down, kiss-web starts a task and broadcasts
  // tasks_updated — the event NEVER reaches this client (lost).
  // The user returns; the shim reconnects and synthesises:
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: true});
  const refetch = lastGetHistory(posted);
  assert.ok(refetch, 'reconnect must refetch history');
  // The server replies with the CURRENT list, including the task
  // started while this client was offline (running, resolved-variant
  // work_dir — the worst case).
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

// ---------------------------------------------------------------------------
// 5b. STALE-REPLY guard: after the reconnect refetch bumps the
//     generation, a late reply to a PRE-disconnect getHistory (old
//     generation) must be dropped, not rendered over the fresh list.
// ---------------------------------------------------------------------------
function testStaleGenerationReplyIsDropped() {
  const {win, posted} = makeWebview({remote: true, desktopMatches: false});
  openDrawer(win);
  const stale = lastGetHistory(posted);
  assert.ok(stale, 'opening the drawer posts getHistory');
  // Socket drops before the reply arrives, then reconnects; the
  // reconnect refetch bumps historyGeneration.
  send(win, {type: 'daemonStatus', connected: false});
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: true});
  const fresh = lastGetHistory(posted);
  assert.ok(fresh.generation > stale.generation, 'refetch bumps generation');
  // The FRESH reply arrives first and renders the current list.
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
  // A LATE reply to the pre-disconnect request (old generation, stale
  // list without the running task) straggles in — it must be ignored.
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

// ---------------------------------------------------------------------------
// 6. Live path regression guard: tasks_updated while the sidebar is
//    open still refetches history.
// ---------------------------------------------------------------------------
function testTasksUpdatedStillRefetches() {
  const {win, posted} = makeWebview({remote: true, desktopMatches: true});
  posted.length = 0;
  send(win, {type: 'tasks_updated', taskId: ''});
  assert.ok(
    lastGetHistory(posted),
    'tasks_updated must still refetch history while the sidebar is open',
  );
  win.close();
  console.log('PASS tasks_updated still refetches while open');
}

// ---------------------------------------------------------------------------
// 7. Isolation: the VS Code extension webview (no remote-chat class)
//    gets daemonStatus connected:true from the extension host on every
//    webview ready / daemon reconnect — with the sidebar closed this
//    must NOT post getHistory (no request spam).
// ---------------------------------------------------------------------------
function testVsCodeWebviewClosedSidebarNoFetch() {
  const {win, posted} = makeWebview({remote: false, desktopMatches: false});
  assert.ok(
    !win.document.getElementById('sidebar').classList.contains('open'),
    'VS Code webview sidebar starts closed',
  );
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: true});
  assert.strictEqual(
    getHistoryMsgs(posted).length,
    0,
    'VS Code webview: daemonStatus connected:true with a closed ' +
      'sidebar must not post getHistory',
  );
  win.close();
  console.log('PASS VS Code webview closed sidebar posts no getHistory');
}

// ---------------------------------------------------------------------------
// 8. VS Code extension webview with the sidebar OPEN: daemonStatus
//    connected:true (posted by the extension host on daemon reconnect
//    and on webview ready) must ALSO refetch — the extension-host UDS
//    path never resends ``ready``, so this dispatch is the only
//    resync trigger after an extension-side daemon reconnect.
// ---------------------------------------------------------------------------
function testVsCodeWebviewOpenSidebarRefetches() {
  const {win, posted} = makeWebview({remote: false, desktopMatches: false});
  openDrawer(win);
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  assert.ok(
    lastGetHistory(posted),
    'VS Code webview: daemon reconnect must refetch history for an ' +
      'OPEN sidebar (the UDS path has no ready-time nudge)',
  );
  win.close();
  console.log('PASS VS Code webview open sidebar refetches on reconnect');
}

testDesktopReconnectRefetchesHistory();
testMobileReconnectRefetchesWithNewGeneration();
testDisconnectDoesNotFetch();
testClosedSidebarReconnectNoFetch();
testMobileLifecycleShowsTaskStartedWhileDisconnected();
testStaleGenerationReplyIsDropped();
testTasksUpdatedStillRefetches();
testVsCodeWebviewClosedSidebarNoFetch();
testVsCodeWebviewOpenSidebarRefetches();
console.log('All remoteHistoryReconnectRefresh tests passed.');
