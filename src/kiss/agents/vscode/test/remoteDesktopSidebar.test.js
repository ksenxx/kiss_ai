// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests: DESKTOP layout of the REMOTE webapp.
//
// Feature under test: on desktop-wide browser windows (matchMedia
// ``(min-width: 900px)``) the remote webapp (body.remote-chat) must
// dock the agent-history sidebar (#sidebar, the burger-menu drawer)
// persistently on the LEFT of the window:
//
//   * on load the body gains the ``remote-desktop`` class, the sidebar
//     gains ``open`` and a ``getHistory`` request is posted so the
//     panel is populated immediately;
//   * the dark ``#sidebar-overlay`` is never shown while docked;
//   * Escape, overlay clicks and history-item clicks must NOT undock
//     the sidebar (mobile drawer semantics do not apply);
//   * the burger #menu-btn and the #sidebar-close button still toggle
//     the docked sidebar (explicit user intent wins);
//   * resizing across the breakpoint docks/undocks live;
//   * NONE of this activates outside body.remote-chat (the VS Code
//     extension webview shares main.js) nor when window.matchMedia is
//     unavailable.
//
// The tests run the real media/chat.html + panelCopy.js + main.js in
// jsdom (same harness as adjacentTaskScroll.test.js) with a
// controllable window.matchMedia stub installed BEFORE main.js is
// evaluated.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/remoteDesktopSidebar.test.js

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
 * @param {boolean} [opts.remote=true]   add class="remote-chat" to body
 * @param {boolean|null} [opts.desktopMatches=true]  initial
 *     matchMedia('(min-width: 900px)').matches; ``null`` = do NOT
 *     install a matchMedia stub at all (jsdom has none natively).
 * @returns {{win: object, posted: Array, fireChange: function(boolean)}}
 */
function makeWebview(opts) {
  const {remote = true, desktopMatches = true} = opts || {};
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
      postMessage: (msg) => posted.push(msg),
      getState: () => state,
      setState: (s) => {
        state = s;
      },
    };
  };
  // Controllable matchMedia stub — installed BEFORE main.js runs so
  // the wiring sees it during initialisation, exactly like a real
  // browser.  jsdom does not implement matchMedia natively.
  const listeners = [];
  const mql = {
    matches: desktopMatches === true,
    media: '(min-width: 900px)',
    addEventListener: (ev, fn) => {
      if (ev === 'change') listeners.push(fn);
    },
    removeEventListener: () => {},
    addListener: (fn) => listeners.push(fn),
    removeListener: () => {},
  };
  if (desktopMatches !== null) {
    // main.js registers on the object returned for '(min-width: 900px)'.
    // Return mql itself for that query so fireChange reaches the
    // registered handler and .matches updates are observed.
    win.matchMedia = function (query) {
      if (query === '(min-width: 900px)') return mql;
      return {
        matches: false,
        media: query,
        addEventListener: () => {},
        removeEventListener: () => {},
        addListener: () => {},
        removeListener: () => {},
      };
    };
  }
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  function fireChange(matches) {
    mql.matches = matches;
    listeners.forEach((fn) => fn(mql));
  }
  return {win, posted, fireChange};
}

/** Deliver a message-event from the extension host to the webview. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function getHistoryMsgs(posted) {
  return posted.filter((m) => m.type === 'getHistory');
}

// ---------------------------------------------------------------------------
// 1. Remote + desktop: on load the sidebar is docked open on the left,
//    the body carries remote-desktop, getHistory is posted, and the
//    overlay stays hidden.
// ---------------------------------------------------------------------------
function testDesktopDocksSidebarOnLoad() {
  const {win, posted} = makeWebview({remote: true, desktopMatches: true});
  const body = win.document.body;
  const sidebar = win.document.getElementById('sidebar');
  const overlay = win.document.getElementById('sidebar-overlay');
  assert.ok(
    body.classList.contains('remote-desktop'),
    'desktop remote page must gain the remote-desktop body class',
  );
  assert.ok(
    sidebar.classList.contains('open'),
    'history sidebar must be docked open on desktop load',
  );
  assert.ok(
    !overlay.classList.contains('open'),
    'the dark overlay must stay hidden while the sidebar is docked',
  );
  assert.ok(
    getHistoryMsgs(posted).length >= 1,
    'docking on load must post getHistory so the panel is populated',
  );
  win.close();
  console.log('PASS desktop remote page docks the history sidebar on load');
}

// ---------------------------------------------------------------------------
// 2. Remote + mobile: nothing docks; drawer semantics preserved.
// ---------------------------------------------------------------------------
function testMobileKeepsDrawerClosed() {
  const {win, posted} = makeWebview({remote: true, desktopMatches: false});
  const body = win.document.body;
  const sidebar = win.document.getElementById('sidebar');
  assert.ok(
    !body.classList.contains('remote-desktop'),
    'narrow remote page must not gain remote-desktop',
  );
  assert.ok(
    !sidebar.classList.contains('open'),
    'sidebar must stay closed on narrow remote page load',
  );
  assert.strictEqual(
    getHistoryMsgs(posted).length,
    0,
    'no auto getHistory on narrow load',
  );
  win.close();
  console.log('PASS narrow remote page keeps the drawer closed');
}

// ---------------------------------------------------------------------------
// 3. Live resize across the breakpoint docks and undocks.
// ---------------------------------------------------------------------------
function testResizeAcrossBreakpoint() {
  const {win, posted, fireChange} = makeWebview({
    remote: true,
    desktopMatches: false,
  });
  const body = win.document.body;
  const sidebar = win.document.getElementById('sidebar');
  assert.ok(!sidebar.classList.contains('open'));
  // Grow the window: mobile → desktop.
  fireChange(true);
  assert.ok(body.classList.contains('remote-desktop'), 'resize wide → dock');
  assert.ok(sidebar.classList.contains('open'), 'resize wide → sidebar open');
  assert.ok(
    getHistoryMsgs(posted).length >= 1,
    'docking after resize must post getHistory',
  );
  // Shrink back: desktop → mobile.
  fireChange(false);
  assert.ok(
    !body.classList.contains('remote-desktop'),
    'resize narrow → remote-desktop removed',
  );
  assert.ok(
    !sidebar.classList.contains('open'),
    'resize narrow → sidebar closes back into drawer mode',
  );
  win.close();
  console.log('PASS resizing across the 900px breakpoint docks/undocks');
}

// ---------------------------------------------------------------------------
// 4. Isolation: the VS Code extension webview (no remote-chat class)
//    must never dock, even on a desktop-wide window.
// ---------------------------------------------------------------------------
function testVsCodeWebviewIsolation() {
  const {win, posted} = makeWebview({remote: false, desktopMatches: true});
  const body = win.document.body;
  const sidebar = win.document.getElementById('sidebar');
  assert.ok(
    !body.classList.contains('remote-desktop'),
    'non-remote webview must never gain remote-desktop',
  );
  assert.ok(
    !sidebar.classList.contains('open'),
    'non-remote webview sidebar must stay closed',
  );
  assert.strictEqual(getHistoryMsgs(posted).length, 0);
  win.close();
  console.log('PASS VS Code webview (no remote-chat) is unaffected');
}

// ---------------------------------------------------------------------------
// 5. No matchMedia at all (older embedder): main.js must not crash and
//    the webview must still boot (ready message posted).
// ---------------------------------------------------------------------------
function testNoMatchMediaNoCrash() {
  const {win, posted} = makeWebview({remote: true, desktopMatches: null});
  assert.ok(
    posted.find((m) => m.type === 'ready'),
    'webview must boot without window.matchMedia',
  );
  assert.ok(
    !win.document.body.classList.contains('remote-desktop'),
    'without matchMedia the desktop mode must stay off',
  );
  win.close();
  console.log('PASS missing window.matchMedia does not crash main.js');
}

// ---------------------------------------------------------------------------
// 6. While docked: Escape and overlay clicks must NOT undock.
// ---------------------------------------------------------------------------
function testEscapeAndOverlayDoNotUndock() {
  const {win} = makeWebview({remote: true, desktopMatches: true});
  const sidebar = win.document.getElementById('sidebar');
  const overlay = win.document.getElementById('sidebar-overlay');
  assert.ok(sidebar.classList.contains('open'));
  win.document.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
  );
  assert.ok(
    sidebar.classList.contains('open'),
    'Escape must not undock the desktop sidebar',
  );
  overlay.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    sidebar.classList.contains('open'),
    'overlay click must not undock the desktop sidebar',
  );
  win.close();
  console.log('PASS Escape / overlay click keep the docked sidebar open');
}

// ---------------------------------------------------------------------------
// 7. While docked: clicking a history row loads the chat but keeps the
//    sidebar docked (Codex/ChatGPT desktop behavior).
// ---------------------------------------------------------------------------
function testHistoryClickKeepsDock() {
  const {win, posted} = makeWebview({remote: true, desktopMatches: true});
  const sidebar = win.document.getElementById('sidebar');
  const gen = getHistoryMsgs(posted).pop().generation;
  send(win, {
    type: 'history',
    generation: gen,
    offset: 0,
    sessions: [
      {
        id: 'chat-1',
        task_id: '7',
        title: 'Old task',
        preview: 'Old task',
        has_events: true,
        ts: Date.now() / 1000,
      },
    ],
  });
  const row = win.document.querySelector('#history-list .sidebar-item');
  assert.ok(row, 'history row must render');
  row.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    posted.find((m) => m.type === 'resumeSession'),
    'clicking the row must resume the session',
  );
  assert.ok(
    sidebar.classList.contains('open'),
    'history click must NOT close the docked desktop sidebar',
  );
  win.close();
  console.log('PASS history-row click keeps the desktop sidebar docked');
}

// ---------------------------------------------------------------------------
// 8. On MOBILE the history click still closes the drawer (regression
//    guard for the pre-existing behavior).
// ---------------------------------------------------------------------------
function testHistoryClickStillClosesOnMobile() {
  const {win, posted} = makeWebview({remote: true, desktopMatches: false});
  const sidebar = win.document.getElementById('sidebar');
  const menuBtn = win.document.getElementById('menu-btn');
  menuBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(sidebar.classList.contains('open'), 'burger opens the drawer');
  const gen = getHistoryMsgs(posted).pop().generation;
  send(win, {
    type: 'history',
    generation: gen,
    offset: 0,
    sessions: [
      {
        id: 'chat-1',
        task_id: '7',
        title: 'Old task',
        preview: 'Old task',
        has_events: true,
        ts: Date.now() / 1000,
      },
    ],
  });
  const row = win.document.querySelector('#history-list .sidebar-item');
  row.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    !sidebar.classList.contains('open'),
    'on mobile a history click must still close the drawer',
  );
  win.close();
  console.log('PASS mobile history-row click still closes the drawer');
}

// ---------------------------------------------------------------------------
// 9. Explicit user intent: the burger #menu-btn toggles the docked
//    sidebar closed and open again (re-open re-fetches history), and
//    the #sidebar-close button also hides it.
// ---------------------------------------------------------------------------
function testMenuAndCloseButtonsStillToggle() {
  const {win, posted} = makeWebview({remote: true, desktopMatches: true});
  const sidebar = win.document.getElementById('sidebar');
  const menuBtn = win.document.getElementById('menu-btn');
  const closeBtn = win.document.getElementById('sidebar-close');
  const overlay = win.document.getElementById('sidebar-overlay');
  assert.ok(sidebar.classList.contains('open'));
  // Burger hides the docked sidebar.
  menuBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    !sidebar.classList.contains('open'),
    'burger click must hide the docked sidebar (explicit intent)',
  );
  const before = getHistoryMsgs(posted).length;
  // Burger shows it again and re-fetches history.
  menuBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(sidebar.classList.contains('open'), 'burger re-opens');
  assert.ok(
    !overlay.classList.contains('open'),
    're-opening on desktop must not show the dark overlay',
  );
  assert.ok(
    getHistoryMsgs(posted).length > before,
    're-opening must post a fresh getHistory',
  );
  // The in-panel close button also hides it.
  closeBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    !sidebar.classList.contains('open'),
    'sidebar-close button must hide the docked sidebar',
  );
  win.close();
  console.log('PASS burger / close button explicitly toggle the dock');
}

// ---------------------------------------------------------------------------
// 10. Escape still closes the drawer on MOBILE (regression guard).
// ---------------------------------------------------------------------------
function testEscapeStillClosesOnMobile() {
  const {win} = makeWebview({remote: true, desktopMatches: false});
  const sidebar = win.document.getElementById('sidebar');
  const menuBtn = win.document.getElementById('menu-btn');
  menuBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(sidebar.classList.contains('open'));
  win.document.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
  );
  assert.ok(
    !sidebar.classList.contains('open'),
    'Escape must still close the mobile drawer',
  );
  win.close();
  console.log('PASS Escape still closes the mobile drawer');
}

testDesktopDocksSidebarOnLoad();
testMobileKeepsDrawerClosed();
testResizeAcrossBreakpoint();
testVsCodeWebviewIsolation();
testNoMatchMediaNoCrash();
testEscapeAndOverlayDoNotUndock();
testHistoryClickKeepsDock();
testHistoryClickStillClosesOnMobile();
testMenuAndCloseButtonsStillToggle();
testEscapeStillClosesOnMobile();
console.log('All remoteDesktopSidebar tests passed.');
