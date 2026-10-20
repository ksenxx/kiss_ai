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
  const {desktopMatches = true} = opts || {};
  // desktopMatches === null: no window.matchMedia at all.
  let fireChange = null;
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts: win => {
      if (desktopMatches !== null) {
        fireChange = fakeMatchMedia(win, desktopMatches === true);
      }
    },
  });
  return {win, posted, fireChange};
}

function getHistoryMsgs(posted) {
  return posted.filter(m => m.type === 'getHistory');
}

function testDesktopDocksSidebarOnLoad() {
  const {win, posted} = makePage({desktopMatches: true});
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

function testMobileKeepsDrawerClosed() {
  const {win, posted} = makePage({desktopMatches: false});
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

function testResizeAcrossBreakpoint() {
  const {win, posted, fireChange} = makePage({
    desktopMatches: false,
  });
  const body = win.document.body;
  const sidebar = win.document.getElementById('sidebar');
  assert.ok(!sidebar.classList.contains('open'));
  fireChange(true);
  assert.ok(body.classList.contains('remote-desktop'), 'resize wide → dock');
  assert.ok(sidebar.classList.contains('open'), 'resize wide → sidebar open');
  assert.ok(
    getHistoryMsgs(posted).length >= 1,
    'docking after resize must post getHistory',
  );
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

function testNoMatchMediaNoCrash() {
  const {win, posted} = makePage({desktopMatches: null});
  assert.ok(
    posted.find(m => m.type === 'ready'),
    'page must boot without window.matchMedia',
  );
  assert.ok(
    !win.document.body.classList.contains('remote-desktop'),
    'without matchMedia the desktop mode must stay off',
  );
  win.close();
  console.log('PASS missing window.matchMedia does not crash main.js');
}

function testEscapeAndOverlayDoNotUndock() {
  const {win} = makePage({desktopMatches: true});
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

function testHistoryClickKeepsDock() {
  const {win, posted} = makePage({desktopMatches: true});
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
    posted.find(m => m.type === 'resumeSession'),
    'clicking the row must resume the session',
  );
  assert.ok(
    sidebar.classList.contains('open'),
    'history click must NOT close the docked desktop sidebar',
  );
  win.close();
  console.log('PASS history-row click keeps the desktop sidebar docked');
}

function testHistoryClickStillClosesOnMobile() {
  const {win, posted} = makePage({desktopMatches: false});
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

function testMenuAndCloseButtonsStillToggle() {
  const {win, posted} = makePage({desktopMatches: true});
  const sidebar = win.document.getElementById('sidebar');
  const menuBtn = win.document.getElementById('menu-btn');
  const closeBtn = win.document.getElementById('sidebar-close');
  const overlay = win.document.getElementById('sidebar-overlay');
  assert.ok(sidebar.classList.contains('open'));
  menuBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    !sidebar.classList.contains('open'),
    'burger click must hide the docked sidebar (explicit intent)',
  );
  const before = getHistoryMsgs(posted).length;
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
  closeBtn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    !sidebar.classList.contains('open'),
    'sidebar-close button must hide the docked sidebar',
  );
  win.close();
  console.log('PASS burger / close button explicitly toggle the dock');
}

function testEscapeStillClosesOnMobile() {
  const {win} = makePage({desktopMatches: false});
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
testNoMatchMediaNoCrash();
testEscapeAndOverlayDoNotUndock();
testHistoryClickKeepsDock();
testHistoryClickStillClosesOnMobile();
testMenuAndCloseButtonsStillToggle();
testEscapeStillClosesOnMobile();
console.log('All remoteDesktopSidebar tests passed.');
