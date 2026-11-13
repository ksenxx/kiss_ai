// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// Helpers shared by the remote meta-drawer suites (remoteMetaMobileDrawer,
// remoteMetaPanelInfo, sidebarChatModeDrawerAndComposer): the drawer's
// open state, the getTaskUpdate polls the page posts and the daemon's
// reply to one.

const assert = require('assert');
const {send, ofType, lastMsg} = require('./simplify2_harness');

// How often main.js polls for the open panel's task update.
const POLL_MS = 5000;

/** Whether the meta drawer is open. */
function drawerOpen(win) {
  return win.document.getElementById('meta-panel').classList.contains('open');
}

/** Whether the meta drawer's overlay is open. */
function overlayOpen(win) {
  return win.document.getElementById('meta-overlay').classList.contains('open');
}

/** Every getTaskUpdate message posted so far, oldest first. */
function polls(wv) {
  return ofType(wv.posted, 'getTaskUpdate');
}

/** How many getTaskUpdate polls the page has posted. */
function pollCount(wv) {
  return polls(wv).length;
}

/** The newest getTaskUpdate poll (asserting one has fired). */
function lastPoll(wv) {
  const poll = lastMsg(wv.posted, 'getTaskUpdate');
  assert.ok(poll, 'a getTaskUpdate poll must have fired');
  return poll;
}

/** A complete taskUpdate reply to *poll*; *fields* override the defaults. */
function replyToPoll(win, poll, fields) {
  send(
    win,
    Object.assign(
      {
        type: 'taskUpdate',
        tabId: poll.tabId,
        token: poll.token,
        taskId: 'task-1',
        exists: true,
        sig: 'sig-1',
        content: '',
        error: '',
        running: false,
        cost: 0,
        updatedAt: 0,
        dueAt: 0,
      },
      fields,
    ),
  );
}

module.exports = {
  POLL_MS,
  drawerOpen,
  overlayOpen,
  polls,
  pollCount,
  lastPoll,
  replyToPoll,
};
