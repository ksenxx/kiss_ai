// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// Helpers shared by the remote meta-drawer suites (remoteMetaMobileDrawer,
// remoteMetaPanelInfo, sidebarChatModeDrawerAndComposer): the drawer's
// open state and the getTaskUpdate polls the page posts.

const assert = require('assert');
const {ofType, lastMsg} = require('./simplify2_harness');

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

module.exports = {drawerOpen, overlayOpen, polls, pollCount, lastPoll};
