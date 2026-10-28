// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The "Loading the chat from <machine>…" line a remote chat's tab
// shows while the daemon fetches the chat (showRemoteChatLoader) goes
// with the answer even when the answer lands while the tab is parked:
// a `status` (the first frame of a replay; the only frame of a chat
// with nothing to replay) or an `error` addressed to the hidden tab
// takes the line out of the parked transcript, so the chat shows
// without it when the user comes back.  The tab is parked with
// createNewTab (an agent's content tab, a question of another task
// bring a tab forward the same way).

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');
const {finishedSession, sendHistory} = require('./history_helpers');

const LOADER = '#remote-chat-loader';

/** Open the page with one remote chat row and click it: its tab shows
 *  the loading line.  Returns the page and the tab's id. */
function openRemoteChat() {
  const {win, posted} = makePage({bodyClass: 'remote-chat'});
  const row = finishedSession('remote', 'Remote task', {
    machine: 'box',
    task_id: 'r-task',
    has_events: true,
  });
  sendHistory(win, posted, [row], {machines: ['box'], localMachine: 'local'});
  win.document.querySelector('.history-chat-header').click();
  const remote = win._testApi.getActiveTabId();
  const loader = win.document.querySelector(LOADER);
  assert(loader, 'the clicked tab says the chat is on its way');
  assert.strictEqual(loader.textContent, 'Loading the chat from box\u2026');
  return {win, remote};
}

/** Park the remote tab behind a new chat; the line goes with it. */
function parkRemoteTab(win, remote) {
  win._testApi.createNewTab();
  assert.notStrictEqual(win._testApi.getActiveTabId(), remote);
  assert(!win.document.querySelector(LOADER), 'the new chat shows no line');
}

function testStatusInTheBackgroundTakesTheLine() {
  const {win, remote} = openRemoteChat();
  parkRemoteTab(win, remote);
  send(win, {type: 'status', running: false, tabId: remote});
  win._testApi.switchToTab(remote);
  assert.strictEqual(win._testApi.getActiveTabId(), remote);
  assert(
    !win.document.querySelector(LOADER),
    'the status that reached the parked tab took the line with it',
  );
  win.close();
  console.log('  ok - a status answering a parked tab takes the loading line');
}

function testErrorInTheBackgroundTakesTheLine() {
  const {win, remote} = openRemoteChat();
  parkRemoteTab(win, remote);
  send(win, {
    type: 'error',
    code: 'prompt_refused',
    text: 'box: connection refused',
    tabId: remote,
  });
  win._testApi.switchToTab(remote);
  assert(
    !win.document.querySelector(LOADER),
    'the error that reached the parked tab took the line with it',
  );
  const err = win.document.querySelector('#output .ev.err');
  assert(err && /box/.test(err.textContent), 'the error is in the transcript');
  win.close();
  console.log('  ok - an error answering a parked tab takes the loading line');
}

function testTheLineStaysUntilTheAnswer() {
  const {win, remote} = openRemoteChat();
  parkRemoteTab(win, remote);
  win._testApi.switchToTab(remote);
  assert(
    win.document.querySelector(LOADER),
    'coming back before the answer, the line is still there',
  );
  send(win, {type: 'status', running: false, tabId: remote});
  assert(!win.document.querySelector(LOADER), 'and goes with the answer');
  win.close();
  console.log('  ok - the line survives a round trip to another tab until the answer');
}

function main() {
  testStatusInTheBackgroundTakesTheLine();
  testErrorInTheBackgroundTakesTheLine();
  testTheLineStaysUntilTheAnswer();
  console.log('remoteChatLoaderBackground.test.js: all assertions passed.');
}

main();
