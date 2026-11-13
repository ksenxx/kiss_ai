// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of the CHATS panel's single-flight history refresh.
//
// The daemon's `getHistory` reply merges every registered machine's
// rows and takes a second or more (it asks each machine over its
// link).  Every refetch opens a new generation and only the newest
// generation's reply is rendered, so the burst of refreshes a page
// load triggers (`daemonStatus`, `tasks_updated`, one `status` per
// replayed tab) used to discard every reply but the last, and a
// steady stream of task news every reply: the panel sat on its
// placeholder, or on rows rendered long ago (local-only ones from a
// reply a remote machine had missed).  Now a refresh asked for while
// page one is in flight folds into it and ONE refetch goes out when
// the reply lands, so every reply sent is shown.

const assert = require('assert');
const {makePage, send, ofType} = require('./simplify2_harness');
const {
  machineSession: session,
  rowTitles,
  lastGetHistory: lastRequest,
  freezeClock,
} = require('./history_helpers');

const FROZEN_NOW_MS = 1_700_500_000_000;

const SESSIONS = [
  session(1, 'local one', ''),
  session(2, 'box task', 'box'),
  session(3, 'lab task', 'lab'),
];

function reply(generation, sessions, offset = 0) {
  return {
    type: 'history',
    sessions,
    offset,
    generation,
    machines: ['box', 'lab'],
    localMachine: 'myhost',
  };
}

// The machines named by the rows' chips (a chat's header and its task
// row each carry one), in list order.
function chips(win) {
  const names = Array.from(
    win.document.querySelectorAll('#history-list .sidebar-item-machine'),
  ).map(el => el.textContent);
  return Array.from(new Set(names));
}

// The page's clock stands still: a flight is given up on after 30 s of
// it, so on the host's clock a stalled run would end the flight under a
// test that folds into it (testAStalledFlightIsGivenUpOn moves the
// clock itself).
function openPage() {
  const {win, posted} = makePage({
    beforeScripts: w => freezeClock(w, FROZEN_NOW_MS),
  });
  win.document.getElementById('sidebar').classList.add('open');
  send(win, {type: 'daemonStatus', connected: true});
  const first = lastRequest(posted);
  assert.ok(first, 'the page asks for page one');
  assert.strictEqual(first.offset, 0);
  return {win, posted, first};
}

function testBurstFoldsIntoTheFlightAndEveryReplyShows() {
  const {win, posted, first} = openPage();
  posted.length = 0;
  // The load burst: task news while page one is still in flight.
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'tasks_updated'});
  send(win, {type: 'status', running: true, attached: true, tabId, startTs: FROZEN_NOW_MS});
  send(win, {type: 'status', running: false, tabId});
  send(win, {type: 'tasks_updated'});
  assert.strictEqual(
    ofType(posted, 'getHistory').length,
    0,
    'a refresh during the flight folds into it',
  );
  // The in-flight reply is the newest generation: it renders, every
  // machine's rows included.
  send(win, reply(first.generation, SESSIONS));
  assert.deepStrictEqual(rowTitles(win), ['lab task', 'box task', 'local one']);
  assert.deepStrictEqual(chips(win), ['lab', 'box']);
  // ... and the folded refresh goes out exactly once, as a new page one.
  const after = ofType(posted, 'getHistory');
  assert.strictEqual(after.length, 1, 'one trailing refetch');
  assert.strictEqual(after[0].offset, 0);
  assert.strictEqual(after[0].generation, first.generation + 1);
  // Its reply shows too and, nothing having been folded, asks nothing more.
  posted.length = 0;
  send(win, reply(after[0].generation, SESSIONS.slice(0, 2)));
  assert.deepStrictEqual(rowTitles(win), ['box task', 'local one']);
  assert.strictEqual(ofType(posted, 'getHistory').length, 0, 'no refetch without a folded refresh');
  // A refresh once nothing is in flight goes out at once.
  send(win, {type: 'tasks_updated'});
  assert.strictEqual(ofType(posted, 'getHistory').length, 1, 'an idle refresh is sent now');
  win.close();
  console.log('  ok - a burst folds into the flight and every reply shows');
}

function testAViewChangeStartsAFreshGenerationAtOnce() {
  const {win, posted, first} = openPage();
  send(win, {type: 'tasks_updated'}); // folded
  posted.length = 0;
  // The user searches: a new view, asked for at once, in a new
  // generation; the folded refresh is satisfied by it.
  const search = win.document.getElementById('history-search');
  search.value = 'box';
  search.dispatchEvent(new win.Event('input', {bubbles: true}));
  const searchReq = lastRequest(posted);
  assert.ok(searchReq, 'the search refetches at once');
  assert.strictEqual(searchReq.query, 'box');
  assert.ok(searchReq.generation > first.generation);
  // The old flight's reply is stale: dropped, and no refetch follows it.
  posted.length = 0;
  send(win, reply(first.generation, SESSIONS));
  assert.deepStrictEqual(rowTitles(win), [], 'stale reply not rendered');
  assert.strictEqual(ofType(posted, 'getHistory').length, 0, 'stale reply triggers nothing');
  // The search's reply renders and nothing was folded into it.
  send(win, reply(searchReq.generation, SESSIONS.slice(1, 2)));
  assert.deepStrictEqual(rowTitles(win), ['box task']);
  assert.strictEqual(ofType(posted, 'getHistory').length, 0);
  win.close();
  console.log('  ok - a search while in flight refetches at once');
}

function testALaterPageDoesNotFlushTheFoldedRefresh() {
  const {win, posted, first} = openPage();
  send(win, reply(first.generation, SESSIONS));
  posted.length = 0;
  send(win, {type: 'tasks_updated'});
  const gen = lastRequest(posted).generation;
  send(win, {type: 'tasks_updated'}); // folded into the new flight
  posted.length = 0;
  // A later page's reply (not page one) flushes nothing.
  send(win, reply(gen, SESSIONS, 1));
  assert.strictEqual(ofType(posted, 'getHistory').length, 0, 'a page-two reply flushes nothing');
  send(win, reply(gen, SESSIONS));
  assert.strictEqual(ofType(posted, 'getHistory').length, 1, 'page one flushes the folded refresh');
  win.close();
  console.log('  ok - only page one flushes the folded refresh');
}

function testAReconnectDoesNotWaitOnALostReply() {
  const {win, posted} = openPage();
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  const reqs = ofType(posted, 'getHistory');
  assert.strictEqual(reqs.length, 1, 'the reconnect refetches page one');
  assert.strictEqual(reqs[0].offset, 0);
  win.close();
  console.log('  ok - a reconnect refetches instead of waiting on a lost reply');
}

function testAStalledFlightIsGivenUpOn() {
  const {win, posted} = openPage();
  posted.length = 0;
  win.Date.now = () => FROZEN_NOW_MS + 29_000;
  send(win, {type: 'tasks_updated'});
  assert.strictEqual(
    ofType(posted, 'getHistory').length,
    0,
    'a page-one request under 30 s old is still waited on',
  );
  win.Date.now = () => FROZEN_NOW_MS + 31_000;
  send(win, {type: 'tasks_updated'});
  assert.strictEqual(
    ofType(posted, 'getHistory').length,
    1,
    'a page-one request 30 s without a reply is not waited on',
  );
  win.close();
  console.log('  ok - a stalled flight is given up on');
}

function testAClosedPanelFoldsNothing() {
  const {win, posted, first} = openPage();
  // Page one has landed: no flight is left for the news to fold into.
  send(win, reply(first.generation, SESSIONS));
  posted.length = 0;
  win.document.getElementById('sidebar').classList.remove('open');
  send(win, {type: 'tasks_updated'});
  assert.strictEqual(ofType(posted, 'getHistory').length, 0, 'a closed panel asks nothing');
  win.close();
  console.log('  ok - a closed panel asks nothing');
}

testBurstFoldsIntoTheFlightAndEveryReplyShows();
testAViewChangeStartsAFreshGenerationAtOnce();
testALaterPageDoesNotFlushTheFoldedRefresh();
testAReconnectDoesNotWaitOnALostReply();
testAStalledFlightIsGivenUpOn();
testAClosedPanelFoldsNothing();
console.log('\n6 passed, 0 failed');
