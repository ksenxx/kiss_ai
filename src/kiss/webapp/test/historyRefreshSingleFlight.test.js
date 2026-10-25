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
const {makePage, send} = require('./simplify2_harness');

function session(id, title, machine) {
  const ts = 1_700_000_000 + id * 100;
  return Object.assign(
    {
      id: 'chat-' + id,
      task_id: 'task-' + id,
      title,
      timestamp: ts,
      preview: title,
      has_events: false,
      failed: false,
      is_running: false,
      tokens: 0,
      cost: 0,
      steps: 0,
      is_favorite: false,
      work_dir: '',
      startTs: ts * 1000,
      endTs: ts * 1000 + 10_000,
    },
    machine ? {machine, machine_url: 'https://' + machine + ':8787'} : {},
  );
}

const SESSIONS = [
  session(1, 'local one', ''),
  session(2, 'box task', 'box'),
  session(3, 'lab task', 'lab'),
];

function historyRequests(posted) {
  return posted.filter(m => m && m.type === 'getHistory');
}

function lastRequest(posted) {
  const reqs = historyRequests(posted);
  return reqs[reqs.length - 1];
}

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

function rowTitles(win) {
  return Array.from(
    win.document.querySelectorAll('#history-list .sidebar-item-text'),
  ).map(el => el.textContent);
}

// The machines named by the rows' chips (a chat's header and its task
// row each carry one), in list order.
function chips(win) {
  const names = Array.from(
    win.document.querySelectorAll('#history-list .sidebar-item-machine'),
  ).map(el => el.textContent);
  return Array.from(new Set(names));
}

function openPage() {
  const {win, posted} = makePage();
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
  const tabId = win.kissActiveTabId();
  send(win, {type: 'tasks_updated'});
  send(win, {type: 'status', running: true, attached: true, tabId, startTs: Date.now()});
  send(win, {type: 'status', running: false, tabId});
  send(win, {type: 'tasks_updated'});
  assert.strictEqual(
    historyRequests(posted).length,
    0,
    'a refresh during the flight folds into it',
  );
  // The in-flight reply is the newest generation: it renders, every
  // machine's rows included.
  send(win, reply(first.generation, SESSIONS));
  assert.deepStrictEqual(rowTitles(win), ['lab task', 'box task', 'local one']);
  assert.deepStrictEqual(chips(win), ['lab', 'box']);
  // ... and the folded refresh goes out exactly once, as a new page one.
  const after = historyRequests(posted);
  assert.strictEqual(after.length, 1, 'one trailing refetch');
  assert.strictEqual(after[0].offset, 0);
  assert.strictEqual(after[0].generation, first.generation + 1);
  // Its reply shows too and, nothing having been folded, asks nothing more.
  posted.length = 0;
  send(win, reply(after[0].generation, SESSIONS.slice(0, 2)));
  assert.deepStrictEqual(rowTitles(win), ['box task', 'local one']);
  assert.strictEqual(historyRequests(posted).length, 0, 'no refetch without a folded refresh');
  // A refresh once nothing is in flight goes out at once.
  send(win, {type: 'tasks_updated'});
  assert.strictEqual(historyRequests(posted).length, 1, 'an idle refresh is sent now');
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
  assert.strictEqual(historyRequests(posted).length, 0, 'stale reply triggers nothing');
  // The search's reply renders and nothing was folded into it.
  send(win, reply(searchReq.generation, SESSIONS.slice(1, 2)));
  assert.deepStrictEqual(rowTitles(win), ['box task']);
  assert.strictEqual(historyRequests(posted).length, 0);
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
  assert.strictEqual(historyRequests(posted).length, 0, 'a page-two reply flushes nothing');
  send(win, reply(gen, SESSIONS));
  assert.strictEqual(historyRequests(posted).length, 1, 'page one flushes the folded refresh');
  win.close();
  console.log('  ok - only page one flushes the folded refresh');
}

function testAReconnectDoesNotWaitOnALostReply() {
  const {win, posted} = openPage();
  posted.length = 0;
  send(win, {type: 'daemonStatus', connected: false});
  send(win, {type: 'daemonStatus', connected: true});
  const reqs = historyRequests(posted);
  assert.strictEqual(reqs.length, 1, 'the reconnect refetches page one');
  assert.strictEqual(reqs[0].offset, 0);
  win.close();
  console.log('  ok - a reconnect refetches instead of waiting on a lost reply');
}

function testAStalledFlightIsGivenUpOn() {
  const {win, posted} = openPage();
  posted.length = 0;
  const realNow = win.Date.now;
  win.Date.now = () => realNow() + 31_000;
  try {
    send(win, {type: 'tasks_updated'});
    assert.strictEqual(
      historyRequests(posted).length,
      1,
      'a page-one request 30 s without a reply is not waited on',
    );
  } finally {
    win.Date.now = realNow;
  }
  win.close();
  console.log('  ok - a stalled flight is given up on');
}

function testAClosedPanelFoldsNothing() {
  const {win, posted} = openPage();
  posted.length = 0;
  win.document.getElementById('sidebar').classList.remove('open');
  send(win, {type: 'tasks_updated'});
  assert.strictEqual(historyRequests(posted).length, 0, 'a closed panel asks nothing');
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
