// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Audit 2026-10-12 (mainjs-2): the daemon replays one `status` per
// chat-bound tab after every `ready` (and again after every reconnect),
// and `tasks_updated` follows each task end.  main.js answers every one
// of them with refreshHistory() -> requestHistoryFromStart(), which
// posts a fresh `getHistory` (a history.db query with chat summaries
// on the daemon) and bumps historyGeneration so that only the LAST
// reply is ever rendered.  A window with N open tabs therefore fires N
// identical history queries in one burst, N-1 of which are thrown
// away on arrival.  The burst must coalesce: one leading request and
// at most one trailing one per tick.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: m => posted.push(m),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function historyRequests(posted) {
  return posted.filter(m => m && m.type === 'getHistory');
}

function nextTick() {
  return new Promise(resolve => setTimeout(resolve, 20));
}

async function testStatusBurstCoalescesIntoOneHistoryFetch() {
  const {win, posted} = makeWebview();
  win.document.getElementById('sidebar').classList.add('open');
  send(win, {type: 'daemonStatus', connected: true});
  const active = win.kissActiveTabId();
  const ids = [active, 'tab-b', 'tab-c', 'tab-d', 'tab-e'];
  send(win, {
    type: 'tabs_state',
    tabs: ids.map((id, i) => ({id, title: 'Task ' + i, chatId: 'chat-' + i})),
  });
  await nextTick();
  posted.length = 0;
  // The ready replay: one status per tab, all in the same tick.
  const now = Date.now();
  for (const id of ids) {
    send(win, {type: 'status', running: true, attached: true, tabId: id, startTs: now});
  }
  await nextTick();
  const n = historyRequests(posted).length;
  assert.ok(
    n <= 2,
    'five replayed statuses in one tick must coalesce into at most a ' +
      'leading and a trailing getHistory, got ' + n,
  );
  // The same holds for the task ends that follow a run.
  posted.length = 0;
  for (const id of ids) {
    send(win, {type: 'status', running: false, tabId: id});
    send(win, {type: 'tasks_updated'});
  }
  await nextTick();
  const m = historyRequests(posted).length;
  assert.ok(
    m <= 2,
    'five task ends in one tick must coalesce into at most two ' +
      'getHistory requests, got ' + m,
  );
  win.close();
  console.log('  ok - a burst of status events fetches history once');
}

async function testASingleStatusStillRefetches() {
  // The coalescing must not swallow the refresh itself: one status
  // with the panel open still asks for page one.
  const {win, posted} = makeWebview();
  win.document.getElementById('sidebar').classList.add('open');
  send(win, {type: 'daemonStatus', connected: true});
  await nextTick();
  posted.length = 0;
  send(win, {
    type: 'status',
    running: true,
    tabId: win.kissActiveTabId(),
    startTs: Date.now(),
  });
  await nextTick();
  const reqs = historyRequests(posted);
  assert.strictEqual(reqs.length, 1, 'exactly one getHistory');
  assert.strictEqual(reqs[0].offset, 0, 'page one');
  win.close();
  console.log('  ok - a single status still refetches page one');
}

async function runTests() {
  await testStatusBurstCoalescesIntoOneHistoryFetch();
  await testASingleStatusStillRefetches();
}

runTests().then(
  () => {
    console.log('\n2 passed, 0 failed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
