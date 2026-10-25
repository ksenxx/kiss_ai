// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of the CHATS panel's Cron chip: the daemon's `history`
// reply marks the tasks the cron scheduler launched with `is_cron`, the
// chip is off by default and those rows are hidden, switching it on
// shows them alongside the others, and switching it off hides them
// again.  The other chips keep working on cron rows (a failed cron row
// needs both Cron and Errored on).

const assert = require('assert');
const harness = require('./simplify2_harness');
const {
  historyRows,
  visibleTitles,
  sendHistory,
} = require('./history_helpers');
const {send} = harness;

function markRemoteChat(win) {
  win.document.body.classList.add('remote-chat');
}

/** The webapp (remote-chat) page over the harness's recording transport. */
function makePage() {
  return harness.makePage({beforeScripts: markRemoteChat});
}

function session(id, title, extra) {
  return Object.assign(
    {
      id: 'chat-' + id,
      task_id: 'task-' + id,
      title,
      timestamp: 1_700_000_000 + id * 100,
      preview: title,
      has_events: false,
      failed: false,
      is_running: false,
      tokens: 0,
      cost: 0,
      steps: 0,
      is_favorite: false,
      work_dir: '',
      startTs: (1_700_000_000 + id * 100) * 1000,
      endTs: (1_700_000_000 + id * 100) * 1000 + 10_000,
    },
    extra || {},
  );
}

const SESSIONS = [
  session(1, 'plain one'),
  session(2, 'cron ok', {is_cron: true}),
  session(3, 'cron failed', {is_cron: true, failed: true}),
  session(4, 'plain two', {is_cron: false}),
  // An older daemon's row has no is_cron at all: a plain row.
  session(5, 'legacy'),
];

function setChip(win, id, checked) {
  const box = win.document.getElementById(id);
  box.checked = checked;
  box.dispatchEvent(new win.Event('change', {bubbles: true}));
}

function testChipIsOffByDefaultAndHidesCronRows() {
  const {win, posted} = makePage();
  const box = win.document.getElementById('hf-cron');
  assert.ok(box && box.type === 'checkbox', 'hf-cron checkbox must exist');
  assert.strictEqual(box.checked, false, 'the Cron chip starts off');
  assert.ok(
    box.closest('.history-filter-chips'),
    'the Cron chip sits with the other status chips',
  );
  sendHistory(win, posted, SESSIONS);
  const seen = {};
  historyRows(win).forEach(r => {
    seen[r.querySelector('.sidebar-item-text').textContent] = r.dataset.cron;
  });
  assert.deepStrictEqual(seen, {
    'plain one': '0',
    'cron ok': '1',
    'cron failed': '1',
    'plain two': '0',
    legacy: '0',
  });
  assert.deepStrictEqual(visibleTitles(win), ['legacy', 'plain one', 'plain two']);
  win.close();
  console.log('  ok - cron rows are hidden while the chip is off');
}

function testSwitchingTheChipShowsAndHidesCronRows() {
  const {win, posted} = makePage();
  sendHistory(win, posted, SESSIONS);
  setChip(win, 'hf-cron', true);
  assert.deepStrictEqual(visibleTitles(win), [
    'cron failed',
    'cron ok',
    'legacy',
    'plain one',
    'plain two',
  ]);
  // The status chips still apply to cron rows.
  setChip(win, 'hf-errors', false);
  assert.deepStrictEqual(visibleTitles(win), [
    'cron ok',
    'legacy',
    'plain one',
    'plain two',
  ]);
  setChip(win, 'hf-errors', true);
  setChip(win, 'hf-cron', false);
  assert.deepStrictEqual(visibleTitles(win), ['legacy', 'plain one', 'plain two']);
  // A refreshed list keeps the chip's state.
  setChip(win, 'hf-cron', true);
  sendHistory(win, posted, SESSIONS);
  assert.deepStrictEqual(visibleTitles(win), [
    'cron failed',
    'cron ok',
    'legacy',
    'plain one',
    'plain two',
  ]);
  win.close();
  console.log('  ok - the Cron chip shows and hides the cron rows');
}

/**
 * Give the history list a layout: JSDOM has none, so clientHeight and
 * scrollHeight are 0.  The list is `height` tall and its content as
 * tall as its visible rows.
 */
function layOutHistoryList(win, height) {
  const list = win.document.getElementById('history-list');
  Object.defineProperty(list, 'clientHeight', {get: () => height});
  Object.defineProperty(list, 'scrollHeight', {
    get: () =>
      Array.from(list.querySelectorAll('.sidebar-item')).filter(
        r => r.style.display !== 'none',
      ).length * 40,
  });
}

function testAFirstPageOfOnlyCronRowsFetchesTheNextPage() {
  const {win, posted} = makePage();
  layOutHistoryList(win, 400);
  const cronPage = [];
  for (let i = 1; i <= 50; i += 1) {
    cronPage.push(session(100 + i, 'tick ' + i, {is_cron: true}));
  }
  const before = posted.filter(m => m && m.type === 'getHistory').length;
  sendHistory(win, posted, cronPage);
  assert.deepStrictEqual(visibleTitles(win), [], 'every row of the page is hidden');
  assert.ok(
    !win.document.querySelector('.sidebar-empty-filter'),
    'no "No tasks match" while more pages may follow',
  );
  const requests = posted.filter(m => m && m.type === 'getHistory');
  assert.strictEqual(requests.length, before + 1, 'the next page is requested');
  assert.strictEqual(requests[requests.length - 1].offset, 50, 'at offset 50');
  assert.ok(
    win.document.getElementById('history-loader'),
    'the Loading row is shown meanwhile',
  );
  // The next page holds a plain task: it shows, and nothing further is
  // requested once the pages run short.
  const second = [session(200, 'a real task')];
  send(win, {
    type: 'history',
    sessions: second,
    offset: 50,
    generation: requests[requests.length - 1].generation,
  });
  assert.deepStrictEqual(visibleTitles(win), ['a real task']);
  assert.strictEqual(
    posted.filter(m => m && m.type === 'getHistory').length,
    before + 1,
    'a short page ends the paging',
  );
  assert.ok(!win.document.getElementById('history-loader'), 'loader retired');
  win.close();
  console.log('  ok - a page of nothing but cron rows pulls in the next page');
}

function testAFilledListDoesNotPageOnItsOwn() {
  const {win, posted} = makePage();
  layOutHistoryList(win, 100);
  const page = [];
  for (let i = 1; i <= 50; i += 1) {
    page.push(session(300 + i, 'task ' + i, {is_cron: i % 2 === 0}));
  }
  const before = posted.filter(m => m && m.type === 'getHistory').length;
  sendHistory(win, posted, page);
  assert.strictEqual(visibleTitles(win).length, 25, 'half the rows show');
  assert.strictEqual(
    posted.filter(m => m && m.type === 'getHistory').length,
    before,
    'a list taller than its viewport waits for a scroll',
  );
  win.close();
  console.log('  ok - a filled list leaves paging to the scroll');
}

testChipIsOffByDefaultAndHidesCronRows();
testSwitchingTheChipShowsAndHidesCronRows();
testAFirstPageOfOnlyCronRowsFetchesTheNextPage();
testAFilledListDoesNotPageOnItsOwn();
console.log('historyCronChip.test.js: all tests passed');
