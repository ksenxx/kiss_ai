// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) test for the Chats list after a favourite click
// (media/main.js renderHistory / mergePendingFlags).  A star click
// flips the row optimistically; a refetch that lands before the daemon
// wrote the favourite is overlaid with the click's value (the daemon's
// own value is stashed on the row as `_kissDaemon_is_favorite`).  The
// page is then identical to what is shown, so the identical-page fast
// path must keep the row elements: it used to rebuild the whole list
// once because the stash key made the incoming row "differ" from the
// shown one.

'use strict';

const assert = require('assert');
const {
  send,
  stubRaf,
  ofType,
  captureTimers,
  fireCapturedTimers,
} = require('./simplify2_harness');
const {
  sendHistory,
  makeRow,
  openChatsPage,
  freezeClock,
} = require('./history_helpers');

const FROZEN_NOW_MS = 1_700_500_000_000;

// The page's clock stands still: a click's star is shielded from stale
// refetches for 10 s of it, so a stalled run must not let the host's
// clock end the shield between a click and the reply that follows it.
function openPage() {
  return openChatsPage(win => {
    stubRaf(win);
    freezeClock(win, FROZEN_NOW_MS);
  });
}

function rows(favorite) {
  return [
    makeRow({
      id: 'A',
      task_id: 1,
      chat_summary: 'Old task',
      is_favorite: favorite,
    }),
    makeRow({
      id: 'B',
      task_id: 2,
      chat_summary: 'Other task',
      is_favorite: false,
    }),
  ];
}

function rowEl(win) {
  return win.document.querySelector(
    '#history-list .history-chat-group[data-chat-id="A"] .sidebar-item',
  );
}

function main() {
  const {win, posted} = openPage();
  sendHistory(win, posted, rows(false));
  const row = rowEl(win);
  assert.ok(row, 'the row is rendered');
  const star = row.querySelector('.sidebar-item-favorite');
  assert.ok(star, 'the row has a star');
  star.click();
  assert.ok(star.classList.contains('favorited'), 'the star flipped at once');
  assert.strictEqual(
    ofType(posted, 'setFavorite').length,
    1,
    'the write was sent',
  );

  // A refetch from BEFORE the write landed: the daemon still says no.
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, rows(false));
  assert.strictEqual(
    rowEl(win),
    row,
    'the identical page kept the row element',
  );
  assert.ok(
    rowEl(win)
      .querySelector('.sidebar-item-favorite')
      .classList.contains('favorited'),
    'the optimistic star is still shown',
  );

  // The write landed: the daemon agrees; still the same rows on screen.
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, rows(true));
  assert.strictEqual(rowEl(win), row, 'the agreeing page kept the row element');
  assert.ok(
    rowEl(win)
      .querySelector('.sidebar-item-favorite')
      .classList.contains('favorited'),
  );
  win.close();
}

function remoteRowEl(win) {
  return win.document.querySelector(
    '#history-list .history-chat-group[data-chat-id="A"] ' +
      '.sidebar-item[data-machine="gpu-box"]',
  );
}

function testARemoteRowOfTheSameTaskIdKeepsItsOwnFlag() {
  // A favourite is a write to THIS daemon's database.  A row of another
  // machine (the MACHINE filter switched within the optimistic window)
  // that happens to carry the same task id neither shows the local
  // star nor acknowledges the local write.
  const {win, posted} = openPage();
  sendHistory(win, posted, rows(false));
  rowEl(win).querySelector('.sidebar-item-favorite').click();
  assert.strictEqual(ofType(posted, 'setFavorite').length, 1);
  const remote = () => [
    makeRow({
      id: 'A',
      task_id: 1,
      chat_summary: 'Remote task',
      is_favorite: false,
      machine: 'gpu-box',
    }),
  ];
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, remote());
  const remoteRow = remoteRowEl(win);
  assert.ok(remoteRow, 'the remote row is rendered');
  assert.strictEqual(
    remoteRow.dataset.favorite,
    '0',
    'BUG: the local favourite was painted onto the remote row',
  );
  // The remote row agreeing with the local toggle is no acknowledgement
  // either: back on the local page, a refetch from before the write
  // still shows the optimistic star.
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, [Object.assign(remote()[0], {is_favorite: true})]);
  send(win, {type: 'tasks_updated'});
  sendHistory(win, posted, rows(false));
  assert.ok(
    rowEl(win).querySelector('.sidebar-item-favorite').classList.contains(
      'favorited',
    ),
    'BUG: the remote row consumed the local overlay',
  );
  win.close();
}

function testAStarEndsAtItsBound() {
  // A row may leave the list (deleted, filtered out) before the
  // daemon's word on its star arrives.  Each click's bookkeeping still
  // ends at its own bound, with the refetch that shows the daemon's
  // value, whatever the pages in between hold.  A row the page still
  // holds then reads what the daemon said of it as soon as it is
  // built again.
  const timers = [];
  let now = FROZEN_NOW_MS;
  const {win, posted} = openChatsPage(w => {
    stubRaf(w);
    freezeClock(w, FROZEN_NOW_MS);
    captureTimers(w, timers);
  });
  win.Date.now = () => now;
  const pending = () => win._testApi.pendingHistoryFlagCount();
  // The delays of the timers of a bound that are armed.
  const armed = () =>
    timers
      .filter(t => !t.cleared && !t.fired && [10100, 3210].includes(t.ms))
      .map(t => t.ms);
  // The bound's timer comes due: whether it asked for the list anew.
  const boundPasses = ms => {
    const asked = ofType(posted, 'getHistory').length;
    timers.find(t => t.ms === ms && !t.cleared && !t.fired).cb();
    return ofType(posted, 'getHistory').length > asked;
  };
  const page = sessions => {
    send(win, {type: 'tasks_updated'});
    sendHistory(win, posted, sessions);
  };
  const starOf = chat =>
    win.document.querySelector(
      '#history-list .history-chat-group[data-chat-id="' +
        chat +
        '"] .sidebar-item-favorite',
    );

  // Two stars, 3.21 s apart, on rows that are then gone from the list.
  sendHistory(win, posted, rows(false));
  starOf('A').click();
  now += 3210;
  starOf('B').click();
  page([]);
  assert.strictEqual(pending(), 2, 'a page without the rows ends no shield');
  assert.deepStrictEqual(armed(), [10100]);

  // The first bound: its star's bookkeeping goes, the second's stays
  // and has its own bound planned, though no page lists its row.
  now = FROZEN_NOW_MS + 10100;
  assert.ok(boundPasses(10100), 'the bound asked for the list anew');
  assert.strictEqual(pending(), 1, 'BUG: the star of a row that left is kept');
  assert.deepStrictEqual(armed(), [3210], 'BUG: the second bound is not planned');
  sendHistory(win, posted, []);
  assert.strictEqual(pending(), 1);
  assert.deepStrictEqual(armed(), [3210]);

  // The second bound: nothing is kept, nothing is planned.
  now += 3210;
  assert.ok(boundPasses(3210));
  sendHistory(win, posted, []);
  assert.strictEqual(pending(), 0);
  assert.deepStrictEqual(armed(), []);

  // A list that is not on screen cannot ask: the bookkeeping goes all
  // the same (the list asks when it is shown again).
  page(rows(false));
  starOf('A').click();
  page([]);
  win.document.getElementById('sidebar').classList.remove('open');
  now += 10100;
  assert.ok(!boundPasses(10100), 'a hidden list asks for nothing');
  assert.strictEqual(pending(), 0, 'BUG: kept for a refetch that is not coming');
  assert.deepStrictEqual(armed(), []);

  // A page the daemon sent meanwhile, parked for a held press, lands
  // after the bound with the daemon's value: no refetch was answered.
  // (The row had left the list on screen: the parked page alone held
  // the star.)
  const starred = () => starOf('A').classList.contains('favorited');
  const list = win.document.getElementById('history-list');
  page(rows(false));
  starOf('A').click();
  page(rows(false).slice(1));
  list.dispatchEvent(new win.MouseEvent('mousedown', {bubbles: true}));
  page(rows(false));
  assert.strictEqual(starOf('A'), null, 'the parked page is not on screen');
  now += 10100;
  assert.ok(!boundPasses(10100));
  const released = timers.length;
  win.dispatchEvent(new win.MouseEvent('mouseup', {bubbles: true}));
  fireCapturedTimers(win, timers, released);
  assert.ok(!starred(), 'BUG: the parked page landed with the star of a click that is over');
  assert.strictEqual(pending(), 0);

  // So does the list built again from the rows it holds (Group by).
  const groupBy = id => {
    const radio = win.document.getElementById(id);
    radio.checked = true;
    radio.dispatchEvent(new win.Event('change', {bubbles: true}));
  };
  starOf('A').click();
  now += 10100;
  assert.ok(!boundPasses(10100));
  groupBy('hf-group-chat');
  assert.ok(!starred(), 'BUG: the list was built again with the star of a click that is over');
  assert.strictEqual(pending(), 0);

  // A row that leaves the list after the bound of its star takes the
  // bookkeeping with it.
  starOf('A').click();
  now += 10100;
  assert.ok(!boundPasses(10100));
  page(rows(false).slice(1));
  assert.strictEqual(pending(), 0, 'BUG: kept for a row that is gone');
  assert.deepStrictEqual(armed(), []);

  // A parked page may be dropped, not landed: a page that reads like
  // the list on screen follows it.  The star it alone held is over at
  // its bound all the same (audit 2026-11-21, W13/F2: it was kept for
  // the parked row, then for ever, with no bound planned).
  page(rows(false));
  starOf('A').click();
  page(rows(false).slice(1));
  list.dispatchEvent(new win.MouseEvent('mousedown', {bubbles: true}));
  page(rows(false));
  now += 10100;
  assert.ok(!boundPasses(10100));
  assert.strictEqual(pending(), 0, 'BUG: kept past its bound for a parked row');
  const shownB = starOf('B');
  page(rows(false).slice(1));
  assert.strictEqual(starOf('B'), shownB, 'the page reads like the list: its rows stay');
  const dropped = timers.length;
  win.dispatchEvent(new win.MouseEvent('mouseup', {bubbles: true}));
  fireCapturedTimers(win, timers, dropped);
  assert.strictEqual(starOf('A'), null, 'the parked page was dropped');
  assert.strictEqual(pending(), 0, 'BUG: kept for a page that was dropped');
  assert.deepStrictEqual(armed(), []);

  // The daemon's word at the bound may be the click's own: the page
  // that then reads like the list on screen makes its rows the
  // daemon's, however the list is built from them again.
  const chatStarred = (sessions, at) => {
    sessions[at].chat_is_favorite = true;
    return sessions;
  };
  page(rows(false));
  starOf('A').click();
  now += 10100;
  assert.ok(!boundPasses(10100));
  const confirmed = starOf('A');
  page(chatStarred(rows(false), 0));
  assert.strictEqual(starOf('A'), confirmed, 'the confirming page kept the row');
  groupBy('hf-group-project');
  assert.ok(starred(), 'BUG: the list was built again without the star the daemon confirmed');

  // A row keeps what the daemon last said of it through clicks whose
  // bounds pass unanswered: the first one's star is not the daemon's.
  page(rows(false));
  for (let click = 0; click < 2; click++) {
    starOf('A').click();
    now += 10100;
    assert.ok(!boundPasses(10100));
  }
  groupBy('hf-group-chat');
  assert.ok(!starred(), 'BUG: the list was built again with the star of an earlier click');

  // So does a later page that agrees within the bound (the list moved
  // down a row since the first page: its last row comes again).
  const fifty = () =>
    Array.from({length: 50}, (_, i) =>
      makeRow({id: 'C' + i, task_id: 100 + i, chat_summary: 'Task ' + i}),
    );
  page(fifty());
  starOf('C49').click();
  sendHistory(win, posted, chatStarred(fifty(), 49).slice(49), {offset: 50});
  assert.strictEqual(pending(), 0, 'the later page agreed');
  groupBy('hf-group-project');
  assert.deepStrictEqual(
    Array.from(
      win.document.querySelectorAll('#history-list .sidebar-item[data-chat-favorite]'),
      row => row.dataset.chatFavorite,
    ).filter(value => value === '1'),
    ['1', '1'],
    'BUG: the list was built again without the star the daemon agreed with',
  );

  // A page parked before the bound that lands after it shows what the
  // daemon said THEN: the bound's refetch is still owed, and asked.
  win.document.getElementById('sidebar').classList.add('open');
  page(rows(false));
  starOf('A').click();
  list.dispatchEvent(new win.MouseEvent('mousedown', {bubbles: true}));
  page(rows(false).reverse());
  now += 10050;
  const late = timers.length;
  win.dispatchEvent(new win.MouseEvent('mouseup', {bubbles: true}));
  fireCapturedTimers(win, timers, late);
  assert.ok(!starred(), 'the parked page landed with what the daemon said');
  // So does a page that arrives then, before the bound's timer.
  sendHistory(win, posted, rows(false));
  assert.deepStrictEqual(armed(), [10100], 'BUG: the bound was called off by an old page');
  assert.ok(boundPasses(10100), 'BUG: the list was not asked anew at the bound');
  assert.strictEqual(pending(), 0);

  // The rows the list holds are no word of the daemon's.  A star
  // clicked on and off again is unanswered still when the list is built
  // again from them, and its bound asks (audit 2026-11-21: the row's
  // own value was taken for the daemon's agreement).
  sendHistory(win, posted, rows(false));
  starOf('A').click();
  starOf('A').click();
  groupBy('hf-group-chat');
  assert.strictEqual(pending(), 1, 'BUG: the list agreed with itself');
  assert.deepStrictEqual(armed(), [10100]);
  now += 10100;
  assert.ok(boundPasses(10100), 'BUG: an unanswered star asked for nothing at its bound');
  assert.strictEqual(pending(), 0);
  win.close();
}

main();
testARemoteRowOfTheSameTaskIdKeepsItsOwnFlag();
testAStarEndsAtItsBound();
console.log('historyFavoriteFastPath_sweep1026.test.js: all tests passed');
