// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end tests of how the CHATS panel loads the rows that fill it.
//
// The list shows fewer rows than it holds (cron tasks are hidden by
// default, a chat's tasks fold into its panel), so one page of 50
// rarely fills the panel.  The page used to ask for 50 rows at every
// refresh and then, the panel not being full, for the next 50 and the
// next, each under a "Loading..." row: a row that came and went with
// every piece of task news, and a chain that task news arriving in
// its middle started over.  Now a refresh asks, in one request, for
// as many rows as the list has asked for so far, those of a page on
// its way included; a list that does not fill its panel asks for as
// many rows again, and shows its "Loading..." row only while it has
// no task to show; a list the user scrolls asks, as soon as less than
// a panel's height of rows is left under the view, for as many rows as
// fill a panel, and again while a page leaves it short of that; and
// the rows of a refresh that a daemon sends in
// several pages (it grants fewer than were asked for: its most for a
// page, or the 50 of an older one, whose reply names no `limit`) are
// shown when the last is in, the list staying as it is until then.

const assert = require('assert');
const {send, ofType} = require('./simplify2_harness');
const {
  openChatsPage,
  historyRows,
  lastGetHistory,
  layOutHistoryList,
  pressFilterOption,
  finishedSession: session,
} = require('./history_helpers');

/**
 * Rows *offset* to *offset* + *count* - 1 of the daemon's stream, the
 * newest first: every fifth is a plain task, the others are cron tasks,
 * which the list hides.
 */
function stream(offset, count) {
  return Array.from({length: count}, (_, i) => {
    const k = offset + i;
    return session(10_000 - k, 'task ' + k, {is_cron: k % 5 !== 0});
  });
}

/** A connected page whose open Chats panel is 1000px tall: 25 rows of 40px fill it. */
function openPage() {
  const {win, posted} = openChatsPage();
  layOutHistoryList(win, 1000);
  send(win, {type: 'daemonStatus', connected: true});
  return {win, posted};
}

/**
 * Answer getHistory *req* (the page's last one by default) with
 * *sessions*: as a daemon that grants the rows asked for (its reply
 * names the `limit`), or as an older one (*old*), which names none.
 */
function answer(win, posted, sessions, old, req = lastGetHistory(posted)) {
  const reply = {
    type: 'history',
    sessions,
    offset: req.offset,
    generation: req.generation,
  };
  if (!old) reply.limit = req.limit;
  send(win, reply);
}

/** The [offset, limit] of every getHistory posted from message *from* on. */
function asks(posted, from) {
  return ofType(posted.slice(from), 'getHistory').map(m => [m.offset, m.limit]);
}

/** The "Loading..." row when it is on show, else null. */
function loader(win) {
  const row = win.document.getElementById('history-loader');
  return row && row.style.display !== 'none' ? row : null;
}

function testARefreshAsksOnceForTheRowsThatFillThePanel() {
  const {win, posted} = openPage();
  // A first visit knows no better than one page, whose ten tasks do not
  // fill the panel: as many rows again are asked for, twice, and no
  // "Loading..." row comes and goes meanwhile.
  assert.deepStrictEqual(asks(posted, 0), [[0, 50]]);
  answer(win, posted, stream(0, 50));
  assert.deepStrictEqual(asks(posted, 0), [[0, 50], [50, 50]]);
  assert.strictEqual(loader(win), null);
  answer(win, posted, stream(50, 50));
  assert.deepStrictEqual(asks(posted, 0), [[0, 50], [50, 50], [100, 100]]);
  assert.strictEqual(loader(win), null);
  answer(win, posted, stream(100, 100));
  assert.strictEqual(asks(posted, 0).length, 3, 'a full panel asks no more');
  assert.strictEqual(historyRows(win).length, 200);
  // Task news: one request for the rows held.  The same rows coming
  // back leave the list as it is.
  let from = posted.length;
  const before = historyRows(win);
  send(win, {type: 'tasks_updated'});
  assert.deepStrictEqual(asks(posted, from), [[0, 200]]);
  answer(win, posted, stream(0, 200));
  historyRows(win).forEach((row, i) => assert.strictEqual(row, before[i]));
  // A new task: the list is built again from the one reply.
  send(win, {type: 'tasks_updated'});
  answer(win, posted, [session(20_000, 'new task')].concat(stream(0, 199)));
  assert.strictEqual(historyRows(win).length, 200);
  assert.notStrictEqual(historyRows(win)[0], before[0]);
  assert.deepStrictEqual(asks(posted, from), [[0, 200], [0, 200]]);
  assert.strictEqual(loader(win), null);
  // Scrolled to its end, the list asks for the rows that fill a panel
  // (41 of its 200 rows show: 122 more bring 25), and says so.
  from = posted.length;
  const list = win.document.getElementById('history-list');
  list.scrollTop = 600;
  list.dispatchEvent(new win.Event('scroll'));
  assert.deepStrictEqual(asks(posted, from), [[200, 122]]);
  assert.strictEqual(loader(win).textContent, 'Loading...');
  // The row stays there while the list is as full: with the cron
  // tasks on show too.
  pressFilterOption(win, 'cron', 'Shown');
  assert.strictEqual(loader(win).textContent, 'Loading...');
  // A search is another view: its first page.
  const search = win.document.getElementById('history-search');
  search.value = 'parser';
  search.dispatchEvent(new win.Event('input', {bubbles: true}));
  assert.deepStrictEqual(asks(posted, from).pop(), [0, 50]);
  win.close();
  console.log('  ok - a refresh asks once for the rows that fill the panel');
}

function testAScrollAsksForTheRowsThatFillAPanel() {
  const {win, posted} = openPage();
  const list = win.document.getElementById('history-list');
  const scrollTo = top => {
    list.scrollTop = top;
    list.dispatchEvent(new win.Event('scroll'));
  };
  const cronRows = (offset, count) =>
    Array.from({length: count}, (_, i) =>
      session(10_000 - offset - i, 'task ' + (offset + i), {is_cron: true}),
    );
  answer(win, posted, stream(0, 50));
  answer(win, posted, stream(50, 50));
  answer(win, posted, stream(100, 100));
  // 200 rows, 40 of them on show: 1600px of rows in a panel of 1000px.
  // At its top the list has what fills the panel, and asks for nothing.
  let from = posted.length;
  scrollTo(0);
  assert.deepStrictEqual(asks(posted, from), []);
  // The first scroll, far from the end: 500px of rows are left under
  // the view, less than a panel's, so the rows that fill one are asked
  // for (one row in five shows: 125 bring 25), under a "Loading..." row.
  scrollTo(100);
  assert.deepStrictEqual(asks(posted, from), [[200, 125]]);
  assert.strictEqual(loader(win).textContent, 'Loading...');
  // Scrolling on while they are on their way asks for nothing more.
  scrollTo(300);
  assert.strictEqual(asks(posted, from).length, 1);
  // They are in: 2600px of rows, 1300px of them under the view.
  answer(win, posted, stream(200, 125));
  assert.strictEqual(historyRows(win).length, 325);
  assert.strictEqual(asks(posted, from).length, 1);
  assert.strictEqual(loader(win), null);
  // A page with no task to show (cron tasks only) leaves the view as
  // short of a panel as it was: the next rows are asked for at once.
  from = posted.length;
  scrollTo(700);
  assert.deepStrictEqual(asks(posted, from), [[325, 125]]);
  answer(win, posted, cronRows(325, 125));
  assert.deepStrictEqual(asks(posted, from), [[325, 125], [450, 174]]);
  assert.strictEqual(loader(win).textContent, 'Loading...');
  answer(win, posted, stream(450, 174));
  assert.strictEqual(asks(posted, from).length, 2, 'a panel of rows is under the view');
  assert.strictEqual(loader(win), null);
  // The stream ends with a short page: no scroll asks after it.
  from = posted.length;
  scrollTo(2500);
  assert.deepStrictEqual(asks(posted, from), [[624, 156]]);
  answer(win, posted, stream(624, 10));
  scrollTo(3000);
  assert.strictEqual(asks(posted, from).length, 1);
  assert.strictEqual(loader(win), null);
  win.close();
  console.log('  ok - a scroll asks for the rows that fill a panel');
}

function testARefreshKeepsTheRowsOfAPageOnItsWay() {
  const {win, posted} = openPage();
  answer(win, posted, stream(0, 50));
  const fill = lastGetHistory(posted);
  assert.deepStrictEqual([fill.offset, fill.limit], [50, 50]);
  // Task news while the second fifty are on their way: the list starts
  // over, and asks for them too.  Their own reply is of no use now.
  send(win, {type: 'tasks_updated'});
  assert.deepStrictEqual(asks(posted, 0), [[0, 50], [50, 50], [0, 100]]);
  const refresh = lastGetHistory(posted);
  answer(win, posted, stream(50, 50), false, fill);
  assert.strictEqual(historyRows(win).length, 50);
  answer(win, posted, stream(0, 100), false, refresh);
  assert.strictEqual(historyRows(win).length, 100);
  // And so again with the next hundred, which fill the panel.
  assert.deepStrictEqual(asks(posted, 0).slice(3), [[100, 100]]);
  send(win, {type: 'tasks_updated'});
  assert.deepStrictEqual(asks(posted, 0).slice(4), [[0, 200]]);
  answer(win, posted, stream(0, 200));
  assert.strictEqual(historyRows(win).length, 200);
  assert.strictEqual(asks(posted, 0).length, 5);
  assert.strictEqual(loader(win), null);
  win.close();
  console.log('  ok - a refresh keeps the rows of a page on its way');
}

function testAListThatHoldsEveryRowAsksForMoreThanItHolds() {
  const {win, posted} = openPage();
  answer(win, posted, stream(0, 50));
  // The stream ends 70 rows in: the panel is not full, and there is
  // nothing left to ask for.
  answer(win, posted, stream(50, 20));
  const from = posted.length;
  assert.strictEqual(historyRows(win).length, 70);
  // The rows held, coming back alone, would not tell of a new one: the
  // hundred asked for so far are asked for again.
  send(win, {type: 'tasks_updated'});
  assert.deepStrictEqual(asks(posted, from), [[0, 100]]);
  answer(win, posted, stream(0, 70));
  // A new task is listed, and the list still holds every row.
  send(win, {type: 'tasks_updated'});
  answer(win, posted, [session(20_000, 'new task')].concat(stream(0, 70)));
  assert.strictEqual(historyRows(win).length, 71);
  assert.deepStrictEqual(asks(posted, from), [[0, 100], [0, 100]]);
  assert.strictEqual(loader(win), null);
  win.close();
  console.log('  ok - a list that holds every row asks for more than it holds');
}

function testTheLoadingRowIsOnShowWhileNoTaskIs() {
  const {win, posted} = openPage();
  const noMatch = () => win.document.querySelector('.sidebar-empty-filter');
  answer(win, posted, stream(0, 50));
  assert.strictEqual(loader(win), null, 'ten tasks, and more rows on their way');
  // Only favourites: none of the rows held is one.  The rows on their
  // way may hold some, so the list says that it is loading, and not
  // that no task matches.
  pressFilterOption(win, 'favorites', 'Only favorites');
  assert.strictEqual(loader(win).textContent, 'Loading...');
  assert.strictEqual(noMatch(), null);
  // They hold none either, and nothing follows them.
  answer(win, posted, stream(50, 20));
  assert.strictEqual(loader(win), null);
  assert.strictEqual(noMatch().textContent, 'No tasks match the filter');
  win.close();
  console.log('  ok - the Loading row is on show while no task is');
}

/**
 * Rows *offset* to *offset* + *count* - 1 of a stream whose newest 120
 * rows are cron tasks, which the list hides: no task to show comes
 * before the third page.
 */
function cronFirst(offset, count) {
  return Array.from({length: count}, (_, i) => {
    const k = offset + i;
    return session(10_000 - k, 'task ' + k, {is_cron: k < 120});
  });
}

function testARefreshSentInPagesIsShownWhenTheLastIsIn() {
  const {win, posted} = openPage();
  // A first visit: the list says that it is loading while it has no
  // task to show, and no longer once it has.
  answer(win, posted, cronFirst(0, 50));
  assert.strictEqual(loader(win).textContent, 'Loading...');
  answer(win, posted, cronFirst(50, 50));
  answer(win, posted, cronFirst(100, 100));
  assert.deepStrictEqual(asks(posted, 0), [[0, 50], [50, 50], [100, 100]]);
  assert.strictEqual(loader(win), null);
  const before = historyRows(win);
  assert.strictEqual(before.length, 200);
  const untouched = () => {
    assert.strictEqual(historyRows(win).length, 200);
    historyRows(win).forEach((row, i) => assert.strictEqual(row, before[i]));
    assert.strictEqual(loader(win), null);
  };
  // Task news, and a daemon that grants a hundred rows a page: the
  // first hundred, none of them a task to show, neither empty the
  // list nor have it say that it is loading.  The rest is asked for.
  const granted = () => Object.assign({}, lastGetHistory(posted), {limit: 100});
  let from = posted.length;
  send(win, {type: 'tasks_updated'});
  const first = granted();
  answer(win, posted, cronFirst(0, 100), false, first);
  assert.deepStrictEqual(asks(posted, from), [[0, 200], [100, 100]]);
  untouched();
  // The same page once more is none of the rows awaited.
  answer(win, posted, cronFirst(0, 100), false, first);
  // More task news waits for the rows on their way.
  send(win, {type: 'tasks_updated'});
  assert.strictEqual(asks(posted, from).length, 2);
  untouched();
  // The last of them: the rows the list holds, which it keeps.  The
  // news that waited is asked after now.
  answer(win, posted, cronFirst(100, 100));
  untouched();
  assert.deepStrictEqual(asks(posted, from).slice(2), [[0, 200]]);
  // Another view is asked for while a refresh's rows come in: they
  // are let go of, and the view's first page is shown as it comes.
  from = posted.length;
  answer(win, posted, cronFirst(0, 100), false, granted());
  const rest = lastGetHistory(posted);
  assert.deepStrictEqual([rest.offset, rest.limit], [100, 100]);
  const search = win.document.getElementById('history-search');
  search.value = 'parser';
  search.dispatchEvent(new win.Event('input', {bubbles: true}));
  assert.deepStrictEqual(asks(posted, from), [[100, 100], [0, 50]]);
  const found = lastGetHistory(posted);
  answer(win, posted, cronFirst(100, 100), false, rest);
  untouched();
  answer(win, posted, [session(1, 'parser fix')], false, found);
  assert.strictEqual(historyRows(win).length, 1);
  assert.strictEqual(asks(posted, from).length, 2);
  win.close();
  console.log('  ok - a refresh sent in pages is shown when the last is in');
}

function testAnOlderDaemonsRefreshIsShownWhenItsLastPageIsIn() {
  const {win, posted} = openPage();
  answer(win, posted, stream(0, 50), true);
  answer(win, posted, stream(50, 50), true);
  // Asked for 100 rows, it sends its 50: more may follow them.
  answer(win, posted, stream(100, 50), true);
  assert.deepStrictEqual(asks(posted, 0), [[0, 50], [50, 50], [100, 100]]);
  assert.strictEqual(historyRows(win).length, 150);
  // Its answer to a refresh is one page too: the rest of the rows
  // asked for are asked for page after page, and the list is left as
  // it is meanwhile.
  const from = posted.length;
  const before = historyRows(win);
  send(win, {type: 'tasks_updated'});
  answer(win, posted, stream(0, 50), true);
  answer(win, posted, stream(50, 50), true);
  answer(win, posted, stream(100, 50), true);
  assert.deepStrictEqual(asks(posted, from), [
    [0, 200],
    [50, 150],
    [100, 100],
    [150, 50],
  ]);
  assert.strictEqual(historyRows(win).length, 150);
  historyRows(win).forEach((row, i) => assert.strictEqual(row, before[i]));
  // The last page: the list is built from the 200 rows at once.
  answer(win, posted, stream(150, 50), true);
  assert.strictEqual(historyRows(win).length, 200);
  assert.strictEqual(asks(posted, from).length, 4);
  assert.strictEqual(loader(win), null);
  // Its stream ends before the rows asked for are in: a short page.
  send(win, {type: 'tasks_updated'});
  answer(win, posted, stream(0, 50), true);
  answer(win, posted, stream(50, 30), true);
  assert.deepStrictEqual(asks(posted, from).slice(4), [[0, 200], [50, 150]]);
  assert.strictEqual(historyRows(win).length, 80);
  assert.strictEqual(loader(win), null);
  win.close();
  console.log("  ok - an older daemon's refresh is shown when its last page is in");
}

testARefreshAsksOnceForTheRowsThatFillThePanel();
testAScrollAsksForTheRowsThatFillAPanel();
testARefreshKeepsTheRowsOfAPageOnItsWay();
testAListThatHoldsEveryRowAsksForMoreThanItHolds();
testTheLoadingRowIsOnShowWhileNoTaskIs();
testARefreshSentInPagesIsShownWhenTheLastIsIn();
testAnOlderDaemonsRefreshIsShownWhenItsLastPageIsIn();
console.log('historyFill.test.js: all tests passed');
