// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for the history panel's two views
// (media/main.js, media/chat.html):
//
// * Group by > None, in the Filter and group menu whose button sits
//   right of the search box, switches to the flat list: every task
//   newest first, no chat panels, no day separators, and no per-chat
//   colour on any row; Group by > Chat switches back (Group by >
//   Project, the default, lists the same chat panels while no project
//   exists; historyProjects.test.js covers the project panels),
// * the choice is remembered in localStorage and honoured on load,
// * switching rebuilds the loaded rows without a refetch — through the
//   identical-refresh fast path, which must not keep the old view — and
//   a later identical refresh keeps the flat rows, and
// * the status filter and the keyboard focus restore work in the flat
//   list too.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  stubRaf,
  fakeMatchMedia,
} = require('./simplify2_harness');
const {
  todayNoonSec,
  dayLabel,
  freezeClock,
  groupedSession: session,
  sendHistoryPage: sendHistory,
  listShape,
  pressFilterOption,
  tickedFilterOptions,
  filterRowValue,
} = require('./history_helpers');

const VIEW_KEY = 'kissSorcar.historyLegacyView';

const {test, report} = require('./ui_antipattern_harness').makeRunner();
const {byId, all} = require('./ui_antipattern_harness');

const DAY = 86400;
// The page's clock stays at the noon the fixture rows are dated from, so
// a run that crosses midnight still files them under Today / Yesterday.
const todayNoon = todayNoonSec();

function makePage(beforeMain) {
  function beforeScripts(win) {
    stubRaf(win);
    fakeMatchMedia(win, true);
    freezeClock(win, todayNoon * 1000);
    if (beforeMain) beforeMain(win);
  }
  return makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts,
    extraScripts: [
      'marked.min.js',
      'contentContextMenu.js',
      'treeContextMenu.js',
    ],
  });
}

function page() {
  return [
    session('A', 'a2', todayNoon),
    session('B', 'b2', todayNoon - 600),
    session('A', 'a1', todayNoon - 1200),
    session('C', 'c1', todayNoon - DAY),
    session('B', 'b1', todayNoon - DAY - 60),
  ];
}

const GROUPED = [
  'sep:Today',
  'chat:A[task a2,task a1]',
  'chat:B[task b2,task b1]',
  'sep:Yesterday',
  'chat:C[task c1]',
];
const FLAT = [
  'sep:Today',
  'row:task a2',
  'row:task b2',
  'row:task a1',
  'sep:Yesterday',
  'row:task c1',
  'row:task b1',
];

function dateSeparator(daysAgo) {
  return 'sep:' + dayLabel(todayNoon - daysAgo * DAY, todayNoon);
}

test('Group by swaps grouped <-> flat without a refetch; its menu button sits right of the search box', () => {
  const {win, posted} = makePage();
  const button = byId(win, 'history-filter-btn');
  const searchWrap = byId(win, 'history-search').closest('.search-wrap');
  assert.strictEqual(
    searchWrap.nextElementSibling,
    button,
    'the button is the search box\u2019s next sibling (to its right)',
  );
  assert.strictEqual(win.document.getElementById('history-view-toggle'), null);
  assert.deepStrictEqual(tickedFilterOptions(win, 'group'), ['Project']);
  assert.strictEqual(filterRowValue(win, 'group'), 'Project');
  assert.ok(!byId(win, 'history-list').classList.contains('legacy-view'));

  sendHistory(win, posted, 0, page());
  assert.deepStrictEqual(listShape(win), GROUPED);
  const fetchesBefore = posted.filter(m => m.type === 'getHistory').length;

  pressFilterOption(win, 'group', 'None');
  assert.deepStrictEqual(tickedFilterOptions(win, 'group'), ['None']);
  assert.strictEqual(filterRowValue(win, 'group'), 'None');
  // Group by arranges the list and hides nothing: no filter is on.
  assert.ok(!button.classList.contains('is-filtered'));
  assert.strictEqual(button.getAttribute('aria-label'), 'Filter and group');
  assert.ok(byId(win, 'history-list').classList.contains('legacy-view'));
  assert.strictEqual(win.localStorage.getItem(VIEW_KEY), '1');
  assert.strictEqual(
    posted.filter(m => m.type === 'getHistory').length,
    fetchesBefore,
    'switching views rebuilds locally, without a refetch',
  );
  assert.deepStrictEqual(listShape(win), FLAT, 'flat, newest first');
  assert.strictEqual(all(win, '#history-list .history-chat-group').length, 0);
  assert.strictEqual(all(win, '#history-list .history-day-sep').length, 2);

  // No per-chat colour: the flat rows carry no inline style at all.
  all(win, '#history-list > .sidebar-item').forEach(r => {
    assert.strictEqual(
      r.getAttribute('style'),
      null,
      'flat rows carry no colour',
    );
  });

  // An identical refresh keeps the flat rows (fast path, same view).
  const rowsBefore = all(win, '#history-list > .sidebar-item');
  sendHistory(win, posted, 0, page());
  assert.deepStrictEqual(listShape(win), FLAT);
  assert.strictEqual(
    all(win, '#history-list > .sidebar-item')[0],
    rowsBefore[0],
    'an identical refresh keeps the DOM rows',
  );

  // The option that is the choice already rebuilds nothing.
  pressFilterOption(win, 'group', 'None');
  assert.strictEqual(all(win, '#history-list > .sidebar-item')[0], rowsBefore[0]);

  // Back to chat panels: the same data, regrouped.
  pressFilterOption(win, 'group', 'Chat');
  assert.deepStrictEqual(tickedFilterOptions(win, 'group'), ['Chat']);
  assert.strictEqual(win.localStorage.getItem(VIEW_KEY), '0');
  assert.ok(!byId(win, 'history-list').classList.contains('legacy-view'));
  assert.deepStrictEqual(listShape(win), GROUPED);
  win.close();
});

test('the flat list is remembered across loads and used for the first page', () => {
  const {win, posted} = makePage(w => {
    w.localStorage.setItem(VIEW_KEY, '1');
  });
  assert.deepStrictEqual(tickedFilterOptions(win, 'group'), ['None']);
  assert.strictEqual(filterRowValue(win, 'group'), 'None');
  assert.ok(byId(win, 'history-list').classList.contains('legacy-view'));
  sendHistory(win, posted, 0, page());
  assert.deepStrictEqual(listShape(win), FLAT);
  // Later pages extend the flat list in order.
  sendHistory(win, posted, 5, [session('D', 'd1', todayNoon - 3 * DAY)]);
  assert.deepStrictEqual(
    listShape(win),
    FLAT.concat([dateSeparator(3), 'row:task d1']),
  );
  // A row without a chat id is listed like any other.
  sendHistory(win, posted, 6, [session('', 'e1', todayNoon - 4 * DAY)]);
  assert.deepStrictEqual(
    listShape(win),
    FLAT.concat([
      dateSeparator(3),
      'row:task d1',
      dateSeparator(4),
      'row:task e1',
    ]),
  );
  win.close();
});

test('Group by with nothing loaded only flips the view; filters and focus work in the flat list', () => {
  const {win, posted} = makePage();
  pressFilterOption(win, 'group', 'None');
  assert.strictEqual(
    byId(win, 'history-list').querySelectorAll('.sidebar-item').length,
    0,
  );
  assert.ok(byId(win, 'history-list').classList.contains('legacy-view'));
  sendHistory(win, posted, 0, [
    session('A', 'a2', todayNoon),
    Object.assign(session('B', 'b1', todayNoon - 600), {failed: true}),
    Object.assign(session('A', 'a1', todayNoon - 1200), {is_running: true}),
  ]);
  assert.deepStrictEqual(listShape(win), [
    'sep:Running',
    'row:task a1',
    'sep:Today',
    'row:task a2',
    'row:task b1',
  ]);
  // Status > Errored, unticked, hides the failed row only.
  pressFilterOption(win, 'status', 'Errored');
  assert.deepStrictEqual(tickedFilterOptions(win, 'status'), ['Running', 'Succeeded']);
  assert.strictEqual(filterRowValue(win, 'status'), 'Running, Succeeded');
  assert.deepStrictEqual(listShape(win), [
    'sep:Running',
    'row:task a1',
    'sep:Today',
    'row:task a2',
  ]);
  pressFilterOption(win, 'status', 'Errored');
  assert.strictEqual(filterRowValue(win, 'status'), 'All');
  // Keyboard focus on a row survives a changed-data rebuild.
  const rowB = all(win, '#history-list > .sidebar-item')[2];
  rowB.focus();
  sendHistory(win, posted, 0, [
    session('A', 'a3', todayNoon + 60),
    session('A', 'a2', todayNoon),
    Object.assign(session('B', 'b1', todayNoon - 600), {failed: true}),
    session('A', 'a1', todayNoon - 1200),
  ]);
  assert.deepStrictEqual(listShape(win), [
    'sep:Today',
    'row:task a3',
    'row:task a2',
    'row:task b1',
    'row:task a1',
  ]);
  const active = win.document.activeElement;
  assert.strictEqual(
    active.querySelector('.sidebar-item-text').textContent,
    'task b1',
    'focus follows the same task after the rebuild',
  );
  win.close();
});

report('historyLegacyViewToggle');
