// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests of the Chats panel's Filter and group menu
// (media/chat.html, media/main.js "The Filter and group menu"): the
// button beside the search box replaces the flat-list switch and the
// Filters panel; its menu has one row per filter and a Group by row;
// a row shows its setting's value and a submenu offers the options,
// the chosen ones ticked; an option sets the hf-* control that holds
// the state, which filters the list; the button says how many filters
// are on.
//
// jsdom has no popovers: opening, placing, the keyboard, hovering and
// Escape are tested in real Chromium
// (src/kiss/tests/webapp/test_history_filter_menu.py).  The filters
// with daemon-side parts have suites of their own (historyTagFilter,
// historyMachineFilter, historyWorkDirFilter, historySeaFilter,
// historyCronChip) and Group by is in historyLegacyViewToggle.

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');
const {test, report} = require('./ui_antipattern_harness').makeRunner();
const {
  visibleTitles,
  makeRow,
  finishedSession,
  STATUS_CHIP_SESSIONS,
  filterSubmenu,
  filterOptions,
  tickedFilterOptions,
  pressFilterOption,
  filterRow,
  filterRowValue,
  freezeClock,
} = require('./history_helpers');

const KEYS = [
  'status',
  'favorites',
  'cron',
  'date',
  'sea',
  'tag',
  'machine',
  'workdir',
  'group',
];

/** The Filter and group button. */
function button(win) {
  return win.document.getElementById('history-filter-btn');
}

/** Set a date box as a typed or picked day does. */
function setDay(win, id, iso) {
  const box = win.document.getElementById(id);
  box.value = iso;
  box.dispatchEvent(new win.Event('change', {bubbles: true}));
}

// The year of the Date test's days.  Its page's clock stands at a noon
// of that year: the Date row names the year of a day only when it is
// not this year, also in a run that crosses New Year's midnight.
const YEAR = 2026;
const NOW_MS = Date.parse(YEAR + '-06-15T12:00:00');

/**
 * The day `iso` (yyyy-mm-dd) as the Date row words it in the locale of
 * this run ("Mar 1" in en-US, "1. März" in de-DE): the short month and
 * the day, with the year when `withYear`.
 */
function dayText(iso, withYear) {
  const opts = {month: 'short', day: 'numeric'};
  if (withYear) opts.year = 'numeric';
  return new Date(iso + 'T00:00:00').toLocaleDateString(undefined, opts);
}

test('the menu has a row and a submenu per filter, then Group by; the Filters panel is gone', () => {
  const {win} = makePage();
  const doc = win.document;
  const menu = doc.getElementById('history-filter-menu');
  assert.strictEqual(button(win).getAttribute('popovertarget'), menu.id);
  assert.strictEqual(menu.getAttribute('popover'), 'auto');
  const rows = Array.from(menu.querySelectorAll(':scope > .fg-row'));
  assert.deepStrictEqual(
    rows.map(r => [r.dataset.fg, r.querySelector('.fg-row-name').textContent]),
    [
      ['status', 'Status'],
      ['favorites', 'Favorites'],
      ['cron', 'Cron tasks'],
      ['date', 'Date'],
      ['sea', 'SEA'],
      ['tag', 'Tag'],
      ['machine', 'Machine'],
      ['workdir', 'Work dir'],
      ['group', 'Group by'],
    ],
  );
  for (const row of rows) {
    const sub = filterSubmenu(win, row.dataset.fg);
    assert.strictEqual(row.nextElementSibling, sub, row.dataset.fg);
    assert.strictEqual(row.getAttribute('popovertarget'), sub.id);
    assert.strictEqual(row.getAttribute('popovertargetaction'), 'show');
    assert.strictEqual(sub.getAttribute('popover'), 'auto');
    assert.strictEqual(row.getAttribute('aria-expanded'), 'false');
  }
  // A hairline parts Group by from the filters.
  assert.ok(filterRow(win, 'group').previousElementSibling.classList.contains('fg-sep'));
  assert.deepStrictEqual(
    KEYS.map(key => filterRowValue(win, key)),
    ['All', 'All', 'Hidden', 'All', 'All', 'All', 'All', 'All', 'Project'],
  );
  assert.deepStrictEqual(filterOptions(win, 'group'), ['Project', 'Chat', 'None']);
  assert.strictEqual(button(win).getAttribute('aria-label'), 'Filter and group');
  assert.strictEqual(button(win).dataset.tooltip, 'Filter and group');
  assert.ok(!button(win).classList.contains('is-filtered'));
  for (const gone of [
    '#history-view-toggle',
    '#history-filters-panel',
    '#history-filters-toggle',
    '.hf-chip',
  ]) {
    assert.strictEqual(doc.querySelector(gone), null, gone);
  }
  win.close();
});

test('Status options are ticked one by one; Favorites offers All or Only favorites', () => {
  const {win} = makePage();
  const fav = makeRow({id: 'chatF', task_id: 4, title: 'starred task', is_favorite: true});
  send(win, {type: 'history', sessions: STATUS_CHIP_SESSIONS.concat([fav]), offset: 0});
  assert.deepStrictEqual(filterOptions(win, 'status'), ['Running', 'Errored', 'Succeeded']);
  assert.deepStrictEqual(tickedFilterOptions(win, 'status'), ['Running', 'Errored', 'Succeeded']);

  pressFilterOption(win, 'status', 'Errored');
  assert.strictEqual(win.document.getElementById('hf-errors').checked, false);
  assert.deepStrictEqual(visibleTitles(win), ['running task', 'starred task', 'succeeded task']);
  assert.strictEqual(filterRowValue(win, 'status'), 'Running, Succeeded');
  assert.strictEqual(button(win).getAttribute('aria-label'), 'Filter and group (1 filter on)');

  pressFilterOption(win, 'status', 'Running');
  pressFilterOption(win, 'status', 'Succeeded');
  assert.deepStrictEqual(tickedFilterOptions(win, 'status'), []);
  assert.strictEqual(filterRowValue(win, 'status'), 'None');
  assert.deepStrictEqual(visibleTitles(win), []);
  for (const label of ['Running', 'Errored', 'Succeeded']) {
    pressFilterOption(win, 'status', label);
  }
  assert.strictEqual(filterRowValue(win, 'status'), 'All');
  assert.ok(!button(win).classList.contains('is-filtered'));

  assert.deepStrictEqual(filterOptions(win, 'favorites'), ['All', 'Only favorites']);
  pressFilterOption(win, 'favorites', 'Only favorites');
  assert.strictEqual(win.document.getElementById('hf-favorite').checked, true);
  assert.deepStrictEqual(tickedFilterOptions(win, 'favorites'), ['Only favorites']);
  assert.strictEqual(filterRowValue(win, 'favorites'), 'Only favorites');
  assert.deepStrictEqual(visibleTitles(win), ['starred task']);
  // Two settings that narrow the list, counted on the button.
  pressFilterOption(win, 'status', 'Running');
  assert.strictEqual(button(win).getAttribute('aria-label'), 'Filter and group (2 filters on)');
  assert.strictEqual(button(win).dataset.tooltip, 'Filter and group (2 filters on)');
  assert.ok(button(win).classList.contains('is-filtered'));
  pressFilterOption(win, 'favorites', 'All');
  assert.strictEqual(win.document.getElementById('hf-favorite').checked, false);
  assert.strictEqual(filterRowValue(win, 'favorites'), 'All');
  win.close();
});

test('the Date row shows the days the user set, and not the range a history reply fills in', () => {
  const {win} = makePage({beforeScripts: w => freezeClock(w, NOW_MS)});
  const clear = win.document.getElementById('hf-date-clear');
  const old = finishedSession(1, 'old task');
  const noon = Date.parse(YEAR + '-03-05T12:00:00');
  const recent = finishedSession(2, 'recent task', {
    timestamp: noon / 1000,
    startTs: noon,
    endTs: noon + 10_000,
  });
  send(win, {
    type: 'history',
    sessions: [old, recent],
    offset: 0,
    dateRange: {min: old.timestamp, max: recent.timestamp},
  });
  // The reply's first and last day fill the boxes; that narrows nothing.
  assert.strictEqual(
    win.document.getElementById('hf-to').value,
    YEAR + '-03-05',
  );
  assert.strictEqual(filterRowValue(win, 'date'), 'All');
  assert.ok(!button(win).classList.contains('is-filtered'));

  setDay(win, 'hf-from', YEAR + '-03-01');
  assert.strictEqual(
    filterRowValue(win, 'date'),
    dayText(YEAR + '-03-01') + ' \u2013 ' + dayText(YEAR + '-03-05'),
  );
  assert.deepStrictEqual(visibleTitles(win), ['recent task']);
  assert.strictEqual(button(win).getAttribute('aria-label'), 'Filter and group (1 filter on)');
  assert.notStrictEqual(clear.style.display, 'none');

  setDay(win, 'hf-to', '');
  assert.strictEqual(
    filterRowValue(win, 'date'),
    'From ' + dayText(YEAR + '-03-01'),
  );
  setDay(win, 'hf-from', '');
  setDay(win, 'hf-to', '2023-11-20');
  // A day of another year is shown with its year.
  assert.strictEqual(
    filterRowValue(win, 'date'),
    'Until ' + dayText('2023-11-20', true),
  );
  assert.deepStrictEqual(visibleTitles(win), ['old task']);

  clear.click();
  assert.strictEqual(filterRowValue(win, 'date'), 'All');
  assert.strictEqual(clear.style.display, 'none');
  assert.ok(!button(win).classList.contains('is-filtered'));
  assert.deepStrictEqual(visibleTitles(win), ['old task', 'recent task']);
  win.close();
});

test('a select\u2019s option rows are rebuilt only when its options change, and the keyboard stays on its option', () => {
  const {win} = makePage();
  const sessions = ['alpha', 'beta'].map((sea, i) => finishedSession(i + 1, sea + ' task', {sea}));
  send(win, {type: 'history', sessions, offset: 0});
  assert.deepStrictEqual(filterOptions(win, 'sea'), ['All', 'None', 'alpha', 'beta']);
  const rowsOf = () => Array.from(filterSubmenu(win, 'sea').querySelectorAll('.fg-option'));
  const before = rowsOf();
  before[3].focus();
  // The same options again (every history reply rebuilds the select):
  // the rows are the elements they were (rows built anew would be
  // deep-equal to them, so each is compared by identity) and the
  // keyboard is on the row it was on.
  send(win, {type: 'history', sessions: sessions.slice().reverse(), offset: 0});
  const after = rowsOf();
  assert.strictEqual(after.length, before.length);
  for (let i = 0; i < before.length; i++) {
    assert.strictEqual(after[i], before[i], 'the same element: option ' + i);
  }
  assert.strictEqual(win.document.activeElement, before[3], 'the keyboard');

  // A new option: new rows, the keyboard on the option it was on.
  sessions.push(finishedSession(3, 'aardvark task', {sea: 'aardvark'}));
  send(win, {type: 'history', sessions, offset: 0});
  assert.deepStrictEqual(filterOptions(win, 'sea'), ['All', 'None', 'aardvark', 'alpha', 'beta']);
  assert.strictEqual(win.document.activeElement, rowsOf()[4]);
  assert.strictEqual(win.document.activeElement.textContent, 'beta');

  // Its option is gone: the first option takes the keyboard.
  send(win, {type: 'history', sessions: sessions.slice(0, 1), offset: 0});
  assert.deepStrictEqual(filterOptions(win, 'sea'), ['All', 'None', 'alpha']);
  assert.strictEqual(win.document.activeElement, rowsOf()[0]);
  win.close();
});

report('historyFilterMenu');
