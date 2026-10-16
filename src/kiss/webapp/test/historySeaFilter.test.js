// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of the History panel's SEA (agent script) filter: the
// <select id="hf-sea"> lists the agent scripts of the loaded rows, each
// row carries its SEA as data-sea and a label, and choosing an option
// hides the rows that were run by another script (or, with "None", the
// rows that were run by any script).

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

function session(id, title, sea, extra) {
  return Object.assign(
    {
      id: 'chat-' + id,
      task_id: id,
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
      sea,
    },
    extra || {},
  );
}

const SESSIONS_FIXTURE = [
  session(1, 'paper task', 'write_paper_sea'),
  session(2, 'plain task', ''),
  session(3, 'cron task', 'cron_agent'),
  session(4, 'second paper task', ' write_paper_sea '),
  session(5, 'legacy row without sea field', undefined),
];

function visibleTitles(win) {
  const rows = win.document
    .getElementById('history-list')
    .querySelectorAll('.sidebar-item');
  return Array.from(rows)
    .filter(r => r.style.display !== 'none')
    .map(r => r.querySelector('.sidebar-item-text').textContent)
    .sort();
}

function optionValues(win) {
  return Array.from(win.document.getElementById('hf-sea').options).map(
    o => o.value,
  );
}

function selectSea(win, value) {
  const sel = win.document.getElementById('hf-sea');
  sel.value = value;
  sel.dispatchEvent(new win.Event('change', {bubbles: true}));
}

function testMarkupAndDefault() {
  const {win} = makePage();
  const doc = win.document;
  const sel = doc.getElementById('hf-sea');
  assert.ok(sel, 'hf-sea select must exist');
  assert.strictEqual(sel.tagName, 'SELECT');
  assert.ok(
    doc.querySelector('.history-filter-bar .history-filter-sea'),
    'the SEA group must live inside the filter bar',
  );
  assert.strictEqual(sel.value, '', 'default selection is All');
  assert.deepStrictEqual(optionValues(win), ['', '__none__']);
  win.close();
  console.log('  ok - SEA select markup and default');
}

function testOptionsBuiltFromRowsAndRowsLabelled() {
  const {win} = makePage();
  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});
  assert.deepStrictEqual(
    optionValues(win),
    ['', '__none__', 'cron_agent', 'write_paper_sea'],
    'options are All, None and the distinct trimmed SEAs, sorted',
  );
  const rows = win.document.querySelectorAll('#history-list .sidebar-item');
  const bySea = {};
  rows.forEach(r => {
    const title = r.querySelector('.sidebar-item-text').textContent;
    const label = r.querySelector('.sidebar-item-sea');
    bySea[title] = {
      sea: r.dataset.sea,
      label: label ? label.textContent : null,
    };
  });
  assert.deepStrictEqual(bySea['paper task'], {
    sea: 'write_paper_sea',
    label: 'write_paper_sea',
  });
  assert.deepStrictEqual(bySea['second paper task'], {
    sea: 'write_paper_sea',
    label: 'write_paper_sea',
  });
  assert.deepStrictEqual(bySea['cron task'], {
    sea: 'cron_agent',
    label: 'cron_agent',
  });
  assert.deepStrictEqual(bySea['plain task'], {sea: '', label: null});
  assert.deepStrictEqual(bySea['legacy row without sea field'], {
    sea: '',
    label: null,
  });
  assert.strictEqual(visibleTitles(win).length, 5, 'All shows every row');
  win.close();
  console.log('  ok - options built from rows, rows carry data-sea and label');
}

function testSelectingSeaFiltersRows() {
  const {win} = makePage();
  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});

  selectSea(win, 'write_paper_sea');
  assert.deepStrictEqual(visibleTitles(win), [
    'paper task',
    'second paper task',
  ]);

  selectSea(win, 'cron_agent');
  assert.deepStrictEqual(visibleTitles(win), ['cron task']);

  selectSea(win, '__none__');
  assert.deepStrictEqual(visibleTitles(win), [
    'legacy row without sea field',
    'plain task',
  ]);

  selectSea(win, '');
  assert.strictEqual(visibleTitles(win).length, 5, 'All restores every row');
  win.close();
  console.log('  ok - selecting a SEA / None / All filters the rows');
}

function testSeaFilterCombinesWithStatusChips() {
  const {win} = makePage();
  const fixture = [
    session(1, 'ok paper', 'write_paper_sea'),
    session(2, 'failed paper', 'write_paper_sea', {failed: true}),
    session(3, 'failed plain', '', {failed: true}),
  ];
  send(win, {type: 'history', sessions: fixture, offset: 0});
  selectSea(win, 'write_paper_sea');
  const errors = win.document.getElementById('hf-errors');
  errors.checked = false;
  errors.dispatchEvent(new win.Event('change', {bubbles: true}));
  assert.deepStrictEqual(visibleTitles(win), ['ok paper']);

  // Nothing matches: the placeholder appears.
  selectSea(win, '__none__');
  assert.deepStrictEqual(visibleTitles(win), []);
  assert.ok(
    win.document.querySelector('#history-list .sidebar-empty-filter'),
    'empty-filter placeholder shown when the SEA filter hides every row',
  );
  win.close();
  console.log('  ok - SEA filter combines with the status chips');
}

function testSelectionSurvivesRefreshWithoutThatSea() {
  const {win} = makePage();
  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});
  selectSea(win, 'cron_agent');

  // A refresh whose rows no longer include cron_agent keeps the choice
  // as an option and hides every row.
  send(win, {
    type: 'history',
    sessions: [session(7, 'only paper', 'write_paper_sea')],
    offset: 0,
  });
  const sel = win.document.getElementById('hf-sea');
  assert.strictEqual(sel.value, 'cron_agent', 'selection is kept');
  assert.deepStrictEqual(optionValues(win), [
    '',
    '__none__',
    'cron_agent',
    'write_paper_sea',
  ]);
  assert.deepStrictEqual(visibleTitles(win), []);

  // A later page (offset > 0) adds its SEAs to the options.
  send(win, {
    type: 'history',
    sessions: [session(8, 'slack task', 'slack_sea')],
    offset: 1,
  });
  assert.deepStrictEqual(optionValues(win), [
    '',
    '__none__',
    'cron_agent',
    'slack_sea',
    'write_paper_sea',
  ]);

  // The None selection also survives a refresh.
  selectSea(win, '__none__');
  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});
  assert.strictEqual(sel.value, '__none__');
  assert.deepStrictEqual(visibleTitles(win), [
    'legacy row without sea field',
    'plain task',
  ]);

  // An empty refresh (no history, or a search with no hits) drops the
  // SEAs of the rows it removed but keeps the selection.
  send(win, {type: 'history', sessions: [], offset: 0});
  assert.deepStrictEqual(optionValues(win), ['', '__none__']);
  assert.strictEqual(sel.value, '__none__');
  selectSea(win, '');
  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});
  selectSea(win, 'cron_agent');
  send(win, {type: 'history', sessions: [], offset: 0});
  assert.deepStrictEqual(optionValues(win), ['', '__none__', 'cron_agent']);
  assert.strictEqual(sel.value, 'cron_agent');
  win.close();
  console.log(
    '  ok - selection survives refreshes and later pages extend options',
  );
}

function runTests() {
  testMarkupAndDefault();
  testOptionsBuiltFromRowsAndRowsLabelled();
  testSelectingSeaFiltersRows();
  testSeaFilterCombinesWithStatusChips();
  testSelectionSurvivesRefreshWithoutThatSea();
}

try {
  runTests();
  console.log('\n5 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
