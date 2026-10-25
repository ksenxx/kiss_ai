// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {loadMainCss, makePage, send} = require('./simplify2_harness');
const {visibleTitles} = require('./history_helpers');

const SESSIONS_FIXTURE = [
  {
    id: 'chatR',
    task_id: 1,
    title: 'running task',
    timestamp: 1_700_000_000,
    preview: 'running task',
    has_events: false,
    failed: false,
    is_running: true,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: '',
    startTs: 1_700_000_000_000,
    endTs: 0,
  },
  {
    id: 'chatE',
    task_id: 2,
    title: 'errored task',
    timestamp: 1_700_000_100,
    preview: 'errored task',
    has_events: false,
    failed: true,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: '',
    startTs: 1_700_000_100_000,
    endTs: 1_700_000_110_000,
  },
  {
    id: 'chatS',
    task_id: 3,
    title: 'succeeded task',
    timestamp: 1_700_000_200,
    preview: 'succeeded task',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: '',
    startTs: 1_700_000_200_000,
    endTs: 1_700_000_210_000,
  },
];

const CHIP_SPEC = [
  {id: 'hf-running', label: 'Running', checked: true, lead: '.hf-chip-dot'},
  {id: 'hf-errors', label: 'Errored', checked: true, lead: '.hf-chip-dot'},
  {
    id: 'hf-completed',
    label: 'Succeeded',
    checked: true,
    lead: '.hf-chip-dot',
  },
  {
    id: 'hf-favorite',
    label: 'Favorites',
    checked: false,
    lead: '.hf-chip-icon',
  },
];

function testChipMarkup() {
  const {win} = makePage();
  const doc = win.document;

  const bar = doc.querySelector('.history-filter-bar');
  assert.ok(bar, '.history-filter-bar must exist');

  const chips = doc.querySelector('.history-filter-chips');
  assert.ok(chips, '.history-filter-chips group must exist');
  assert.strictEqual(
    chips.parentElement,
    bar,
    '.history-filter-chips must be a DIRECT child of .history-filter-bar',
  );

  for (const spec of CHIP_SPEC) {
    const input = doc.getElementById(spec.id);
    assert.ok(input, `#${spec.id} checkbox must exist`);
    assert.strictEqual(input.type, 'checkbox', `#${spec.id} is a checkbox`);
    assert.strictEqual(
      input.checked,
      spec.checked,
      `#${spec.id} default checked must be ${spec.checked}`,
    );
    const chip = input.parentElement;
    assert.ok(
      chip &&
        chip.tagName === 'LABEL' &&
        chip.classList.contains('hf-chip'),
      `#${spec.id} must be wrapped in a <label class="hf-chip">`,
    );
    assert.strictEqual(
      chip.parentElement,
      chips,
      `#${spec.id} chip must live inside .history-filter-chips`,
    );
    const lead = chip.querySelector(spec.lead);
    assert.ok(
      lead,
      `#${spec.id} chip must contain a leading "${spec.lead}" element`,
    );
    assert.strictEqual(
      chip.textContent.trim(),
      spec.label,
      `#${spec.id} chip label text`,
    );
  }

  const favChip = doc.getElementById('hf-favorite').parentElement;
  assert.ok(
    favChip.querySelector('svg.hf-chip-icon'),
    "#hf-favorite chip's leading icon must be an inline <svg>",
  );

  const ids = Array.from(
    bar.querySelectorAll('input[type="checkbox"]'),
  ).map(i => i.id);
  assert.strictEqual(
    ids.indexOf('hf-workspace'),
    -1,
    'the Workspace chip is gone from the CHATS filter bar — got ' +
      JSON.stringify(ids),
  );
  assert.strictEqual(
    ids.indexOf('hf-favorite'),
    ids.indexOf('hf-completed') + 1,
    'Favorites chip must sit immediately after Succeeded — got ' +
      JSON.stringify(ids),
  );

  win.close();
  console.log('  ok - chip markup, hidden checkbox ids, defaults, order');
}

function testChipTogglingFiltersRows() {
  const {win} = makePage();
  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});

  assert.deepStrictEqual(
    visibleTitles(win).sort(),
    ['errored task', 'running task', 'succeeded task'],
    'all three rows visible with default chips',
  );

  const toggle = (id, checked) => {
    const chk = win.document.getElementById(id);
    chk.checked = checked;
    chk.dispatchEvent(new win.Event('change', {bubbles: true}));
  };

  toggle('hf-errors', false);
  assert.deepStrictEqual(
    visibleTitles(win).sort(),
    ['running task', 'succeeded task'],
    'unchecking the Errored chip hides the errored row',
  );

  toggle('hf-completed', false);
  assert.deepStrictEqual(
    visibleTitles(win),
    ['running task'],
    'unchecking the Succeeded chip hides the succeeded row',
  );

  toggle('hf-errors', true);
  toggle('hf-completed', true);
  assert.strictEqual(
    visibleTitles(win).length,
    3,
    're-checking the chips shows every row again',
  );

  win.close();
  console.log('  ok - toggling chip checkboxes filters history rows');
}

function testChipCss() {
  const {win} = makePage();
  loadMainCss(win);
  const doc = win.document;
  const chips = Array.from(doc.querySelectorAll('.hf-chip'));
  assert.ok(chips.length >= 2, 'the filter bar renders status chips');
  const [first, second] = chips;
  const chip = win.getComputedStyle(first);
  assert.ok(
    chip.borderRadius && chip.borderRadius !== '0px',
    '.hf-chip must be pill-shaped (border-radius)',
  );
  assert.strictEqual(
    chip.cursor,
    'pointer',
    '.hf-chip must look interactive (cursor: pointer)',
  );
  assert.strictEqual(
    win.getComputedStyle(first.querySelector('input[type="checkbox"]')).opacity,
    '0',
    'the chip checkbox must be visually hidden (opacity: 0)',
  );

  first.querySelector('input').checked = true;
  second.querySelector('input').checked = false;
  assert.notStrictEqual(
    win.getComputedStyle(first).background,
    win.getComputedStyle(second).background,
    'a selected chip is painted differently from an unselected one',
  );

  assert.strictEqual(
    win.getComputedStyle(doc.querySelector('.history-filter-chips')).flexWrap,
    'wrap',
    '.history-filter-chips must wrap chips to new rows',
  );

  win.close();
  console.log('  ok - chip CSS: pill, hidden checkbox, selected state');
}

function testNoEmojiCalendarButtons() {
  const {win} = makePage();
  const doc = win.document;
  assert.ok(
    !doc.body.textContent.includes('\u{1F4C5}'),
    'the page must not show the 📅 emoji for the date picker buttons',
  );
  const range = doc.querySelector('.history-filter-daterange');
  assert.ok(
    range,
    'the From/To pickers must live inside one ' +
      '.history-filter-daterange group',
  );
  assert.strictEqual(
    range.parentElement,
    doc.querySelector('.history-filter-bar'),
    '.history-filter-daterange must be a direct child of the filter bar',
  );
  ['hf-from-btn', 'hf-to-btn'].forEach(id => {
    const btn = doc.getElementById(id);
    assert.ok(btn, `#${id} must exist`);
    assert.ok(
      btn.querySelector('svg'),
      `#${id} must render an inline SVG calendar icon`,
    );
  });
  win.close();
  console.log('  ok - emoji buttons replaced by SVG icons in a date group');
}

function main() {
  testChipMarkup();
  testChipTogglingFiltersRows();
  testChipCss();
  testNoEmojiCalendarButtons();
  console.log('historyFilterChips.test.js: all assertions passed.');
}

main();
