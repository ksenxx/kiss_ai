// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');
const {visibleTitles} = require('./history_helpers');

function makePage(preSetup) {
  return harness.makePage({beforeScripts: preSetup});
}

const send = harness.send;

function clickToggle(win) {
  const btn = win.document.getElementById('history-filters-toggle');
  btn.dispatchEvent(
    new win.MouseEvent('click', {bubbles: true, cancelable: true}),
  );
}

const FILTER_CONTROL_IDS = [
  'hf-running',
  'hf-errors',
  'hf-completed',
  'hf-favorite',
  'hf-from',
  'hf-from-btn',
  'hf-to',
  'hf-to-btn',
  'hf-tag',
];

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

function testFiltersPanelMarkup() {
  const {win} = makePage();
  const doc = win.document;

  const panel = doc.getElementById('history-filters-panel');
  assert.ok(panel, '#history-filters-panel wrapper must exist');
  assert.strictEqual(
    panel.parentElement,
    doc.getElementById('sidebar-tab-history-panel'),
    '#history-filters-panel must sit directly inside the History ' +
      'sidebar panel',
  );

  const historyPanel = doc.getElementById('sidebar-tab-history-panel');
  const kids = Array.from(historyPanel.children);
  // The search box (with the view toggle beside it) sits in its own row.
  const searchIdx = kids.findIndex(
    el =>
      el.classList.contains('history-search-row') &&
      el.querySelector('.search-wrap #history-search'),
  );
  const panelIdx = kids.indexOf(panel);
  const listIdx = kids.findIndex(el => el.id === 'history-list');
  assert.ok(
    searchIdx !== -1 && searchIdx < panelIdx && panelIdx < listIdx,
    'the Filters panel must sit between the search box and the ' +
      `history list — got indexes search=${searchIdx} ` +
      `panel=${panelIdx} list=${listIdx}`,
  );

  const toggle = doc.getElementById('history-filters-toggle');
  assert.ok(toggle, '#history-filters-toggle must exist');
  assert.strictEqual(
    toggle.tagName,
    'BUTTON',
    'the Filters header must be a real <button>',
  );
  assert.strictEqual(
    toggle.getAttribute('type'),
    'button',
    'the toggle must be type="button" so it never submits forms',
  );
  assert.strictEqual(
    toggle.parentElement,
    panel,
    'the toggle must be a direct child of #history-filters-panel',
  );
  assert.match(
    toggle.textContent.trim(),
    /^Filters$/,
    'the panel header must be titled exactly "Filters"',
  );
  assert.ok(
    toggle.querySelector('svg'),
    'the toggle must carry an inline SVG chevron',
  );
  assert.strictEqual(
    toggle.getAttribute('aria-controls'),
    'history-filters-body',
    'the toggle must declare aria-controls="history-filters-body"',
  );

  const body = doc.getElementById('history-filters-body');
  assert.ok(body, '#history-filters-body must exist');
  assert.strictEqual(
    body.parentElement,
    panel,
    'the body must be a direct child of #history-filters-panel',
  );

  const bar = doc.querySelector('.history-filter-bar');
  assert.ok(bar, '.history-filter-bar must exist');
  assert.strictEqual(
    bar.parentElement,
    body,
    '.history-filter-bar must be a DIRECT child of ' +
      '#history-filters-body so collapsing hides every filter control',
  );
  for (const id of FILTER_CONTROL_IDS) {
    const el = doc.getElementById(id);
    assert.ok(el, `#${id} must exist`);
    assert.ok(
      body.contains(el),
      `#${id} must live inside #history-filters-body`,
    );
  }

  win.close();
  console.log('  ok - Filters panel markup: toggle + body wrap the bar');
}

function testDefaultCollapsedThenControlsVisibleOnExpand() {
  const {win} = makePage();
  const doc = win.document;

  doc.getElementById('app').style.display = '';

  const toggle = doc.getElementById('history-filters-toggle');
  const body = doc.getElementById('history-filters-body');
  assert.strictEqual(
    toggle.getAttribute('aria-expanded'),
    'false',
    'the Filters panel must be collapsed by default',
  );
  assert.strictEqual(
    body.hidden,
    true,
    'the body must carry the hidden attribute while collapsed',
  );

  clickToggle(win);
  for (const id of FILTER_CONTROL_IDS) {
    let el = doc.getElementById(id);
    for (; el; el = el.parentElement) {
      assert.ok(
        !el.hidden && el.style.display !== 'none',
        `#${id} must be visible when the panel is uncollapsed — ` +
          `ancestor <${el.tagName.toLowerCase()} id="${el.id}"> hides it`,
      );
    }
  }

  win.close();
  console.log('  ok - collapsed by default; expanding reveals controls');
}

function testToggleExpandsAndCollapses() {
  const {win} = makePage();
  const doc = win.document;
  const toggle = doc.getElementById('history-filters-toggle');
  const body = doc.getElementById('history-filters-body');

  clickToggle(win);
  assert.strictEqual(
    toggle.getAttribute('aria-expanded'),
    'true',
    'clicking the header must expand the panel',
  );
  assert.strictEqual(
    body.hidden,
    false,
    'the expanded body must not be hidden',
  );

  clickToggle(win);
  assert.strictEqual(
    toggle.getAttribute('aria-expanded'),
    'false',
    'clicking the header again must collapse the panel',
  );
  assert.strictEqual(
    body.hidden,
    true,
    'the re-collapsed body must carry the hidden attribute',
  );

  win.close();
  console.log('  ok - toggle expands and re-collapses the panel');
}

function testExpandedStatePersists() {
  const first = makePage();
  clickToggle(first.win);
  const persistedItems = {};
  for (let i = 0; i < first.win.localStorage.length; i++) {
    const k = first.win.localStorage.key(i);
    persistedItems[k] = first.win.localStorage.getItem(k);
  }
  first.win.close();
  assert.ok(
    Object.keys(persistedItems).length > 0,
    'expanding must persist the choice in localStorage',
  );

  const second = makePage(win => {
    for (const [k, v] of Object.entries(persistedItems)) {
      win.localStorage.setItem(k, v);
    }
  });
  assert.strictEqual(
    second.win.document
      .getElementById('history-filters-toggle')
      .getAttribute('aria-expanded'),
    'true',
    'a reloaded page must restore the expanded state',
  );
  assert.strictEqual(
    second.win.document.getElementById('history-filters-body').hidden,
    false,
    'a reloaded page must keep the body visible',
  );

  clickToggle(second.win);
  const collapsedItems = {};
  for (let i = 0; i < second.win.localStorage.length; i++) {
    const k = second.win.localStorage.key(i);
    collapsedItems[k] = second.win.localStorage.getItem(k);
  }
  second.win.close();

  const third = makePage(win => {
    for (const [k, v] of Object.entries(collapsedItems)) {
      win.localStorage.setItem(k, v);
    }
  });
  assert.strictEqual(
    third.win.document
      .getElementById('history-filters-toggle')
      .getAttribute('aria-expanded'),
    'false',
    'a reloaded page must restore the re-collapsed state',
  );
  assert.strictEqual(
    third.win.document.getElementById('history-filters-body').hidden,
    true,
    'a reloaded page must hide the body again',
  );
  third.win.close();

  console.log('  ok - collapsed/expanded state persists across reloads');
}

// Chip toggling itself is pinned by historyFilterChips.test.js; this
// only checks that collapsing the panel does not reset the filters.
function testFiltersStayAppliedWhileCollapsed() {
  const {win} = makePage();
  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});

  clickToggle(win);
  const chk = win.document.getElementById('hf-completed');
  chk.checked = false;
  chk.dispatchEvent(new win.Event('change', {bubbles: true}));
  clickToggle(win);
  assert.deepStrictEqual(
    visibleTitles(win).sort(),
    ['errored task', 'running task'],
    'filters stay applied while the panel is collapsed',
  );

  win.close();
  console.log('  ok - filters stay applied while the panel is collapsed');
}

function testLocalStorageUnavailableFallsBackToCollapsed() {
  const {win} = makePage(w => {
    Object.defineProperty(w, 'localStorage', {
      configurable: true,
      get() {
        throw new Error('localStorage denied');
      },
    });
  });
  const toggle = win.document.getElementById('history-filters-toggle');
  const body = win.document.getElementById('history-filters-body');
  assert.strictEqual(
    toggle.getAttribute('aria-expanded'),
    'false',
    'without localStorage the panel must still default to collapsed',
  );
  clickToggle(win);
  assert.strictEqual(
    body.hidden,
    false,
    'without localStorage the toggle must still expand the panel',
  );
  clickToggle(win);
  assert.strictEqual(
    body.hidden,
    true,
    'without localStorage the toggle must still collapse the panel',
  );
  win.close();
  console.log('  ok - denied localStorage still yields a working panel');
}

function testFiltersPanelCss() {
  const {win} = makePage();
  harness.loadMainCss(win);
  const doc = win.document;
  const toggle = doc.getElementById('history-filters-toggle');
  const chevron = toggle.querySelector('.history-filters-chevron');
  const body = doc.getElementById('history-filters-body');
  assert.strictEqual(
    win.getComputedStyle(toggle).cursor,
    'pointer',
    'the toggle must look interactive (cursor: pointer)',
  );
  assert.strictEqual(
    win.getComputedStyle(body).display,
    'none',
    'the collapsed body is not displayed',
  );
  const collapsedChevron = win.getComputedStyle(chevron).transform;
  clickToggle(win);
  assert.ok(toggle.classList.contains('expanded'));
  assert.notStrictEqual(
    win.getComputedStyle(chevron).transform,
    collapsedChevron,
    'the chevron turns in the expanded state',
  );
  assert.notStrictEqual(
    win.getComputedStyle(body).display,
    'none',
    'the expanded body is displayed',
  );
  win.close();
  console.log('  ok - Filters panel CSS: pointer toggle, chevron, hidden');
}

function main() {
  testFiltersPanelMarkup();
  testDefaultCollapsedThenControlsVisibleOnExpand();
  testToggleExpandsAndCollapses();
  testExpandedStatePersists();
  testFiltersStayAppliedWhileCollapsed();
  testLocalStorageUnavailableFallsBackToCollapsed();
  testFiltersPanelCss();
  console.log('historyFiltersCollapsible.test.js: all assertions passed.');
}

main();
