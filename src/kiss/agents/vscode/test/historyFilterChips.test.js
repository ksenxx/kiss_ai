// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the redesigned History filter bar: the five
// category checkboxes (Running / Errored / Succeeded / Workspace /
// Favorites) are rendered as Material-style FILTER CHIPS — pill
// toggles with a hidden checkbox, a leading status dot (or icon)
// and a text label — instead of bare native checkboxes.
//
// Requirements driven by this test:
//
//   1. The chips live in a `.history-filter-chips` group that is a
//      DIRECT child of `.history-filter-bar`.
//
//   2. Each chip is a `<label class="hf-chip">` containing a hidden
//      `<input type="checkbox">` that KEEPS the pre-redesign ids
//      (hf-running, hf-errors, hf-completed, hf-workspace,
//      hf-favorite) and default checked states (all checked except
//      Favorites) so the existing filter logic keeps working.
//
//   3. Workspace still sits immediately before Favorites.
//
//   4. Each of the three status chips (Running / Errored /
//      Succeeded) carries a `.hf-chip-dot` status dot; the
//      Workspace and Favorites chips carry a `.hf-chip-icon`
//      leading SVG icon.  Every chip has a `.hf-chip-label`.
//
//   5. Toggling a chip's checkbox still filters the history rows
//      (client-side, via ``applyHistoryFilterVisibility``).
//
//   6. main.css styles the chips: pill shape, hidden checkbox, and
//      a `:has(input:checked)` selected state.
//
//   7. The ugly emoji calendar buttons are gone: chat.html contains
//      no 📅 character; the pickers are SVG icon buttons inside one
//      `.history-filter-daterange` group.
//
// This test drives the production ``media/main.js`` + real
// ``media/chat.html`` markup inside jsdom — no mocks of project
// code — exactly like ``historyWorkspaceFilter.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/historyFilterChips.test.js

'use strict';

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
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Visible (display !== 'none') history row titles. */
function visibleTitles(win) {
  const list = win.document.getElementById('history-list');
  const out = [];
  list.querySelectorAll('.sidebar-item').forEach(r => {
    if (r.style.display !== 'none') {
      out.push(r.querySelector('.sidebar-item-text').textContent);
    }
  });
  return out;
}

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
    id: 'hf-workspace',
    label: 'Workspace',
    checked: true,
    lead: '.hf-chip-icon',
  },
  {
    id: 'hf-favorite',
    label: 'Favorites',
    checked: false,
    lead: '.hf-chip-icon',
  },
];

function testChipMarkup() {
  const {win} = makeWebview();
  const doc = win.document;

  const bar = doc.querySelector('.history-filter-bar');
  assert.ok(bar, '.history-filter-bar must exist');

  // 1) Chips group is a direct child of the filter bar.
  const chips = doc.querySelector('.history-filter-chips');
  assert.ok(chips, '.history-filter-chips group must exist');
  assert.strictEqual(
    chips.parentElement,
    bar,
    '.history-filter-chips must be a DIRECT child of .history-filter-bar',
  );

  // 2) Each chip: label.hf-chip > hidden checkbox (stable id +
  //    default) + leading dot/icon + .hf-chip-label text.
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
    const label = chip.querySelector('.hf-chip-label');
    assert.ok(label, `#${spec.id} chip must contain a .hf-chip-label`);
    assert.strictEqual(
      label.textContent.trim(),
      spec.label,
      `#${spec.id} chip label text`,
    );
  }

  // The two icon chips carry inline SVG icons (not emoji).
  ['hf-workspace', 'hf-favorite'].forEach(id => {
    const chip = doc.getElementById(id).parentElement;
    assert.ok(
      chip.querySelector('svg.hf-chip-icon'),
      `#${id} chip's leading icon must be an inline <svg>`,
    );
  });

  // 3) Workspace immediately before Favorites among bar checkboxes.
  const ids = Array.from(
    bar.querySelectorAll('input[type="checkbox"]'),
  ).map(i => i.id);
  assert.strictEqual(
    ids.indexOf('hf-favorite'),
    ids.indexOf('hf-workspace') + 1,
    'Workspace chip must sit immediately before Favorites — got ' +
      JSON.stringify(ids),
  );

  win.close();
  console.log('  ok - chip markup, hidden checkbox ids, defaults, order');
}

function testChipTogglingFiltersRows() {
  const {win} = makeWebview();
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
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');

  // Pill container rule.
  const chipRule = /\.hf-chip\s*\{([^}]*)\}/.exec(css);
  assert.ok(chipRule, 'main.css must declare a .hf-chip rule');
  assert.ok(
    /border-radius\s*:/.test(chipRule[1]),
    '.hf-chip must be pill-shaped (border-radius)',
  );
  assert.ok(
    /cursor\s*:\s*pointer/.test(chipRule[1]),
    '.hf-chip must look interactive (cursor: pointer)',
  );

  // The native checkbox is visually hidden inside the chip.
  const hiddenRule =
    /\.hf-chip\s+input\[type="checkbox"\]\s*\{([^}]*)\}/.exec(css);
  assert.ok(
    hiddenRule,
    'main.css must declare a .hf-chip input[type="checkbox"] rule',
  );
  assert.ok(
    /opacity\s*:\s*0/.test(hiddenRule[1]),
    'the chip checkbox must be visually hidden (opacity: 0)',
  );

  // Selected state via :has(input:checked).
  assert.ok(
    /\.hf-chip:has\([^)]*input:checked\)/.test(css),
    'main.css must style the selected chip via .hf-chip:has(input:checked)',
  );

  // Chips group wraps.
  const groupRule = /\.history-filter-chips\s*\{([^}]*)\}/.exec(css);
  assert.ok(groupRule, 'main.css must declare .history-filter-chips');
  assert.ok(
    /flex-wrap\s*:\s*wrap/.test(groupRule[1]),
    '.history-filter-chips must wrap chips to new rows',
  );

  console.log('  ok - chip CSS: pill, hidden checkbox, :has selected state');
}

function testNoEmojiCalendarButtons() {
  const html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  assert.ok(
    !html.includes('\u{1F4C5}'),
    'chat.html must not use the 📅 emoji for the date picker buttons',
  );

  const {win} = makeWebview();
  const doc = win.document;
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
