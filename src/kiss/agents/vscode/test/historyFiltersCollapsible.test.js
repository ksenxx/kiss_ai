// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for the collapsible "Filters" panel in the task
// History sidebar.  The status filter chips (Running / Errored /
// Succeeded / Workspace / Favorites) and the From/To date range
// used for filtering tasks must live inside a collapsible panel
// titled "Filters" — in BOTH the VS Code extension webview and the
// remote web chat, which share this very ``media/chat.html`` +
// ``media/main.js`` pair (see ``buildChatHtml`` in SorcarTab.ts and
// ``_build_html`` in src/kiss/server/web_server.py).
//
// Requirements driven by this test:
//
//   1. chat.html wraps the whole ``.history-filter-bar`` (chips +
//      date range) inside a ``#history-filters-body`` container that
//      belongs to a ``#history-filters-panel`` wrapper sitting
//      between the history search box and ``#history-list``.
//
//   2. The panel header is a real ``<button id=
//      "history-filters-toggle">`` labelled "Filters" with proper
//      disclosure semantics: ``aria-expanded`` reflecting the state
//      and ``aria-controls="history-filters-body"``.
//
//   3. The panel is EXPANDED by default: the body is not hidden and
//      every filter control (five chip checkboxes, two date inputs,
//      two picker buttons) is present inside it — i.e. the filter
//      buttons and dates are visible whenever the panel is
//      uncollapsed.
//
//   4. Clicking the toggle collapses the panel (``hidden`` body,
//      ``aria-expanded="false"``) and clicking again expands it.
//
//   5. The collapsed/expanded choice persists across webview
//      reloads via ``localStorage``.
//
//   6. Filtering still works with the panel wrapper in place:
//      unchecking a chip inside the expanded panel hides matching
//      history rows.
//
//   7. main.css styles the toggle as an interactive disclosure
//      header (pointer cursor, rotating chevron) and guarantees a
//      hidden body stays ``display: none``.
//
// This test drives the production ``media/main.js`` + real
// ``media/chat.html`` markup inside jsdom — no mocks of project
// code — exactly like ``historyFilterChips.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/historyFiltersCollapsible.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Load the real chat.html + main.js into a jsdom webview.
 *
 * ``preSetup(win)`` runs after the DOM exists but BEFORE main.js is
 * evaluated — tests use it to seed ``localStorage`` and verify the
 * persisted-state restore path.
 */
function makeWebview(preSetup) {
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

  if (preSetup) preSetup(win);

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
  'hf-workspace',
  'hf-favorite',
  'hf-from',
  'hf-from-btn',
  'hf-to',
  'hf-to-btn',
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
  const {win} = makeWebview();
  const doc = win.document;

  const panel = doc.getElementById('history-filters-panel');
  assert.ok(panel, '#history-filters-panel wrapper must exist');
  assert.strictEqual(
    panel.parentElement,
    doc.getElementById('sidebar-tab-history-panel'),
    '#history-filters-panel must sit directly inside the History ' +
      'sidebar panel',
  );

  // The panel sits between the search box and the history list.
  const historyPanel = doc.getElementById('sidebar-tab-history-panel');
  const kids = Array.from(historyPanel.children);
  const searchIdx = kids.findIndex(el => el.classList.contains('search-wrap'));
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

  // The whole filter bar (chips + dates) lives INSIDE the body.
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

function testDefaultExpandedControlsVisible() {
  const {win} = makeWebview();
  const doc = win.document;

  // ``#app`` ships hidden in the static HTML and is revealed once the
  // host connects; flip it here exactly like ``setServerLoading(false)``
  // does in production so ancestor-visibility checks are meaningful.
  doc.getElementById('app').style.display = '';

  const toggle = doc.getElementById('history-filters-toggle');
  const body = doc.getElementById('history-filters-body');
  assert.strictEqual(
    toggle.getAttribute('aria-expanded'),
    'true',
    'the Filters panel must be expanded by default',
  );
  assert.strictEqual(
    body.hidden,
    false,
    'the body must NOT be hidden while the panel is uncollapsed',
  );
  // Every filter button and date input is visible (not inside any
  // hidden ancestor and not display:none itself).
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
  console.log('  ok - expanded by default; buttons and dates visible');
}

function testToggleCollapsesAndExpands() {
  const {win} = makeWebview();
  const doc = win.document;
  const toggle = doc.getElementById('history-filters-toggle');
  const body = doc.getElementById('history-filters-body');

  clickToggle(win);
  assert.strictEqual(
    toggle.getAttribute('aria-expanded'),
    'false',
    'clicking the header must collapse the panel',
  );
  assert.strictEqual(
    body.hidden,
    true,
    'the collapsed body must carry the hidden attribute',
  );

  clickToggle(win);
  assert.strictEqual(
    toggle.getAttribute('aria-expanded'),
    'true',
    'clicking the header again must expand the panel',
  );
  assert.strictEqual(
    body.hidden,
    false,
    'the re-expanded body must not be hidden',
  );

  win.close();
  console.log('  ok - toggle collapses and re-expands the panel');
}

function testCollapsedStatePersists() {
  // Collapse the panel and capture what was persisted.
  const first = makeWebview();
  clickToggle(first.win);
  const persistedItems = {};
  for (let i = 0; i < first.win.localStorage.length; i++) {
    const k = first.win.localStorage.key(i);
    persistedItems[k] = first.win.localStorage.getItem(k);
  }
  first.win.close();
  assert.ok(
    Object.keys(persistedItems).length > 0,
    'collapsing must persist the choice in localStorage',
  );

  // A fresh webview seeded with the SAME storage restores collapsed.
  const second = makeWebview(win => {
    for (const [k, v] of Object.entries(persistedItems)) {
      win.localStorage.setItem(k, v);
    }
  });
  assert.strictEqual(
    second.win.document
      .getElementById('history-filters-toggle')
      .getAttribute('aria-expanded'),
    'false',
    'a reloaded webview must restore the collapsed state',
  );
  assert.strictEqual(
    second.win.document.getElementById('history-filters-body').hidden,
    true,
    'a reloaded webview must keep the body hidden',
  );

  // Expanding again also persists, and a third webview restores it.
  clickToggle(second.win);
  const expandedItems = {};
  for (let i = 0; i < second.win.localStorage.length; i++) {
    const k = second.win.localStorage.key(i);
    expandedItems[k] = second.win.localStorage.getItem(k);
  }
  second.win.close();

  const third = makeWebview(win => {
    for (const [k, v] of Object.entries(expandedItems)) {
      win.localStorage.setItem(k, v);
    }
  });
  assert.strictEqual(
    third.win.document
      .getElementById('history-filters-toggle')
      .getAttribute('aria-expanded'),
    'true',
    'a reloaded webview must restore the re-expanded state',
  );
  assert.strictEqual(
    third.win.document.getElementById('history-filters-body').hidden,
    false,
    'a reloaded webview must show the body again',
  );
  third.win.close();

  console.log('  ok - collapsed/expanded state persists across reloads');
}

function testFilteringStillWorksInsidePanel() {
  const {win} = makeWebview();
  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});

  assert.deepStrictEqual(
    visibleTitles(win).sort(),
    ['errored task', 'running task', 'succeeded task'],
    'all three rows visible with the default expanded panel',
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
    'unchecking the Errored chip inside the panel still filters rows',
  );
  toggle('hf-errors', true);
  assert.strictEqual(
    visibleTitles(win).length,
    3,
    're-checking the chip shows every row again',
  );

  // Collapsing the panel must not clear the active filters.
  toggle('hf-completed', false);
  clickToggle(win);
  assert.deepStrictEqual(
    visibleTitles(win).sort(),
    ['errored task', 'running task'],
    'filters stay applied while the panel is collapsed',
  );

  win.close();
  console.log('  ok - filtering still works inside the Filters panel');
}

function testLocalStorageUnavailableFallsBackToExpanded() {
  // Some embedders deny localStorage entirely (the accessor throws).
  // The panel must still default to expanded and stay toggleable —
  // this drives the two catch branches of the persistence wiring.
  const {win} = makeWebview(w => {
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
    'true',
    'without localStorage the panel must still default to expanded',
  );
  clickToggle(win);
  assert.strictEqual(
    body.hidden,
    true,
    'without localStorage the toggle must still collapse the panel',
  );
  clickToggle(win);
  assert.strictEqual(
    body.hidden,
    false,
    'without localStorage the toggle must still expand the panel',
  );
  win.close();
  console.log('  ok - denied localStorage still yields a working panel');
}

function testFiltersPanelCss() {
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');

  const toggleRule = /\.history-filters-toggle\s*\{([^}]*)\}/.exec(css);
  assert.ok(toggleRule, 'main.css must declare .history-filters-toggle');
  assert.ok(
    /cursor\s*:\s*pointer/.test(toggleRule[1]),
    'the toggle must look interactive (cursor: pointer)',
  );

  assert.ok(
    /\.history-filters-toggle\.expanded\s+\.history-filters-chevron/.test(css),
    'main.css must rotate the chevron in the expanded state',
  );
  assert.ok(
    /\.history-filters-body\[hidden\]\s*\{[^}]*display\s*:\s*none/.test(css),
    'main.css must force display:none on the hidden body',
  );

  console.log('  ok - Filters panel CSS: pointer toggle, chevron, hidden');
}

function main() {
  testFiltersPanelMarkup();
  testDefaultExpandedControlsVisible();
  testToggleCollapsesAndExpands();
  testCollapsedStatePersists();
  testFilteringStillWorksInsidePanel();
  testLocalStorageUnavailableFallsBackToExpanded();
  testFiltersPanelCss();
  console.log('historyFiltersCollapsible.test.js: all assertions passed.');
}

main();
