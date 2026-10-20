// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (jsdom) tests for the task-history collapse/uncollapse
// toggle that replaced the delete button.  A history task panel is
// collapsed by default: it shows only the line-clamped task text and
// hides the metadata block (.running-item-info: metrics, workspace,
// ids).  Clicking the chevron expands the panel and shows the
// metadata; clicking again collapses it back.  The toggle must not
// bubble into the row click handler (which would open the chat).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {makePage, send} = require('./simplify2_harness.js');

const MEDIA = path.join(__dirname, '..', 'media');

function makeSession() {
  return {
    id: 'chat-1',
    task_id: 'task-1',
    preview: 'refactor the parser',
    title: 'refactor the parser',
    has_events: true,
    tokens: 1234,
    cost: 0.5678,
    steps: 7,
    timestamp: 1700000000,
    work_dir: '/home/user/proj',
    model: 'test-model',
    is_parallel: false,
  };
}

function loadHistory(win, sessions) {
  send(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: sessions,
  });
}

function historyRow(win) {
  return win.document.querySelector('#history-list .sidebar-item');
}

function testCollapsedByDefaultAndToggles() {
  const {win, posted} = makePage();
  loadHistory(win, [makeSession()]);

  const row = historyRow(win);
  assert.ok(row, 'history row rendered');

  const text = row.querySelector('.sidebar-item-text');
  assert.ok(text, 'task text span rendered');
  assert.strictEqual(text.textContent, 'refactor the parser');

  const info = row.querySelector('.running-item-info');
  assert.ok(info, 'metadata block rendered');
  const metrics = info.querySelector('.running-item-metrics');
  assert.ok(metrics, 'metrics rendered inside metadata block');
  assert.ok(/7 steps/.test(metrics.textContent), 'metrics show steps');
  assert.ok(/1\.23K tok/.test(metrics.textContent), 'metrics show tokens');
  assert.ok(/\$0\.57/.test(metrics.textContent), 'metrics show cost');
  const workspace = info.querySelector('.running-item-workspace');
  assert.ok(workspace, 'workspace metadata rendered');
  assert.ok(
    workspace.textContent.indexOf('/home/user/proj') === 0,
    'workspace metadata shows the work dir',
  );
  const ids = info.querySelector('.running-item-ids');
  assert.ok(ids, 'ids metadata rendered');
  assert.ok(/chat chat-1/.test(ids.textContent), 'ids show chat id');
  assert.ok(/task task-1/.test(ids.textContent), 'ids show task id');

  assert.ok(
    row.classList.contains('collapsed'),
    'task panel is collapsed by default (metadata hidden)',
  );

  const toggle = row.querySelector('.sidebar-item-collapse');
  assert.ok(toggle, 'collapse toggle rendered');
  assert.strictEqual(toggle.dataset.tooltip, 'Show details');
  assert.strictEqual(toggle.getAttribute('aria-expanded'), 'false');
  assert.strictEqual(
    toggle.getAttribute('aria-label'),
    'Expand task details',
  );

  toggle.click();
  assert.ok(
    !row.classList.contains('collapsed'),
    'uncollapse shows the metadata',
  );
  assert.strictEqual(toggle.dataset.tooltip, 'Hide details');
  assert.strictEqual(toggle.getAttribute('aria-expanded'), 'true');
  assert.strictEqual(
    toggle.getAttribute('aria-label'),
    'Collapse task details',
  );

  toggle.click();
  assert.ok(
    row.classList.contains('collapsed'),
    'collapse hides the metadata again',
  );
  assert.strictEqual(toggle.dataset.tooltip, 'Show details');
  assert.strictEqual(toggle.getAttribute('aria-expanded'), 'false');

  assert.ok(
    !posted.some(m => m.type === 'resumeSession'),
    'toggle clicks must not bubble into the row click handler',
  );
  win.close();
  console.log('PASS collapsed by default; toggle expands and collapses');
}

function testRowClickStillOpensChat() {
  const {win, posted} = makePage();
  loadHistory(win, [makeSession()]);
  const row = historyRow(win);
  row.querySelector('.sidebar-item-collapse').click();
  row.click();
  assert.ok(
    posted.some(m => m.type === 'resumeSession'),
    'clicking the row body still opens the chat after uncollapsing',
  );
  win.close();
  console.log('PASS row click still opens the chat');
}

function testExpandedStateSurvivesRerender() {
  const {win} = makePage();
  loadHistory(win, [makeSession()]);
  historyRow(win).querySelector('.sidebar-item-collapse').click();
  assert.ok(!historyRow(win).classList.contains('collapsed'));
  // Re-render with a FRESH, serialized session payload, exactly as a
  // backend-driven history refresh delivers it (the server constructs
  // new session dicts on every getHistory response, so no client-side
  // property stored on the old object can survive).
  send(win, {
    type: 'history',
    offset: 0,
    generation: 1,
    sessions: [JSON.parse(JSON.stringify(makeSession()))],
  });
  assert.ok(
    !historyRow(win).classList.contains('collapsed'),
    'expanded state keyed by task id survives a fresh-payload re-render',
  );
  // Collapsing again must also survive a further fresh re-render.
  historyRow(win).querySelector('.sidebar-item-collapse').click();
  assert.ok(historyRow(win).classList.contains('collapsed'));
  send(win, {
    type: 'history',
    offset: 0,
    generation: 2,
    sessions: [JSON.parse(JSON.stringify(makeSession()))],
  });
  assert.ok(
    historyRow(win).classList.contains('collapsed'),
    're-collapsed state survives a fresh-payload re-render',
  );
  win.close();
  console.log('PASS expanded state survives fresh-payload re-renders');
}

function testCollapseStateIsPerTask() {
  const {win} = makePage();
  const other = makeSession();
  other.id = 'chat-9';
  other.task_id = 'task-9';
  other.preview = 'another task';
  other.title = 'another task';
  loadHistory(win, [makeSession(), other]);
  const rows = win.document.querySelectorAll('#history-list .sidebar-item');
  assert.strictEqual(rows.length, 2);
  rows[0].querySelector('.sidebar-item-collapse').click();
  assert.ok(!rows[0].classList.contains('collapsed'), 'first expanded');
  assert.ok(
    rows[1].classList.contains('collapsed'),
    'expanding one task must not expand the other',
  );
  win.close();
  console.log('PASS collapse state is tracked per task');
}

function testRowWithoutTaskIdStillCollapsible() {
  const {win} = makePage();
  loadHistory(win, [
    {
      id: 'chat-2',
      preview: 'old imported chat',
      title: 'old imported chat',
      has_events: false,
      timestamp: 1700000000,
    },
  ]);
  const row = historyRow(win);
  assert.ok(row, 'row without task_id rendered');
  assert.ok(
    row.classList.contains('collapsed'),
    'row without task_id is collapsed by default',
  );
  const toggle = row.querySelector('.sidebar-item-collapse');
  assert.ok(toggle, 'row without task_id still gets a collapse toggle');
  toggle.click();
  assert.ok(!row.classList.contains('collapsed'), 'toggle works');
  win.close();
  console.log('PASS rows without task_id are collapsible too');
}

/** Append a media/ stylesheet to *win*'s document. */
function loadCss(win, name) {
  const styleEl = win.document.createElement('style');
  styleEl.textContent = fs.readFileSync(path.join(MEDIA, name), 'utf8');
  win.document.head.appendChild(styleEl);
}

function testCssHidesMetadataWhenCollapsed() {
  const {win} = makePage();
  loadCss(win, 'main.css');
  loadCss(win, 'remote-codex.css');
  win.document.body.classList.add('remote-chat');
  loadHistory(win, [makeSession()]);
  const row = historyRow(win);
  const info = row.querySelector('.running-item-info');
  const text = row.querySelector('.sidebar-item-text');
  const toggle = row.querySelector('.sidebar-item-collapse');
  assert.ok(row.classList.contains('collapsed'), 'row starts collapsed');
  assert.strictEqual(
    win.getComputedStyle(info).display,
    'none',
    'a collapsed panel hides its metadata',
  );
  assert.strictEqual(
    win.getComputedStyle(text).getPropertyValue('-webkit-line-clamp'),
    '3',
    'a collapsed panel clamps the task text to 3 lines',
  );
  toggle.click();
  assert.notStrictEqual(
    win.getComputedStyle(info).display,
    'none',
    'an expanded panel shows its metadata',
  );
  const toggleStyle = win.getComputedStyle(toggle);
  const copyStyle = win.getComputedStyle(
    row.querySelector('.sidebar-item-copy'),
  );
  for (const prop of ['cursor', 'border-radius', 'background-image']) {
    assert.strictEqual(
      toggleStyle.getPropertyValue(prop),
      copyStyle.getPropertyValue(prop),
      `the collapse toggle's ${prop} matches the other row buttons'`,
    );
  }
  assert.strictEqual(
    toggleStyle.color,
    'var(--fg)',
    'the webapp theme paints the collapse toggle in the foreground colour',
  );
  const del = row.querySelector('.sidebar-item-actions > .sidebar-item-delete');
  assert.ok(del, 'a finished history row has a delete button in its action strip');
  for (const prop of ['cursor', 'border-radius', 'background-image']) {
    assert.strictEqual(
      win.getComputedStyle(del).getPropertyValue(prop),
      copyStyle.getPropertyValue(prop),
      `the delete button's ${prop} matches the other row buttons'`,
    );
  }
  win.close();
  console.log('PASS css hides collapsed metadata in the webapp');
}

function main() {
  testCollapsedByDefaultAndToggles();
  testRowClickStillOpensChat();
  testExpandedStateSurvivesRerender();
  testCollapseStateIsPerTask();
  testRowWithoutTaskIdStillCollapsible();
  testCssHidesMetadataWhenCollapsed();
  console.log('All historyTaskCollapse tests passed');
}

main();
