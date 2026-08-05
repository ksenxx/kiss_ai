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
const {makeWebview, send} = require('./simplify2_harness.js');

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
    is_worktree: true,
    is_parallel: false,
    auto_commit_mode: true,
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
  const {win, posted} = makeWebview();
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
  assert.ok(/1,234 tok/.test(metrics.textContent), 'metrics show tokens');
  assert.ok(/\$0\.5678/.test(metrics.textContent), 'metrics show cost');
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
  const {win, posted} = makeWebview();
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
  const {win} = makeWebview();
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
  const {win} = makeWebview();
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
  const {win} = makeWebview();
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

function testCssHidesMetadataWhenCollapsed() {
  const css = fs.readFileSync(
    path.join(__dirname, '..', 'media', 'main.css'),
    'utf8',
  );
  assert.ok(
    /\.running-item\.collapsed\s*>\s*\.running-item-info\s*\{\s*display:\s*none;\s*\}/.test(
      css,
    ),
    'main.css must hide .running-item-info of collapsed panels',
  );
  const textRule = css.slice(
    css.indexOf('.running-item > .sidebar-item-text'),
  );
  assert.ok(
    /-webkit-line-clamp:\s*3;/.test(textRule.slice(0, 400)) &&
      /[^-]line-clamp:\s*3;/.test(textRule.slice(0, 400)),
    'collapsed panels must clamp the task text to 3 lines',
  );
  assert.ok(
    css.indexOf('.sidebar-item-collapse,') >= 0,
    'main.css must style the collapse toggle like the other row buttons',
  );
  const remote = fs.readFileSync(
    path.join(__dirname, '..', 'media', 'remote-codex.css'),
    'utf8',
  );
  assert.ok(
    remote.indexOf('.running-item .sidebar-item-collapse') >= 0,
    'remote-codex.css must theme the collapse toggle for the webapp',
  );
  assert.ok(
    remote.indexOf('.running-item .sidebar-item-delete') < 0,
    'remote-codex.css must not theme a history delete button anymore',
  );
  console.log('PASS css hides collapsed metadata in extension and webapp');
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
