// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function openSidebar(win) {
  const btn = win.document.getElementById('menu-btn');
  assert.ok(btn, 'burger menu button (#menu-btn) must exist');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  const sidebar = win.document.getElementById('sidebar');
  assert.ok(
    sidebar.classList.contains('open'),
    'burger click must open the sidebar',
  );
}

function lastGetHistory(posted) {
  for (let i = posted.length - 1; i >= 0; i--) {
    if (posted[i] && posted[i].type === 'getHistory') return posted[i];
  }
  return null;
}

function visibleRows(win) {
  const list = win.document.getElementById('history-list');
  const rows = list.querySelectorAll('.sidebar-item');
  const out = [];
  rows.forEach(r => {
    if (r.style.display !== 'none') {
      out.push(r.querySelector('.sidebar-item-text').textContent);
    }
  });
  return out;
}

function rowByTitle(win, title) {
  const list = win.document.getElementById('history-list');
  const rows = list.querySelectorAll('.sidebar-item');
  for (const r of rows) {
    const t = r.querySelector('.sidebar-item-text');
    if (t && t.textContent === title) return r;
  }
  return null;
}

function makeRow(overrides) {
  return Object.assign(
    {
      id: 'chat-' + (overrides.task_id || 0),
      task_id: 0,
      title: 'untitled',
      timestamp: 1_700_000_000,
      preview: 'untitled',
      has_events: false,
      failed: false,
      is_running: false,
      tokens: 0,
      cost: 0,
      steps: 0,
      is_favorite: false,
      work_dir: '',
      startTs: 1_700_000_000_000,
      endTs: 1_700_000_010_000,
    },
    overrides,
  );
}

function enableWorkspaceFilter(win) {
  const ws = win.document.getElementById('hf-workspace');
  ws.checked = true;
  ws.dispatchEvent(new win.Event('change', {bubbles: true}));
}

function testRunningRowVisibleDespiteResolvedWorkDirVariant() {
  const {win, posted} = makeWebview();

  send(win, {
    type: 'configData',
    config: {work_dir: '/var/repo/alpha'},
    apiKeys: {},
  });

  openSidebar(win);
  enableWorkspaceFilter(win);
  assert.ok(lastGetHistory(posted), 'opening the sidebar must post getHistory');

  posted.length = 0;
  send(win, {type: 'tasks_updated', taskId: ''});
  const fetched = lastGetHistory(posted);
  assert.ok(
    fetched,
    'start-time tasks_updated must post getHistory while the sidebar is open',
  );

  send(win, {
    type: 'history',
    sessions: [
      makeRow({
        task_id: 101,
        title: 'just-started task',
        is_running: true,
        work_dir: '/private/var/repo/alpha',
        endTs: 0,
      }),
    ],
    offset: 0,
    generation: fetched.generation,
  });

  const row = rowByTitle(win, 'just-started task');
  assert.ok(row, 'the running row must be rendered');
  assert.strictEqual(
    row.dataset.category,
    'running',
    'row must be stamped data-category=running',
  );
  assert.notStrictEqual(
    row.style.display,
    'none',
    'REGRESSION: a just-started RUNNING task must be VISIBLE in the ' +
      'History panel even when its early-persisted work_dir is a ' +
      'resolved path variant of the client workspace — the Workspace ' +
      'filter must never hide a running row',
  );

  win.close();
  console.log('  ok - running row visible despite resolved work_dir variant');
}

function testRunningRowVisibleFromOtherWorkspace() {
  const {win, posted} = makeWebview();

  send(win, {
    type: 'configData',
    config: {work_dir: '/repo/alpha'},
    apiKeys: {},
  });
  openSidebar(win);
  enableWorkspaceFilter(win);
  const fetched = lastGetHistory(posted);

  send(win, {
    type: 'history',
    sessions: [
      makeRow({
        task_id: 102,
        title: 'running elsewhere',
        is_running: true,
        work_dir: '/repo/beta',
        endTs: 0,
      }),
      makeRow({
        task_id: 103,
        title: 'completed elsewhere',
        is_running: false,
        work_dir: '/repo/beta',
      }),
    ],
    offset: 0,
    generation: fetched.generation,
  });

  const visible = visibleRows(win).sort();
  assert.deepStrictEqual(
    visible,
    ['running elsewhere'],
    'a RUNNING row must always pass the Workspace filter while a ' +
      'completed row from another workspace stays hidden; got ' +
      JSON.stringify(visible),
  );

  win.close();
  console.log('  ok - running row visible from other workspace, completed hidden');
}

function testUncheckingRunningStillHidesRunningRow() {
  const {win, posted} = makeWebview();

  send(win, {
    type: 'configData',
    config: {work_dir: '/repo/alpha'},
    apiKeys: {},
  });
  openSidebar(win);
  enableWorkspaceFilter(win);
  const fetched = lastGetHistory(posted);

  send(win, {
    type: 'history',
    sessions: [
      makeRow({
        task_id: 104,
        title: 'running task',
        is_running: true,
        work_dir: '/private/repo/alpha',
        endTs: 0,
      }),
    ],
    offset: 0,
    generation: fetched.generation,
  });

  assert.deepStrictEqual(visibleRows(win), ['running task']);

  const hfRunning = win.document.getElementById('hf-running');
  hfRunning.checked = false;
  hfRunning.dispatchEvent(new win.Event('change', {bubbles: true}));

  assert.deepStrictEqual(
    visibleRows(win),
    [],
    'unchecking the Running filter must still hide running rows — ' +
      'the workspace always-pass rule must not defeat the category filter',
  );

  win.close();
  console.log('  ok - unchecking Running still hides the running row');
}

function testTasksUpdatedRefetchesHistoryWhileSidebarOpen() {
  const {win, posted} = makeWebview();

  send(win, {
    type: 'configData',
    config: {work_dir: '/repo/alpha'},
    apiKeys: {},
  });
  openSidebar(win);

  posted.length = 0;
  send(win, {type: 'tasks_updated', taskId: ''});
  assert.ok(
    lastGetHistory(posted),
    'tasks_updated with the sidebar open must post getHistory so the ' +
      'just-started task appears immediately',
  );

  win.close();
  console.log('  ok - tasks_updated re-fetches history while sidebar open');
}

function testTrailingSlashNormalizedMatchForCompletedRows() {
  const {win, posted} = makeWebview();

  send(win, {
    type: 'configData',
    config: {work_dir: '/repo/alpha'},
    apiKeys: {},
  });
  openSidebar(win);
  enableWorkspaceFilter(win);
  const fetched = lastGetHistory(posted);

  send(win, {
    type: 'history',
    sessions: [
      makeRow({
        task_id: 105,
        title: 'completed with trailing slash',
        is_running: false,
        work_dir: '/repo/alpha/',
      }),
      makeRow({
        task_id: 106,
        title: 'completed different dir',
        is_running: false,
        work_dir: '/repo/beta',
      }),
    ],
    offset: 0,
    generation: fetched.generation,
  });

  const visible = visibleRows(win).sort();
  assert.deepStrictEqual(
    visible,
    ['completed with trailing slash'],
    'the Workspace comparison must normalize trailing slashes ' +
      '(/repo/alpha/ matches /repo/alpha) while still hiding rows ' +
      `from a genuinely different dir; got ${JSON.stringify(visible)}`,
  );

  win.close();
  console.log('  ok - trailing-slash normalized match for completed rows');
}

function testClientTrailingSlashNormalizes() {
  const {win, posted} = makeWebview();

  send(win, {
    type: 'configData',
    config: {work_dir: '/repo/alpha/'},
    apiKeys: {},
  });
  openSidebar(win);
  enableWorkspaceFilter(win);
  const fetched = lastGetHistory(posted);

  send(win, {
    type: 'history',
    sessions: [
      makeRow({
        task_id: 107,
        title: 'completed exact',
        is_running: false,
        work_dir: '/repo/alpha',
      }),
    ],
    offset: 0,
    generation: fetched.generation,
  });

  assert.deepStrictEqual(
    visibleRows(win),
    ['completed exact'],
    'a trailing slash on the CLIENT work_dir must also normalize away',
  );

  win.close();
  console.log('  ok - client-side trailing slash normalizes');
}

function testWindowsPathVariantsNormalize() {
  const {win, posted} = makeWebview();

  send(win, {
    type: 'configData',
    config: {work_dir: 'C:\\Repo\\Alpha\\'},
    apiKeys: {},
  });
  openSidebar(win);
  enableWorkspaceFilter(win);
  const fetched = lastGetHistory(posted);

  send(win, {
    type: 'history',
    sessions: [
      makeRow({
        task_id: 108,
        title: 'completed Windows path variant',
        is_running: false,
        work_dir: 'c:/repo/alpha',
      }),
    ],
    offset: 0,
    generation: fetched.generation,
  });

  assert.deepStrictEqual(
    visibleRows(win),
    ['completed Windows path variant'],
    'Windows backslash/forward-slash, drive-case, component-case, and ' +
      'trailing-separator variants must compare as the same workspace',
  );

  win.close();
  console.log('  ok - Windows path separator/case variants normalize');
}

function runTests() {
  testRunningRowVisibleDespiteResolvedWorkDirVariant();
  testRunningRowVisibleFromOtherWorkspace();
  testUncheckingRunningStillHidesRunningRow();
  testTasksUpdatedRefetchesHistoryWhileSidebarOpen();
  testTrailingSlashNormalizedMatchForCompletedRows();
  testClientTrailingSlashNormalizes();
  testWindowsPathVariantsNormalize();
}

try {
  runTests();
  console.log('\n7 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
