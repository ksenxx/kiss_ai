// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePage, send} = require('./simplify2_harness');

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

// The CHATS list is not scoped to the working directory: every row of
// the history shows whatever folder its task ran in, running or not,
// and whatever the client's own working directory is.
function testRowsFromEveryWorkDirVisible() {
  const {win, posted} = makePage();

  send(win, {
    type: 'configData',
    config: {work_dir: '/repo/alpha'},
    apiKeys: {},
  });
  openSidebar(win);
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
      makeRow({
        task_id: 102,
        title: 'other workspace task',
        work_dir: '/repo/beta',
      }),
      makeRow({
        task_id: 103,
        title: 'windows task',
        work_dir: 'C:\\Users\\me\\proj',
      }),
      makeRow({task_id: 104, title: 'no work dir task', work_dir: ''}),
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
  assert.deepStrictEqual(
    visibleRows(win),
    ['just-started task', 'other workspace task', 'windows task', 'no work dir task'],
    'every row must be visible whatever its work_dir: the CHATS list has ' +
      'no workspace filter',
  );
  assert.strictEqual(
    win.document.getElementById('hf-workspace'),
    null,
    'the Workspace chip is gone from the CHATS filter bar',
  );

  // A change of the working directory does not hide anything either.
  send(win, {
    type: 'configData',
    config: {work_dir: '/repo/gamma'},
    apiKeys: {},
  });
  assert.strictEqual(visibleRows(win).length, 4);

  win.close();
  console.log('  ok - rows from every work_dir are visible');
}

function testUncheckingRunningStillHidesRunningRow() {
  const {win, posted} = makePage();

  send(win, {
    type: 'configData',
    config: {work_dir: '/repo/alpha'},
    apiKeys: {},
  });
  openSidebar(win);
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
    'unchecking the Running filter must hide running rows',
  );

  win.close();
  console.log('  ok - unchecking Running still hides the running row');
}

function testTasksUpdatedRefetchesHistoryWhileSidebarOpen() {
  const {win, posted} = makePage();

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

function runTests() {
  testRowsFromEveryWorkDirVisible();
  testUncheckingRunningStillHidesRunningRow();
  testTasksUpdatedRefetchesHistoryWhileSidebarOpen();
}

try {
  runTests();
  console.log('\n3 passed, 0 failed');
  process.exit(0);
} catch (err) {
  console.error('FAIL:', err && err.stack ? err.stack : err);
  process.exit(1);
}
