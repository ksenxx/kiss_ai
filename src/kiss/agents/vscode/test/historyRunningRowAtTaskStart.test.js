// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test: the History sidebar MUST show a task
// the moment kiss-web starts it — even when the running row's
// persisted ``work_dir`` is a path VARIANT of the client's configured
// workspace (macOS /var → /private/var symlink resolution via
// ``git rev-parse --show-toplevel`` in ``discover_repo``, trailing
// slashes, etc.).
//
// User-reported bug: "The task history panel is not getting updated
// as soon as kiss-web starts a task.  It doesn't show the task in the
// task history."
//
// Root cause: since ``ChatSorcarAgent.run`` persists ``extra.work_dir``
// EARLY (at ``_add_task``), a freshly-started running row always
// carries a NON-EMPTY work_dir; worktree runs redirect through
// ``discover_repo`` which resolves symlinks, so the persisted path can
// strictly mismatch the client's configured workspace path.  The
// default-CHECKED ``#hf-workspace`` filter in
// ``applyHistoryFilterVisibility`` then used a strict
// ``rowWorkDir === clientWorkDir`` string comparison and hid the
// just-started running row (rendered but ``display:none``).
//
// Requirements driven by this test:
//
//   1. A RUNNING row (``data-category === 'running'``) must ALWAYS
//      pass the Workspace filter, whatever its ``work_dir``, so a
//      just-started task is never invisible in History.
//
//   2. ``tasks_updated`` (broadcast by ``ChatSorcarAgent.run`` right
//      after ``_add_task``) must trigger a ``getHistory`` fetch while
//      the sidebar is open.
//
//   3. The Workspace comparison for NON-running rows must normalize
//      trailing slashes, so ``/repo/alpha/`` matches ``/repo/alpha``.
//
//   4. Completed rows with a genuinely different work_dir must STILL
//      be hidden (the filter stays functional).
//
// This test drives the production ``media/main.js`` (plus the real
// ``media/chat.html`` markup and ``media/panelCopy.js``) inside jsdom —
// no mocks of project code — exactly like the existing
// ``historyWorkspaceFilter.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/historyRunningRowAtTaskStart.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview.
 *
 * The real ``chat.html`` body is loaded (placeholders blanked), then
 * ``panelCopy.js`` and ``main.js`` are evaluated in the window, and a
 * recording ``acquireVsCodeApi`` stub is installed (the only host API
 * the webview has).
 */
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

/** Dispatch a backend→webview event exactly like the extension does. */
function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** Open the History sidebar via the burger button like a real user. */
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

/** Return the most recent posted ``getHistory`` command, or null. */
function lastGetHistory(posted) {
  for (let i = posted.length - 1; i >= 0; i--) {
    if (posted[i] && posted[i].type === 'getHistory') return posted[i];
  }
  return null;
}

/** Collect the visible history rows (display !== 'none'). */
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

/** Find a rendered row (visible or hidden) by its title text. */
function rowByTitle(win, title) {
  const list = win.document.getElementById('history-list');
  const rows = list.querySelectorAll('.sidebar-item');
  for (const r of rows) {
    const t = r.querySelector('.sidebar-item-text');
    if (t && t.textContent === title) return r;
  }
  return null;
}

/** Build a full history-session row with overrides. */
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

// --- Test 1: THE user-reported regression ------------------------------
//
// Client workspace is /var/repo/alpha; kiss-web starts a task whose
// EARLY-persisted work_dir is the RESOLVED variant
// /private/var/repo/alpha (what ``git rev-parse --show-toplevel``
// returns through macOS's /var → /private/var symlink).  The running
// row must be VISIBLE in the History panel immediately.
function testRunningRowVisibleDespiteResolvedWorkDirVariant() {
  const {win, posted} = makeWebview();

  send(win, {
    type: 'configData',
    config: {work_dir: '/var/repo/alpha'},
    apiKeys: {},
  });

  openSidebar(win);
  assert.ok(lastGetHistory(posted), 'opening the sidebar must post getHistory');

  // The agent has now inserted the task_history row and broadcasts
  // the real start-time system event.  This is the production trigger
  // that must make an already-open History panel fetch the new row.
  posted.length = 0;
  send(win, {type: 'tasks_updated', taskId: ''});
  const fetched = lastGetHistory(posted);
  assert.ok(
    fetched,
    'start-time tasks_updated must post getHistory while the sidebar is open',
  );

  // Backend reply: the JUST-STARTED running task.  Its work_dir was
  // persisted early by ChatSorcarAgent.run via strip_worktree_suffix
  // on the worktree path — a symlink-RESOLVED variant of the client
  // workspace.  endTs 0 = still running.
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

// --- Test 2: running row with a COMPLETELY different work_dir ----------
//
// Even a running task from another workspace must stay visible: the
// user must never lose sight of a live agent.  (The Running checkbox —
// not the Workspace filter — is the intended control for running rows.)
function testRunningRowVisibleFromOtherWorkspace() {
  const {win, posted} = makeWebview();

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

// --- Test 3: unchecking Running still hides the running row ------------
//
// The always-pass rule applies to the WORKSPACE filter only; the
// Running category checkbox must retain full control over running rows.
function testUncheckingRunningStillHidesRunningRow() {
  const {win, posted} = makeWebview();

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

  // Visible while Running is checked (default).
  assert.deepStrictEqual(visibleRows(win), ['running task']);

  // Uncheck Running → the row must hide (Workspace always-pass must
  // NOT override the category filter).
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

// --- Test 4: tasks_updated triggers a history re-fetch ------------------
//
// ChatSorcarAgent.run broadcasts ``tasks_updated`` right after
// ``_add_task``; with the sidebar open the webview must re-fetch.
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

// --- Test 5: trailing-slash normalization for completed rows -----------
function testTrailingSlashNormalizedMatchForCompletedRows() {
  const {win, posted} = makeWebview();

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

// --- Test 6: client work_dir with trailing slash also normalizes -------
function testClientTrailingSlashNormalizes() {
  const {win, posted} = makeWebview();

  send(win, {
    type: 'configData',
    config: {work_dir: '/repo/alpha/'},
    apiKeys: {},
  });
  openSidebar(win);
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

// --- Test 7: Windows separator/case variants normalize -----------------
//
// VS Code's ``Uri.fsPath`` uses backslashes on Windows, while Git's
// ``rev-parse --show-toplevel`` (and strip_worktree_suffix) can yield
// forward slashes.  Windows drive paths are also case-insensitive.
function testWindowsPathVariantsNormalize() {
  const {win, posted} = makeWebview();

  send(win, {
    type: 'configData',
    config: {work_dir: 'C:\\Repo\\Alpha\\'},
    apiKeys: {},
  });
  openSidebar(win);
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
