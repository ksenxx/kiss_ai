// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {makePage: makeSharedPage, send, MEDIA} = require('./simplify2_harness');
const {inlineDesignTokens} = require('./designTokens');

const FROZEN_NOW_MS = 1_700_500_000_000;

const pages = [];

function makePage() {
  const {win, posted} = makeSharedPage({beforeScripts: freezeClockAndStyle});
  pages.push(win);
  return {win, posted};
}

/** Freeze the page's Date at FROZEN_NOW_MS and inline main.css. */
function freezeClockAndStyle(win) {
  const RealDate = win.Date;
  const FakeDate = function (...args) {
    if (args.length === 0) return new RealDate(FROZEN_NOW_MS);
    return new RealDate(...args);
  };
  FakeDate.prototype = RealDate.prototype;
  FakeDate.now = () => FROZEN_NOW_MS;
  FakeDate.parse = RealDate.parse;
  FakeDate.UTC = RealDate.UTC;
  win.Date = FakeDate;

  const cssText = inlineDesignTokens(
    fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8'),
  );
  const styleEl = win.document.createElement('style');
  styleEl.textContent = cssText;
  win.document.head.appendChild(styleEl);
}

function sendEmptyWorkDir(win) {
  send(win, {
    type: 'configData',
    config: {work_dir: ''},
    apiKeys: {},
  });
}

function configureClientWorkDir(win, workDir) {
  send(win, {
    type: 'configData',
    config: {work_dir: workDir},
    apiKeys: {},
  });
}

function makeRow(overrides) {
  return Object.assign(
    {
      id: 'chat-' + (overrides.task_id || 0),
      task_id: overrides.task_id || 0,
      title: overrides.title || 'untitled',
      timestamp: 1_700_000_000,
      preview: overrides.title || 'untitled',
      has_events: false,
      failed: false,
      is_running: false,
      tokens: 1,
      cost: 0,
      steps: 1,
      is_favorite: false,
      work_dir: '',
      startTs: 1_700_000_000_000,
      endTs: 1_700_000_010_000,
    },
    overrides,
  );
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

function clickBurgerMenu(win) {
  const btn = win.document.getElementById('menu-btn');
  assert.ok(btn, 'burger menu button (#menu-btn) must exist');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function lastGetHistory(posted) {
  for (let i = posted.length - 1; i >= 0; i--) {
    if (posted[i] && posted[i].type === 'getHistory') return posted[i];
  }
  return null;
}

function findDot(row) {
  return row.querySelector('.sidebar-item-running');
}

function findCompletedDot(row) {
  return row.querySelector('.sidebar-item-completed');
}

function testRunningTaskAppearsWhenBurgerOpened() {
  const {win, posted} = makePage();
  sendEmptyWorkDir(win);
  const sidebar = win.document.getElementById('sidebar');
  assert.ok(sidebar, 'sidebar element must exist');
  assert.ok(
    !sidebar.classList.contains('open'),
    'sidebar must start CLOSED to reproduce the regression flow',
  );

  posted.length = 0;
  send(win, {
    type: 'status',
    running: true,
    startTs: FROZEN_NOW_MS - 3_000,
    tabId: undefined,
  });
  assert.strictEqual(
    lastGetHistory(posted),
    null,
    'with sidebar closed, status running:true must NOT post getHistory; ' +
      'the burger click is what triggers the fetch',
  );

  posted.length = 0;
  clickBurgerMenu(win);
  assert.ok(
    sidebar.classList.contains('open'),
    'burger click must open the sidebar',
  );
  const fetched = lastGetHistory(posted);
  assert.ok(
    fetched,
    'burger click must post a getHistory command so the History ' +
      'panel can list the freshly-started running task',
  );
  const generation = fetched.generation;

  const sessions = [
    makeRow({
      task_id: 42,
      title: 'my running task',
      is_running: true,
      endTs: 0,
    }),
  ];
  send(win, {
    type: 'history',
    sessions: sessions,
    offset: 0,
    generation: generation,
  });

  const row = rowByTitle(win, 'my running task');
  assert.ok(
    row,
    'the running task row MUST appear in the History list after ' +
      'the burger menu opens — this is the user-reported invariant',
  );
  const dot = findDot(row);
  assert.ok(dot, 'running task row MUST carry a .sidebar-item-running spinner');
  assert.strictEqual(
    row.firstElementChild,
    dot,
    'spinner MUST be the first child of the row so it sits to ' +
      'the LEFT of the task title (middle-left layout)',
  );
  assert.strictEqual(row.dataset.category, 'running');
  // The spinner's colour and animation are historyRunningPulsingDot's.
  assert.ok(
    dot.classList.contains('status-spinner'),
    'the running indicator must be the shared .status-spinner icon',
  );

  posted.length = 0;
  send(win, {type: 'status', running: false, tabId: undefined});
  const refetch = lastGetHistory(posted);
  assert.ok(
    refetch,
    'status running:false with sidebar open MUST post getHistory ' +
      'so the row can swap pulsing for solid dot',
  );

  const finished = [
    makeRow({
      task_id: 42,
      title: 'my running task',
      is_running: false,
      endTs: FROZEN_NOW_MS - 1_000,
    }),
  ];
  send(win, {
    type: 'history',
    sessions: finished,
    offset: 0,
    generation: refetch.generation,
  });

  const row2 = rowByTitle(win, 'my running task');
  assert.ok(row2, 'row must still exist after status running:false');
  assert.strictEqual(
    findDot(row2),
    null,
    'pulsing .sidebar-item-running dot MUST disappear once the task ends',
  );
  const completed = findCompletedDot(row2);
  assert.ok(
    completed,
    'finished task row MUST show a .sidebar-item-completed green tick ' +
      '(no animation)',
  );
  assert.strictEqual(
    row2.firstElementChild,
    completed,
    'green tick MUST be the first child of the row',
  );
  assert.ok(
    completed.classList.contains('status-tick'),
    'the completed indicator must be the shared .status-tick icon',
  );

  const cs2 = win.getComputedStyle(completed);
  assert.strictEqual(
    cs2.color,
    'var(--green)',
    `tick colour must be the theme green (--status-ok -> var(--green)); got: ${cs2.color}`,
  );
  const anim2Name = cs2.getPropertyValue('animation-name') || '';
  const anim2Short = cs2.getPropertyValue('animation') || '';
  assert.ok(
    anim2Name.indexOf('status-spin') < 0 &&
      anim2Short.indexOf('status-spin') < 0,
    `tick must NOT animate; got animation-name="${anim2Name}" ` +
      `animation="${anim2Short}"`,
  );

  console.log(
    '  ok - running task appears via burger menu and swaps to a tick on finish',
  );
}

function testRunningTaskVisibleUnderDefaultWorkspaceFilter() {
  {
    const {win, posted} = makePage();
    configureClientWorkDir(win, '/Users/me/repo');
    clickBurgerMenu(win);
    const fetched = lastGetHistory(posted);
    assert.ok(fetched, 'burger click must post getHistory');
    send(win, {
      type: 'history',
      sessions: [
        makeRow({
          task_id: 1,
          title: 'matching wd',
          is_running: true,
          work_dir: '/Users/me/repo',
          endTs: 0,
        }),
      ],
      offset: 0,
      generation: fetched.generation,
    });
    const row = rowByTitle(win, 'matching wd');
    assert.ok(row, 'row with matching work_dir MUST be visible');
    assert.notStrictEqual(
      row.style.display,
      'none',
      'matching-work_dir row MUST be visible under default Workspace filter',
    );
    assert.ok(
      findDot(row),
      'matching-work_dir running row MUST carry the spinner',
    );
  }

  {
    const {win, posted} = makePage();
    configureClientWorkDir(win, '/Users/me/repo');
    clickBurgerMenu(win);
    const fetched = lastGetHistory(posted);
    assert.ok(fetched, 'burger click must post getHistory');
    send(win, {
      type: 'history',
      sessions: [
        makeRow({
          task_id: 2,
          title: 'empty wd row',
          is_running: true,
          work_dir: '',
          endTs: 0,
        }),
      ],
      offset: 0,
      generation: fetched.generation,
    });
    const row = rowByTitle(win, 'empty wd row');
    assert.ok(row, 'row with empty work_dir MUST render in DOM');
    assert.notStrictEqual(
      row.style.display,
      'none',
      'empty-row-work_dir RUNNING task MUST be visible under default ' +
        'Workspace filter — the comment in applyHistoryFilterVisibility ' +
        'says "an empty row work_dir passes the filter" but the strict ' +
        '=== test hides it.  This is the user-reported regression.',
    );
  }

  {
    const {win, posted} = makePage();
    configureClientWorkDir(win, '');
    clickBurgerMenu(win);
    const fetched = lastGetHistory(posted);
    assert.ok(fetched, 'burger click must post getHistory');
    send(win, {
      type: 'history',
      sessions: [
        makeRow({
          task_id: 3,
          title: 'no folder open',
          is_running: true,
          work_dir: '/some/abs/path',
          endTs: 0,
        }),
      ],
      offset: 0,
      generation: fetched.generation,
    });
    const row = rowByTitle(win, 'no folder open');
    assert.ok(row, 'row MUST render');
    assert.notStrictEqual(
      row.style.display,
      'none',
      'when client work_dir is empty, every row MUST pass the ' +
        'Workspace filter regardless of its own work_dir',
    );
  }

  console.log(
    '  ok - running task visible under default Workspace filter (3 sub-cases)',
  );
}

function main() {
  testRunningTaskAppearsWhenBurgerOpened();
  testRunningTaskVisibleUnderDefaultWorkspaceFilter();
  console.log('historyBurgerMenuRunningTask.test.js: all assertions passed.');
}

main();
for (const win of pages) win.close();
