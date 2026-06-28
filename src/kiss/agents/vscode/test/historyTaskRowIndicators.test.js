// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test pinning two intertwined History-sidebar invariants
// that regressed together:
//
//   (a) When the user starts a task and then opens the burger menu,
//       the freshly-started running task MUST appear at the TOP of
//       the History list (no other row may come before it).
//
//   (b) Each row's status dot follows this spec:
//         * ``is_running:true``  → ``.sidebar-item-running``
//           (pulsing green via @keyframes sidebar-running-pulse).
//         * ``failed:true``      → ``.sidebar-item-failed``
//           (solid red).
//         * Finished cleanly AND the user just witnessed the
//           running→completed transition in this page session →
//           ``.sidebar-item-completed`` (SOLID green, no animation),
//           which then STAYS solid green for the rest of the session.
//         * Finished cleanly on a FRESH history load (never seen
//           running in this session) → NO dot at all.  The History
//           panel must not display a sea of solid green circles on
//           old completed tasks the user didn't run this session.
//       When present, the indicator MUST be the FIRST child of the
//       row so it sits to the left of the task title.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/historyTaskRowIndicators.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const FROZEN_NOW_MS = 1_700_500_000_000;

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

  // Inline main.css so getComputedStyle reflects the real
  // .sidebar-item-* indicator rules in jsdom.
  const cssText = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const styleEl = win.document.createElement('style');
  styleEl.textContent = cssText;
  win.document.head.appendChild(styleEl);

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function makeRow(overrides) {
  return Object.assign(
    {
      id: 'chat-' + (overrides.task_id || 0),
      task_id: overrides.task_id || 0,
      title: overrides.title || 'untitled',
      timestamp: overrides.timestamp || 1_700_000_000,
      preview: overrides.title || 'untitled',
      has_events: false,
      failed: false,
      is_running: false,
      tokens: 1,
      cost: 0,
      steps: 1,
      is_favorite: false,
      work_dir: '',
      startTs: (overrides.timestamp || 1_700_000_000) * 1000,
      endTs: 1_700_000_010_000,
    },
    overrides,
  );
}

function uncheckWorkspaceFilter(win) {
  // Workspace filter is on by default; switch it off so empty
  // work_dir rows (fresh running tasks) and finished rows alike
  // pass the filter and become visible.  Without this, the
  // workspace filter would hide rows whose ``work_dir`` does not
  // match the client's configured workspace.
  send(win, {
    type: 'configData',
    config: {work_dir: ''},
    apiKeys: {},
  });
  const ws = win.document.getElementById('hf-workspace');
  if (ws && ws.checked) {
    ws.checked = false;
    ws.dispatchEvent(new win.Event('change', {bubbles: true}));
  }
}

function openBurgerMenu(win) {
  // ``menu-btn`` is the burger button under the input box; clicking
  // it opens the History sidebar (#sidebar) and posts ``getHistory``
  // to the backend.
  const btn = win.document.getElementById('menu-btn');
  assert.ok(btn, 'burger menu button (#menu-btn) must exist');
  btn.click();
}

function rows(win) {
  const list = win.document.getElementById('history-list');
  return list.querySelectorAll('.sidebar-item');
}

function indicatorOf(row) {
  return (
    row.querySelector('.sidebar-item-running') ||
    row.querySelector('.sidebar-item-completed') ||
    row.querySelector('.sidebar-item-failed')
  );
}

function testRunningTaskShowsAtTopAfterBurgerOpen() {
  // Scenario: an older finished task is already in history; the
  // user kicks off a new task whose row is persisted at the top
  // of ``task_history`` (most-recent timestamp).  When the user
  // clicks the burger menu, the backend's ``getHistory`` reply
  // delivers sessions in ``ORDER BY timestamp DESC`` — the
  // running task therefore arrives FIRST.  The frontend MUST
  // preserve that ordering: the running row MUST be rendered as
  // the FIRST ``.sidebar-item`` in the History list.
  const {win, posted} = makeWebview();

  // Simulate: user clicks the burger menu while a task is running.
  // This must open the sidebar AND post ``getHistory``.
  openBurgerMenu(win);
  uncheckWorkspaceFilter(win);
  const getHist = posted.find(m => m && m.type === 'getHistory');
  assert.ok(
    getHist,
    'opening the burger menu must post getHistory; got: ' +
      JSON.stringify(posted),
  );

  // Backend reply (timestamp-DESC order): the newly-started
  // running task comes first, the older finished task second.
  const sessions = [
    makeRow({
      task_id: 101,
      title: 'NEW running task',
      is_running: true,
      timestamp: 1_700_100_000,
      endTs: 0,
    }),
    makeRow({
      task_id: 100,
      title: 'OLD finished task',
      is_running: false,
      timestamp: 1_700_000_000,
    }),
  ];
  send(win, {
    type: 'history',
    sessions,
    offset: 0,
    generation: getHist.generation,
  });

  const r = rows(win);
  assert.strictEqual(r.length, 2, 'both rows must render');
  const firstText = r[0].querySelector('.sidebar-item-text');
  assert.ok(firstText, 'first row must have a text span');
  assert.strictEqual(
    firstText.textContent,
    'NEW running task',
    'the freshly-started running task MUST be the FIRST row in ' +
      'the History list when the burger menu is opened',
  );

  // And the running row must carry the pulsing green dot as its
  // first child (middle-left of the row).
  const dot = r[0].querySelector('.sidebar-item-running');
  assert.ok(
    dot,
    'first (running) row must carry .sidebar-item-running dot',
  );
  assert.strictEqual(
    r[0].firstElementChild,
    dot,
    'pulsing dot must be the first child of the row',
  );

  win.close();
  console.log(
    '  ok - running task is FIRST in History list when burger menu opens',
  );
}

function testFinishedTaskShowsSolidGreenCircle() {
  // Scenario: the user starts a task (rendered as running with the
  // pulsing dot), then the same task finishes (re-rendered as
  // ``is_running:false`` in the SAME page session).  The dot MUST
  // swap from pulsing green to SOLID green and STAY solid green
  // across subsequent history refreshes.  Meanwhile, a row the
  // session NEVER saw running (a fresh completed row from the
  // backend) MUST render with NO dot at all — the History panel
  // must not surface solid green circles on every old completed
  // task.
  const {win, posted} = makeWebview();
  openBurgerMenu(win);
  uncheckWorkspaceFilter(win);
  const getHist = posted.find(m => m && m.type === 'getHistory');
  assert.ok(getHist, 'burger menu open must post getHistory');

  // Step 1: deliver a batch with one fresh-completed row (never seen
  // running this session), one running row, and one failed row.
  const initialSessions = [
    makeRow({task_id: 1, title: 'fresh completed task', is_running: false}),
    makeRow({
      task_id: 2,
      title: 'running task',
      is_running: true,
      endTs: 0,
    }),
    makeRow({
      task_id: 3,
      title: 'failed task',
      is_running: false,
      failed: true,
    }),
  ];
  send(win, {
    type: 'history',
    sessions: initialSessions,
    offset: 0,
    generation: getHist.generation,
  });

  let byTitle = {};
  rows(win).forEach(r => {
    const t = r.querySelector('.sidebar-item-text');
    if (t) byTitle[t.textContent] = r;
  });

  // The fresh-completed row MUST NOT render any solid green dot.
  const freshRow = byTitle['fresh completed task'];
  assert.ok(freshRow, 'fresh completed row must render');
  assert.strictEqual(
    freshRow.querySelector('.sidebar-item-completed'),
    null,
    'fresh history load of a completed task MUST NOT render a ' +
      'solid green circle — that is reserved for tasks the user ' +
      'just watched transition from running to completed',
  );
  assert.strictEqual(
    freshRow.querySelector('.sidebar-item-running'),
    null,
    'fresh completed row must not render a pulsing dot either',
  );

  // Running row gets the pulsing dot.
  const runningRow = byTitle['running task'];
  assert.ok(
    runningRow.querySelector('.sidebar-item-running'),
    'running row must carry .sidebar-item-running',
  );

  // Failed row keeps the red dot.
  const failedRow = byTitle['failed task'];
  assert.ok(
    failedRow.querySelector('.sidebar-item-failed'),
    'failed row must carry .sidebar-item-failed',
  );

  // Step 2: the running task finishes — deliver a follow-up event
  // with task_id=2 now ``is_running:false``.  The session WITNESSED
  // the running state, so its row MUST now show a SOLID green dot.
  const finishedSessions = [
    makeRow({task_id: 1, title: 'fresh completed task', is_running: false}),
    makeRow({
      task_id: 2,
      title: 'running task',
      is_running: false,
      endTs: 1_700_000_010_000,
    }),
    makeRow({
      task_id: 3,
      title: 'failed task',
      is_running: false,
      failed: true,
    }),
  ];
  send(win, {
    type: 'history',
    sessions: finishedSessions,
    offset: 0,
    generation: getHist.generation,
  });
  byTitle = {};
  rows(win).forEach(r => {
    const t = r.querySelector('.sidebar-item-text');
    if (t) byTitle[t.textContent] = r;
  });

  const transitionedRow = byTitle['running task'];
  const completedDot = transitionedRow.querySelector(
    '.sidebar-item-completed',
  );
  assert.ok(
    completedDot,
    'a row whose running→completed transition the session ' +
      'witnessed MUST render a .sidebar-item-completed solid green dot',
  );
  assert.strictEqual(
    transitionedRow.firstElementChild,
    completedDot,
    'solid green circle must be the FIRST child (middle-left) of the row',
  );

  // Computed style: solid green, no animation.
  const cs = win.getComputedStyle(completedDot);
  assert.strictEqual(
    cs.backgroundColor,
    'rgb(46, 125, 50)',
    `solid circle background must be #2e7d32 ` +
      `(rgb(46, 125, 50)); got: ${cs.backgroundColor}`,
  );
  const animName = cs.getPropertyValue('animation-name') || '';
  const animShort = cs.getPropertyValue('animation') || '';
  assert.ok(
    animName.indexOf('sidebar-running-pulse') < 0 &&
      animShort.indexOf('sidebar-running-pulse') < 0,
    'solid (completed) circle MUST NOT animate via ' +
      `sidebar-running-pulse; got animation-name="${animName}" ` +
      `animation="${animShort}"`,
  );

  // The fresh-completed row STILL must not carry a solid green dot
  // (unrelated to the witnessed transition).
  const stillFreshRow = byTitle['fresh completed task'];
  assert.strictEqual(
    stillFreshRow.querySelector('.sidebar-item-completed'),
    null,
    'an unrelated fresh-completed row MUST not inherit the solid ' +
      'green circle just because another row transitioned',
  );

  // Step 3: a third reload (everything still finished) must KEEP the
  // solid green dot on the transitioned row.
  send(win, {
    type: 'history',
    sessions: finishedSessions,
    offset: 0,
    generation: getHist.generation,
  });
  const persisted = rows(win)[1];
  assert.ok(
    persisted.querySelector('.sidebar-item-completed'),
    'solid green circle MUST persist across subsequent ' +
      'history reloads once it has appeared',
  );

  win.close();
  console.log(
    '  ok - solid green circle only after witnessed running→completed transition',
  );
}

function testIndicatorsAreVerticallyCenteredInTaskPanels() {
  // Reproduces the visual regression reported by the user: History
  // rows use the same multi-line ``running-item`` task-panel layout
  // for running, completed, and failed tasks.  The status indicator
  // is the row's first child at the left edge; it must be vertically
  // centered in the panel, not pinned to the top-left.  This drives
  // the real chat.html + main.css + main.js inside jsdom and checks
  // the rendered task panels for all three status variants.
  const {win, posted} = makeWebview();
  openBurgerMenu(win);
  uncheckWorkspaceFilter(win);
  const getHist = posted.find(m => m && m.type === 'getHistory');
  assert.ok(getHist, 'burger menu open must post getHistory');

  // Prime: send task_id 11 as running first so the session
  // witnesses its later running→completed transition.  Without this
  // priming step the fresh-completed row would render with NO dot
  // (per the no-solid-green-on-fresh-load invariant) and would have
  // nothing to vertically center.
  send(win, {
    type: 'history',
    sessions: [
      makeRow({
        task_id: 11,
        title: 'completed centered task',
        is_running: true,
        endTs: 0,
      }),
      makeRow({
        task_id: 12,
        title: 'running centered task',
        is_running: true,
        endTs: 0,
      }),
      makeRow({
        task_id: 13,
        title: 'failed centered task',
        failed: true,
      }),
    ],
    offset: 0,
    generation: getHist.generation,
  });

  send(win, {
    type: 'history',
    sessions: [
      makeRow({task_id: 11, title: 'completed centered task'}),
      makeRow({
        task_id: 12,
        title: 'running centered task',
        is_running: true,
        endTs: 0,
      }),
      makeRow({
        task_id: 13,
        title: 'failed centered task',
        failed: true,
      }),
    ],
    offset: 0,
    generation: getHist.generation,
  });

  rows(win).forEach(row => {
    const title = row.querySelector('.sidebar-item-text').textContent;
    const indicator = indicatorOf(row);
    assert.ok(indicator, `row ${title} must render a status indicator`);
    assert.strictEqual(
      row.firstElementChild,
      indicator,
      `row ${title} indicator must stay at the left edge as first child`,
    );
    const style = win.getComputedStyle(indicator);
    assert.strictEqual(
      style.top,
      '50%',
      `row ${title} indicator must be vertically centered in the task ` +
        `panel, not top-aligned; got top=${style.top}`,
    );
    assert.strictEqual(
      style.transform,
      'translateY(-50%)',
      `row ${title} indicator must translate by half its own height ` +
        `to sit at panel middle-left; got transform=${style.transform}`,
    );
  });

  win.close();
  console.log(
    '  ok - history task-panel indicators are centered at middle-left',
  );
}

function testCompletedDotKeyframesNotShared() {
  // Regression guard: the @keyframes ``sidebar-running-pulse`` rule
  // belongs to the running dot ONLY.  The completed dot is a
  // separate class; main.css MUST define it as solid green and MUST
  // NOT attach the pulsing animation to it.
  const cssText = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  assert.ok(
    /\.sidebar-item-completed\s*\{/.test(cssText),
    'main.css must define .sidebar-item-completed for the solid ' +
      'green finished-task circle',
  );

  // Pull out the .sidebar-item-completed block and assert it does
  // NOT contain ``animation:`` or ``animation-name:`` referencing
  // the pulse keyframe.
  const m = cssText.match(/\.sidebar-item-completed\s*\{([^}]*)\}/);
  assert.ok(m, 'expected a single-rule .sidebar-item-completed block');
  const body = m[1];
  assert.ok(
    body.indexOf('sidebar-running-pulse') < 0,
    '.sidebar-item-completed MUST NOT use sidebar-running-pulse; ' +
      'the solid circle is static',
  );
  assert.ok(
    /background\s*:\s*#2e7d32/i.test(body),
    '.sidebar-item-completed MUST use the #2e7d32 green background',
  );

  console.log('  ok - .sidebar-item-completed is defined as solid green');
}

function main() {
  testRunningTaskShowsAtTopAfterBurgerOpen();
  testFinishedTaskShowsSolidGreenCircle();
  testIndicatorsAreVerticallyCenteredInTaskPanels();
  testCompletedDotKeyframesNotShared();
  console.log('historyTaskRowIndicators.test.js: all assertions passed.');
}

main();
