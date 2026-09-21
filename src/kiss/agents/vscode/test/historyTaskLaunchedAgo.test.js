// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (jsdom) tests for the "launched N units ago" label that
// every task panel in the task-history panel shows next to its
// show-details chevron (``.sidebar-item-collapse``).
//
// The label reports how long ago the task was launched using the unit
// ladder minutes -> hours -> days -> weeks -> months -> years (and
// "just now" under a minute).  It prefers the session's ``startTs``
// (epoch ms) and falls back to the row's insertion ``timestamp``
// (epoch seconds); a session with neither renders no label.  A 30 s
// sweep keeps on-screen labels current between history refreshes.
//
// The same ``media/`` bundle is served to the VS Code webview and to
// the remote webapp (``web_server.py`` renders ``chat.html`` with
// ``<body class="remote-chat">`` plus ``remote-codex.css``), so the
// rendering assertions run against both surfaces.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const MINUTE_MS = 60 * 1000;
const HOUR_MS = 60 * MINUTE_MS;
const DAY_MS = 24 * HOUR_MS;

function makeWebview(remote) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  if (remote) html = html.replace('<body', '<body class="remote-chat"');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const style = win.document.createElement('style');
  style.textContent = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  win.document.head.appendChild(style);
  if (remote) {
    const remoteStyle = win.document.createElement('style');
    remoteStyle.textContent = fs.readFileSync(
      path.join(MEDIA, 'remote-codex.css'),
      'utf8',
    );
    win.document.head.appendChild(remoteStyle);
  }

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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function makeSession(overrides) {
  return Object.assign(
    {
      id: 'chat-1',
      task_id: 'task-1',
      title: 'refactor the parser',
      preview: 'refactor the parser',
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
      is_favorite: false,
    },
    overrides || {},
  );
}

function loadHistory(win, sessions) {
  win.dispatchEvent(
    new win.MessageEvent('message', {
      data: {type: 'history', offset: 0, generation: 0, sessions: sessions},
    }),
  );
}

function historyRows(win) {
  return Array.prototype.slice.call(
    win.document.querySelectorAll('#history-list .sidebar-item'),
  );
}

// ---------------------------------------------------------------------------
// 1. Every unit of the ladder renders, next to the show-details button,
//    on both surfaces.
// ---------------------------------------------------------------------------

function testUnitLadder(remote) {
  const label = remote ? 'webapp' : 'extension';
  const {win} = makeWebview(remote);
  const now = Date.now();
  // One session per rung of the ladder, oldest first so each maps to a
  // distinct rendered row (renderHistory keeps the given order).
  const cases = [
    {ago: 30 * 1000, text: 'launched just now'},
    {ago: 5 * MINUTE_MS, text: 'launched 5 minutes ago'},
    {ago: 1 * MINUTE_MS, text: 'launched 1 minute ago'},
    {ago: 3 * HOUR_MS, text: 'launched 3 hours ago'},
    {ago: 1 * HOUR_MS, text: 'launched 1 hour ago'},
    {ago: 2 * DAY_MS, text: 'launched 2 days ago'},
    {ago: 1 * DAY_MS, text: 'launched 1 day ago'},
    {ago: 8 * DAY_MS, text: 'launched 1 week ago'},
    {ago: 22 * DAY_MS, text: 'launched 3 weeks ago'},
    {ago: 45 * DAY_MS, text: 'launched 1 month ago'},
    {ago: 200 * DAY_MS, text: 'launched 6 months ago'},
    {ago: 400 * DAY_MS, text: 'launched 1 year ago'},
    {ago: 800 * DAY_MS, text: 'launched 2 years ago'},
  ];
  const sessions = cases.map((c, i) =>
    makeSession({
      id: 'chat-' + i,
      task_id: 'task-' + i,
      startTs: now - c.ago,
      timestamp: Math.floor((now - c.ago) / 1000),
    }),
  );
  loadHistory(win, sessions);

  const rows = historyRows(win);
  assert.strictEqual(
    rows.length,
    cases.length,
    `${label}: one history row per session`,
  );
  rows.forEach((row, i) => {
    const ago = row.querySelector('.sidebar-item-launched');
    assert.ok(ago, `${label}: row ${i} renders the launched-ago label`);
    assert.strictEqual(
      ago.textContent,
      cases[i].text,
      `${label}: row ${i} (${cases[i].ago} ms old)`,
    );
    // "Next to the show details button": the label lives in the same
    // action strip, immediately after the collapse toggle.
    const actions = row.querySelector('.sidebar-item-actions');
    assert.strictEqual(
      ago.parentElement,
      actions,
      `${label}: row ${i} label sits in the action strip`,
    );
    const toggle = row.querySelector('.sidebar-item-collapse');
    assert.strictEqual(
      toggle.nextElementSibling,
      ago,
      `${label}: row ${i} label is adjacent to the show-details button`,
    );
    // The tooltip carries the absolute launch time.
    assert.ok(
      ago.title.indexOf('Launched ') === 0,
      `${label}: row ${i} tooltip shows the absolute launch time`,
    );
  });
  win.close();
  console.log(`  ok - ${label}: full unit ladder renders next to the toggle`);
}

// ---------------------------------------------------------------------------
// 2. startTs (ms) wins; timestamp (s) is the fallback; neither -> no label.
// ---------------------------------------------------------------------------

function testTimestampSources() {
  const {win} = makeWebview(false);
  const now = Date.now();
  loadHistory(win, [
    makeSession({
      id: 'chat-a',
      task_id: 'task-a',
      startTs: now - 2 * HOUR_MS,
      // A deliberately different (much older) row timestamp: startTs
      // must win.
      timestamp: Math.floor(now / 1000) - 10 * 24 * 3600,
    }),
    makeSession({
      id: 'chat-b',
      task_id: 'task-b',
      startTs: 0,
      timestamp: Math.floor((now - 3 * DAY_MS) / 1000),
    }),
    // Absent timestamp: no launch instant, so no label.
    makeSession({id: 'chat-c', task_id: 'task-c', startTs: 0, timestamp: null}),
    // Epoch zero is a VALID launch instant (imported databases carry
    // it; see explorerRootsHistoryGroups: "epoch-zero timestamps remain
    // valid task dates") — it must render as N years ago, not vanish.
    makeSession({id: 'chat-d', task_id: 'task-d', startTs: 0, timestamp: 0}),
    // Corrupt timestamps beyond the JavaScript Date range must render
    // no label (the date renderer files such rows under "Undated"),
    // instead of a bogus "just now" with an Invalid Date tooltip.
    makeSession({
      id: 'chat-e',
      task_id: 'task-e',
      startTs: 0,
      timestamp: Infinity,
    }),
    makeSession({id: 'chat-f', task_id: 'task-f', startTs: 0, timestamp: 1e20}),
  ]);
  const rows = historyRows(win);
  assert.strictEqual(
    rows[0].querySelector('.sidebar-item-launched').textContent,
    'launched 2 hours ago',
    'startTs (ms) takes precedence over the row timestamp',
  );
  assert.strictEqual(
    rows[1].querySelector('.sidebar-item-launched').textContent,
    'launched 3 days ago',
    'timestamp (epoch seconds) is the fallback launch instant',
  );
  assert.strictEqual(
    rows[2].querySelector('.sidebar-item-launched'),
    null,
    'a session with no usable launch instant renders no label',
  );
  const epochYears = Math.floor(Math.floor(now / DAY_MS) / 365);
  assert.strictEqual(
    rows[3].querySelector('.sidebar-item-launched').textContent,
    'launched ' + epochYears + ' years ago',
    'an epoch-zero launch instant is valid and renders in years',
  );
  [4, 5].forEach(i => {
    assert.strictEqual(
      rows[i].querySelector('.sidebar-item-launched'),
      null,
      'a timestamp outside the Date range renders no label (row ' + i + ')',
    );
  });
  win.close();
  console.log(
    '  ok - startTs wins, timestamp falls back, epoch zero valid, ' +
      'corrupt/missing -> none',
  );
}

// ---------------------------------------------------------------------------
// 3. A future launch instant (clock skew) clamps to "just now".
// ---------------------------------------------------------------------------

function testFutureClockSkew() {
  const {win} = makeWebview(false);
  loadHistory(win, [makeSession({startTs: Date.now() + 5 * MINUTE_MS})]);
  assert.strictEqual(
    historyRows(win)[0].querySelector('.sidebar-item-launched').textContent,
    'launched just now',
    'a launch instant in the future must clamp, not go negative',
  );
  win.close();
  console.log('  ok - future launch instants clamp to "just now"');
}

// ---------------------------------------------------------------------------
// 4. The 30 s sweep keeps a quiet panel's labels ticking.
// ---------------------------------------------------------------------------

function testPeriodicRefresh() {
  const {win} = makeWebview(false);
  const now = Date.now();
  loadHistory(win, [
    makeSession({startTs: now - 45 * 1000}), // "just now" at render time
  ]);
  const ago = historyRows(win)[0].querySelector('.sidebar-item-launched');
  assert.strictEqual(ago.textContent, 'launched just now');
  // Age the label past the minute boundary by rewinding its stored
  // launch instant, then let the sweep run (jsdom timers are real).
  // At check time (~31 s from now) the age is ~121 s -> "2 minutes".
  ago.dataset.launchTs = String(now - 90 * 1000);
  return new Promise(resolve => {
    setTimeout(() => {
      assert.strictEqual(
        ago.textContent,
        'launched 2 minutes ago',
        'the periodic sweep must refresh on-screen labels in place',
      );
      win.close();
      console.log('  ok - the 30s sweep keeps labels current');
      resolve();
    }, 31 * 1000);
  });
}

// ---------------------------------------------------------------------------
// 5. The label must not break the action strip's behaviour: the toggle
//    still expands/collapses and clicks do not bubble into the row.
// ---------------------------------------------------------------------------

function testActionsStillWork(remote) {
  const label = remote ? 'webapp' : 'extension';
  const {win, posted} = makeWebview(remote);
  loadHistory(win, [makeSession({startTs: Date.now() - 5 * MINUTE_MS})]);
  const row = historyRows(win)[0];
  const toggle = row.querySelector('.sidebar-item-collapse');
  assert.ok(row.classList.contains('collapsed'), `${label}: collapsed first`);
  toggle.click();
  assert.ok(
    !row.classList.contains('collapsed'),
    `${label}: the show-details toggle still expands the panel`,
  );
  assert.ok(
    !posted.some(m => m.type === 'resumeSession'),
    `${label}: the toggle click must not bubble into the row click`,
  );
  row.click();
  assert.ok(
    posted.some(m => m.type === 'resumeSession'),
    `${label}: clicking the row body still opens the chat`,
  );
  win.close();
  console.log(`  ok - ${label}: action strip behaviour is unchanged`);
}

// ---------------------------------------------------------------------------
// 6. A window that never renders history must not keep the host process
//    alive: the refresh sweep is an on-demand timeout chain, not a
//    permanent interval. Regression: a plain setInterval(…, 30000) at
//    load time made every jsdom suite that leaves a window open hang
//    (node kept alive forever; the test runner waits on the process).
// ---------------------------------------------------------------------------

function testNoTimerWithoutLabels() {
  const {execFileSync} = require('child_process');
  const script = [
    "const fs = require('fs');",
    "const path = require('path');",
    "const {JSDOM} = require('jsdom');",
    'const MEDIA = ' + JSON.stringify(MEDIA) + ';',
    "let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');",
    "html = html.replace(/\\{\\{MODEL_NAME\\}\\}/g, 'test-model');",
    "html = html.replace(/\\{\\{[A-Z_]+\\}\\}/g, '');",
    "html = html.replace(/<script[^>]*>[\\s\\S]*?<\\/script>/g, '');",
    'const dom = new JSDOM(html, {',
    "  runScripts: 'dangerously',",
    "  url: 'https://localhost/',",
    '});',
    'const win = dom.window;',
    'win.Element.prototype.scrollIntoView = function () {};',
    'win.acquireVsCodeApi = function () {',
    '  return {postMessage() {}, getState() {}, setState() {}};',
    '};',
    "win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));",
    "win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));",
    "win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));",
    '// Deliberately NO win.close(): the process must exit by itself.',
    "console.log('loaded');",
  ].join('\n');
  const out = execFileSync(process.execPath, ['-e', script], {
    cwd: path.join(__dirname, '..'),
    timeout: 20000,
    encoding: 'utf8',
  });
  assert.ok(
    out.indexOf('loaded') !== -1,
    'the probe process ran main.js to completion',
  );
  console.log('  ok - no history rendered -> no timer keeps node alive');
}

async function main() {
  [false, true].forEach(remote => {
    testUnitLadder(remote);
    testActionsStillWork(remote);
  });
  testTimestampSources();
  testFutureClockSkew();
  testNoTimerWithoutLabels();
  await testPeriodicRefresh();
  console.log('All historyTaskLaunchedAgo tests passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
