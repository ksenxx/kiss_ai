// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (jsdom) tests for the "N units ago" launch-age label that
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
// The page is booted the way ``web_server.py`` serves ``chat.html``
// (``<body class="remote-chat">`` plus ``remote-codex.css``).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {
  makePage: makeChatPage,
  captureTimers,
  loadMainCss,
  MEDIA,
} = require('./simplify2_harness');

const MINUTE_MS = 60 * 1000;
const HOUR_MS = 60 * MINUTE_MS;
const DAY_MS = 24 * HOUR_MS;

/**
 * Build a chat page under jsdom.  When *captureTimeouts* is set, every
 * `setTimeout` the page arms is also returned in `timeouts` as `{cb, ms}`
 * so a test can fire a long timer on demand instead of waiting for it.
 */
function makePage(captureTimeouts) {
  let timeouts;
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts(win) {
      loadMainCss(win);
      const remoteStyle = win.document.createElement('style');
      remoteStyle.textContent = fs.readFileSync(
        path.join(MEDIA, 'remote-codex.css'),
        'utf8',
      );
      win.document.head.appendChild(remoteStyle);
      if (captureTimeouts) timeouts = captureTimers(win);
    },
  });
  return {win, posted, timeouts};
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
      is_parallel: false,
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

const label = 'webapp';

// ---------------------------------------------------------------------------
// 1. Every unit of the ladder renders, next to the show-details button.
// ---------------------------------------------------------------------------

function testUnitLadder() {
  const {win} = makePage();
  const now = Date.now();
  // One session per rung. Identify each row by chat, independent of
  // the history renderer's running-first/chronological section order.
  const cases = [
    {ago: 30 * 1000, text: 'just now'},
    {ago: 5 * MINUTE_MS, text: '5 minutes ago'},
    {ago: 1 * MINUTE_MS, text: '1 minute ago'},
    {ago: 3 * HOUR_MS, text: '3 hours ago'},
    {ago: 1 * HOUR_MS, text: '1 hour ago'},
    {ago: 2 * DAY_MS, text: '2 days ago'},
    {ago: 1 * DAY_MS, text: '1 day ago'},
    {ago: 8 * DAY_MS, text: '1 week ago'},
    {ago: 22 * DAY_MS, text: '3 weeks ago'},
    {ago: 45 * DAY_MS, text: '1 month ago'},
    {ago: 200 * DAY_MS, text: '6 months ago'},
    {ago: 400 * DAY_MS, text: '1 year ago'},
    {ago: 800 * DAY_MS, text: '2 years ago'},
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
  cases.forEach((testCase, i) => {
    const row = win.document.querySelector(
      '[data-chat-id="chat-' + i + '"] .sidebar-item',
    );
    const ago = row.querySelector('.sidebar-item-launched');
    assert.ok(ago, `${label}: row ${i} renders the launched-ago label`);
    assert.strictEqual(
      ago.textContent,
      testCase.text,
      `${label}: row ${i} (${testCase.ago} ms old)`,
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
  const {win} = makePage();
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
  const rows = ['a', 'b', 'c', 'd', 'e', 'f'].map(id =>
    win.document.querySelector(
      '[data-chat-id="chat-' + id + '"] .sidebar-item',
    ),
  );
  assert.strictEqual(
    rows[0].querySelector('.sidebar-item-launched').textContent,
    '2 hours ago',
    'startTs (ms) takes precedence over the row timestamp',
  );
  assert.strictEqual(
    rows[1].querySelector('.sidebar-item-launched').textContent,
    '3 days ago',
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
    String(epochYears) + ' years ago',
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
  const {win} = makePage();
  loadHistory(win, [makeSession({startTs: Date.now() + 5 * MINUTE_MS})]);
  assert.strictEqual(
    historyRows(win)[0].querySelector('.sidebar-item-launched').textContent,
    'just now',
    'a launch instant in the future must clamp, not go negative',
  );
  win.close();
  console.log('  ok - future launch instants clamp to "just now"');
}

// ---------------------------------------------------------------------------
// 4. The 30 s sweep keeps a quiet panel's labels ticking.
// ---------------------------------------------------------------------------

function testPeriodicRefresh() {
  const {win, timeouts} = makePage(true);
  const now = Date.now();
  loadHistory(win, [
    makeSession({startTs: now - 45 * 1000}), // "just now" at render time
  ]);
  const ago = historyRows(win)[0].querySelector('.sidebar-item-launched');
  assert.strictEqual(ago.textContent, 'just now');
  const sweeps = timeouts.filter(t => t.ms === 30000);
  assert.strictEqual(
    sweeps.length,
    1,
    'rendering a label arms exactly one 30 s sweep',
  );
  // Age the label past the minute boundary by rewinding its stored
  // launch instant, then fire the armed sweep on demand instead of
  // waiting 30 s of wall-clock time for it.
  ago.dataset.launchTs = String(now - 120 * 1000);
  sweeps[0].cb();
  assert.strictEqual(
    ago.textContent,
    '2 minutes ago',
    'the periodic sweep must refresh on-screen labels in place',
  );
  assert.strictEqual(
    timeouts.filter(t => t.ms === 30000).length,
    2,
    'the sweep re-arms itself while a label is still on screen',
  );
  win.close();
  console.log('  ok - the 30s sweep keeps labels current');
}

// ---------------------------------------------------------------------------
// 5. The label must not break the action strip's behaviour: the toggle
//    still expands/collapses and clicks do not bubble into the row.
// ---------------------------------------------------------------------------

function testActionsStillWork() {
  const {win, posted} = makePage();
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
  // The page is built the way every suite builds it; deliberately NO
  // win.close(): the process must exit by itself.  The runner's V8 flags
  // (process.execArgv) are passed on, since they are what keeps a node
  // process from hanging at exit.
  const script =
    "require('./test/simplify2_harness').makePage(); console.log('loaded');";
  const out = execFileSync(
    process.execPath,
    [...process.execArgv, '-e', script],
    {
      cwd: path.join(__dirname, '..'),
      timeout: 20000,
      encoding: 'utf8',
    },
  );
  assert.ok(
    out.indexOf('loaded') !== -1,
    'the probe process ran main.js to completion',
  );
  console.log('  ok - no history rendered -> no timer keeps node alive');
}

testUnitLadder();
testActionsStillWork();
testTimestampSources();
testFutureClockSkew();
testNoTimerWithoutLabels();
testPeriodicRefresh();
console.log('All historyTaskLaunchedAgo tests passed');
