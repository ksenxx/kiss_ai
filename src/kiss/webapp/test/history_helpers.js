// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Helpers shared by the Chats-panel (history list) suites: reading the
// rows main.js painted, the page's getHistory requests, and the fixture
// rows the daemon's `history` reply carries.

'use strict';

const assert = require('assert');
const {send, rowsByTitle} = require('./simplify2_harness');
const {installTokenisedCss} = require('./designTokens');

/** Every row of the history list, in list order. */
function historyRows(win) {
  return Array.from(
    win.document
      .getElementById('history-list')
      .querySelectorAll('.sidebar-item'),
  );
}

/** The titles of the history rows on screen, sorted. */
function visibleTitles(win) {
  return historyRows(win)
    .filter(r => r.style.display !== 'none')
    .map(r => r.querySelector('.sidebar-item-text').textContent)
    .sort();
}

/** The history row titled `title`, or null. */
function rowByTitle(win, title) {
  return rowsByTitle(win)[title] || null;
}

/** The last getHistory the page posted, or undefined. */
function lastGetHistory(posted) {
  return posted.filter(m => m && m.type === 'getHistory').pop();
}

/**
 * Answer the page's last getHistory (its generation) with `sessions`;
 * `extra` adds further reply fields (`workDirs`, `machines`, ...).
 */
function sendHistory(win, posted, sessions, extra) {
  const req = lastGetHistory(posted);
  send(
    win,
    Object.assign(
      {
        type: 'history',
        offset: 0,
        generation: req ? req.generation : undefined,
        sessions,
      },
      extra || {},
    ),
  );
}

/** Deliver a first history page (generation 0) of `sessions`. */
function loadHistory(win, sessions) {
  sendHistory(win, [], sessions);
}

/**
 * Freeze `win.Date` at `nowMs`, so "ago" / "in 5m" labels and running
 * timers are deterministic.  `new Date(args)` and the static helpers
 * still work; `Date.now` can still be reassigned by a test.
 */
function freezeClock(win, nowMs) {
  const RealDate = win.Date;
  const FakeDate = function (...args) {
    if (args.length === 0) return new RealDate(nowMs);
    return new RealDate(...args);
  };
  FakeDate.prototype = RealDate.prototype;
  FakeDate.now = () => nowMs;
  FakeDate.parse = RealDate.parse;
  FakeDate.UTC = RealDate.UTC;
  win.Date = FakeDate;
}

/**
 * freezeClock plus main.css applied with its design tokens inlined, for
 * computed-style asserts.
 */
function freezeClockAndStyle(win, nowMs) {
  freezeClock(win, nowMs);
  installTokenisedCss(win, 'main.css');
}

/**
 * A history row carrying every field main.js reads, keyed on
 * `overrides.task_id` / `overrides.title`, with `overrides` on top.
 */
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

/** A finished "refactor the parser" row with metrics, `overrides` on top. */
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

/** A finished "payment gateway retries" row with every flag off. */
function gatewaySession(overrides) {
  return Object.assign(
    {
      id: 'chat-1',
      task_id: 1,
      title: 'Refactor the payment gateway retries',
      preview: 'Refactor the payment gateway retries',
      has_events: true,
      failed: false,
      is_running: false,
      tokens: 0,
      cost: 0,
      steps: 0,
      is_favorite: false,
      timestamp: 1_700_000_000,
      work_dir: '',
    },
    overrides || {},
  );
}

/** Pick `value` in the history filter <select id=`id`> (hf-tag, hf-workdir, ...). */
function selectFilter(win, id, value) {
  const sel = win.document.getElementById(id);
  sel.value = value;
  sel.dispatchEvent(new win.Event('change', {bubbles: true}));
}

/** Open the Chats panel through the burger button, as the user does. */
function openChatsPanel(win) {
  const btn = win.document.getElementById('menu-btn');
  assert.ok(btn, 'burger menu button (#menu-btn) must exist');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
  assert.ok(
    win.document.getElementById('sidebar').classList.contains('open'),
    'burger click must open the Chats panel',
  );
}

/**
 * Answer the page's last getHistory (its generation) with `sessions`,
 * failing when nothing was requested: main.js drops a reply whose
 * generation is not the one it asked for, so answering an absent request
 * would render nothing and fail later, far from the cause.
 */
function answerHistory(win, posted, sessions) {
  assert.ok(lastGetHistory(posted), 'the page must have requested history');
  sendHistory(win, posted, sessions);
}

/** The local calendar day of the unix time `sec` as `YYYY-MM-DD`, the date-input form. */
function localIsoDate(sec) {
  const d = new Date(sec * 1000);
  const p = n => String(n).padStart(2, '0');
  return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate());
}

// ---- The grouped Chats panel (historyGroups, historyLegacyViewToggle) ----

/**
 * Noon today as unix seconds, so "yesterday" and "three days ago" stay
 * on their days whatever the local time zone offset does to the clock.
 */
function todayNoonSec() {
  const d = new Date();
  d.setHours(12, 0, 0, 0);
  return Math.floor(d.getTime() / 1000);
}

/** A row of chat `chat` for task `task` at unix time `ts`; `extra` on top. */
function groupedSession(chat, task, ts, extra) {
  return Object.assign(
    {
      id: chat,
      task_id: task,
      title: 'task ' + task,
      preview: 'task ' + task,
      has_events: true,
      timestamp: ts,
      tokens: 1,
      cost: 0.1,
      steps: 1,
    },
    extra || {},
  );
}

/**
 * Deliver a history page at `offset` stamped with the generation the
 * panel last asked for (0 when it never asked).
 */
function sendHistoryPage(win, posted, offset, sessions) {
  sendHistory(win, posted, sessions, {offset});
}

/**
 * The visible top-level children of #history-list, described:
 * `sep:<label>` for a day separator, `chat:<id>[<visible row titles>]`
 * for a chat panel, `row:<title>` for a flat-list row, else the class.
 */
function listShape(win) {
  const rowTitle = r => r.querySelector('.sidebar-item-text').textContent;
  return Array.from(win.document.getElementById('history-list').children)
    .filter(el => el.style.display !== 'none')
    .map(el => {
      if (el.classList.contains('history-day-sep')) {
        return 'sep:' + el.textContent;
      }
      if (el.classList.contains('history-chat-group')) {
        const titles = Array.from(el.querySelectorAll('.sidebar-item'))
          .filter(r => r.style.display !== 'none')
          .map(rowTitle);
        return 'chat:' + el.dataset.chatId + '[' + titles.join(',') + ']';
      }
      if (el.classList.contains('sidebar-item')) return 'row:' + rowTitle(el);
      return el.className;
    });
}

/**
 * A finished row `chat-<id>` / `task-<id>` titled `title` whose task ran
 * at 1_700_000_000 + id * 100 s (so rows order by id); `extra` on top.
 */
function finishedSession(id, title, extra) {
  const ts = 1_700_000_000 + id * 100;
  return makeRow(
    Object.assign(
      {
        id: 'chat-' + id,
        task_id: 'task-' + id,
        title,
        timestamp: ts,
        startTs: ts * 1000,
        endTs: ts * 1000 + 10_000,
      },
      extra || {},
    ),
  );
}

module.exports = {
  historyRows,
  visibleTitles,
  rowByTitle,
  lastGetHistory,
  sendHistory,
  loadHistory,
  freezeClock,
  freezeClockAndStyle,
  makeRow,
  makeSession,
  gatewaySession,
  selectFilter,
  openChatsPanel,
  answerHistory,
  localIsoDate,
  todayNoonSec,
  groupedSession,
  sendHistoryPage,
  listShape,
  finishedSession,
};
