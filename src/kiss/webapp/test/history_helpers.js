// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Helpers shared by the Chats-panel (history list) suites: reading the
// rows main.js painted, the page's getHistory requests, and the fixture
// rows the daemon's `history` reply carries.

'use strict';

const fs = require('fs');
const path = require('path');
const {MEDIA, send, rowsByTitle} = require('./simplify2_harness');
const {inlineDesignTokens} = require('./designTokens');

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

/** Answer the page's last getHistory (its generation) with `sessions`. */
function sendHistory(win, posted, sessions) {
  const req = lastGetHistory(posted);
  send(win, {
    type: 'history',
    offset: 0,
    generation: req ? req.generation : undefined,
    sessions,
  });
}

/** Deliver a first history page (generation 0) of `sessions`. */
function loadHistory(win, sessions) {
  send(win, {type: 'history', offset: 0, generation: 0, sessions});
}

/**
 * Freeze `win.Date` at `nowMs` (so "ago" labels and running timers are
 * deterministic) and apply main.css with its design tokens inlined, for
 * computed-style asserts.
 */
function freezeClockAndStyle(win, nowMs) {
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

  const styleEl = win.document.createElement('style');
  styleEl.textContent = inlineDesignTokens(
    fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8'),
  );
  win.document.head.appendChild(styleEl);
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

module.exports = {
  historyRows,
  visibleTitles,
  rowByTitle,
  lastGetHistory,
  sendHistory,
  loadHistory,
  freezeClockAndStyle,
  makeRow,
  makeSession,
  gatewaySession,
};
