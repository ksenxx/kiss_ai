// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  rowsByTitle,
} = require('./simplify2_harness');
const {freezeClockAndStyle} = require('./history_helpers');

const FROZEN_NOW_MS = 1_700_500_000_000;

function freezeDate(win) {
  freezeClockAndStyle(win, FROZEN_NOW_MS);
}

function makePage() {
  return makeChatPage({beforeScripts: freezeDate});
}

const SESSIONS_FIXTURE = [
  {
    id: 'chatA',
    task_id: 1,
    title: 'ten second task',
    timestamp: 1_700_000_000,
    preview: 'ten second task',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 1234,
    cost: 0.1234,
    steps: 3,
    is_favorite: false,
    work_dir: '',
    startTs: 1_700_000_000_000,
    endTs: 1_700_000_010_000,
  },
  {
    id: 'chatB',
    task_id: 2,
    title: 'sixty five second task',
    timestamp: 1_700_000_100,
    preview: 'sixty five second task',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 100,
    cost: 0.01,
    steps: 1,
    is_favorite: false,
    work_dir: '',
    startTs: 1_700_000_100_000,
    endTs: 1_700_000_165_000,
  },
  {
    id: 'chatC',
    task_id: 3,
    title: 'long hour task',
    timestamp: 1_700_000_200,
    preview: 'long hour task',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 99999,
    cost: 1.2345,
    steps: 50,
    is_favorite: false,
    work_dir: '',
    startTs: 1_700_000_200_000,
    endTs: 1_700_000_200_000 + 3_725_000,
  },
  {
    id: 'chatD',
    task_id: 4,
    title: 'still running task',
    timestamp: Math.floor((FROZEN_NOW_MS - 65_000) / 1000),
    preview: 'still running task',
    has_events: false,
    failed: false,
    is_running: true,
    tokens: 42,
    cost: 0,
    steps: 1,
    is_favorite: false,
    work_dir: '',
    startTs: FROZEN_NOW_MS - 65_000,
    endTs: 0,
  },
  {
    id: 'chatE',
    task_id: 5,
    title: 'legacy row with no timestamps',
    timestamp: 0,
    preview: 'legacy row',
    has_events: false,
    failed: false,
    is_running: false,
    tokens: 0,
    cost: 0,
    steps: 0,
    is_favorite: false,
    work_dir: '',
    startTs: 0,
    endTs: 0,
  },
];

function metricsText(row) {
  const m = row.querySelector('.running-item-metrics');
  assert.ok(m, 'every row must have a .running-item-metrics span');
  return m.textContent;
}

function testDurationAppearsAfterCost() {
  const {win} = makePage();
  send(win, {
    type: 'configData',
    config: {work_dir: ''},
    apiKeys: {},
  });

  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});

  const rows = rowsByTitle(win);
  assert.ok(rows['ten second task'], 'row A must render');
  assert.ok(rows['sixty five second task'], 'row B must render');
  assert.ok(rows['long hour task'], 'row C must render');
  assert.ok(rows['still running task'], 'row D must render');
  assert.ok(rows['legacy row with no timestamps'], 'row E must render');

  const a = metricsText(rows['ten second task']);
  const b = metricsText(rows['sixty five second task']);
  const c = metricsText(rows['long hour task']);
  const d = metricsText(rows['still running task']);
  const e = metricsText(rows['legacy row with no timestamps']);

  assert.match(
    a,
    /\$0\.12 • 00:00:10\b/,
    `row A metrics must read "$0.12 • 00:00:10 …"; got: ${a}`,
  );
  assert.match(
    b,
    /\$0\.01 • 00:01:05\b/,
    `row B metrics must read "$0.01 • 00:01:05 …"; got: ${b}`,
  );
  assert.match(
    c,
    /\$1\.23 • 01:02:05\b/,
    `row C metrics must read "$1.23 • 01:02:05 …"; got: ${c}`,
  );
  assert.match(
    c.slice(c.indexOf('01:02:05') + '01:02:05'.length),
    / • /,
    `date suffix " • <localised>" must follow the duration; got: ${c}`,
  );
  assert.match(
    d,
    /\$0\.00 • 00:01:05\b/,
    `row D (running) metrics must read "$0.00 • 00:01:05 …" using ` +
      `Date.now()-startTs; got: ${d}`,
  );

  assert.ok(
    !/00:00:00/.test(e),
    `legacy row must omit duration when no usable timestamps; got: ${e}`,
  );

  win.close();
  console.log('  ok - duration shown in hh:mm:ss between cost and date');
}

testDurationAppearsAfterCost();
console.log('historyTaskDuration.test.js: all assertions passed.');
