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

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
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

function rowsByTitle(win) {
  const list = win.document.getElementById('history-list');
  const rows = list.querySelectorAll('.sidebar-item');
  const map = {};
  rows.forEach(r => {
    const t = r.querySelector('.sidebar-item-text');
    if (!t) return;
    map[t.textContent] = r;
  });
  return map;
}

function metricsText(row) {
  const m = row.querySelector('.running-item-metrics');
  assert.ok(m, 'every row must have a .running-item-metrics span');
  return m.textContent;
}

function testDurationAppearsAfterCost() {
  const {win} = makeWebview();
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
    /\$0\.1234 • 00:00:10\b/,
    `row A metrics must read "$0.1234 • 00:00:10 …"; got: ${a}`,
  );
  assert.match(
    b,
    /\$0\.0100 • 00:01:05\b/,
    `row B metrics must read "$0.0100 • 00:01:05 …"; got: ${b}`,
  );
  assert.match(
    c,
    /\$1\.2345 • 01:02:05\b/,
    `row C metrics must read "$1.2345 • 01:02:05 …"; got: ${c}`,
  );
  assert.match(
    d,
    /\$0\.0000 • 00:01:05\b/,
    `row D (running) metrics must read "$0.0000 • 00:01:05 …" using ` +
      `Date.now()-startTs; got: ${d}`,
  );

  assert.ok(
    !/00:00:00/.test(e),
    `legacy row must omit duration when no usable timestamps; got: ${e}`,
  );

  win.close();
  console.log('  ok - duration shown in hh:mm:ss after cost');
}

function testDurationBeforeDateSuffix() {
  const {win} = makeWebview();
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

  send(win, {type: 'history', sessions: SESSIONS_FIXTURE, offset: 0});

  const rows = rowsByTitle(win);
  const c = metricsText(rows['long hour task']);

  const costIdx = c.indexOf('$1.2345');
  const durIdx = c.indexOf('01:02:05');
  assert.ok(costIdx >= 0, `cost must appear; got: ${c}`);
  assert.ok(durIdx > costIdx, `duration must follow cost; got: ${c}`);

  const tail = c.slice(durIdx + '01:02:05'.length);
  assert.ok(
    / • /.test(tail),
    `date suffix " • <localised>" must follow duration; got tail: ${tail}`,
  );

  win.close();
  console.log('  ok - duration sits between cost and date suffix');
}

function main() {
  testDurationAppearsAfterCost();
  testDurationBeforeDateSuffix();
  console.log('historyTaskDuration.test.js: all assertions passed.');
}

main();
