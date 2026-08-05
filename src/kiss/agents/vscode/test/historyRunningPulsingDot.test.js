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

  const cssText = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const styleEl = win.document.createElement('style');
  styleEl.textContent = cssText;
  win.document.head.appendChild(styleEl);

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function openSidebar(win) {
  const sidebar = win.document.getElementById('sidebar');
  if (sidebar && !sidebar.classList.contains('open')) {
    sidebar.classList.add('open');
  }
}

function uncheckWorkspaceFilter(win) {
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

function rowsByTaskId(win) {
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

function dotOf(row) {
  return row.querySelector('.sidebar-item-running');
}

function testDotRendersForRunningRow() {
  const {win} = makeWebview();
  openSidebar(win);
  uncheckWorkspaceFilter(win);

  const SESSIONS = [
    makeRow({task_id: 1, title: 'finished task', is_running: false}),
    makeRow({task_id: 2, title: 'running task', is_running: true, endTs: 0}),
    makeRow({task_id: 3, title: 'failed task', is_running: false, failed: true}),
  ];
  send(win, {type: 'history', sessions: SESSIONS, offset: 0});

  const rows = rowsByTaskId(win);
  const finished = rows['finished task'];
  const running = rows['running task'];
  const failed = rows['failed task'];
  assert.ok(finished && running && failed, 'all rows must render');

  assert.strictEqual(
    dotOf(finished),
    null,
    'finished row must NOT carry .sidebar-item-running',
  );
  assert.strictEqual(
    dotOf(failed),
    null,
    'failed row must NOT carry .sidebar-item-running',
  );

  const dot = dotOf(running);
  assert.ok(dot, 'running row must carry .sidebar-item-running dot');

  assert.strictEqual(
    running.firstElementChild,
    dot,
    'running dot must be the first child of the row (left of text)',
  );

  const cs = win.getComputedStyle(dot);
  assert.strictEqual(
    cs.backgroundColor,
    'rgb(46, 125, 50)',
    `dot background must be #2e7d32 (rgb(46, 125, 50)); got: ${cs.backgroundColor}`,
  );
  const animName = cs.getPropertyValue('animation-name') || '';
  const animShort = cs.getPropertyValue('animation') || '';
  assert.ok(
    animName.indexOf('running-pulse') >= 0 ||
      animShort.indexOf('running-pulse') >= 0,
    `dot must animate via 'running-pulse'; got animation-name=` +
      `"${animName}" animation="${animShort}"`,
  );

  const cssText = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  assert.ok(
    /@keyframes\s+running-pulse\b/.test(cssText),
    'main.css must define @keyframes running-pulse',
  );

  assert.strictEqual(running.dataset.category, 'running');

  win.close();
  console.log('  ok - pulsing green dot renders on is_running rows');
}

function testDotAppearsLiveOnStatusRunningTrue() {
  const {win, posted} = makeWebview();
  openSidebar(win);
  uncheckWorkspaceFilter(win);

  const initial = [
    makeRow({task_id: 7, title: 'live task', is_running: false}),
  ];
  send(win, {type: 'history', sessions: initial, offset: 0});

  let rows = rowsByTaskId(win);
  assert.ok(rows['live task'], 'row must render initially');
  assert.strictEqual(
    dotOf(rows['live task']),
    null,
    'no dot before the task starts running',
  );

  posted.length = 0;
  send(win, {
    type: 'status',
    running: true,
    startTs: FROZEN_NOW_MS - 5_000,
    tabId: undefined,
  });
  const sent = posted.find(m => m && m.type === 'getHistory');
  assert.ok(
    sent,
    'status running:true must trigger a getHistory refetch ' +
      'so the History panel can repaint with is_running:true; ' +
      'posted=' + JSON.stringify(posted),
  );

  const generation = sent.generation;
  const live = [
    makeRow({task_id: 7, title: 'live task', is_running: true, endTs: 0}),
  ];
  send(win, {
    type: 'history',
    sessions: live,
    offset: 0,
    generation: generation,
  });

  rows = rowsByTaskId(win);
  const row = rows['live task'];
  assert.ok(row, 'row must still render after the live refresh');
  const dot = dotOf(row);
  assert.ok(
    dot,
    'pulsing green dot must appear on the row after status running:true',
  );
  assert.strictEqual(
    row.firstElementChild,
    dot,
    'live-added dot must still be the first child of the row',
  );

  win.close();
  console.log('  ok - dot appears live when status flips to running');
}

function testDotDisappearsLiveOnStatusRunningFalse() {
  const {win, posted} = makeWebview();
  openSidebar(win);
  uncheckWorkspaceFilter(win);

  const initial = [
    makeRow({task_id: 9, title: 'ending task', is_running: true, endTs: 0}),
  ];
  send(win, {type: 'history', sessions: initial, offset: 0});

  let rows = rowsByTaskId(win);
  assert.ok(dotOf(rows['ending task']), 'dot must be present initially');

  posted.length = 0;
  send(win, {type: 'status', running: false, tabId: undefined});
  const sent = posted.find(m => m && m.type === 'getHistory');
  assert.ok(
    sent,
    'status running:false must trigger a getHistory refetch ' +
      'so the History panel can drop the pulsing dot',
  );

  const generation = sent.generation;
  const finished = [
    makeRow({
      task_id: 9,
      title: 'ending task',
      is_running: false,
      endTs: 1_700_000_010_000,
    }),
  ];
  send(win, {
    type: 'history',
    sessions: finished,
    offset: 0,
    generation: generation,
  });

  rows = rowsByTaskId(win);
  assert.strictEqual(
    dotOf(rows['ending task']),
    null,
    'pulsing dot must be removed after status running:false reply',
  );

  win.close();
  console.log('  ok - dot disappears live when status flips to not-running');
}

function main() {
  testDotRendersForRunningRow();
  testDotAppearsLiveOnStatusRunningTrue();
  testDotDisappearsLiveOnStatusRunningFalse();
  console.log('historyRunningPulsingDot.test.js: all assertions passed.');
}

main();
