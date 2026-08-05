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
  const {win, posted} = makeWebview();

  openBurgerMenu(win);
  uncheckWorkspaceFilter(win);
  const getHist = posted.find(m => m && m.type === 'getHistory');
  assert.ok(
    getHist,
    'opening the burger menu must post getHistory; got: ' +
      JSON.stringify(posted),
  );

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
  const {win, posted} = makeWebview();
  openBurgerMenu(win);
  uncheckWorkspaceFilter(win);
  const getHist = posted.find(m => m && m.type === 'getHistory');
  assert.ok(getHist, 'burger menu open must post getHistory');

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

  const runningRow = byTitle['running task'];
  assert.ok(
    runningRow.querySelector('.sidebar-item-running'),
    'running row must carry .sidebar-item-running',
  );

  const failedRow = byTitle['failed task'];
  assert.ok(
    failedRow.querySelector('.sidebar-item-failed'),
    'failed row must carry .sidebar-item-failed',
  );

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
    animName.indexOf('running-pulse') < 0 &&
      animShort.indexOf('running-pulse') < 0,
    'solid (completed) circle MUST NOT animate via ' +
      `running-pulse; got animation-name="${animName}" ` +
      `animation="${animShort}"`,
  );

  const stillFreshRow = byTitle['fresh completed task'];
  assert.strictEqual(
    stillFreshRow.querySelector('.sidebar-item-completed'),
    null,
    'an unrelated fresh-completed row MUST not inherit the solid ' +
      'green circle just because another row transitioned',
  );

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
  const {win, posted} = makeWebview();
  openBurgerMenu(win);
  uncheckWorkspaceFilter(win);
  const getHist = posted.find(m => m && m.type === 'getHistory');
  assert.ok(getHist, 'burger menu open must post getHistory');

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
    // The action buttons occupy a line of their own below the task
    // text, so the panel is taller than its title: the indicator is
    // centered on the first line of the text (the panel's padding-top
    // plus half a line box) instead of on the whole panel.
    assert.strictEqual(
      style.top,
      'calc(0.5lh + 7px)',
      `row ${title} indicator must sit on the first line of the task ` +
        `text, not at the panel middle; got top=${style.top}`,
    );
    assert.strictEqual(
      style.transform,
      'translateY(-50%)',
      `row ${title} indicator must translate by half its own height ` +
        `to center on that line; got transform=${style.transform}`,
    );
  });

  win.close();
  console.log(
    '  ok - history task-panel indicators are centered at middle-left',
  );
}

function testCompletedDotKeyframesNotShared() {
  const cssText = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  assert.ok(
    /\.sidebar-item-completed\s*\{/.test(cssText),
    'main.css must define .sidebar-item-completed for the solid ' +
      'green finished-task circle',
  );

  const m = cssText.match(/\.sidebar-item-completed\s*\{([^}]*)\}/);
  assert.ok(m, 'expected a single-rule .sidebar-item-completed block');
  const body = m[1];
  assert.ok(
    body.indexOf('running-pulse') < 0,
    '.sidebar-item-completed MUST NOT use running-pulse; ' +
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
