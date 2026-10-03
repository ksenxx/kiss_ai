// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the right sidebar's Schedule, Apps and Spend
// subpanels (the sidebarpanels block of media/main.js), on every
// surface that shows the panel (remote webapp, sidebar-chat drawer,
// editor-tabs Task Info view) and the ones that must stay quiet (chat
// editor panels, the history panel):
//
// * boot and a daemon (re)connect request getCronJobs + getAppsStatus
//   + getSpendReport; the poll timer starts on the first reply, once,
//   and skips hidden pages;
// * spendReport replies render the all-time line, the daily heatmap
//   (Monday-first week columns back to the oldest day, at least 98
//   days, more when the panel is wide; shaded by cost), the
//   cost-by-model bars, the hover tooltips and the pagers;
// * cronJobs replies render rows (running / paused badges, next / last
//   run, tooltip with the prompt or command and the last outcome) or
//   the empty-state hint;
// * appsStatus replies render connected apps first, the "N of M
//   connected" line or the failure hint, and a button per app that is
//   not connected;
// * the refresh buttons re-request (apps with refresh: true, spinning
//   until the reply);
// * clicking an app that is not connected launches the connect task:
//   a new internal tab that submits the prompt (drawer closed), or, in
//   the Task Info view, an openChatPanel{autoSubmit} to the host; while
//   the app waits to be connected (at most 30 minutes) polls re-probe.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

let failures = 0;

async function test(name, fn) {
  try {
    await fn();
    console.log(`  ok - ${name}`);
  } catch (e) {
    failures++;
    console.log(`  FAIL - ${name}`);
    console.log(`      ${e.stack || e.message}`);
  }
}

/**
 * Boot a chat webview with the given body attributes.  setInterval is
 * recorded (not run) so a test can fire the poll by hand.
 */
function makeWebview(bodyAttrs) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  html = html.replace('<body', '<body' + bodyAttrs);
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  win.cancelAnimationFrame = function () {};
  const intervals = [];
  win.setInterval = function (fn, ms) {
    intervals.push({fn, ms});
    return intervals.length;
  };
  win.clearInterval = function () {};
  // jsdom has no ResizeObserver: record the observers so a test can
  // fire a resize by hand.
  const observers = [];
  win.ResizeObserver = class {
    constructor(fn) {
      this.fn = fn;
      observers.push(this);
    }
    observe(target) {
      this.target = target;
    }
  };
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
  win.matchMedia = function (query) {
    return {
      matches: query === '(min-width: 900px)',
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=sidebarpanels-main.js',
  );
  return {win, posted, intervals, observers};
}

const REMOTE = ' class="remote-chat"';
const SIDEBAR = '';
const META_VIEW =
  ' class="editor-tab-mode meta-panel-mode" data-kiss-tab-id="meta-panel"';
const EDITOR_PANEL = ' class="editor-tab-mode" data-kiss-tab-id="panel-1"';
const HISTORY =
  ' class="history-panel-mode" data-kiss-tab-id="history-panel"';

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

/** deepStrictEqual across realms: jsdom objects have foreign prototypes. */
function eq(actual, expected, message) {
  assert.deepStrictEqual(JSON.parse(JSON.stringify(actual)), expected, message);
}

function el(win, id) {
  return win.document.getElementById(id);
}

function types(posted) {
  return posted.map(m => m.type);
}

/** Epoch ms *offsetMinutes* from now, like the daemon's rows. */
function iso(offsetMinutes) {
  return Date.now() + offsetMinutes * 60000;
}

// The Schedule subpanel's Pacific-time rendering of a run time.
function pt(t) {
  return new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/Los_Angeles',
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    timeZoneName: 'short',
  }).format(new Date(t));
}

function job(fields) {
  return Object.assign(
    {
      id: 'j',
      name: 'Job',
      schedule: 'every 5m',
      kind: 'prompt',
      what: 'do it',
      enabled: true,
      running: false,
      nextRunAt: 0,
      lastRunAt: 0,
      lastStatus: '',
      workDir: '',
    },
    fields,
  );
}

const APPS = [
  {name: 'slack', label: 'Slack', authenticated: false, error: ''},
  {name: 'gmail', label: 'Gmail', authenticated: true, error: ''},
  {name: 'matrix', label: 'Matrix', authenticated: null, error: 'TimeoutError'},
  {name: 'brave', label: 'Brave Search', authenticated: true, error: ''},
];

function appRows(win) {
  return Array.from(win.document.querySelectorAll('#meta-apps-list .app-row'));
}

const PANEL_COMMANDS = ['getCronJobs', 'getAppsStatus', 'getSpendReport'];

/** "YYYY-MM-DD" of the local day *daysAgo* days before today. */
function dayKey(daysAgo) {
  const now = new Date();
  const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() - daysAgo);
  return (
    d.getFullYear() +
    '-' +
    String(d.getMonth() + 1).padStart(2, '0') +
    '-' +
    String(d.getDate()).padStart(2, '0')
  );
}

/** "Sep 27" for a "YYYY-MM-DD" key, as the tooltip shows it. */
function dayLabel(key) {
  const [y, m, d] = key.split('-').map(Number);
  return new Date(y, m - 1, d).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
  });
}

function sum(fields) {
  return Object.assign({cost: 0, tokens: 0, tasks: 0}, fields);
}

/** A spendReport reply: today, yesterday and a day 200 days back. */
function spendReport() {
  const [today, yesterday, old] = [dayKey(0), dayKey(1), dayKey(200)];
  return {
    type: 'spendReport',
    total: sum({cost: 2037.92, tokens: 1234567, tasks: 6}),
    days: [
      sum({date: old, cost: 0, tokens: 12, tasks: 1}),
      sum({date: yesterday, cost: 2000, tokens: 1000000, tasks: 4}),
      sum({date: today, cost: 37.92, tokens: 234555, tasks: 1}),
    ],
    totalByModel: [
      sum({model: 'claude-fable-5', cost: 2000, tokens: 1000000, tasks: 4}),
      sum({model: 'gpt-6-astra', cost: 37.92, tokens: 234555, tasks: 1}),
      sum({model: 'unknown', cost: 0, tokens: 12, tasks: 1}),
    ],
    daysByModel: {
      [old]: [sum({model: 'unknown', cost: 0, tokens: 12, tasks: 1})],
      [yesterday]: [
        sum({model: 'claude-fable-5', cost: 1500, tokens: 900000, tasks: 3}),
        sum({model: 'gpt-6-astra', cost: 500, tokens: 100000, tasks: 1}),
      ],
    },
  };
}

function cells(win) {
  return Array.from(win.document.querySelectorAll('#meta-spend-graph .spend-cell'));
}

function cellOf(win, key) {
  return win.document.querySelector(`.spend-cell[data-spend-date="${key}"]`);
}

function hover(win, target) {
  target.dispatchEvent(new win.MouseEvent('mouseover', {bubbles: true}));
}

function tipText(win) {
  const tip = win.document.querySelector('#meta-spend-graph .spend-tip');
  return tip.hidden ? null : tip.innerHTML.split('<br>');
}

async function main() {
  for (const [label, attrs] of [
    ['remote webapp', REMOTE],
    ['sidebar-chat drawer', SIDEBAR],
    ['Task Info view', META_VIEW],
  ]) {
    await test(`${label}: boot requests the three subpanels; the poll starts on the first reply`, () => {
      const {win, posted, intervals} = makeWebview(attrs);
      const boot = posted.filter(m => PANEL_COMMANDS.includes(m.type));
      eq(boot, [
        {type: 'getCronJobs'},
        {type: 'getAppsStatus'},
        {type: 'getSpendReport'},
      ]);
      assert.strictEqual(el(win, 'meta-apps-status').textContent, 'Checking\u2026');
      assert.strictEqual(el(win, 'meta-spend-status').textContent, 'Loading\u2026');
      const polls = () => intervals.filter(i => i.ms === 30000);
      assert.strictEqual(polls().length, 0, 'no poll before the daemon answers');
      send(win, {type: 'cronJobs', jobs: []});
      send(win, {type: 'cronJobs', jobs: []});
      assert.strictEqual(polls().length, 1, 'one poll timer, started once');
      posted.length = 0;
      polls()[0].fn();
      eq(types(posted), PANEL_COMMANDS);
      Object.defineProperty(win.document, 'hidden', {
        value: true,
        configurable: true,
      });
      posted.length = 0;
      polls()[0].fn();
      eq(posted, [], 'a hidden page does not poll');
    });
  }

  await test('the empty Schedule shows a hint; rows show badges, runs and tooltips', () => {
    const {win} = makeWebview(REMOTE);
    send(win, {type: 'cronJobs', jobs: []});
    assert.match(el(win, 'meta-schedule-status').textContent, /^No scheduled jobs/);
    assert.strictEqual(el(win, 'meta-schedule-list').children.length, 0);
    send(win, {type: 'cronJobs', jobs: 'bogus'});
    assert.match(el(win, 'meta-schedule-status').textContent, /^No scheduled jobs/);

    const times = {
      a: iso(5.2),
      b: iso(3 * 60 + 1),
      c: iso(3 * 1440 + 1),
      d: iso(0.2),
      f: iso(-125),
    };
    send(win, {
      type: 'cronJobs',
      jobs: [
        job({id: 'a', name: 'Digest', nextRunAt: times.a, lastStatus: 'ok'}),
        job({
          id: 'b',
          name: 'Backup',
          kind: 'command',
          what: 'rsync -a src dst',
          running: true,
          nextRunAt: times.b,
        }),
        job({id: 'c', name: 'Weekly', nextRunAt: times.c}),
        job({id: 'd', name: 'Soon', nextRunAt: times.d}),
        job({id: 'e', name: 'No next run'}),
        job({id: 'f', name: 'Old', enabled: false, lastRunAt: times.f}),
        job({id: 'g', name: 'Never ran', enabled: false}),
      ],
    });
    assert.strictEqual(el(win, 'meta-schedule-status').textContent, '');
    const rows = Array.from(el(win, 'meta-schedule-list').children);
    const sub = r => r.querySelector('.sidebar-panel-sub').textContent;
    const badge = r => {
      const b = r.querySelector('.sidebar-panel-badge');
      return b ? b.textContent : '';
    };
    assert.strictEqual(rows.length, 7);
    assert.strictEqual(rows[0].querySelector('.sidebar-panel-name').textContent, 'Digest');
    assert.strictEqual(sub(rows[0]), `every 5m \u00b7 next ${pt(times.a)} (in 5m)`);
    assert.strictEqual(rows[0].dataset.tooltip, 'do it\n\nLast run: ok');
    assert.strictEqual(rows[0].title, '', 'no native title: the custom tooltip shows the task');
    assert.strictEqual(badge(rows[0]), '');
    assert.strictEqual(badge(rows[1]), 'running');
    assert.strictEqual(sub(rows[1]), `every 5m \u00b7 next ${pt(times.b)} (in 3h)`);
    assert.strictEqual(rows[1].dataset.tooltip, '$ rsync -a src dst');
    assert.strictEqual(sub(rows[2]), `every 5m \u00b7 next ${pt(times.c)} (in 3d)`);
    assert.strictEqual(sub(rows[3]), `every 5m \u00b7 next ${pt(times.d)} (now)`);
    assert.strictEqual(sub(rows[4]), 'every 5m');
    assert.ok(rows[5].classList.contains('paused'));
    assert.strictEqual(badge(rows[5]), 'paused');
    assert.strictEqual(sub(rows[5]), `every 5m \u00b7 last ${pt(times.f)} (2h ago)`);
    assert.strictEqual(sub(rows[6]), 'every 5m');
  });

  await test('Schedule run times are Pacific time whatever the viewer zone', () => {
    const {win} = makeWebview(REMOTE);
    // 2026-09-27 12:00 UTC is 5:00 AM PDT; 2026-12-01 17:00 UTC is 9:00 AM PST.
    send(win, {
      type: 'cronJobs',
      jobs: [
        job({id: 's', name: 'Summer', nextRunAt: Date.UTC(2026, 8, 27, 12, 0)}),
        job({id: 'w', name: 'Winter', nextRunAt: Date.UTC(2026, 11, 1, 17, 0)}),
      ],
    });
    const subs = Array.from(el(win, 'meta-schedule-list').children).map(
      r => r.querySelector('.sidebar-panel-sub').textContent,
    );
    assert.match(subs[0], /^every 5m \u00b7 next Sun, Sep 27, 5:00\s?AM PDT \(/);
    assert.match(subs[1], /^every 5m \u00b7 next Tue, Dec 1, 9:00\s?AM PST \(/);
  });

  for (const [label, attrs] of [
    ['remote webapp', REMOTE],
    ['sidebar chat', SIDEBAR],
    ['Task Info view', META_VIEW],
  ]) {
    await test(`${label}: each Schedule row copies its exact task and shows it in the tooltip`, async () => {
      const {win} = makeWebview(attrs);
      const writes = [];
      Object.defineProperty(win.navigator, 'clipboard', {
        configurable: true,
        value: {
          writeText(text) {
            writes.push(String(text));
            return Promise.resolve();
          },
        },
      });
      // A prompt with newlines, quotes and a shell-looking line: the
      // clipboard and the tooltip get it verbatim, never a summary.
      const prompt =
        'Call the run_agent tool IMMEDIATELY with:\n' +
        "  agent = 'src/kiss/agents/seas/rsi7d_sea.py'\n" +
        '  task  = "all. Work inside this checkout."\n\n' +
        'Use \'claude-fable-5-1\' for all tasks.';
      send(win, {
        type: 'cronJobs',
        jobs: [
          job({id: 'p', name: 'Weekly rsi7d', what: prompt, lastStatus: 'ok in 12m'}),
          job({id: 'c', name: 'Sync', kind: 'command', what: 'rsync -a src dst'}),
          job({id: 'n', name: 'No text', what: null}),
        ],
      });
      const rows = Array.from(el(win, 'meta-schedule-list').children);
      const copies = rows.map(r => r.querySelector('.sidebar-panel-row-top > .sidebar-item-copy'));
      assert.ok(copies.every(Boolean), 'every row has a copy button on its name line');
      assert.strictEqual(
        copies[0].getAttribute('aria-label'),
        'Copy the scheduled task to clipboard',
      );
      assert.strictEqual(copies[0].dataset.tooltip, 'Copy the scheduled task');
      assert.strictEqual(
        copies[1].getAttribute('aria-label'),
        'Copy the scheduled command to clipboard',
      );
      assert.strictEqual(copies[1].dataset.tooltip, 'Copy the scheduled command');

      copies[0].click();
      copies[1].click();
      copies[2].click();
      await new Promise(r => setImmediate(r));
      eq(writes, [prompt, 'rsync -a src dst', ''], 'the exact prompt or command, unchanged');
      assert.ok(copies[0].classList.contains('copied'), 'the button flashes a check mark');

      // The row's tooltip is the same text (commands prefixed "$ "),
      // then the last run's outcome; it opens on keyboard focus too.
      assert.strictEqual(rows[0].dataset.tooltip, prompt + '\n\nLast run: ok in 12m');
      assert.strictEqual(rows[1].dataset.tooltip, '$ rsync -a src dst');
      assert.strictEqual(rows[2].dataset.tooltip, '');
      assert.strictEqual(rows[0].tabIndex, 0, 'rows take keyboard focus');
      const tip = win.document.getElementById('custom-tooltip');
      rows[0].dispatchEvent(new win.FocusEvent('focusin', {bubbles: true}));
      assert.ok(tip.classList.contains('visible'));
      assert.strictEqual(tip.textContent, prompt + '\n\nLast run: ok in 12m');
      rows[0].dispatchEvent(new win.FocusEvent('focusout', {bubbles: true}));
      assert.ok(!tip.classList.contains('visible'));
      // Hovering the copy button shows its own tooltip, not the task.
      copies[1].dispatchEvent(new win.FocusEvent('focusin', {bubbles: true}));
      assert.strictEqual(tip.textContent, 'Copy the scheduled command');
    });
  }

  await test('a tooltip that would run off the bottom or right edge stays on screen', () => {
    const {win} = makeWebview(REMOTE);
    send(win, {type: 'cronJobs', jobs: [job({id: 'p', name: 'Tall', what: 'x'.repeat(2000)})]});
    const row = el(win, 'meta-schedule-list').firstElementChild;
    const tip = win.document.getElementById('custom-tooltip');
    // jsdom has no layout: give the row and the tooltip sizes by hand.
    // The tooltip is measured at the origin (a fixed box near the right
    // edge would shrink to the room left of it), so the stub reports
    // its natural size and records where the measurement happened.
    Object.defineProperty(win, 'innerHeight', {configurable: true, value: 600});
    Object.defineProperty(win, 'innerWidth', {configurable: true, value: 800});
    let measuredAt = null;
    let size = {width: 350, height: 300};
    tip.getBoundingClientRect = () => {
      measuredAt = tip.style.left + ' ' + tip.style.top;
      return {left: 0, top: 0, right: size.width, bottom: size.height, ...size};
    };
    // The row sits near the bottom right; the tooltip is tall and wide.
    row.getBoundingClientRect = () => ({left: 600, top: 500, bottom: 540, right: 800, width: 200, height: 40});
    row.dispatchEvent(new win.FocusEvent('focusin', {bubbles: true}));
    assert.strictEqual(measuredAt, '0px 0px', 'measured at the origin');
    assert.ok(tip.classList.contains('visible'));
    assert.strictEqual(tip.style.top, 500 - 4 - 300 + 'px', 'flipped above the row');
    assert.strictEqual(tip.style.left, 800 - 350 - 8 + 'px', 'slid left of the right edge');
    row.dispatchEvent(new win.FocusEvent('focusout', {bubbles: true}));

    // Room below and to the right: left-aligned with the row, under it.
    row.getBoundingClientRect = () => ({left: 20, top: 100, bottom: 140, right: 220, width: 200, height: 40});
    row.dispatchEvent(new win.FocusEvent('focusin', {bubbles: true}));
    assert.strictEqual(tip.style.top, '144px');
    assert.strictEqual(tip.style.left, '20px');
    row.dispatchEvent(new win.FocusEvent('focusout', {bubbles: true}));

    // Too tall for either side: it slides up to end at the bottom edge
    // (the CSS caps its height at the viewport, so this never goes
    // above the top edge).
    size = {width: 350, height: 500};
    row.dispatchEvent(new win.FocusEvent('focusin', {bubbles: true}));
    assert.strictEqual(tip.style.top, 600 - 500 - 8 + 'px');
    assert.strictEqual(tip.style.left, '20px');
    row.dispatchEvent(new win.FocusEvent('focusout', {bubbles: true}));
    size = {width: 350, height: 600};
    row.dispatchEvent(new win.FocusEvent('focusin', {bubbles: true}));
    assert.strictEqual(tip.style.top, '0px', 'never above the top edge');

    const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
    const rule = css.match(/#custom-tooltip \{[^}]*\}/)[0];
    assert.match(rule, /max-height: calc\(100vh - 16px\)/, 'capped at the viewport height');
    assert.match(rule, /overflow: hidden/);
  });

  await test('Apps: connected first, status line, buttons for the rest, failure hint', () => {
    const {win} = makeWebview(REMOTE);
    send(win, {type: 'appsStatus', apps: APPS, checkedAt: 1790000000000});
    assert.strictEqual(el(win, 'meta-apps-status').textContent, '2 of 4 connected');
    const rows = appRows(win);
    eq(
      rows.map(r => r.dataset.app),
      ['brave', 'gmail', 'matrix', 'slack'],
    );
    const main0 = rows[0].querySelector('.app-row-main');
    assert.strictEqual(main0.tagName, 'DIV');
    assert.strictEqual(main0.querySelector('.app-state').textContent, 'Connected');
    assert.ok(main0.querySelector('.app-dot.connected'));
    const matrix = rows[2].querySelector('.app-row-main');
    assert.strictEqual(matrix.tagName, 'BUTTON');
    assert.ok(matrix.querySelector('.app-dot.unknown'));
    assert.strictEqual(
      matrix.title,
      'Status unknown (TimeoutError). Click to connect Matrix in a new task',
    );
    const slack = rows[3].querySelector('.app-row-main');
    assert.ok(slack.querySelector('.app-dot.disconnected'));
    assert.strictEqual(slack.querySelector('.app-state').textContent, 'Connect');
    assert.strictEqual(slack.title, 'Click to connect Slack in a new task');

    send(win, {type: 'appsStatus', apps: 'bogus', checkedAt: 0});
    assert.strictEqual(appRows(win).length, 0);
    assert.match(el(win, 'meta-apps-status').textContent, /^Could not check the apps/);
  });

  await test('refresh buttons re-request; the apps one spins until the reply', () => {
    const {win, posted} = makeWebview(REMOTE);
    posted.length = 0;
    el(win, 'meta-schedule-refresh').click();
    eq(posted, [{type: 'getCronJobs'}]);
    posted.length = 0;
    const btn = el(win, 'meta-apps-refresh');
    btn.click();
    eq(posted, [
      {type: 'getCronJobs'},
      {type: 'getAppsStatus', refresh: true},
      {type: 'getSpendReport'},
    ]);
    assert.ok(btn.disabled && btn.classList.contains('spinning'));
    posted.length = 0;
    el(win, 'meta-spend-refresh').click();
    eq(posted, [{type: 'getSpendReport'}]);
    send(win, {type: 'appsStatus', apps: APPS, checkedAt: 1});
    assert.ok(!btn.disabled && !btn.classList.contains('spinning'));
    // A dropped connection never answers: the button recovers.
    btn.click();
    send(win, {type: 'daemonStatus', connected: false});
    assert.ok(!btn.disabled && !btn.classList.contains('spinning'));
  });

  await test('a daemon (re)connect requests the three subpanels again', () => {
    const {win, posted} = makeWebview(REMOTE);
    posted.length = 0;
    send(win, {type: 'daemonStatus', connected: true});
    for (const cmd of PANEL_COMMANDS) assert.ok(types(posted).includes(cmd), cmd);
  });

  await test('Spend: an empty history draws the all-time line and a blank 98-day grid', () => {
    const {win, intervals} = makeWebview(REMOTE);
    send(win, {
      type: 'spendReport',
      total: 'bogus',
      days: 'bogus',
      totalByModel: 'bogus',
    });
    assert.strictEqual(intervals.filter(i => i.ms === 30000).length, 1, 'poll started');
    assert.strictEqual(el(win, 'meta-spend-status').textContent, 'No spend recorded yet.');
    assert.strictEqual(
      win.document.querySelector('.spend-total').textContent,
      'All time \u00b7 $0.00 \u00b7 0 tok \u00b7 0 tasks',
    );
    const dated = cells(win).filter(c => c.dataset.spendDate);
    assert.strictEqual(dated.length, 98);
    assert.strictEqual(dated[97].dataset.spendDate, dayKey(0));
    assert.strictEqual(dated[0].dataset.spendDate, dayKey(97));
    assert.ok(!dated.some(c => /\bl[1-4]\b/.test(c.className)), 'no shaded cell');
    assert.strictEqual(win.document.querySelector('.spend-models'), null, 'no model bars');
    // Every column is a Monday-first week: the first one is padded
    // with blanks up to the first day's weekday.
    const cols = Array.from(win.document.querySelectorAll('.spend-col'));
    const [y, m, d] = dayKey(97).split('-').map(Number);
    const offset = (new Date(y, m - 1, d).getDay() + 6) % 7;
    assert.strictEqual(cols[0].querySelectorAll('.spend-cell.blank').length, offset);
    for (const col of cols.slice(0, -1))
      assert.strictEqual(col.children.length, 7);
    assert.strictEqual(cols.length, Math.ceil((offset + 98) / 7));
    // Hovering a blank cell or the grid itself shows nothing.
    hover(win, win.document.querySelector('.spend-grid'));
    assert.strictEqual(tipText(win), null);
    hover(win, dated[0]);
    eq(tipText(win), [dayLabel(dayKey(97)) + ' \u00b7 no usage']);
    if (offset) {
      hover(win, cols[0].firstElementChild);
      assert.strictEqual(tipText(win), null);
    }
  });

  await test('Spend: the tooltip sits above the cell, below it near the top, and never over it when it can help', () => {
    const {win} = makeWebview(REMOTE);
    send(win, spendReport());
    const graph = el(win, 'meta-spend-graph');
    const tip = graph.querySelector('.spend-tip');
    const cell = cellOf(win, dayKey(0));
    // jsdom has no layout: the graph is 200 x 180, the tooltip 120 x 90.
    graph.getBoundingClientRect = () => ({left: 0, top: 0, right: 200, bottom: 180, width: 200, height: 180});
    Object.defineProperty(tip, 'offsetWidth', {configurable: true, value: 120});
    Object.defineProperty(tip, 'offsetHeight', {configurable: true, value: 90});
    const cellAt = top => {
      cell.getBoundingClientRect = () => ({left: 100, top, right: 111, bottom: top + 11, width: 11, height: 11});
      hover(win, cell);
      return tip.style.top;
    };
    // Room above: 6px over the cell, centered on it.
    assert.strictEqual(cellAt(120), 120 - 90 - 6 + 'px');
    assert.strictEqual(tip.style.left, 100 + 11 / 2 - 60 + 'px');
    // A cell in the top rows: below it instead.
    assert.strictEqual(cellAt(10), 10 + 11 + 6 + 'px');
    // Neither side has room inside the graph: above (clamped to the
    // graph's top) ends 2px short of the cell, below (clamped to the
    // bottom) would cover it, so above wins.
    assert.strictEqual(cellAt(92), '0px');
    // ...and below wins when it is the side that leaves the cell visible.
    assert.strictEqual(cellAt(78), 180 - 90 + 'px');
    // Both sides cover the cell: the one covering less of it wins.
    assert.strictEqual(cellAt(88), '0px', 'above covers 2px, below covers 9px');
    assert.strictEqual(cellAt(82), 180 - 90 + 'px', 'above covers 8px, below covers 3px');
  });

  await test('Spend: the heatmap reaches back to the oldest day, shaded by cost, with model bars', () => {
    const {win} = makeWebview(REMOTE);
    const report = spendReport();
    send(win, report);
    assert.strictEqual(el(win, 'meta-spend-status').textContent, '');
    assert.strictEqual(
      win.document.querySelector('.spend-total').textContent,
      'All time \u00b7 $2.04K \u00b7 1.23M tok \u00b7 6 tasks',
    );
    const dated = cells(win).filter(c => c.dataset.spendDate);
    assert.strictEqual(dated.length, 201, '200 days back to the oldest day, plus today');
    assert.strictEqual(dated[0].dataset.spendDate, dayKey(200));
    // Levels: the dearest day is l4, a day with a task but no cost l1,
    // today's $37.92 of a $2000 max is l1 too, and days without tasks
    // have no level.
    assert.ok(cellOf(win, dayKey(1)).classList.contains('l4'));
    assert.ok(cellOf(win, dayKey(200)).classList.contains('l1'));
    assert.ok(cellOf(win, dayKey(0)).classList.contains('l1'));
    assert.strictEqual(cellOf(win, dayKey(2)).className, 'spend-cell');
    // Model bars: dearest first, width = share of the total (2% floor
    // for a model with any cost, none for a free one), shaded against
    // the dearest model, value abbreviated.
    const rows = Array.from(win.document.querySelectorAll('.spend-model-row'));
    eq(
      rows.map(r => r.dataset.spendModel),
      ['claude-fable-5', 'gpt-6-astra', 'unknown'],
    );
    eq(
      rows.map(r => r.querySelector('.spend-model-value').textContent),
      ['$2.00K', '$37.92', '$0.00'],
    );
    const bars = rows.map(r => r.querySelector('.spend-model-bar'));
    assert.ok(bars[0].classList.contains('l4'));
    assert.ok(bars[1].classList.contains('l1'));
    assert.ok(bars[2].classList.contains('l1'));
    assert.strictEqual(bars[0].style.width, (2000 / 2037.92) * 100 + '%');
    assert.strictEqual(bars[1].style.width, '2%');
    assert.strictEqual(bars[2].style.width, '0%');

    // Tooltips: a day lists its totals then its models with their
    // share of the day; a free day's models have no share; a model bar
    // shows its all-time sum and share of the total.
    hover(win, cellOf(win, dayKey(1)));
    eq(tipText(win), [
      dayLabel(dayKey(1)) + ' \u00b7 $2.00K \u00b7 1.00M tok \u00b7 4 tasks',
      'claude-fable-5 \u00b7 $1.50K \u00b7 900K tok \u00b7 3 tasks \u00b7 75%',
      'gpt-6-astra \u00b7 $500.00 \u00b7 100K tok \u00b7 1 task \u00b7 25%',
    ]);
    hover(win, cellOf(win, dayKey(200)));
    eq(tipText(win), [
      dayLabel(dayKey(200)) + ' \u00b7 $0.00 \u00b7 12 tok \u00b7 1 task',
      'unknown \u00b7 $0.00 \u00b7 12 tok \u00b7 1 task',
    ]);
    // A day the reply has no model breakdown for lists only its totals.
    hover(win, cellOf(win, dayKey(0)));
    eq(tipText(win), [dayLabel(dayKey(0)) + ' \u00b7 $37.92 \u00b7 235K tok \u00b7 1 task']);
    hover(win, rows[1].querySelector('.spend-model-name'));
    eq(tipText(win), ['gpt-6-astra \u00b7 $37.92 \u00b7 235K tok \u00b7 1 task \u00b7 2%']);
    el(win, 'meta-spend-graph').dispatchEvent(new win.MouseEvent('mouseleave'));
    assert.strictEqual(tipText(win), null);

    // Pagers scroll the viewport (56px steps at jsdom's zero width);
    // the scroll offset from the right edge survives a redraw.
    const viewport = win.document.querySelector('.spend-viewport');
    assert.strictEqual(viewport.scrollLeft, 0);
    win.document.querySelector('.spend-nav.right').click();
    assert.strictEqual(viewport.scrollLeft, 56);
    win.document.querySelector('.spend-nav.left').click();
    assert.strictEqual(viewport.scrollLeft, 0);
    win.document.querySelector('.spend-grid').click();
    assert.strictEqual(viewport.scrollLeft, 0);
    // Scrolled off the left end, the left pager shows; the right one
    // hides at the right end (which is everywhere at jsdom's zero width).
    viewport.scrollLeft = 30;
    viewport.dispatchEvent(new win.Event('scroll'));
    const hidden = dir =>
      win.document.querySelector('.spend-nav.' + dir).classList.contains('nav-hidden');
    assert.ok(!hidden('left') && hidden('right'));
    viewport.scrollLeft = 0;
    viewport.dispatchEvent(new win.Event('scroll'));
    assert.ok(hidden('left') && hidden('right'));
    send(win, report);
    assert.strictEqual(win.document.querySelectorAll('.spend-total').length, 1, 'redrawn, not appended');
  });

  await test('Spend: a wider panel redraws with more week columns; a narrower one keeps the view', () => {
    const {win, observers} = makeWebview(REMOTE);
    const graph = el(win, 'meta-spend-graph');
    const resize = observers.find(o => o.target === graph);
    assert.ok(resize, 'the graph is observed');
    resize.fn();
    assert.strictEqual(win.document.querySelector('.spend-grid'), null, 'nothing drawn yet');
    send(win, spendReport());
    const columns = () => win.document.querySelectorAll('.spend-col').length;
    const before = columns();
    resize.fn();
    assert.strictEqual(columns(), before, 'zero width: same grid');
    // 1000px fits 71 columns of 11px cells with 3px gaps: the grid
    // grows to fill them (the oldest day is only 29 weeks back).
    Object.defineProperty(graph, 'clientWidth', {value: 1000, configurable: true});
    resize.fn();
    assert.strictEqual(columns(), 71);
    const dated = cells(win).filter(c => c.dataset.spendDate);
    assert.strictEqual(dated[dated.length - 1].dataset.spendDate, dayKey(0));
    assert.ok(dated.length >= 71 * 7 - 6 && dated.length <= 71 * 7);
    Object.defineProperty(graph, 'clientWidth', {value: 0, configurable: true});
    resize.fn();
    assert.strictEqual(columns(), 71, 'narrower: no redraw');
  });

  await test('Spend: a daemon day ahead of the viewer (another time zone) ends the grid', () => {
    const {win} = makeWebview(REMOTE);
    const report = spendReport();
    report.days.push(sum({date: dayKey(-1), cost: 1, tokens: 1, tasks: 1}));
    send(win, report);
    const dated = cells(win).filter(c => c.dataset.spendDate);
    assert.strictEqual(dated[dated.length - 1].dataset.spendDate, dayKey(-1));
    assert.strictEqual(dated[0].dataset.spendDate, dayKey(200));
    assert.strictEqual(dated.length, 202);
    assert.ok(cellOf(win, dayKey(-1)).classList.contains('l1'));
    // The last column still ends on that day: Monday-first columns of
    // seven, the last one cut after it.
    const cols = Array.from(win.document.querySelectorAll('.spend-col'));
    const [y, m, d] = dayKey(-1).split('-').map(Number);
    const weekday = (new Date(y, m - 1, d).getDay() + 6) % 7;
    assert.strictEqual(cols[cols.length - 1].children.length, weekday + 1);
  });

  await test('Spend: the grid reaches back at most three years, however old the history', () => {
    const {win} = makeWebview(REMOTE);
    const report = spendReport();
    report.days.unshift(sum({date: '2009-02-13', cost: 0, tokens: 0, tasks: 2}));
    send(win, report);
    const dated = cells(win).filter(c => c.dataset.spendDate);
    assert.strictEqual(dated.length, 3 * 366);
    assert.strictEqual(dated[0].dataset.spendDate, dayKey(3 * 366 - 1));
    assert.strictEqual(dated[dated.length - 1].dataset.spendDate, dayKey(0));
  });

  await test('Spend: a history that cost nothing still shows its day and its model', () => {
    const {win} = makeWebview(REMOTE);
    send(win, {
      type: 'spendReport',
      total: sum({cost: 0, tokens: 5, tasks: 1}),
      days: [sum({date: dayKey(0), cost: 0, tokens: 5, tasks: 1})],
      totalByModel: [sum({model: 'unknown', cost: 0, tokens: 5, tasks: 1})],
      daysByModel: {},
    });
    assert.strictEqual(el(win, 'meta-spend-status').textContent, '');
    // Nothing cost anything: the recorded day and the model still show.
    assert.ok(cellOf(win, dayKey(0)).classList.contains('l1'));
    const bar = win.document.querySelector('.spend-model-bar');
    assert.ok(bar.classList.contains('l1'));
    assert.strictEqual(bar.style.width, '0%');
    hover(win, win.document.querySelector('.spend-model-row'));
    eq(tipText(win), ['unknown \u00b7 $0.00 \u00b7 5 tok \u00b7 1 task']);
  });

  for (const [label, attrs] of [
    ['remote webapp', REMOTE],
    ['sidebar-chat drawer', SIDEBAR],
  ]) {
    await test(`${label}: clicking an app that is not connected submits a connect task in a new tab`, () => {
      const {win, posted, intervals} = makeWebview(attrs);
      const inp = el(win, 'task-input');
      inp.value = 'my draft';
      inp.dispatchEvent(new win.Event('input', {bubbles: true}));
      el(win, 'meta-panel').classList.add('open');
      send(win, {type: 'appsStatus', apps: APPS, checkedAt: 1});
      const tabsBefore = win.document.querySelectorAll('#tab-list .chat-tab').length;
      posted.length = 0;
      win.document.querySelector('.app-row[data-app="slack"] button').click();
      const submit = posted.find(m => m.type === 'submit');
      assert.ok(submit, 'a submit must be posted');
      assert.ok(
        submit.prompt.startsWith(
          'Connect my Slack app: authenticate the "slack" third-party agent (run_agent with agent "slack").',
        ),
        submit.prompt,
      );
      assert.match(submit.prompt, /OAuth consent or device-code flow/);
      assert.match(submit.prompt, /Never ask for my password in chat/);
      assert.match(submit.prompt, /Never retry a failed sign-in in a loop/);
      assert.match(submit.prompt, /solve or bypass a CAPTCHA/);
      assert.strictEqual(
        win.document.querySelectorAll('#tab-list .chat-tab').length,
        tabsBefore + 1,
      );
      assert.ok(!el(win, 'meta-panel').classList.contains('open'), 'drawer closed');

      // While Slack waits for its connect task, every poll re-probes;
      // once a reply shows it connected, polls use the cache again.
      send(win, {type: 'cronJobs', jobs: []});
      const poll = intervals.find(i => i.ms === 30000);
      posted.length = 0;
      poll.fn();
      eq(posted.filter(m => m.type === 'getAppsStatus'), [
        {type: 'getAppsStatus', refresh: true},
      ]);
      const connected = APPS.map(a =>
        a.name === 'slack' ? Object.assign({}, a, {authenticated: true}) : a,
      );
      send(win, {type: 'appsStatus', apps: connected, checkedAt: 1});
      posted.length = 0;
      poll.fn();
      eq(posted.filter(m => m.type === 'getAppsStatus'), [{type: 'getAppsStatus'}]);

      // A connect task that never completes stops forcing re-probes
      // after 30 minutes.
      win.document.querySelector('.app-row[data-app="matrix"] button').click();
      const realNow = win.Date.now;
      win.Date.now = () => realNow() + 31 * 60000;
      posted.length = 0;
      poll.fn();
      win.Date.now = realNow;
      eq(posted.filter(m => m.type === 'getAppsStatus'), [{type: 'getAppsStatus'}]);
    });
  }

  await test('Task Info view: the click asks the host for an auto-submitting chat panel', () => {
    const {win, posted} = makeWebview(META_VIEW);
    send(win, {type: 'appsStatus', apps: APPS, checkedAt: 1});
    posted.length = 0;
    win.document.querySelector('.app-row[data-app="matrix"] button').click();
    assert.strictEqual(posted.length, 1);
    assert.strictEqual(posted[0].type, 'openChatPanel');
    assert.strictEqual(posted[0].autoSubmit, true);
    assert.ok(posted[0].pendingText.startsWith('Connect my Matrix app'));
  });

  for (const [label, attrs] of [
    ['chat editor panel', EDITOR_PANEL],
    ['history panel', HISTORY],
  ]) {
    await test(`${label}: never requests nor renders the subpanels`, () => {
      const {win, posted, intervals} = makeWebview(attrs);
      send(win, {type: 'daemonStatus', connected: true});
      for (const cmd of PANEL_COMMANDS) assert.ok(!types(posted).includes(cmd), cmd);
      send(win, {type: 'cronJobs', jobs: [job({})]});
      send(win, {type: 'appsStatus', apps: APPS, checkedAt: 1});
      send(win, spendReport());
      assert.strictEqual(el(win, 'meta-schedule-list').children.length, 0);
      assert.strictEqual(appRows(win).length, 0);
      assert.strictEqual(cells(win).length, 0);
      assert.ok(!intervals.some(i => i.ms === 30000));
    });
  }

  if (failures) {
    console.log(`\n${failures} test(s) failed`);
    process.exit(1);
  }
  console.log('\nall sidebarPanels tests passed');
  process.exit(0);
}

main();
