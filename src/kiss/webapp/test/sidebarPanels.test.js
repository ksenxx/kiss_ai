// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the right sidebar's Schedule, Apps, Spend and Recommendations
// subpanels (the sidebarpanels block of media/main.js):
//
// * boot and a daemon (re)connect request getCronJobs + getMachines +
//   getAppsStatus + getSpendReport + getRecommendations; the poll
//   timer starts on the first reply, once,
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
//   a new internal tab that submits the prompt (drawer closed); while
//   the app waits to be connected (at most 30 minutes) polls re-probe.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {
  makePage: makeChatPage,
  send,
  stubRaf,
  fakeMatchMedia,
  MEDIA,
} = require('./simplify2_harness');

const {makeRunner} = require('./ui_antipattern_harness');

const {test, report} = makeRunner();

/**
 * Boot a chat page with the given body attributes.  setInterval is
 * recorded (not run) so a test can fire the poll by hand.  `desktop`
 * (default true) is the (min-width: 900px) media query: false boots
 * the phone layout, where the task-info panel is a drawer.
 */
function makePage(opts) {
  const {desktop = true} = opts || {};
  const intervals = [];
  // jsdom has no ResizeObserver: record the observers so a test can
  // fire a resize by hand.
  const observers = [];
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js'],
    sourceUrl: 'sidebarpanels-main.js',
    beforeScripts(win) {
      stubRaf(win);
      win.setInterval = function (fn, ms) {
        intervals.push({fn, ms});
        return intervals.length;
      };
      win.clearInterval = function () {};
      win.ResizeObserver = class {
        constructor(fn) {
          this.fn = fn;
          observers.push(this);
        }
        observe(target) {
          this.target = target;
        }
      };
      fakeMatchMedia(win, desktop);
    },
  });
  return {win, posted, intervals, observers};
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

const MACHINES = [
  {
    name: 'laptop',
    address: 'https://10.0.0.5:8787',
    ip: '10.0.0.5',
    port: 8787,
    local: true,
    status: 'reachable',
    error: '',
  },
  {
    name: 'gpu-box',
    address: 'ssh me@gpu',
    ip: '10.0.0.9',
    port: 41234,
    local: false,
    status: 'reachable',
    error: '',
  },
  {
    // Registered by its address: the daemon labels it by its host name.
    name: 'office',
    label: 'office-pc',
    address: 'ssh me@office',
    ip: 'office',
    port: 0,
    local: false,
    status: 'unreachable',
    error: 'office: ssh exited',
  },
];

function appRows(win) {
  return Array.from(win.document.querySelectorAll('#meta-apps-list .app-row'));
}

const PANEL_COMMANDS = [
  'getCronJobs',
  'getMachines',
  'getAppsStatus',
  'getSpendReport',
  'getRecommendations',
];

function recommendationRows(win) {
  return Array.from(
    win.document.querySelectorAll(
      '#meta-recommendations-list .recommendation-row',
    ),
  );
}

const FIVE_TASKS = [
  'Fix the failing parser test in parser.py',
  'Add a changelog entry for /recommendations',
  'Run the full test suite in /home/me/proj',
  'Review the open pull request on the cron scheduler',
  'Document the sidebar panels in README.md',
];

/** "YYYY-MM-DD" of the local day *daysAgo* days before today. */
function dayKey(daysAgo) {
  const now = new Date();
  const d = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate() - daysAgo,
  );
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
  return Array.from(
    win.document.querySelectorAll('#meta-spend-graph .spend-cell'),
  );
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
  await test('the daemon coming up requests the five subpanels once; the poll starts on the first reply', () => {
    const {win, posted, intervals} = makePage();
    // Audit 2026-10-18: the boot made its own request on top of the
    // one the `daemonStatus connected` handler makes (transport.js
    // announces it once the socket is authenticated), so every
    // subpanel was asked twice at page load.
    const panelPosts = () =>
      posted.filter(m => PANEL_COMMANDS.includes(m.type));
    eq(panelPosts(), [], 'nothing is asked before the daemon is up');
    send(win, {type: 'daemonStatus', connected: true});
    eq(panelPosts(), [
      {type: 'getCronJobs'},
      {type: 'getMachines'},
      {type: 'getAppsStatus'},
      {type: 'getSpendReport'},
      {type: 'getRecommendations'},
    ]);
    assert.strictEqual(
      el(win, 'meta-apps-status').textContent,
      'Checking\u2026',
    );
    assert.strictEqual(
      el(win, 'meta-machines-status').textContent,
      'Checking\u2026',
    );
    assert.strictEqual(
      el(win, 'meta-spend-status').textContent,
      'Loading\u2026',
    );
    assert.strictEqual(
      el(win, 'meta-recommendations-status').textContent,
      'Loading\u2026',
    );
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

  await test('the empty Schedule shows a hint; rows show badges, runs and tooltips', () => {
    const {win} = makePage();
    send(win, {type: 'cronJobs', jobs: []});
    assert.match(
      el(win, 'meta-schedule-status').textContent,
      /^No scheduled jobs/,
    );
    assert.strictEqual(el(win, 'meta-schedule-list').children.length, 0);
    send(win, {type: 'cronJobs', jobs: 'bogus'});
    assert.match(
      el(win, 'meta-schedule-status').textContent,
      /^No scheduled jobs/,
    );

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
    assert.strictEqual(
      rows[0].querySelector('.sidebar-panel-name').textContent,
      'Digest',
    );
    assert.strictEqual(
      sub(rows[0]),
      `every 5m \u00b7 next ${pt(times.a)} (in 5m)`,
    );
    assert.strictEqual(rows[0].dataset.tooltip, 'do it\n\nLast run: ok');
    assert.strictEqual(
      rows[0].title,
      '',
      'no native title: the custom tooltip shows the task',
    );
    assert.strictEqual(badge(rows[0]), '');
    assert.strictEqual(badge(rows[1]), 'running');
    assert.strictEqual(
      sub(rows[1]),
      `every 5m \u00b7 next ${pt(times.b)} (in 3h)`,
    );
    assert.strictEqual(rows[1].dataset.tooltip, '$ rsync -a src dst');
    assert.strictEqual(
      sub(rows[2]),
      `every 5m \u00b7 next ${pt(times.c)} (in 3d)`,
    );
    assert.strictEqual(
      sub(rows[3]),
      `every 5m \u00b7 next ${pt(times.d)} (now)`,
    );
    assert.strictEqual(sub(rows[4]), 'every 5m');
    assert.ok(rows[5].classList.contains('paused'));
    assert.strictEqual(badge(rows[5]), 'paused');
    assert.strictEqual(
      sub(rows[5]),
      `every 5m \u00b7 last ${pt(times.f)} (2h ago)`,
    );
    assert.strictEqual(sub(rows[6]), 'every 5m');
  });

  await test('Schedule run times are Pacific time whatever the viewer zone', () => {
    const {win} = makePage();
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
    assert.match(
      subs[0],
      /^every 5m \u00b7 next Sun, Sep 27, 5:00\s?AM PDT \(/,
    );
    assert.match(subs[1], /^every 5m \u00b7 next Tue, Dec 1, 9:00\s?AM PST \(/);
  });

  await test('Schedule rows: edit, run now, pause / resume, copy and delete buttons', () => {
    const {win, posted} = makePage();
    send(win, {
      type: 'cronJobs',
      jobs: [
        job({id: 'a', name: 'Digest'}),
        job({
          id: 'b',
          name: 'Backup',
          kind: 'command',
          what: 'rsync',
          running: true,
        }),
        job({id: 'c', name: 'Old', enabled: false}),
      ],
    });
    const rows = Array.from(el(win, 'meta-schedule-list').children);
    const labels = r =>
      Array.from(r.querySelectorAll('.sidebar-panel-row-top > button')).map(b =>
        b.getAttribute('aria-label'),
      );
    eq(labels(rows[0]), [
      'Edit the job',
      'Run the job now',
      'Disable the job',
      'Copy the scheduled task to clipboard',
      'Delete the job',
    ]);
    eq(labels(rows[1]), [
      'Edit the job',
      'Run the job now',
      'Disable the job',
      'Copy the scheduled command to clipboard',
      'Delete the job',
    ]);
    assert.strictEqual(labels(rows[2])[2], 'Enable the job');
    const button = (r, label) => r.querySelector(`[aria-label="${label}"]`);
    assert.ok(
      button(rows[1], 'Run the job now').disabled,
      'a running job cannot be run again',
    );
    assert.ok(!button(rows[0], 'Run the job now').disabled);
    posted.length = 0;
    button(rows[0], 'Run the job now').click();
    button(rows[0], 'Disable the job').click();
    button(rows[2], 'Enable the job').click();
    eq(posted, [
      {type: 'cronAction', action: 'run_now', jobId: 'a'},
      {type: 'cronAction', action: 'pause', jobId: 'a'},
      {type: 'cronAction', action: 'resume', jobId: 'c'},
    ]);
    // Delete is a two-step inline confirm: the trash icon only reveals
    // Delete / Cancel; Cancel posts nothing, Delete posts the removal.
    posted.length = 0;
    const del = button(rows[0], 'Delete the job');
    del.click();
    const confirm = rows[0].querySelector('.sidebar-item-confirm');
    assert.strictEqual(confirm.style.display, '');
    assert.strictEqual(del.style.display, 'none');
    confirm.querySelector('.sidebar-confirm-no').click();
    eq(posted, []);
    assert.strictEqual(confirm.style.display, 'none');
    del.click();
    confirm.querySelector('.sidebar-confirm-yes').click();
    eq(posted, [{type: 'cronAction', action: 'remove', jobId: 'a'}]);
    // Answering Delete spends the button but keeps the question in
    // place until the reply rebuilds the row: a double click can
    // neither post the removal twice nor land on the row under it.
    assert.strictEqual(confirm.style.display, '');
    assert.ok(confirm.querySelector('.sidebar-confirm-yes').disabled);
    confirm.querySelector('.sidebar-confirm-yes').click();
    eq(posted, [{type: 'cronAction', action: 'remove', jobId: 'a'}]);
    // The reply to an action carries an error when the daemon refused
    // it: shown on the status line, cleared by the next clean reply.
    send(win, {
      type: 'cronJobs',
      jobs: [job({id: 'a'})],
      error: 'job is already running',
    });
    const status = el(win, 'meta-schedule-status');
    assert.strictEqual(status.textContent, 'job is already running');
    assert.ok(status.classList.contains('is-error'));
    send(win, {type: 'cronJobs', jobs: [job({id: 'a'})]});
    assert.strictEqual(status.textContent, '');
    assert.ok(!status.classList.contains('is-error'));
    // The refused removal's reply rebuilt the row: the daemon has
    // answered, so the question comes back with a live Delete.
    const row = el(win, 'meta-schedule-list').querySelector(
      '.sidebar-panel-row',
    );
    assert.notStrictEqual(row, rows[0], 'the reply rebuilt the row');
    assert.strictEqual(
      row.querySelector('.sidebar-item-confirm').style.display,
      '',
    );
    assert.ok(!row.querySelector('.sidebar-confirm-yes').disabled);
    // Cancel folds it; the trash icon is back.
    row.querySelector('.sidebar-confirm-no').click();
    assert.strictEqual(
      row.querySelector('.sidebar-item-confirm').style.display,
      'none',
    );
    assert.strictEqual(
      row.querySelector('.sidebar-item-delete').style.display,
      '',
    );
    // A question still open when a poll reply rebuilds the list is asked
    // again in the new row, with the keyboard where it was.
    button(row, 'Delete the job').click();
    row.querySelector('.sidebar-confirm-yes').focus();
    send(win, {type: 'cronJobs', jobs: [job({id: 'a'})]});
    const rebuilt = el(win, 'meta-schedule-list').querySelector(
      '.sidebar-panel-row',
    );
    assert.notStrictEqual(rebuilt, row, 'the reply rebuilt the row');
    assert.strictEqual(
      rebuilt.querySelector('.sidebar-item-confirm').style.display,
      '',
    );
    assert.strictEqual(
      win.document.activeElement,
      rebuilt.querySelector('.sidebar-confirm-yes'),
    );
    // A row gone from the reply has nothing to restore.
    send(win, {type: 'cronJobs', jobs: []});
    assert.strictEqual(
      el(win, 'meta-schedule-list').querySelector('.sidebar-item-confirm'),
      null,
    );
  });

  await test('Schedule rows: the edit button opens an in-place editor that saves or cancels', () => {
    const {win, posted} = makePage();
    const jobs = [
      job({id: 'a', name: 'Digest', schedule: '0 9 * * *', what: 'summarize'}),
      job({id: 'b', name: 'Backup', kind: 'command', what: 'rsync'}),
    ];
    send(win, {type: 'cronJobs', jobs});
    const list = el(win, 'meta-schedule-list');
    const edit = r => r.querySelector('[aria-label="Edit the job"]');
    edit(list.children[0]).click();
    let form = list.children[0].querySelector('form.sidebar-inline-form');
    assert.ok(form, 'the editor sits under its row');
    eq(
      Array.from(form.querySelectorAll('.sidebar-inline-form-label')).map(
        l => l.textContent,
      ),
      ['Name', 'Schedule', 'Task'],
    );
    assert.strictEqual(form.elements.name.value, 'Digest');
    assert.strictEqual(form.elements.schedule.value, '0 9 * * *');
    assert.strictEqual(form.elements.what.value, 'summarize');
    assert.strictEqual(form.elements.what.tagName, 'TEXTAREA');
    assert.strictEqual(win.document.activeElement, form.elements.name);
    // A second edit click (or one on another row) opens one editor at a time.
    edit(list.children[1]).click();
    assert.strictEqual(list.querySelectorAll('form').length, 1);
    form = list.children[1].querySelector('form');
    assert.strictEqual(
      form.querySelectorAll('.sidebar-inline-form-label')[2].textContent,
      'Command',
    );
    // Cancel (the button, or Escape anywhere in the form) closes it.
    form.querySelector('.sidebar-confirm-no').click();
    assert.strictEqual(list.querySelectorAll('form').length, 0);
    assert.strictEqual(win.document.activeElement, list.children[1]);
    edit(list.children[0]).click();
    form = list.querySelector('form');
    form.elements.schedule.dispatchEvent(
      new win.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
    );
    assert.strictEqual(list.querySelectorAll('form').length, 0);
    // Save posts the update; Enter in an input saves, Enter in the
    // textarea does not (Ctrl+Enter does).
    edit(list.children[0]).click();
    form = list.querySelector('form');
    form.elements.name.value = 'Morning digest';
    form.elements.schedule.value = 'every 2h';
    form.elements.what.value = 'summarize my email';
    posted.length = 0;
    form.elements.what.dispatchEvent(
      new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
    );
    eq(posted, []);
    form.elements.what.dispatchEvent(
      new win.KeyboardEvent('keydown', {
        key: 'Enter',
        ctrlKey: true,
        bubbles: true,
      }),
    );
    const update = {
      type: 'cronAction',
      action: 'update',
      jobId: 'a',
      name: 'Morning digest',
      schedule: 'every 2h',
      what: 'summarize my email',
    };
    eq(posted, [update]);
    // The update is in flight: Save is disabled and a repeated Enter
    // or click sends nothing more until the reply re-renders the form.
    assert.ok(form.querySelector('.sidebar-inline-form-save').disabled);
    posted.length = 0;
    form.elements.schedule.dispatchEvent(
      new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
    );
    form.querySelector('.sidebar-inline-form-save').click();
    eq(posted, []);
    // A refused update re-opens the editor with the typed values; the
    // periodic poll (no action) and another row's action keep it too,
    // with whatever was typed meanwhile; the reply acknowledging the
    // update closes it.
    send(win, {
      type: 'cronJobs',
      jobs,
      action: 'update',
      error: "schedule 'every 2h' never fires",
    });
    form = list.children[0].querySelector('form');
    assert.ok(form, 'the editor is back under its row');
    assert.strictEqual(form.elements.name.value, 'Morning digest');
    assert.strictEqual(form.elements.what.value, 'summarize my email');
    // Enter in an input and the Save button each send the update.
    assert.ok(!form.querySelector('.sidebar-inline-form-save').disabled);
    form.elements.schedule.dispatchEvent(
      new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
    );
    eq(posted, [update]);
    send(win, {type: 'cronJobs', jobs, action: 'update', error: 'still no'});
    form = list.children[0].querySelector('form');
    posted.length = 0;
    form.querySelector('.sidebar-inline-form-save').click();
    eq(posted, [update]);
    send(win, {
      type: 'cronJobs',
      jobs,
      action: 'update',
      error: "schedule 'every 2h' never fires",
    });
    form = list.children[0].querySelector('form');
    assert.strictEqual(
      el(win, 'meta-schedule-status').textContent,
      "schedule 'every 2h' never fires",
    );
    form.elements.schedule.value = 'every 3h';
    send(win, {type: 'cronJobs', jobs});
    form = list.children[0].querySelector('form');
    assert.ok(form, 'a poll keeps the editor open');
    assert.strictEqual(
      form.elements.schedule.value,
      'every 3h',
      'with the draft',
    );
    assert.ok(!form.querySelector('.sidebar-inline-form-save').disabled);
    // Sent again: a poll and another row's action arriving while the
    // update is in flight keep the editor AND keep its Save disabled
    // (the rebuilt form is still waiting for the same reply).
    posted.length = 0;
    form.querySelector('.sidebar-inline-form-save').click();
    eq(posted, [Object.assign({}, update, {schedule: 'every 3h'})]);
    send(win, {type: 'cronJobs', jobs});
    form = list.children[0].querySelector('form');
    assert.ok(
      form.querySelector('.sidebar-inline-form-save').disabled,
      'in flight across a poll',
    );
    send(win, {type: 'cronJobs', jobs, action: 'pause', error: ''});
    form = list.children[0].querySelector('form');
    assert.ok(form, "another row's action keeps it");
    assert.ok(form.querySelector('.sidebar-inline-form-save').disabled);
    posted.length = 0;
    form.querySelector('.sidebar-inline-form-save').click();
    eq(posted, [], 'no resend while in flight');
    // The daemon drops: no reply is coming, so Save is re-armed, and a
    // save while disconnected is refused with a note (the draft stays).
    send(win, {type: 'daemonStatus', connected: false});
    form = list.children[0].querySelector('form');
    assert.ok(!form.querySelector('.sidebar-inline-form-save').disabled);
    form.querySelector('.sidebar-inline-form-save').click();
    eq(posted, []);
    assert.strictEqual(
      el(win, 'meta-schedule-status').textContent,
      'Not connected: try again when reconnected',
    );
    assert.ok(el(win, 'meta-schedule-status').classList.contains('is-error'));
    assert.strictEqual(form.elements.schedule.value, 'every 3h');
    send(win, {type: 'daemonStatus', connected: true});
    posted.length = 0; // the reconnect re-requests the panels
    form.querySelector('.sidebar-inline-form-save').click();
    eq(posted, [Object.assign({}, update, {schedule: 'every 3h'})]);
    send(win, {type: 'cronJobs', jobs, action: 'update', error: ''});
    assert.strictEqual(list.querySelectorAll('form').length, 0);
    // A poll with no editor open changes nothing.
    send(win, {type: 'cronJobs', jobs});
    assert.strictEqual(list.querySelectorAll('form').length, 0);
  });

  await test('Schedule: the header + opens an in-place form that schedules a job', () => {
    const {win, posted} = makePage();
    const jobs = [
      job({id: 'a', name: 'Digest', schedule: '0 9 * * *', what: 'summarize'}),
      job({id: 'b', name: 'Backup', kind: 'command', what: 'rsync'}),
    ];
    send(win, {type: 'cronJobs', jobs: []});
    assert.match(
      el(win, 'meta-schedule-status').textContent,
      /^No scheduled jobs\. Click \+ to schedule one/,
      'the empty hint points at the +',
    );
    send(win, {type: 'cronJobs', jobs});
    const list = el(win, 'meta-schedule-list');
    const addBtn = el(win, 'meta-schedule-add');
    assert.strictEqual(addBtn.getAttribute('aria-label'), 'Add job');
    assert.strictEqual(
      addBtn.closest('.meta-section-hdr'),
      el(win, 'meta-schedule-refresh').closest('.meta-section-hdr'),
      'the + sits in the Schedule header, next to the refresh button',
    );
    const topForm = () =>
      list.firstElementChild.classList.contains('sched-add-row')
        ? list.firstElementChild.querySelector('form.sidebar-inline-form')
        : null;
    const labels = form =>
      Array.from(form.querySelectorAll('.sidebar-inline-form-label')).map(
        l => l.textContent,
      );
    addBtn.click();
    let form = topForm();
    assert.ok(form, 'the form opens at the top of the list');
    assert.strictEqual(list.children.length, 3, 'above the two rows');
    eq(labels(form), ['Name', 'Schedule', 'Run', 'Task']);
    assert.strictEqual(form.elements.kind.tagName, 'SELECT');
    eq(
      Array.from(form.elements.kind.options).map(o => [o.value, o.textContent]),
      [
        ['prompt', 'A task (an LLM prompt)'],
        ['command', 'A shell command'],
      ],
    );
    assert.strictEqual(form.elements.kind.value, 'prompt');
    assert.strictEqual(form.elements.what.tagName, 'TEXTAREA');
    assert.strictEqual(win.document.activeElement, form.elements.name);
    // Picking a shell command relabels the task field.
    form.elements.kind.value = 'command';
    form.elements.kind.dispatchEvent(new win.Event('change', {bubbles: true}));
    eq(labels(form), ['Name', 'Schedule', 'Run', 'Command']);
    form.elements.kind.value = 'prompt';
    form.elements.kind.dispatchEvent(new win.Event('change', {bubbles: true}));
    eq(labels(form), ['Name', 'Schedule', 'Run', 'Task']);
    // One form at a time: a second press replaces it, blank; a row's
    // edit button closes it (and the + closes an open editor).
    form.elements.name.value = 'typed';
    addBtn.click();
    assert.strictEqual(list.querySelectorAll('form').length, 1);
    form = topForm();
    assert.strictEqual(form.elements.name.value, '');
    list.children[1].querySelector('[aria-label="Edit the job"]').click();
    assert.strictEqual(list.querySelectorAll('form').length, 1);
    assert.strictEqual(topForm(), null, 'the add row is gone');
    assert.ok(list.children[0].querySelector('form'), 'the editor is open');
    addBtn.click();
    assert.strictEqual(list.querySelectorAll('form').length, 1);
    assert.ok(topForm(), 'the add form replaced the editor');
    // Cancel closes it and returns the focus to the +.
    form = topForm();
    form.querySelector('.sidebar-confirm-no').click();
    assert.strictEqual(list.querySelectorAll('form').length, 0);
    assert.strictEqual(list.children.length, 2);
    assert.strictEqual(win.document.activeElement, addBtn);
    // Save posts a create with the kind; Enter in an input saves too.
    addBtn.click();
    form = topForm();
    form.elements.name.value = 'Nightly';
    form.elements.schedule.value = 'every 1d';
    form.elements.kind.value = 'command';
    form.elements.what.value = 'echo night';
    posted.length = 0;
    form.elements.schedule.dispatchEvent(
      new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
    );
    const create = {
      type: 'cronAction',
      action: 'create',
      name: 'Nightly',
      schedule: 'every 1d',
      what: 'echo night',
      kind: 'command',
    };
    eq(posted, [create]);
    // The create is in flight: a second click sends nothing more (the
    // daemon would refuse the duplicate and paint that over the first
    // create's success).
    assert.ok(form.querySelector('.sidebar-inline-form-save').disabled);
    posted.length = 0;
    form.querySelector('.sidebar-inline-form-save').click();
    eq(posted, []);
    // A refused create re-opens the form, at the top, with the typed
    // values (the kind included); the periodic poll and another job's
    // action keep it, with whatever was typed meanwhile; the reply
    // acknowledging the create (the job listed) closes it.
    send(win, {
      type: 'cronJobs',
      jobs,
      action: 'create',
      error: 'duplicate: job a',
    });
    form = topForm();
    assert.ok(form, 'the form is back at the top');
    assert.ok(
      !form.querySelector('.sidebar-inline-form-save').disabled,
      'a refusal re-enables Save',
    );
    form.querySelector('.sidebar-inline-form-save').click();
    eq(posted, [create], 'the Save button sends the create too');
    send(win, {
      type: 'cronJobs',
      jobs,
      action: 'create',
      error: 'duplicate: job a',
    });
    form = topForm();
    assert.strictEqual(form.elements.kind.value, 'command');
    assert.strictEqual(form.elements.what.value, 'echo night');
    eq(
      labels(form),
      ['Name', 'Schedule', 'Run', 'Command'],
      'relabelled for the kind',
    );
    assert.strictEqual(
      el(win, 'meta-schedule-status').textContent,
      'duplicate: job a',
    );
    form.elements.name.value = 'Nightly 2';
    send(win, {type: 'cronJobs', jobs});
    form = topForm();
    assert.ok(form, 'a poll keeps the form open');
    assert.strictEqual(form.elements.name.value, 'Nightly 2', 'with the draft');
    send(win, {type: 'cronJobs', jobs, action: 'pause', error: ''});
    assert.ok(topForm(), "another job's action keeps it");
    const created = jobs.concat([
      job({id: 'c', name: 'Nightly 2', kind: 'command', what: 'echo night'}),
    ]);
    posted.length = 0;
    topForm().querySelector('.sidebar-inline-form-save').click();
    assert.strictEqual(
      posted[0].name,
      'Nightly 2',
      'the retry sends the draft',
    );
    send(win, {type: 'cronJobs', jobs: created, action: 'create', error: ''});
    assert.strictEqual(list.querySelectorAll('form').length, 0);
    assert.strictEqual(list.children.length, 3);
    // A create acknowledged after the + already opened a fresh form
    // (its own create still pending, or never sent) leaves that form
    // and its draft alone; only the form that sent the create closes.
    addBtn.click();
    form = topForm();
    form.elements.name.value = 'Weekly';
    form.elements.schedule.value = 'every 7d';
    form.elements.what.value = 'report';
    form.querySelector('.sidebar-inline-form-save').click();
    addBtn.click();
    form = topForm();
    form.elements.name.value = 'Fresh';
    send(win, {type: 'cronJobs', jobs: created, action: 'create', error: ''});
    form = topForm();
    assert.ok(form, "an earlier form's acknowledgment keeps the fresh form");
    assert.strictEqual(form.elements.name.value, 'Fresh', 'with its draft');
    assert.strictEqual(list.querySelectorAll('form').length, 1);
    form.elements.schedule.value = 'every 2d';
    form.elements.what.value = 'f';
    form.querySelector('.sidebar-inline-form-save').click();
    send(win, {
      type: 'cronJobs',
      jobs: created,
      action: 'create',
      error: 'nope',
    });
    form = topForm();
    assert.ok(form, 'its refused create re-opens it');
    assert.strictEqual(form.elements.name.value, 'Fresh');
    send(win, {type: 'cronJobs', jobs: created, action: 'create', error: ''});
    assert.ok(topForm(), 'a stray acknowledgment after the refusal keeps it');
    form = topForm();
    form.querySelector('.sidebar-inline-form-save').click();
    send(win, {type: 'cronJobs', jobs: created, action: 'create', error: ''});
    assert.strictEqual(
      list.querySelectorAll('form').length,
      0,
      'its own closes it',
    );
  });

  await test('each Schedule row copies its exact task and shows it in the tooltip', async () => {
    const {win} = makePage();
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
      "Use 'claude-fable-5-1' for all tasks.";
    send(win, {
      type: 'cronJobs',
      jobs: [
        job({
          id: 'p',
          name: 'Weekly rsi7d',
          what: prompt,
          lastStatus: 'ok in 12m',
        }),
        job({id: 'c', name: 'Sync', kind: 'command', what: 'rsync -a src dst'}),
        job({id: 'n', name: 'No text', what: null}),
      ],
    });
    const rows = Array.from(el(win, 'meta-schedule-list').children);
    const copies = rows.map(r =>
      r.querySelector('.sidebar-panel-row-top > .sidebar-item-copy'),
    );
    assert.ok(
      copies.every(Boolean),
      'every row has a copy button on its name line',
    );
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
    eq(
      writes,
      [prompt, 'rsync -a src dst', ''],
      'the exact prompt or command, unchanged',
    );
    assert.ok(
      copies[0].classList.contains('copied'),
      'the button flashes a check mark',
    );

    // The row's tooltip is the same text (commands prefixed "$ "),
    // then the last run's outcome; it opens on keyboard focus too.
    assert.strictEqual(
      rows[0].dataset.tooltip,
      prompt + '\n\nLast run: ok in 12m',
    );
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

  await test('a tooltip that would run off the bottom or right edge stays on screen', () => {
    const {win} = makePage();
    send(win, {
      type: 'cronJobs',
      jobs: [job({id: 'p', name: 'Tall', what: 'x'.repeat(2000)})],
    });
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
    row.getBoundingClientRect = () => ({
      left: 600,
      top: 500,
      bottom: 540,
      right: 800,
      width: 200,
      height: 40,
    });
    row.dispatchEvent(new win.FocusEvent('focusin', {bubbles: true}));
    assert.strictEqual(measuredAt, '0px 0px', 'measured at the origin');
    assert.ok(tip.classList.contains('visible'));
    assert.strictEqual(
      tip.style.top,
      500 - 4 - 300 + 'px',
      'flipped above the row',
    );
    assert.strictEqual(
      tip.style.left,
      800 - 350 - 8 + 'px',
      'slid left of the right edge',
    );
    row.dispatchEvent(new win.FocusEvent('focusout', {bubbles: true}));

    // Room below and to the right: left-aligned with the row, under it.
    row.getBoundingClientRect = () => ({
      left: 20,
      top: 100,
      bottom: 140,
      right: 220,
      width: 200,
      height: 40,
    });
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
    assert.match(
      rule,
      /max-height: calc\(100vh - 16px\)/,
      'capped at the viewport height',
    );
    assert.match(rule, /overflow: hidden/);
  });

  await test('Machines: rows show the status, the name, a local badge, the tooltip and the buttons', async () => {
    const {win, posted} = makePage();
    const copied = [];
    Object.defineProperty(win.navigator, 'clipboard', {
      value: {writeText: t => (copied.push(t), Promise.resolve())},
      configurable: true,
    });
    send(win, {type: 'machines', machines: 'bogus', checkedAt: 0});
    assert.strictEqual(
      el(win, 'meta-machines-status').textContent,
      'No machines.',
    );
    assert.strictEqual(el(win, 'meta-machines-list').children.length, 0);
    send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
    assert.strictEqual(el(win, 'meta-machines-status').textContent, '');
    const rows = Array.from(el(win, 'meta-machines-list').children);
    assert.strictEqual(rows.length, 3);
    const name = r => r.querySelector('.sidebar-panel-name').textContent;
    const sub = r => r.querySelector('.sidebar-panel-sub').textContent;
    const dot = r => r.querySelector('.machine-status');
    eq(
      rows.map(name),
      ['laptop', 'gpu-box', 'office-pc'],
      'a row shows the label',
    );
    assert.strictEqual(
      rows[2].dataset.machine,
      'office',
      'the row keeps the name',
    );
    assert.strictEqual(
      rows[0].querySelector('.sidebar-panel-badge').textContent,
      'local',
    );
    assert.strictEqual(rows[1].querySelector('.sidebar-panel-badge'), null);
    assert.ok(dot(rows[0]).classList.contains('reachable'));
    assert.strictEqual(dot(rows[0]).getAttribute('aria-label'), 'reachable');
    assert.ok(dot(rows[2]).classList.contains('unreachable'));
    assert.ok(rows[2].classList.contains('unreachable'));
    assert.strictEqual(sub(rows[0]), 'https://10.0.0.5:8787 \u00b7 reachable');
    assert.strictEqual(sub(rows[2]), 'ssh me@office \u00b7 unreachable');
    assert.strictEqual(rows[0].dataset.tooltip, 'IP 10.0.0.5 \u00b7 port 8787');
    assert.strictEqual(
      rows[1].dataset.tooltip,
      'IP 10.0.0.9 \u00b7 port 41234',
    );
    assert.strictEqual(
      rows[2].dataset.tooltip,
      'IP office \u00b7 port none (tunnel closed)\n\noffice: ssh exited',
    );
    const labels = r =>
      Array.from(r.querySelectorAll('.sidebar-panel-row-top > button')).map(b =>
        b.getAttribute('aria-label'),
      );
    eq(labels(rows[1]), [
      'Copy the machine address to clipboard',
      'Delete the machine',
    ]);
    assert.strictEqual(
      rows[1].querySelector('[aria-label="Add a machine"]'),
      null,
      'a row has no add button: the header has the one +',
    );
    const button = (r, label) => r.querySelector(`[aria-label="${label}"]`);
    const localDel = button(rows[0], 'Delete the machine');
    assert.ok(localDel.disabled, 'the local machine cannot be deleted');
    assert.strictEqual(
      localDel.dataset.tooltip,
      'The local machine cannot be deleted',
    );
    assert.ok(!button(rows[1], 'Delete the machine').disabled);
    button(rows[1], 'Copy the machine address to clipboard').click();
    await new Promise(r => setTimeout(r, 0));
    eq(copied, ['ssh me@gpu']);
    posted.length = 0;
    button(rows[1], 'Delete the machine').click();
    rows[1].querySelector('.sidebar-confirm-yes').click();
    eq(posted, [{type: 'machineAction', action: 'remove', name: 'gpu-box'}]);
    send(win, {
      type: 'machines',
      machines: MACHINES,
      checkedAt: 2,
      error: "no machine named 'gpu-box'",
    });
    const status = el(win, 'meta-machines-status');
    assert.strictEqual(status.textContent, "no machine named 'gpu-box'");
    assert.ok(status.classList.contains('is-error'));
    send(win, {type: 'machines', machines: MACHINES, checkedAt: 3});
    assert.strictEqual(status.textContent, '');
    assert.ok(!status.classList.contains('is-error'));
  });

  await test('Machines: the header + opens an in-place form that registers a machine', () => {
    const {win, posted} = makePage();
    send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
    const list = el(win, 'meta-machines-list');
    const addBtn = el(win, 'meta-machines-add');
    assert.strictEqual(addBtn.getAttribute('aria-label'), 'Add machine');
    assert.ok(
      addBtn.closest('.meta-section-hdr'),
      'the + sits in the Machines header',
    );
    const topForm = () =>
      list.firstElementChild.classList.contains('machine-add-row')
        ? list.firstElementChild.querySelector('form.sidebar-inline-form')
        : null;
    addBtn.click();
    let form = topForm();
    assert.ok(form, 'the form opens at the top of the list');
    assert.strictEqual(list.children.length, 4, 'above the three rows');
    eq(
      Array.from(form.querySelectorAll('.sidebar-inline-form-label')).map(
        l => l.textContent,
      ),
      ['Name', 'Address', 'Password'],
    );
    assert.strictEqual(form.elements.password.type, 'password');
    assert.strictEqual(win.document.activeElement, form.elements.name);
    // One form at a time: a second press replaces it.
    form.elements.name.value = 'typed';
    addBtn.click();
    assert.strictEqual(list.querySelectorAll('form').length, 1);
    form = topForm();
    assert.strictEqual(form.elements.name.value, '');
    form.querySelector('.sidebar-confirm-no').click();
    assert.strictEqual(list.querySelectorAll('form').length, 0);
    assert.strictEqual(list.children.length, 3);
    assert.strictEqual(win.document.activeElement, addBtn);
    addBtn.click();
    form = topForm();
    form.elements.name.value = 'lab';
    form.elements.address.value = 'https://lab.example.org:8787';
    form.elements.password.value = 'pw';
    posted.length = 0;
    form.elements.address.dispatchEvent(
      new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
    );
    const addCmd = {
      type: 'machineAction',
      action: 'add',
      name: 'lab',
      address: 'https://lab.example.org:8787',
      machinePassword: 'pw',
    };
    eq(posted, [addCmd]);
    // The add is in flight: a repeated Enter sends nothing more.
    assert.ok(form.querySelector('.sidebar-inline-form-save').disabled);
    form.elements.address.dispatchEvent(
      new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
    );
    eq(posted, [addCmd]);
    // A refused add re-opens the form, at the top, with the typed
    // values; the periodic poll keeps it (with the draft); the reply
    // acknowledging the retried add (the machine registered) closes
    // it, while one acknowledging nothing this form sent (another
    // window's add) leaves it alone.
    send(win, {
      type: 'machines',
      machines: MACHINES,
      checkedAt: 2,
      action: 'add',
      error: 'lab: unreachable',
    });
    form = topForm();
    assert.ok(form, 'the form is back at the top');
    assert.strictEqual(
      form.elements.address.value,
      'https://lab.example.org:8787',
    );
    assert.strictEqual(
      el(win, 'meta-machines-status').textContent,
      'lab: unreachable',
    );
    form.elements.name.value = 'lab2';
    send(win, {type: 'machines', machines: MACHINES.slice(0, 2), checkedAt: 3});
    form = topForm();
    assert.ok(form, 'a poll keeps the form open');
    assert.strictEqual(form.elements.name.value, 'lab2', 'with the draft');
    send(win, {
      type: 'machines',
      machines: MACHINES,
      checkedAt: 4,
      action: 'add',
      error: '',
    });
    form = topForm();
    assert.ok(form, 'an add this form did not send leaves it open');
    posted.length = 0;
    form.querySelector('.sidebar-inline-form-save').click();
    assert.strictEqual(posted[0].name, 'lab2', 'the retry sends the draft');
    send(win, {
      type: 'machines',
      machines: MACHINES,
      checkedAt: 5,
      action: 'add',
      error: '',
    });
    assert.strictEqual(list.querySelectorAll('form').length, 0);
  });

  await test('Machines: a machine added since the last reply brings the subpanel on screen', () => {
    // Desktop: the section expands and the docked panel slides back
    // (it is slid off while a content tab holds the pane).
    const {win} = makePage();
    const section = el(win, 'meta-machines');
    send(win, {type: 'machines', machines: MACHINES.slice(0, 1), checkedAt: 1});
    section.querySelector('.meta-section-toggle').click();
    assert.ok(section.classList.contains('collapsed'), 'folded by the user');
    send(win, {
      type: 'tabs_state',
      tabs: [{tabId: 'a1', chatId: 'chat-a1', title: 'a1', workDir: '/ws'}],
    });
    send(win, {
      type: 'fileContent',
      tabId: 'a1',
      path: '/ws/notes.md',
      name: 'notes.md',
      content: '# Notes',
    });
    assert.ok(win.document.body.classList.contains('content-pane-open'));
    assert.ok(win.document.body.classList.contains('meta-hidden'), 'slid off');
    // A poll naming the same machines changes nothing.
    send(win, {type: 'machines', machines: MACHINES.slice(0, 1), checkedAt: 2});
    assert.ok(section.classList.contains('collapsed'));
    assert.ok(win.document.body.classList.contains('meta-hidden'));
    send(win, {type: 'machines', machines: MACHINES, checkedAt: 3});
    assert.ok(!section.classList.contains('collapsed'), 'expanded');
    assert.ok(!win.document.body.classList.contains('meta-hidden'), 'docked');
    assert.ok(!el(win, 'meta-panel').classList.contains('open'), 'no drawer');

    // Phone: the task-info drawer opens.
    const phone = makePage({desktop: false}).win;
    const panel = el(phone, 'meta-panel');
    send(phone, {
      type: 'machines',
      machines: MACHINES.slice(0, 1),
      checkedAt: 1,
    });
    assert.ok(!panel.classList.contains('open'), 'the drawer starts closed');
    send(phone, {type: 'machines', machines: MACHINES, checkedAt: 2});
    assert.ok(panel.classList.contains('open'), 'the drawer opened');
    assert.ok(!panel.hasAttribute('inert'));
    assert.strictEqual(
      el(phone, 'meta-drawer-btn').getAttribute('aria-expanded'),
      'true',
    );
  });

  await test('Machines: the chat header lists the reachable ones and a pick is per tab', () => {
    const {win, posted} = makePage();
    const select = el(win, 'chat-machine');
    send(win, {type: 'configData', config: {}, machine: 'laptop'});
    send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
    eq(
      Array.from(select.options).map(o => o.value),
      ['', 'gpu-box'],
      'this machine, then the reachable registered ones',
    );
    const pick = value => {
      select.value = value;
      select.dispatchEvent(new win.Event('change', {bubbles: true}));
    };
    pick('gpu-box');
    assert.ok(select.classList.contains('remote'));
    pick('');
    assert.ok(!select.classList.contains('remote'));
    posted.length = 0;
    select.dispatchEvent(new win.Event('focus'));
    eq(
      types(posted),
      ['getMachines'],
      'reaching for the list asks for a fresh one',
    );
  });

  await test('Apps: connected first, status line, buttons for the rest, failure hint', () => {
    const {win} = makePage();
    send(win, {type: 'appsStatus', apps: APPS, checkedAt: 1790000000000});
    assert.strictEqual(
      el(win, 'meta-apps-status').textContent,
      '2 of 4 connected',
    );
    const rows = appRows(win);
    eq(
      rows.map(r => r.dataset.app),
      ['brave', 'gmail', 'matrix', 'slack'],
    );
    const main0 = rows[0].querySelector('.app-row-main');
    assert.strictEqual(main0.tagName, 'DIV');
    assert.strictEqual(
      main0.querySelector('.app-state').textContent,
      'Connected',
    );
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
    assert.strictEqual(
      slack.querySelector('.app-state').textContent,
      'Connect',
    );
    assert.strictEqual(slack.title, 'Click to connect Slack in a new task');

    send(win, {type: 'appsStatus', apps: 'bogus', checkedAt: 0});
    assert.strictEqual(appRows(win).length, 0);
    assert.match(
      el(win, 'meta-apps-status').textContent,
      /^Could not check the apps/,
    );
  });

  await test('Apps: a Connect row is disabled while its task runs; a second click launches nothing', () => {
    // Audit 2026-10-18: the row stayed an enabled "Connect" after the
    // click, so a second click (or one on the row the next probe
    // repainted) opened a second connect chat for the same app.
    const {win, posted} = makePage();
    send(win, {type: 'appsStatus', apps: APPS, checkedAt: 1});
    const slackRow = () =>
      win.document.querySelector('.app-row[data-app="slack"] button');
    const submits = () => posted.filter(m => m.type === 'submit');
    slackRow().click();
    assert.strictEqual(
      submits().length,
      1,
      'the first click opens the connect chat',
    );
    assert.ok(slackRow().disabled, 'the row is disabled while the task runs');
    assert.strictEqual(
      slackRow().querySelector('.app-state').textContent,
      'Connecting\u2026',
    );
    assert.strictEqual(slackRow().title, 'A task is connecting Slack');
    // A scripted activation of the disabled row (a `click()` is a
    // no-op on it) launches nothing either.
    slackRow().dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
    assert.strictEqual(
      submits().length,
      1,
      'REGRESSION: a second click launched another task',
    );
    // The next probe repaints the row still waiting ...
    send(win, {type: 'appsStatus', apps: APPS, checkedAt: 2});
    assert.ok(slackRow().disabled);
    assert.strictEqual(
      slackRow().querySelector('.app-state').textContent,
      'Connecting\u2026',
    );
    slackRow().click();
    assert.strictEqual(submits().length, 1);
    // ... and the other apps' rows are untouched.
    const matrix = win.document.querySelector(
      '.app-row[data-app="matrix"] button',
    );
    assert.ok(!matrix.disabled);
    assert.strictEqual(
      matrix.querySelector('.app-state').textContent,
      'Connect',
    );
    // Slack found connected ends the wait: its row is the plain state.
    const connected = APPS.map(a =>
      a.name === 'slack' ? Object.assign({}, a, {authenticated: true}) : a,
    );
    send(win, {type: 'appsStatus', apps: connected, checkedAt: 3});
    assert.strictEqual(slackRow(), null, 'a connected app has no button');
    assert.strictEqual(
      win.document.querySelector('.app-row[data-app="slack"] .app-state')
        .textContent,
      'Connected',
    );
    // Disconnected again later (a revoked token): Connect works again.
    send(win, {type: 'appsStatus', apps: APPS, checkedAt: 4});
    assert.ok(!slackRow().disabled);
    slackRow().click();
    assert.strictEqual(submits().length, 2);
  });

  await test('Apps: a Connect click while the daemon is away launches nothing and Connect works after the reconnect', () => {
    // Review 2026-10-18: the click marked the app as connecting although
    // sendMessage holds prompts back while the daemon is away, so no
    // task existed and the row stayed "Connecting…" until the 30 min
    // wait lapsed; it also left the prompt in a new chat's composer.
    const {win, posted} = makePage();
    send(win, {type: 'appsStatus', apps: APPS, checkedAt: 1});
    const slackRow = () =>
      win.document.querySelector('.app-row[data-app="slack"] button');
    const submits = () => posted.filter(m => m.type === 'submit');
    send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
    const activeTab = win._testApi.getActiveTabId();
    slackRow().click();
    assert.strictEqual(
      submits().length,
      0,
      'nothing is submitted while the daemon is away',
    );
    assert.ok(
      !slackRow().disabled,
      'REGRESSION: the row says Connecting without a task',
    );
    assert.strictEqual(
      slackRow().querySelector('.app-state').textContent,
      'Connect',
    );
    assert.strictEqual(
      win._testApi.getActiveTabId(),
      activeTab,
      'no chat is opened',
    );
    assert.strictEqual(
      el(win, 'task-input').value,
      '',
      'the composer is untouched',
    );
    // The reconnect's probe repaints the row enabled, and Connect works.
    send(win, {type: 'daemonStatus', connected: true});
    send(win, {type: 'appsStatus', apps: APPS, checkedAt: 2});
    assert.ok(!slackRow().disabled);
    slackRow().click();
    assert.strictEqual(
      submits().length,
      1,
      'REGRESSION: Connect stayed dead after the reconnect',
    );
    assert.ok(slackRow().disabled);
  });

  await test('refresh buttons re-request; the apps one spins until the reply', () => {
    const {win, posted} = makePage();
    posted.length = 0;
    el(win, 'meta-schedule-refresh').click();
    eq(posted, [{type: 'getCronJobs'}]);
    posted.length = 0;
    const btn = el(win, 'meta-apps-refresh');
    btn.click();
    eq(posted, [
      {type: 'getCronJobs'},
      {type: 'getMachines'},
      {type: 'getAppsStatus', refresh: true},
      {type: 'getSpendReport'},
      {type: 'getRecommendations'},
    ]);
    assert.ok(btn.disabled && btn.classList.contains('spinning'));
    posted.length = 0;
    el(win, 'meta-spend-refresh').click();
    eq(posted, [{type: 'getSpendReport'}]);
    posted.length = 0;
    const recBtn = el(win, 'meta-recommendations-refresh');
    recBtn.click();
    eq(posted, [{type: 'getRecommendations'}]);
    assert.ok(recBtn.disabled && recBtn.classList.contains('spinning'));
    send(win, {type: 'recommendations', generatedAt: 0, tasks: []});
    assert.ok(!recBtn.disabled && !recBtn.classList.contains('spinning'));
    recBtn.click();
    send(win, {type: 'daemonStatus', connected: false});
    assert.ok(
      !recBtn.disabled,
      'a dropped connection stops the recommendations spin',
    );
    // While disconnected a click neither sends (nothing would answer)
    // nor spins.
    posted.length = 0;
    const machinesBtn = el(win, 'meta-machines-refresh');
    machinesBtn.click();
    recBtn.click();
    btn.click();
    eq(posted, []);
    assert.ok(!machinesBtn.disabled && !recBtn.disabled && !btn.disabled);
    send(win, {type: 'daemonStatus', connected: true});
    posted.length = 0;
    machinesBtn.click();
    eq(posted, [{type: 'getMachines', refresh: true}]);
    assert.ok(
      machinesBtn.disabled && machinesBtn.classList.contains('spinning'),
    );
    send(win, {type: 'appsStatus', apps: APPS, checkedAt: 1});
    assert.ok(!btn.disabled && !btn.classList.contains('spinning'));
    assert.ok(
      machinesBtn.disabled,
      'an apps reply does not stop the machines spin',
    );
    send(win, {type: 'machines', machines: [], checkedAt: 1});
    assert.ok(
      !machinesBtn.disabled && !machinesBtn.classList.contains('spinning'),
    );
    // A dropped connection never answers: the buttons recover.
    btn.click();
    machinesBtn.click();
    send(win, {type: 'daemonStatus', connected: false});
    assert.ok(!btn.disabled && !btn.classList.contains('spinning'));
    assert.ok(
      !machinesBtn.disabled && !machinesBtn.classList.contains('spinning'),
    );
  });

  await test('a daemon (re)connect requests the five subpanels again', () => {
    const {win, posted} = makePage();
    posted.length = 0;
    send(win, {type: 'daemonStatus', connected: true});
    for (const cmd of PANEL_COMMANDS)
      assert.ok(types(posted).includes(cmd), cmd);
  });

  await test('the subpanels are requested once per connection: a repeated connected status asks nothing more', () => {
    // Review 2026-10-18: a password-protected transport announces
    // `daemonStatus connected` on `auth_required` (to reveal the page)
    // and again on `auth_ok`; each announcement requested every
    // subpanel, so a password-required startup asked twice.
    const {win, posted} = makePage();
    const panelPosts = () =>
      posted.filter(m => PANEL_COMMANDS.includes(m.type));
    send(win, {type: 'daemonStatus', connected: true});
    assert.strictEqual(
      panelPosts().length,
      PANEL_COMMANDS.length,
      'one request each on the first announcement',
    );
    send(win, {type: 'daemonStatus', connected: true});
    assert.strictEqual(
      panelPosts().length,
      PANEL_COMMANDS.length,
      'REGRESSION: the auth_ok announcement asked for every subpanel again',
    );
    // An outage and the reconnect still refresh them.
    send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
    send(win, {type: 'daemonStatus', connected: true});
    assert.strictEqual(
      panelPosts().length,
      2 * PANEL_COMMANDS.length,
      'the reconnect asks once',
    );
    eq(types(panelPosts()), PANEL_COMMANDS.concat(PANEL_COMMANDS));
  });

  await test('Spend: an empty history draws the all-time line and a blank 98-day grid', () => {
    const {win, intervals} = makePage();
    send(win, {
      type: 'spendReport',
      total: 'bogus',
      days: 'bogus',
      totalByModel: 'bogus',
    });
    assert.strictEqual(
      intervals.filter(i => i.ms === 30000).length,
      1,
      'poll started',
    );
    assert.strictEqual(
      el(win, 'meta-spend-status').textContent,
      'No spend recorded yet.',
    );
    assert.strictEqual(
      win.document.querySelector('.spend-total').textContent,
      'All time \u00b7 $0.00 \u00b7 0 tok \u00b7 0 tasks',
    );
    const dated = cells(win).filter(c => c.dataset.spendDate);
    assert.strictEqual(dated.length, 98);
    assert.strictEqual(dated[97].dataset.spendDate, dayKey(0));
    assert.strictEqual(dated[0].dataset.spendDate, dayKey(97));
    assert.ok(
      !dated.some(c => /\bl[1-4]\b/.test(c.className)),
      'no shaded cell',
    );
    assert.strictEqual(
      win.document.querySelector('.spend-models'),
      null,
      'no model bars',
    );
    // Every column is a Monday-first week: the first one is padded
    // with blanks up to the first day's weekday.
    const cols = Array.from(win.document.querySelectorAll('.spend-col'));
    const [y, m, d] = dayKey(97).split('-').map(Number);
    const offset = (new Date(y, m - 1, d).getDay() + 6) % 7;
    assert.strictEqual(
      cols[0].querySelectorAll('.spend-cell.blank').length,
      offset,
    );
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
    const {win} = makePage();
    send(win, spendReport());
    const graph = el(win, 'meta-spend-graph');
    const tip = graph.querySelector('.spend-tip');
    const cell = cellOf(win, dayKey(0));
    // jsdom has no layout: the graph is 200 x 180, the tooltip 120 x 90.
    graph.getBoundingClientRect = () => ({
      left: 0,
      top: 0,
      right: 200,
      bottom: 180,
      width: 200,
      height: 180,
    });
    Object.defineProperty(tip, 'offsetWidth', {configurable: true, value: 120});
    Object.defineProperty(tip, 'offsetHeight', {configurable: true, value: 90});
    const cellAt = top => {
      cell.getBoundingClientRect = () => ({
        left: 100,
        top,
        right: 111,
        bottom: top + 11,
        width: 11,
        height: 11,
      });
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
    assert.strictEqual(
      cellAt(82),
      180 - 90 + 'px',
      'above covers 8px, below covers 3px',
    );
  });

  await test('Spend: the heatmap reaches back to the oldest day, shaded by cost, with model bars', () => {
    const {win} = makePage();
    const report = spendReport();
    send(win, report);
    assert.strictEqual(el(win, 'meta-spend-status').textContent, '');
    assert.strictEqual(
      win.document.querySelector('.spend-total').textContent,
      'All time \u00b7 $2.04K \u00b7 1.23M tok \u00b7 6 tasks',
    );
    const dated = cells(win).filter(c => c.dataset.spendDate);
    assert.strictEqual(
      dated.length,
      201,
      '200 days back to the oldest day, plus today',
    );
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
    eq(tipText(win), [
      dayLabel(dayKey(0)) + ' \u00b7 $37.92 \u00b7 235K tok \u00b7 1 task',
    ]);
    hover(win, rows[1].querySelector('.spend-model-name'));
    eq(tipText(win), [
      'gpt-6-astra \u00b7 $37.92 \u00b7 235K tok \u00b7 1 task \u00b7 2%',
    ]);
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
      win.document
        .querySelector('.spend-nav.' + dir)
        .classList.contains('nav-hidden');
    assert.ok(!hidden('left') && hidden('right'));
    viewport.scrollLeft = 0;
    viewport.dispatchEvent(new win.Event('scroll'));
    assert.ok(hidden('left') && hidden('right'));
    send(win, report);
    assert.strictEqual(
      win.document.querySelectorAll('.spend-total').length,
      1,
      'redrawn, not appended',
    );
  });

  await test('Spend: a wider panel redraws with more week columns; a narrower one keeps the view', () => {
    const {win, observers} = makePage();
    const graph = el(win, 'meta-spend-graph');
    const resize = observers.find(o => o.target === graph);
    assert.ok(resize, 'the graph is observed');
    resize.fn();
    assert.strictEqual(
      win.document.querySelector('.spend-grid'),
      null,
      'nothing drawn yet',
    );
    send(win, spendReport());
    const columns = () => win.document.querySelectorAll('.spend-col').length;
    const before = columns();
    resize.fn();
    assert.strictEqual(columns(), before, 'zero width: same grid');
    // 1000px fits 71 columns of 11px cells with 3px gaps: the grid
    // grows to fill them (the oldest day is only 29 weeks back).
    Object.defineProperty(graph, 'clientWidth', {
      value: 1000,
      configurable: true,
    });
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
    const {win} = makePage();
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
    const {win} = makePage();
    const report = spendReport();
    report.days.unshift(
      sum({date: '2009-02-13', cost: 0, tokens: 0, tasks: 2}),
    );
    send(win, report);
    const dated = cells(win).filter(c => c.dataset.spendDate);
    assert.strictEqual(dated.length, 3 * 366);
    assert.strictEqual(dated[0].dataset.spendDate, dayKey(3 * 366 - 1));
    assert.strictEqual(dated[dated.length - 1].dataset.spendDate, dayKey(0));
  });

  await test('Spend: a history that cost nothing still shows its day and its model', () => {
    const {win} = makePage();
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

  await test('clicking an app that is not connected submits a connect task in a new tab', () => {
    const {win, posted, intervals} = makePage();
    const inp = el(win, 'task-input');
    inp.value = 'my draft';
    inp.dispatchEvent(new win.Event('input', {bubbles: true}));
    el(win, 'meta-panel').classList.add('open');
    send(win, {type: 'appsStatus', apps: APPS, checkedAt: 1});
    const chatTabs = () =>
      win._testApi.openTabs().filter(t => !t.isContentTab && !t.isSubagentTab);
    const firstTab = win._testApi.getActiveTabId();
    eq(
      chatTabs().map(t => t.id),
      [firstTab],
    );
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
    // The task runs in a fresh chat.  The connect prompt replaces the
    // composer text, so the draft stays with the chat left behind,
    // which therefore stays open (an idle chat without a draft would
    // be retired); switching back shows the draft again.
    const connectTab = win._testApi.getActiveTabId();
    assert.notStrictEqual(connectTab, firstTab, 'a new chat is on screen');
    assert.strictEqual(
      submit.tabId,
      connectTab,
      'the submit names the new chat',
    );
    eq(
      chatTabs().map(t => t.id),
      [firstTab, connectTab],
    );
    assert.ok(
      !posted.some(m => m.type === 'closeTab' && m.tabId === firstTab),
      'the chat holding the draft is kept',
    );
    assert.ok(
      !el(win, 'meta-panel').classList.contains('open'),
      'drawer closed',
    );
    win._testApi.switchToTab(firstTab);
    assert.strictEqual(inp.value, 'my draft', 'the draft survives');
    win._testApi.switchToTab(connectTab);

    // While Slack waits for its connect task, every poll re-probes;
    // once a reply shows it connected, polls use the cache again.
    send(win, {type: 'cronJobs', jobs: []});
    const poll = intervals.find(i => i.ms === 30000);
    posted.length = 0;
    poll.fn();
    eq(
      posted.filter(m => m.type === 'getAppsStatus'),
      [{type: 'getAppsStatus', refresh: true}],
    );
    const connected = APPS.map(a =>
      a.name === 'slack' ? Object.assign({}, a, {authenticated: true}) : a,
    );
    send(win, {type: 'appsStatus', apps: connected, checkedAt: 1});
    posted.length = 0;
    poll.fn();
    eq(
      posted.filter(m => m.type === 'getAppsStatus'),
      [{type: 'getAppsStatus'}],
    );

    // A connect task that never completes stops forcing re-probes
    // after 30 minutes.
    win.document.querySelector('.app-row[data-app="matrix"] button').click();
    const realNow = win.Date.now;
    win.Date.now = () => realNow() + 31 * 60000;
    posted.length = 0;
    poll.fn();
    win.Date.now = realNow;
    eq(
      posted.filter(m => m.type === 'getAppsStatus'),
      [{type: 'getAppsStatus'}],
    );
  });

  await test('Recommendations: the empty state shows a hint; rows show the sentences and when they were suggested', () => {
    const {win} = makePage();
    const status = () => el(win, 'meta-recommendations-status').textContent;
    send(win, {type: 'recommendations', generatedAt: 0, tasks: []});
    assert.match(status(), /^No recommendations yet: \/recommendations now/);
    assert.strictEqual(recommendationRows(win).length, 0);
    // A malformed reply (no list, blank entries) is the empty state too.
    send(win, {type: 'recommendations', tasks: 'bogus'});
    assert.match(status(), /^No recommendations yet/);
    send(win, {
      type: 'recommendations',
      generatedAt: 1,
      tasks: ['', '  ', null],
    });
    assert.match(status(), /^No recommendations yet/);

    const generatedAt = Date.now() - 23 * 60000;
    send(win, {
      type: 'recommendations',
      generatedAt,
      tasks: [' ' + FIVE_TASKS[0] + ' ', ...FIVE_TASKS.slice(1)],
    });
    assert.strictEqual(status(), 'Suggested 23m ago');
    const rows = recommendationRows(win);
    eq(
      rows.map(r => r.textContent),
      FIVE_TASKS,
    );
    eq(
      rows.map(r => r.dataset.task),
      FIVE_TASKS,
    );
    for (const row of rows) {
      assert.strictEqual(row.tabIndex, 0);
      assert.strictEqual(row.dataset.tooltip, 'Open a new chat with this task');
      assert.ok(row.querySelector('.sidebar-panel-name'));
    }
    // A reply without a time shows the rows and no status line.
    send(win, {type: 'recommendations', tasks: FIVE_TASKS.slice(0, 2)});
    assert.strictEqual(status(), '');
    assert.strictEqual(recommendationRows(win).length, 2);
    // A later empty reply clears the rows again.
    send(win, {type: 'recommendations', generatedAt: 0, tasks: []});
    assert.strictEqual(recommendationRows(win).length, 0);
  });

  await test('activating a recommendation opens a new chat with the task in the composer, unsent', () => {
    const {win, posted} = makePage();
    const inp = el(win, 'task-input');
    inp.value = 'my draft';
    inp.dispatchEvent(new win.Event('input', {bubbles: true}));
    el(win, 'meta-panel').classList.add('open');
    send(win, {
      type: 'recommendations',
      generatedAt: Date.now(),
      tasks: FIVE_TASKS,
    });
    const chatTabs = () =>
      win._testApi.openTabs().filter(t => !t.isContentTab && !t.isSubagentTab);
    const firstTab = win._testApi.getActiveTabId();
    const list = el(win, 'meta-recommendations-list');
    posted.length = 0;
    // Other keys, and a click on the list outside a row, do nothing.
    recommendationRows(win)[1].dispatchEvent(
      new win.KeyboardEvent('keydown', {key: 'a', bubbles: true}),
    );
    list.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
    assert.strictEqual(win._testApi.getActiveTabId(), firstTab);
    eq(
      chatTabs().map(t => t.id),
      [firstTab],
    );

    recommendationRows(win)[1].click();
    const newTab = win._testApi.getActiveTabId();
    assert.notStrictEqual(newTab, firstTab, 'a new chat is on screen');
    assert.strictEqual(inp.value, FIVE_TASKS[1], 'the task is in the composer');
    assert.ok(!posted.some(m => m.type === 'submit'), 'nothing is submitted');
    eq(
      chatTabs().map(t => t.id),
      [firstTab, newTab],
    );
    assert.ok(
      !el(win, 'meta-panel').classList.contains('open'),
      'drawer closed',
    );
    win._testApi.switchToTab(firstTab);
    assert.strictEqual(inp.value, 'my draft', 'the draft survives');
    win._testApi.switchToTab(newTab);

    // Enter and Space on a focused row do the same.
    posted.length = 0;
    recommendationRows(win)[2].dispatchEvent(
      new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
    );
    assert.strictEqual(inp.value, FIVE_TASKS[2]);
    assert.notStrictEqual(win._testApi.getActiveTabId(), newTab);
    recommendationRows(win)[3].dispatchEvent(
      new win.KeyboardEvent('keydown', {key: ' ', bubbles: true}),
    );
    assert.strictEqual(inp.value, FIVE_TASKS[3]);
    assert.ok(
      !posted.some(m => m.type === 'submit'),
      'still nothing submitted',
    );
  });

  await test('Machines: a transcript file link is checked on the machine the daemon bound the tab to', () => {
    const {win, posted} = makePage();
    send(win, {type: 'daemonStatus', connected: true});
    send(win, {
      type: 'tabs_state',
      tabs: [
        {tabId: 'gpu1', chatId: 'chat-gpu1', title: 'gpu', workDir: '/gpu/ws'},
        {tabId: 'home1', chatId: 'chat-home1', title: 'home', workDir: '/ws'},
      ],
    });
    send(win, {type: 'tabMachine', tabId: 'gpu1', machine: 'gpu-box'});
    posted.length = 0;
    send(win, {type: 'prompt', text: 'see src/train.py please', tabId: 'gpu1'});
    send(win, {type: 'text_end', tabId: 'gpu1'});
    const checks = posted.filter(m => m.type === 'checkPaths');
    eq(checks, [
      {
        type: 'checkPaths',
        paths: ['src/train.py'],
        workDir: '/gpu/ws',
        tabId: 'gpu1',
        machine: 'gpu-box',
      },
    ]);
    const span = win.document.querySelector(
      '#output [data-path-candidate="src/train.py"]',
    );
    assert.strictEqual(span.getAttribute('data-path-machine'), 'gpu-box');

    // A tab the daemon never bound checks on this daemon: no machine.
    posted.length = 0;
    send(win, {
      type: 'prompt',
      text: 'see src/local.py please',
      tabId: 'home1',
    });
    send(win, {type: 'text_end', tabId: 'home1'});
    const home = posted.filter(m => m.type === 'checkPaths');
    assert.strictEqual(home.length, 1);
    assert.strictEqual(home[0].machine, undefined, 'no machine stamped');
    assert.strictEqual(home[0].tabId, 'home1');
  });

  // ---- The Schedule, Apps, Spend and Recommendations subpanels show
  // the machine the chat header's drop-down has picked ----

  /** The subpanel commands posted since *from*, as {type, machine}. */
  function panelRequests(posted, from) {
    return posted
      .slice(from)
      .filter(m => PANEL_COMMANDS.includes(m.type))
      .map(m => ({type: m.type, machine: m.machine}));
  }

  /** A page whose drop-down lists gpu-box, with every subpanel loaded for this daemon. */
  function pageShowingThisMachine(opts) {
    const {win, posted, intervals} = makePage(opts);
    send(win, {type: 'daemonStatus', connected: true});
    send(win, {type: 'configData', config: {}, machine: 'laptop'});
    send(win, {type: 'machines', machines: MACHINES, checkedAt: 1});
    send(win, {
      type: 'cronJobs',
      jobs: [job({id: 'home-job', name: 'Home job'})],
    });
    send(win, {type: 'appsStatus', apps: APPS, checkedAt: 1});
    send(win, {type: 'spendReport', ...spendReport()});
    send(win, {
      type: 'recommendations',
      generatedAt: Date.now(),
      tasks: FIVE_TASKS,
    });
    const pick = value => {
      const select = el(win, 'chat-machine');
      select.value = value;
      select.dispatchEvent(new win.Event('change', {bubbles: true}));
    };
    return {win, posted, intervals, pick};
  }

  const jobNames = win =>
    Array.from(
      win.document.querySelectorAll(
        '#meta-schedule-list .sched-row .sidebar-panel-name',
      ),
    ).map(n => n.textContent);

  await test('Machines: picking a machine empties the four subpanels and asks for its data', () => {
    const {win, posted, pick} = pageShowingThisMachine();
    eq(jobNames(win), ['Home job']);
    assert.strictEqual(appRows(win).length, APPS.length);
    assert.strictEqual(recommendationRows(win).length, FIVE_TASKS.length);
    const from = posted.length;
    pick('gpu-box');
    // The previous machine's rows are gone while gpu-box answers.
    eq(jobNames(win), []);
    assert.strictEqual(appRows(win).length, 0);
    assert.strictEqual(recommendationRows(win).length, 0);
    assert.strictEqual(el(win, 'meta-spend-graph').childElementCount, 0);
    assert.strictEqual(
      el(win, 'meta-schedule-status').textContent,
      'Loading\u2026',
    );
    assert.strictEqual(
      el(win, 'meta-apps-status').textContent,
      'Checking\u2026',
    );
    assert.strictEqual(
      el(win, 'meta-spend-status').textContent,
      'Loading\u2026',
    );
    assert.strictEqual(
      el(win, 'meta-recommendations-status').textContent,
      'Loading\u2026',
    );
    // Four commands stamped with the machine; the Machines list is
    // this daemon's registry and is not.
    eq(panelRequests(posted, from), [
      {type: 'getCronJobs', machine: 'gpu-box'},
      {type: 'getMachines'},
      {type: 'getAppsStatus', machine: 'gpu-box'},
      {type: 'getSpendReport', machine: 'gpu-box'},
      {type: 'getRecommendations', machine: 'gpu-box'},
    ]);
    // Picking the same machine again asks nothing more.
    const again = posted.length;
    pick('gpu-box');
    eq(panelRequests(posted, again), []);
  });

  await test("Machines: only the shown machine's replies are painted", () => {
    const {win, pick} = pageShowingThisMachine();
    pick('gpu-box');
    // This daemon's own reply (a poll answered late, a push) is dropped.
    send(win, {
      type: 'cronJobs',
      jobs: [job({id: 'home-job', name: 'Home job'})],
    });
    eq(jobNames(win), []);
    assert.strictEqual(
      el(win, 'meta-schedule-status').textContent,
      'Loading\u2026',
    );
    // Another machine's forwarded reply is dropped too.
    send(win, {
      type: 'cronJobs',
      forwarded: true,
      machine: 'office',
      jobs: [job({name: 'Office job'})],
    });
    eq(jobNames(win), []);
    // gpu-box's reply is the one shown.
    send(win, {
      type: 'cronJobs',
      forwarded: true,
      machine: 'gpu-box',
      jobs: [job({id: 'g1', name: 'GPU job'})],
    });
    eq(jobNames(win), ['GPU job']);
    send(win, {type: 'appsStatus', apps: APPS, checkedAt: 2});
    assert.strictEqual(
      appRows(win).length,
      0,
      "this daemon's apps are not gpu-box's",
    );
    send(win, {
      type: 'appsStatus',
      forwarded: true,
      machine: 'gpu-box',
      apps: APPS.slice(0, 2),
      checkedAt: 2,
    });
    assert.strictEqual(appRows(win).length, 2);
    send(win, {
      type: 'recommendations',
      generatedAt: Date.now(),
      tasks: FIVE_TASKS,
    });
    assert.strictEqual(recommendationRows(win).length, 0);
    send(win, {
      type: 'recommendations',
      forwarded: true,
      machine: 'gpu-box',
      generatedAt: Date.now(),
      tasks: FIVE_TASKS.slice(0, 1),
    });
    assert.strictEqual(recommendationRows(win).length, 1);
    send(win, {type: 'spendReport', ...spendReport()});
    assert.strictEqual(el(win, 'meta-spend-graph').childElementCount, 0);
    send(win, {
      type: 'spendReport',
      forwarded: true,
      machine: 'gpu-box',
      ...spendReport(),
    });
    assert.ok(
      el(win, 'meta-spend-graph').querySelector('.spend-total'),
      "gpu-box's spend is drawn",
    );
    // Back on this machine: the forwarded replies are the ones dropped.
    pick('');
    eq(jobNames(win), []);
    send(win, {
      type: 'cronJobs',
      forwarded: true,
      machine: 'gpu-box',
      jobs: [job({id: 'g1', name: 'GPU job'})],
    });
    eq(jobNames(win), []);
    send(win, {
      type: 'cronJobs',
      jobs: [job({id: 'home-job', name: 'Home job'})],
    });
    eq(jobNames(win), ['Home job']);
  });

  await test('Machines: the refresh buttons, the poll and the Schedule actions go to the shown machine', () => {
    const {win, posted, intervals, pick} = pageShowingThisMachine();
    pick('gpu-box');
    send(win, {
      type: 'cronJobs',
      forwarded: true,
      machine: 'gpu-box',
      jobs: [job({id: 'g1', name: 'GPU job'})],
    });
    let from = posted.length;
    el(win, 'meta-schedule-refresh').click();
    el(win, 'meta-apps-refresh').click();
    el(win, 'meta-spend-refresh').click();
    el(win, 'meta-recommendations-refresh').click();
    eq(panelRequests(posted, from), [
      {type: 'getCronJobs', machine: 'gpu-box'},
      {type: 'getCronJobs', machine: 'gpu-box'},
      {type: 'getMachines'},
      {type: 'getAppsStatus', machine: 'gpu-box'},
      {type: 'getSpendReport', machine: 'gpu-box'},
      {type: 'getRecommendations', machine: 'gpu-box'},
      {type: 'getSpendReport', machine: 'gpu-box'},
      {type: 'getRecommendations', machine: 'gpu-box'},
    ]);
    from = posted.length;
    intervals[0].fn();
    assert.ok(
      panelRequests(posted, from).every(
        m => m.type === 'getMachines' || m.machine === 'gpu-box',
      ),
      'the poll asks gpu-box',
    );
    from = posted.length;
    win.document.querySelector('#meta-schedule-list .sched-play').click();
    win.document.querySelector('#meta-schedule-list .sched-toggle').click();
    eq(
      posted.slice(from).filter(m => m.type === 'cronAction'),
      [
        {
          type: 'cronAction',
          action: 'run_now',
          jobId: 'g1',
          machine: 'gpu-box',
        },
        {type: 'cronAction', action: 'pause', jobId: 'g1', machine: 'gpu-box'},
      ],
    );
    // The add form creates the job on the machine.
    from = posted.length;
    el(win, 'meta-schedule-add').click();
    const form = win.document.querySelector(
      '#meta-schedule-list .sidebar-inline-form',
    );
    form.elements.name.value = 'Nightly';
    form.elements.schedule.value = 'every 1d';
    form.elements.what.value = 'clean up';
    form.dispatchEvent(
      new win.Event('submit', {bubbles: true, cancelable: true}),
    );
    const create = posted.slice(from).find(m => m.type === 'cronAction');
    assert.strictEqual(create.action, 'create');
    assert.strictEqual(create.machine, 'gpu-box');
    assert.strictEqual(create.name, 'Nightly');
  });

  await test('Machines: a machine that did not answer is reported on each subpanel', () => {
    const {win, pick} = pageShowingThisMachine();
    pick('gpu-box');
    const error = 'gpu-box did not answer';
    send(win, {
      type: 'cronJobs',
      forwarded: true,
      machine: 'gpu-box',
      jobs: [],
      error,
    });
    send(win, {
      type: 'appsStatus',
      forwarded: true,
      machine: 'gpu-box',
      apps: [],
      checkedAt: 0,
      error,
    });
    send(win, {
      type: 'spendReport',
      forwarded: true,
      machine: 'gpu-box',
      error,
    });
    send(win, {
      type: 'recommendations',
      forwarded: true,
      machine: 'gpu-box',
      generatedAt: 0,
      tasks: [],
      error,
    });
    for (const id of [
      'meta-schedule-status',
      'meta-apps-status',
      'meta-spend-status',
      'meta-recommendations-status',
    ]) {
      assert.strictEqual(el(win, id).textContent, error, id);
      assert.ok(el(win, id).classList.contains('is-error'), id + ' is red');
    }
    // Once the machine answers, the lines are plain again.
    send(win, {
      type: 'appsStatus',
      forwarded: true,
      machine: 'gpu-box',
      apps: APPS,
      checkedAt: 2,
    });
    assert.ok(!el(win, 'meta-apps-status').classList.contains('is-error'));
    send(win, {
      type: 'spendReport',
      forwarded: true,
      machine: 'gpu-box',
      ...spendReport(),
    });
    assert.ok(!el(win, 'meta-spend-status').classList.contains('is-error'));
    send(win, {
      type: 'recommendations',
      forwarded: true,
      machine: 'gpu-box',
      generatedAt: Date.now(),
      tasks: FIVE_TASKS,
    });
    assert.ok(
      !el(win, 'meta-recommendations-status').classList.contains('is-error'),
    );
  });

  await test("Machines: an app connected from a machine's subpanel, or its recommendation, opens a chat on that machine", () => {
    const {win, posted, pick} = pageShowingThisMachine();
    el(win, 'meta-panel').classList.add('open');
    pick('gpu-box');
    send(win, {
      type: 'appsStatus',
      forwarded: true,
      machine: 'gpu-box',
      apps: APPS,
      checkedAt: 2,
    });
    send(win, {
      type: 'recommendations',
      forwarded: true,
      machine: 'gpu-box',
      generatedAt: Date.now(),
      tasks: FIVE_TASKS,
    });
    const firstTab = win._testApi.getActiveTabId();
    let from = posted.length;
    win.document.querySelector('.app-row[data-app="slack"] button').click();
    const submit = posted.slice(from).find(m => m.type === 'submit');
    assert.strictEqual(
      submit.machine,
      'gpu-box',
      'the connect task runs on gpu-box',
    );
    assert.notStrictEqual(win._testApi.getActiveTabId(), firstTab);
    assert.strictEqual(
      el(win, 'chat-machine').value,
      'gpu-box',
      'the new chat shows gpu-box picked',
    );
    // The subpanels still show gpu-box: nothing was asked again.
    eq(panelRequests(posted, from), []);
    from = posted.length;
    recommendationRows(win)[0].click();
    assert.strictEqual(
      posted.slice(from).find(m => m.type === 'submit'),
      undefined,
      'a recommendation is not sent',
    );
    assert.strictEqual(el(win, 'chat-machine').value, 'gpu-box');
    assert.strictEqual(el(win, 'task-input').value, FIVE_TASKS[0]);
    eq(panelRequests(posted, from), []);
  });

  await test('Machines: a machine change while the daemon is away empties the subpanels; the reconnect asks once', () => {
    const {win, posted, pick} = pageShowingThisMachine();
    pick('gpu-box');
    send(win, {
      type: 'cronJobs',
      forwarded: true,
      machine: 'gpu-box',
      jobs: [job({id: 'g1', name: 'GPU job'})],
    });
    send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
    // Back to this machine during the outage (a new chat follows the
    // global pick, so the user picks it): gpu-box's rows go, but no
    // panel request is posted (the shim would queue it and send it on
    // top of the reconnect's own request).
    let from = posted.length;
    win._testApi.createNewTab();
    assert.strictEqual(
      el(win, 'chat-machine').value,
      'gpu-box',
      'a new chat follows the global pick',
    );
    eq(jobNames(win), ['GPU job']);
    pick('');
    eq(jobNames(win), []);
    assert.strictEqual(
      el(win, 'meta-schedule-status').textContent,
      'Loading\u2026',
    );
    eq(panelRequests(posted, from), []);
    from = posted.length;
    send(win, {type: 'daemonStatus', connected: true});
    eq(panelRequests(posted, from), [
      {type: 'getCronJobs'},
      {type: 'getMachines'},
      {type: 'getAppsStatus'},
      {type: 'getSpendReport'},
      {type: 'getRecommendations'},
    ]);
    // gpu-box's late reply is not this machine's.
    send(win, {
      type: 'cronJobs',
      forwarded: true,
      machine: 'gpu-box',
      jobs: [job({id: 'g1', name: 'GPU job'})],
    });
    eq(jobNames(win), []);
    send(win, {
      type: 'cronJobs',
      jobs: [job({id: 'home-job', name: 'Home job'})],
    });
    eq(jobNames(win), ['Home job']);
  });

  await test("Machines: a tab switch and the daemon's binding move the subpanels too", () => {
    const {win, posted} = pageShowingThisMachine();
    const home = win._testApi.getActiveTabId();
    // The chat ran a task on gpu-box (the daemon's binding): the
    // subpanels show gpu-box.
    send(win, {type: 'tabMachine', tabId: home, machine: 'gpu-box'});
    send(win, {
      type: 'cronJobs',
      forwarded: true,
      machine: 'gpu-box',
      jobs: [job({id: 'g1', name: 'GPU job'})],
    });
    // A new chat (the global pick: this machine) brings the panels home ...
    let from = posted.length;
    win._testApi.createNewTab();
    eq(jobNames(win), []);
    assert.ok(
      panelRequests(posted, from).some(
        m => m.type === 'getCronJobs' && m.machine === undefined,
      ),
    );
    send(win, {
      type: 'cronJobs',
      jobs: [job({id: 'home-job', name: 'Home job'})],
    });
    eq(jobNames(win), ['Home job']);
    // ... and switching back to the gpu-box chat asks gpu-box again.
    from = posted.length;
    win._testApi.switchToTab(home);
    eq(jobNames(win), []);
    assert.ok(
      panelRequests(posted, from).some(
        m => m.type === 'getCronJobs' && m.machine === 'gpu-box',
      ),
    );
    // The global pick changing on another surface moves a new chat's
    // panels (a chat that shows a task keeps its own machine).
    const blank = win._testApi.openTabs().find(t => t.id !== home).id;
    from = posted.length;
    send(win, {type: 'machineChanged', machine: 'gpu-box'});
    eq(panelRequests(posted, from), [], 'the gpu-box chat is unmoved');
    win._testApi.switchToTab(blank);
    assert.strictEqual(el(win, 'chat-machine').value, 'gpu-box');
    send(win, {type: 'machineChanged', machine: ''});
    assert.strictEqual(el(win, 'chat-machine').value, '');
    assert.ok(
      panelRequests(posted, from).some(
        m => m.type === 'getCronJobs' && m.machine === undefined,
      ),
    );
    // The daemon binding the active tab elsewhere (another surface's
    // submit of a chat with no pick of its own here) moves them as well.
    const fresh = win._testApi.openTabs().find(t => t.id !== home).id;
    win._testApi.switchToTab(fresh);
    from = posted.length;
    send(win, {type: 'tabMachine', tabId: fresh, machine: 'office'});
    assert.ok(
      panelRequests(posted, from).some(
        m => m.type === 'getCronJobs' && m.machine === 'office',
      ),
    );
    assert.strictEqual(el(win, 'chat-machine').value, 'office');
  });

  await test('Machines: a machine change keeps a Schedule add draft; a row editor closes', () => {
    // Audit 2026-10-18: a change the user did not make (the daemon
    // binding the tab elsewhere, the 30 s poll finding the header on
    // another machine) threw away a job being typed into the add form.
    const {win, posted, pick} = pageShowingThisMachine();
    const list = el(win, 'meta-schedule-list');
    const topForm = () =>
      list.firstElementChild &&
      list.firstElementChild.classList.contains('sched-add-row')
        ? list.firstElementChild.querySelector('form.sidebar-inline-form')
        : null;
    const active = win._testApi.getActiveTabId();
    el(win, 'meta-schedule-add').click();
    let form = topForm();
    form.elements.name.value = 'Nightly';
    form.elements.schedule.value = 'every 1d';
    form.elements.kind.value = 'command';
    form.elements.what.value = 'echo night';
    // The daemon binds the tab to office (a submit on another surface).
    send(win, {type: 'tabMachine', tabId: active, machine: 'office'});
    eq(jobNames(win), [], 'the subpanel is emptied');
    send(win, {
      type: 'cronJobs',
      forwarded: true,
      machine: 'office',
      jobs: [job({id: 'o1', name: 'Office job'})],
    });
    form = topForm();
    assert.ok(form, 'REGRESSION: the add form is gone with the draft');
    eq(
      [
        form.elements.name.value,
        form.elements.schedule.value,
        form.elements.kind.value,
        form.elements.what.value,
      ],
      ['Nightly', 'every 1d', 'command', 'echo night'],
      'the typed values are back',
    );
    assert.ok(!form.querySelector('.sidebar-inline-form-save').disabled);
    // Saved now, the job is created on office, where the form shows.
    posted.length = 0;
    form.querySelector('.sidebar-inline-form-save').click();
    eq(
      posted.map(m => ({
        type: m.type,
        action: m.action,
        machine: m.machine,
        name: m.name,
      })),
      [
        {
          type: 'cronAction',
          action: 'create',
          machine: 'office',
          name: 'Nightly',
        },
      ],
    );
    assert.ok(
      form.querySelector('.sidebar-inline-form-save').disabled,
      'the create is in flight',
    );
    // A new chat (this machine) while that create is in flight:
    // office's reply is dropped here, so Save works again once home
    // answers, with the draft; a second change before any reply
    // (back to the office chat) keeps it as well.
    win._testApi.createNewTab();
    assert.strictEqual(
      topForm(),
      null,
      'nothing is painted while the machine answers',
    );
    win._testApi.switchToTab(active);
    send(win, {
      type: 'cronJobs',
      forwarded: true,
      machine: 'office',
      jobs: [],
      action: 'create',
    });
    form = topForm();
    assert.strictEqual(form.elements.name.value, 'Nightly');
    assert.ok(
      !form.querySelector('.sidebar-inline-form-save').disabled,
      'Save works again',
    );
    // Cancel drops the draft for good: the next change re-opens nothing.
    form.querySelector('.sidebar-confirm-no').click();
    const fresh = win._testApi.openTabs().find(t => t.id !== active).id;
    win._testApi.switchToTab(fresh);
    send(win, {
      type: 'cronJobs',
      jobs: [job({id: 'home-job', name: 'Home job'})],
    });
    assert.strictEqual(topForm(), null);
    // A user's pick keeps the draft too (the job may well be meant for
    // the machine just picked); a row's editor edits a job of the
    // machine left behind and closes.
    el(win, 'meta-schedule-add').click();
    topForm().elements.name.value = 'Picked';
    pick('gpu-box');
    send(win, {
      type: 'cronJobs',
      forwarded: true,
      machine: 'gpu-box',
      jobs: [job({id: 'g1', name: 'GPU job'})],
    });
    assert.strictEqual(topForm().elements.name.value, 'Picked');
    topForm().querySelector('.sidebar-confirm-no').click();
    list.children[0].querySelector('[aria-label="Edit the job"]').click();
    assert.ok(
      list.querySelector('form.sidebar-inline-form'),
      'the editor is open',
    );
    win._testApi.switchToTab(active);
    send(win, {
      type: 'cronJobs',
      forwarded: true,
      machine: 'office',
      jobs: [job({id: 'o1', name: 'Office job'})],
    });
    assert.strictEqual(
      list.querySelector('form.sidebar-inline-form'),
      null,
      'no editor on office',
    );
    eq(jobNames(win), ['Office job']);
  });

  await test('Machines: a connect waiting on one machine does not re-probe another, and only its machine ends the wait', () => {
    const {win, posted, intervals, pick} = pageShowingThisMachine();
    el(win, 'meta-panel').classList.add('open');
    const home = win._testApi.getActiveTabId();
    const poll = intervals.find(i => i.ms === 30000);
    pick('gpu-box');
    send(win, {
      type: 'appsStatus',
      forwarded: true,
      machine: 'gpu-box',
      apps: APPS,
      checkedAt: 2,
    });
    // A draft keeps the home chat open when the connect chat opens.
    el(win, 'task-input').value = 'my draft';
    el(win, 'task-input').dispatchEvent(
      new win.Event('input', {bubbles: true}),
    );
    win.document.querySelector('.app-row[data-app="slack"] button').click();
    // The connect chat (on gpu-box) is shown: its polls re-probe gpu-box.
    const connectTab = win._testApi.getActiveTabId();
    let from = posted.length;
    poll.fn();
    eq(
      posted.slice(from).filter(m => m.type === 'getAppsStatus'),
      [{type: 'getAppsStatus', refresh: true, machine: 'gpu-box'}],
    );
    // Back on the home chat, brought home: this daemon is not
    // re-probed for gpu-box's wait, and this daemon's connected Slack
    // does not end it.
    win._testApi.switchToTab(home);
    pick('');
    const connected = APPS.map(a =>
      a.name === 'slack' ? Object.assign({}, a, {authenticated: true}) : a,
    );
    send(win, {type: 'appsStatus', apps: connected, checkedAt: 3});
    from = posted.length;
    poll.fn();
    eq(
      posted.slice(from).filter(m => m.type === 'getAppsStatus'),
      [{type: 'getAppsStatus'}],
    );
    win._testApi.switchToTab(connectTab);
    from = posted.length;
    poll.fn();
    eq(
      posted.slice(from).filter(m => m.type === 'getAppsStatus'),
      [{type: 'getAppsStatus', refresh: true, machine: 'gpu-box'}],
    );
    // gpu-box reporting Slack connected ends the wait.
    send(win, {
      type: 'appsStatus',
      forwarded: true,
      machine: 'gpu-box',
      apps: connected,
      checkedAt: 4,
    });
    from = posted.length;
    poll.fn();
    eq(
      posted.slice(from).filter(m => m.type === 'getAppsStatus'),
      [{type: 'getAppsStatus', machine: 'gpu-box'}],
    );
  });

  await test('Machines: a poll that finds the header on another machine empties the subpanels first', () => {
    // The owner chat of the shown report closed on another surface:
    // the side views fall back to the report's own machine with no
    // refresh of the panels; the next poll must not leave this
    // daemon's rows actionable against that machine.  (The stacked
    // layout: there a content tab outlives its owner chat.)
    const {win, posted, intervals, pick} = pageShowingThisMachine({
      desktop: false,
    });
    el(win, 'meta-panel').classList.add('open');
    const poll = intervals.find(i => i.ms === 30000);
    const home = win._testApi.getActiveTabId();
    pick('gpu-box');
    send(win, {
      type: 'machineWorkDirs',
      machine: 'gpu-box',
      workDir: '/srv/proj',
      recentWorkDirs: [],
      error: '',
    });
    send(win, {
      type: 'tabs_state',
      tabs: [{tabId: home, chatId: 'c-home', title: 'home', workDir: '/ws'}],
    });
    // A file of the gpu-box chat opens as a content tab owned by it.
    send(win, {
      type: 'fileContent',
      tabId: home,
      machine: 'gpu-box',
      path: '/srv/proj/notes.txt',
      name: 'notes.txt',
      content: 'hi\n',
      version: 'v1',
    });
    const report = win._testApi.openTabs().find(t => t.isContentTab);
    assert.ok(report, 'the file opened as a content tab');
    // Its owner is brought home while the report is shown ...
    win._testApi.switchToTab(home);
    pick('');
    send(win, {
      type: 'cronJobs',
      jobs: [job({id: 'local-only', name: 'Home job'})],
    });
    eq(jobNames(win), ['Home job']);
    win._testApi.switchToTab(report.id);
    eq(
      jobNames(win),
      ['Home job'],
      "a content tab keeps its owner chat's machine",
    );
    // ... then closed elsewhere: the report keeps the machine it was
    // opened on, and the next poll targets gpu-box.
    send(win, {type: 'tabs_state', tabs: []});
    assert.strictEqual(
      win._testApi.getActiveTabId(),
      report.id,
      'the report survives',
    );
    const from = posted.length;
    poll.fn();
    const asked = panelRequests(posted, from).find(
      m => m.type === 'getCronJobs',
    );
    assert.strictEqual(
      asked.machine,
      'gpu-box',
      "the poll asks the file's machine",
    );
    eq(jobNames(win), [], "this daemon's rows are gone before gpu-box answers");
    assert.strictEqual(
      el(win, 'meta-schedule-status').textContent,
      'Loading\u2026',
    );
    send(win, {
      type: 'cronJobs',
      forwarded: true,
      machine: 'gpu-box',
      jobs: [job({id: 'g1', name: 'GPU job'})],
    });
    eq(jobNames(win), ['GPU job']);
  });

  report('sidebarPanels');
  process.exit(0);
}

main();
