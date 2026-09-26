// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the right sidebar's Schedule and Apps subpanels
// (the sidebarpanels block of media/main.js), on every surface that
// shows the panel (remote webapp, sidebar-chat drawer, editor-tabs
// Task Info view) and the ones that must stay quiet (chat editor
// panels, the history panel):
//
// * boot and a daemon (re)connect request getCronJobs + getAppsStatus;
//   the poll timer starts on the first reply, once, and skips hidden
//   pages;
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
  return {win, posted, intervals};
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

async function main() {
  for (const [label, attrs] of [
    ['remote webapp', REMOTE],
    ['sidebar-chat drawer', SIDEBAR],
    ['Task Info view', META_VIEW],
  ]) {
    await test(`${label}: boot requests both subpanels; the poll starts on the first reply`, () => {
      const {win, posted, intervals} = makeWebview(attrs);
      const boot = posted.filter(
        m => m.type === 'getCronJobs' || m.type === 'getAppsStatus',
      );
      eq(boot, [
        {type: 'getCronJobs'},
        {type: 'getAppsStatus'},
      ]);
      assert.strictEqual(el(win, 'meta-apps-status').textContent, 'Checking\u2026');
      const polls = () => intervals.filter(i => i.ms === 30000);
      assert.strictEqual(polls().length, 0, 'no poll before the daemon answers');
      send(win, {type: 'cronJobs', jobs: []});
      send(win, {type: 'cronJobs', jobs: []});
      assert.strictEqual(polls().length, 1, 'one poll timer, started once');
      posted.length = 0;
      polls()[0].fn();
      eq(types(posted), ['getCronJobs', 'getAppsStatus']);
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
    assert.strictEqual(rows[0].title, 'do it\nLast run: ok');
    assert.strictEqual(badge(rows[0]), '');
    assert.strictEqual(badge(rows[1]), 'running');
    assert.strictEqual(sub(rows[1]), `every 5m \u00b7 next ${pt(times.b)} (in 3h)`);
    assert.strictEqual(rows[1].title, '$ rsync -a src dst');
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
    ]);
    assert.ok(btn.disabled && btn.classList.contains('spinning'));
    send(win, {type: 'appsStatus', apps: APPS, checkedAt: 1});
    assert.ok(!btn.disabled && !btn.classList.contains('spinning'));
    // A dropped connection never answers: the button recovers.
    btn.click();
    send(win, {type: 'daemonStatus', connected: false});
    assert.ok(!btn.disabled && !btn.classList.contains('spinning'));
  });

  await test('a daemon (re)connect requests both subpanels again', () => {
    const {win, posted} = makeWebview(REMOTE);
    posted.length = 0;
    send(win, {type: 'daemonStatus', connected: true});
    assert.ok(types(posted).includes('getCronJobs'));
    assert.ok(types(posted).includes('getAppsStatus'));
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
      assert.ok(!types(posted).includes('getCronJobs'));
      assert.ok(!types(posted).includes('getAppsStatus'));
      send(win, {type: 'cronJobs', jobs: [job({})]});
      send(win, {type: 'appsStatus', apps: APPS, checkedAt: 1});
      assert.strictEqual(el(win, 'meta-schedule-list').children.length, 0);
      assert.strictEqual(appRows(win).length, 0);
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
