// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the mobile remote webapp's task-info drawer
// (media/chat.html + media/main.js, remote mode below 900px):
//
// * #meta-drawer-btn (the tab bar's right edge) opens the #meta-panel
//   drawer with its #meta-overlay backdrop and flips aria-expanded;
//   the close button and the backdrop both dismiss it;
// * the 5s getTaskUpdate poll runs only while the drawer is OPEN (and a
//   task runs): opening polls immediately, closing stops the timer; the
//   taskUpdate reply renders in the drawer's #meta-info subpanel;
// * the subpanel's refresh button posts a getTaskUpdate with
//   refresh:true and shows the agent's running state (disabled,
//   spinning, "Updating…") until a reply with running:false arrives;
// * the status-bar mirror still fills the drawer's #meta-list rows on
//   mobile (the hidden #tab-status-bar keeps updating underneath);
// * switching to the desktop layout (matchMedia flip) closes the
//   drawer state so the docked panel never sits over a live backdrop.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const POLL_MS = 5000;

let passed = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.stack || e.message}`);
  }
}

/**
 * Replace the window's setInterval/clearInterval with a fake clock so
 * the 5s task-update poll can be advanced deterministically instead of
 * waited out. setTimeout stays real (message handlers debounce with it).
 *
 * @param {Window} win The jsdom window, before main.js is evaluated.
 * @returns {{tick: function(number), intervals: Array}} `tick(ms)`
 *   advances the clock and fires every due interval callback;
 *   `intervals` lists the live intervals as {cb, ms, next}.
 */
function installFakeIntervals(win) {
  let now = 0;
  let nextId = 1;
  const intervals = [];
  win.setInterval = function (cb, ms) {
    const iv = {id: nextId++, cb, ms: Number(ms) || 0, next: now + (Number(ms) || 0)};
    intervals.push(iv);
    return iv.id;
  };
  win.clearInterval = function (id) {
    const i = intervals.findIndex(iv => iv.id === id);
    if (i >= 0) intervals.splice(i, 1);
  };
  function tick(ms) {
    const target = now + ms;
    for (;;) {
      const due = intervals.filter(iv => iv.next <= target);
      if (due.length === 0) break;
      due.sort((a, b) => a.next - b.next);
      const iv = due[0];
      now = iv.next;
      iv.next += iv.ms;
      iv.cb();
    }
    now = target;
  }
  return {tick, intervals};
}

function makeWebview(opts) {
  const {desktopMatches = false} = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  html = html.replace('<body', '<body class="remote-chat"');
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
  const mqlListeners = [];
  const mql = {
    matches: desktopMatches,
    media: '(min-width: 900px)',
    addEventListener: (_type, cb) => mqlListeners.push(cb),
    removeEventListener: () => {},
    addListener: cb => mqlListeners.push(cb),
    removeListener: () => {},
  };
  win.matchMedia = query =>
    query === '(min-width: 900px)'
      ? mql
      : {
          matches: false,
          media: query,
          addEventListener: () => {},
          removeEventListener: () => {},
          addListener: () => {},
          removeListener: () => {},
        };
  const timers = installFakeIntervals(win);
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=metadrawer-main.js',
  );
  const setDesktop = matches => {
    mql.matches = matches;
    for (const cb of mqlListeners.slice()) cb({matches});
  };
  return {win, posted, setDesktop, tick: timers.tick, intervals: timers.intervals};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function click(win, id) {
  win.document
    .getElementById(id)
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function drawerOpen(win) {
  return win.document.getElementById('meta-panel').classList.contains('open');
}

function overlayOpen(win) {
  return win.document
    .getElementById('meta-overlay')
    .classList.contains('open');
}

function polls(wv) {
  return wv.posted.filter(m => m.type === 'getTaskUpdate');
}

function pollCount(wv) {
  return polls(wv).length;
}

function lastPoll(wv) {
  const all = polls(wv);
  return all[all.length - 1];
}

/** A taskUpdate reply answering *poll*, with the given overrides. */
function reply(win, poll, fields) {
  send(
    win,
    Object.assign(
      {
        type: 'taskUpdate',
        tabId: poll.tabId,
        token: poll.token,
        taskId: 'task-1',
        exists: true,
        sig: '',
        content: '',
        error: '',
        running: false,
        cost: 0,
        updatedAt: 0,
      },
      fields,
    ),
  );
}

/** Open the drawer over a running task; returns the webview. */
function openRunningDrawer() {
  const wv = makeWebview();
  send(wv.win, {
    type: 'configData',
    config: {work_dir: '/cfg/dir', max_budget: 42},
    apiKeys: {},
  });
  send(wv.win, {type: 'status', running: true});
  click(wv.win, 'meta-drawer-btn');
  return wv;
}

async function main() {
  await test(
    'the tab-bar button toggles the drawer; close button and backdrop ' +
      'dismiss it',
    () => {
      const wv = makeWebview();
      const win = wv.win;
      const btn = win.document.getElementById('meta-drawer-btn');
      assert.ok(btn, 'the toggle exists in the tab bar');
      assert.strictEqual(
        btn.parentElement.id,
        'tab-bar',
        'the toggle rides the tab bar, not the composer footer',
      );
      assert.strictEqual(btn.getAttribute('aria-expanded'), 'false');
      assert.ok(!drawerOpen(win));

      click(win, 'meta-drawer-btn');
      assert.ok(drawerOpen(win), 'button opens the drawer');
      assert.ok(overlayOpen(win), 'the backdrop opens with the drawer');
      assert.strictEqual(btn.getAttribute('aria-expanded'), 'true');

      click(win, 'meta-close');
      assert.ok(!drawerOpen(win), 'the close button dismisses the drawer');
      assert.ok(!overlayOpen(win));
      assert.strictEqual(btn.getAttribute('aria-expanded'), 'false');

      click(win, 'meta-drawer-btn');
      assert.ok(drawerOpen(win));
      click(win, 'meta-overlay');
      assert.ok(!drawerOpen(win), 'the backdrop dismisses the drawer');
      assert.ok(!overlayOpen(win));
    },
  );

  await test(
    'the drawer mirrors the (hidden) status-bar values on mobile',
    async () => {
      const wv = makeWebview();
      const win = wv.win;
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
        machine: 'phone-host',
      });
      send(win, {type: 'usage_info', total_tokens: 2000, cost: 0.25});
      await sleep(50);
      const text = id => win.document.getElementById(id).textContent;
      assert.strictEqual(text('meta-tokens'), '2.00K');
      assert.strictEqual(text('meta-cost'), '$0.25');
      assert.strictEqual(text('meta-machine'), 'phone-host');
      assert.strictEqual(text('meta-workdir'), '/cfg/dir');
      assert.strictEqual(text('meta-max-budget'), '$42.00');
    },
  );

  await test(
    'getTaskUpdate polls every 5s only while the drawer is open, and ' +
      'the reply renders in the drawer',
    () => {
      const wv = makeWebview();
      const win = wv.win;
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
      send(win, {type: 'status', running: true});
      wv.tick(POLL_MS * 2);
      assert.strictEqual(
        pollCount(wv),
        0,
        'a closed drawer must not poll (mobile network)',
      );

      click(win, 'meta-drawer-btn');
      assert.strictEqual(pollCount(wv), 1, 'opening the drawer polls at once');
      assert.ok(
        wv.intervals.some(iv => iv.ms === POLL_MS),
        'the poll timer ticks every 5000 ms',
      );
      const poll = lastPoll(wv);
      assert.deepStrictEqual(Object.keys(poll).sort(), [
        'knownSig',
        'refresh',
        'tabId',
        'token',
        'type',
      ]);
      assert.ok(!('workDir' in poll), 'the poll names a tab, not a workdir');
      assert.strictEqual(typeof poll.tabId, 'string');
      assert.ok(poll.tabId, 'the poll targets the visible chat tab');
      assert.strictEqual(poll.knownSig, '');
      assert.strictEqual(typeof poll.token, 'string');
      assert.strictEqual(poll.refresh, false, 'a timer poll is not a refresh');

      wv.tick(POLL_MS - 1);
      assert.strictEqual(pollCount(wv), 1, 'no poll before the interval');
      wv.tick(1);
      assert.strictEqual(pollCount(wv), 2, 'the timer polls at 5000 ms');

      reply(win, poll, {
        sig: '5:9',
        content: 'working on **it**\n',
        updatedAt: Date.now(),
        cost: 0.12,
      });
      const info = win.document.getElementById('meta-info');
      assert.ok(info.classList.contains('visible'));
      assert.ok(
        win.document
          .getElementById('meta-info-content')
          .innerHTML.includes('<strong>it</strong>'),
        'markdown content goes through marked',
      );
      const status = win.document.getElementById('meta-info-status').textContent;
      assert.ok(/^Updated \d/.test(status), `status names the time: ${status}`);
      assert.ok(status.endsWith(' \u00b7 $0.12'), `status names the cost: ${status}`);

      wv.tick(POLL_MS);
      assert.strictEqual(
        lastPoll(wv).knownSig,
        '5:9',
        'the next poll carries the reply\'s sig',
      );

      reply(win, lastPoll(wv), {unchanged: true, sig: 'ignored'});
      assert.ok(
        win.document
          .getElementById('meta-info-content')
          .innerHTML.includes('<strong>it</strong>'),
        'an unchanged reply leaves the report alone',
      );

      reply(win, lastPoll(wv), {
        sig: '6:0',
        content:
          '  <p>Report <b>html</b></p><img src=x onerror="alert(1)">' +
          '<script>alert(2)</script>',
        updatedAt: Date.now(),
      });
      const html = win.document.getElementById('meta-info-content').innerHTML;
      assert.ok(html.includes('<b>html</b>'), 'HTML content renders as-is');
      assert.ok(!html.includes('<script'), 'scripts are stripped');
      assert.ok(!html.includes('onerror'), 'event handlers are stripped');

      reply(win, lastPoll(wv), {exists: false});
      assert.ok(!info.classList.contains('visible'), 'exists:false hides it');
      assert.strictEqual(
        win.document.getElementById('meta-info-content').innerHTML,
        '',
      );

      click(win, 'meta-close');
      const count = pollCount(wv);
      wv.tick(POLL_MS * 3);
      assert.strictEqual(
        pollCount(wv),
        count,
        'closing the drawer stops the poll',
      );
      assert.ok(
        !wv.intervals.some(iv => iv.ms === POLL_MS),
        'the poll timer is cleared',
      );
    },
  );

  await test(
    'the refresh button posts getTaskUpdate refresh:true and mirrors ' +
      'the agent\'s running state',
    () => {
      const wv = openRunningDrawer();
      const win = wv.win;
      const btn = win.document.getElementById('meta-info-refresh');
      const status = win.document.getElementById('meta-info-status');
      assert.ok(btn, 'the refresh button sits in the subpanel header');
      assert.strictEqual(
        btn.parentElement.parentElement.id,
        'meta-info',
        'the button belongs to the task-update subpanel',
      );
      const before = pollCount(wv);
      click(win, 'meta-info-refresh');
      assert.strictEqual(pollCount(wv), before + 1, 'the click polls at once');
      const refresh = lastPoll(wv);
      assert.strictEqual(refresh.type, 'getTaskUpdate');
      assert.strictEqual(refresh.refresh, true, 'the click asks for a run');
      assert.ok(!('workDir' in refresh));

      reply(win, refresh, {sig: 'r1', running: true});
      assert.strictEqual(btn.disabled, true, 'no second run while one runs');
      assert.ok(btn.classList.contains('spinning'));
      assert.strictEqual(status.textContent, 'Updating\u2026');
      assert.ok(
        win.document.getElementById('meta-info').classList.contains('visible'),
        'the subpanel shows while the first run is pending',
      );
      assert.ok(
        win.document
          .getElementById('meta-info-content')
          .innerHTML.includes('Preparing the first update'),
      );

      wv.tick(POLL_MS);
      assert.strictEqual(
        lastPoll(wv).refresh,
        false,
        'the timer poll after the click is a plain poll',
      );
      reply(win, lastPoll(wv), {
        sig: 'r2',
        running: false,
        content: '<p>Half <em>done</em></p>',
        updatedAt: Date.now(),
        cost: 0.05,
      });
      assert.strictEqual(btn.disabled, false, 'the button is usable again');
      assert.ok(!btn.classList.contains('spinning'));
      assert.ok(/^Updated \d/.test(status.textContent), status.textContent);
      assert.ok(status.textContent.endsWith(' \u00b7 $0.05'));
      assert.ok(
        win.document
          .getElementById('meta-info-content')
          .innerHTML.includes('<em>done</em>'),
      );

      reply(win, lastPoll(wv), {
        sig: 'r3',
        running: false,
        content: '<p>Half <em>done</em></p>',
        error: 'budget exceeded',
        updatedAt: Date.now(),
      });
      assert.ok(
        status.textContent.endsWith(' \u00b7 Last run failed: budget exceeded'),
        status.textContent,
      );
      assert.strictEqual(btn.disabled, false);

      send(win, {type: 'status', running: false});
      assert.ok(
        !win.document.getElementById('meta-info').classList.contains('visible'),
        'the task ending empties the subpanel',
      );
      assert.strictEqual(status.textContent, '');
      assert.strictEqual(btn.disabled, false);
    },
  );

  await test('switching to the desktop layout closes the drawer', () => {
    const wv = makeWebview();
    const win = wv.win;
    click(win, 'meta-drawer-btn');
    assert.ok(drawerOpen(win) && overlayOpen(win));
    wv.setDesktop(true);
    assert.ok(
      win.document.body.classList.contains('remote-desktop'),
      'the desktop class follows the media query',
    );
    assert.ok(!drawerOpen(win), 'desktop docking drops the drawer state');
    assert.ok(!overlayOpen(win), 'the backdrop goes with it');
  });

  await test(
    'desktop remote keeps polling without the drawer (docked panel)',
    () => {
      const wv = makeWebview({desktopMatches: true});
      const win = wv.win;
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
      send(win, {type: 'status', running: true});
      assert.ok(
        pollCount(wv) >= 1,
        'the docked desktop panel polls with no drawer involved',
      );
      const count = pollCount(wv);
      wv.tick(POLL_MS);
      assert.strictEqual(pollCount(wv), count + 1, 'and keeps polling');
      assert.strictEqual(lastPoll(wv).type, 'getTaskUpdate');
      assert.ok(!('workDir' in lastPoll(wv)));
    },
  );

  console.log(`\n${passed} passed, ${failures.length} failed`);
  process.exit(failures.length > 0 ? 1 : 0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
