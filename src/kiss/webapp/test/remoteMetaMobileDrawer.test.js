// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the mobile remote webapp's task-info drawer
// (media/chat.html + media/main.js, remote mode below 900px):
//
// * the status-bar mirror still fills the drawer's #meta-list rows on
//   mobile (the hidden #tab-status-bar keeps updating underneath);
// * switching to the desktop layout (matchMedia flip) closes the
//   drawer state so the docked panel never sits over a live backdrop;
// * a desktop-sized remote page polls getTaskUpdate with no drawer.
//
// The drawer open/close flows, the drawer-gated 5s poll and the refresh
// button are covered by sidebarChatModeDrawerAndComposer.test.js.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {waitFor} = require('./simplify2_harness');

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

function makePage(opts) {
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
  win.acquireTransport = function () {
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

async function main() {
  await test(
    'the drawer mirrors the (hidden) status-bar values on mobile',
    async () => {
      const wv = makePage();
      const win = wv.win;
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
        machine: 'phone-host',
      });
      send(win, {type: 'usage_info', total_tokens: 2000, cost: 0.25});
      const text = id => win.document.getElementById(id).textContent;
      // main.js mirrors the status-bar values into the panel through a
      // MutationObserver, which is delivered as a microtask.
      await waitFor(
        () => text('meta-tokens') === '2.00K' && text('meta-cost') === '$0.25',
        'the drawer never mirrored the status-bar tokens and cost',
      );
      assert.strictEqual(text('meta-tokens'), '2.00K');
      assert.strictEqual(text('meta-cost'), '$0.25');
      // The machine name stays in the status bar: no Machine row.
      assert.strictEqual(text('status-machine'), 'phone-host');
      assert.strictEqual(win.document.getElementById('meta-machine'), null);
      assert.strictEqual(text('meta-workdir'), '/cfg/dir');
      assert.strictEqual(text('meta-max-budget'), '$42.00');
    },
  );

  await test('switching to the desktop layout closes the drawer', () => {
    const wv = makePage();
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
      const wv = makePage({desktopMatches: true});
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
