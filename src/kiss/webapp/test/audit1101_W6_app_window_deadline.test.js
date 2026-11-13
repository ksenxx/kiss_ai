// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E regression for audit 2026-11-01 W6 F1: media/appWindow.html asks
// the daemon /api/app-window whether the popup is open before it opens
// (or shows) the webapp, and its documented fallback is to take a daemon
// that does not answer for one without a popup.  The request had no
// deadline: a daemon (or proxy) that accepted the connection but never
// sent its headers, or sent them and then stalled the JSON body, kept
// this empty page waiting for the browser's own network timeout
// (minutes) with no window on screen.  The sibling appStart.html bounds
// its probe with AbortSignal.timeout(2000), and the Python side asks the
// same question with a 2 s timeout (standalone_app.ask_app_window_open).
//
// The page's real inline script runs against a real loopback HTTP
// server with Node's native fetch; only the window surface it touches
// (location, window.open/close, localStorage) is a recording stand-in.
// The two stalled cases hand the page an AbortSignal the test aborts
// itself (recording the deadline the page asked for) once the daemon
// has received the request, or the page the headers: the page's
// handling of the deadline is checked at that boundary, not against
// the clock, so a scheduling pause cannot fail a correct page (round 2,
// F6).  Both stalled cases failed against the pre-fix page (no action
// after the abort); the answered cases guard the existing behaviour
// under a deadline that never expires, since a pause of 2 s between the
// request and its answer would otherwise make a correct page take the
// fallback (WT5 F2).

const assert = require('assert');
const fs = require('fs');
const http = require('http');
const path = require('path');
const vm = require('vm');
const {makeRunner} = require('./ui_antipattern_harness');
const {MEDIA, waitFor} = require('./simplify2_harness');

const {test, report} = makeRunner();

const SCRIPT = fs
  .readFileSync(path.join(MEDIA, 'appWindow.html'), 'utf8')
  .match(/<script>([\s\S]*?)<\/script>/)[1];

/** The longest any step of these tests waits for an event (a watchdog, not an envelope). */
const WATCHDOG_MS = 10000;

/** A loopback daemon stand-in whose /api/app-window `answer` decides. */
async function startDaemon(answer) {
  const sockets = new Set();
  const server = http.createServer((req, res) => {
    answer(req, res);
  });
  server.on('connection', s => {
    sockets.add(s);
    s.on('close', () => sockets.delete(s));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = 'http://127.0.0.1:' + server.address().port;
  return {
    base,
    async close() {
      for (const s of sockets) s.destroy();
      await new Promise(resolve => server.close(resolve));
    },
  };
}

/**
 * The deadline under the test's control: `AbortSignal.timeout(ms)` as
 * the page sees it hands out `signal` and records `ms` in `deadlines`;
 * `abort()` is the deadline expiring.  `headers` resolves once the
 * page's fetch received the daemon's response headers.
 */
function abortControl() {
  const controller = new AbortController();
  let headersReceived = null;
  const headers = new Promise(resolve => {
    headersReceived = resolve;
  });
  return {
    deadlines: [],
    signal: controller.signal,
    abort: () => controller.abort(),
    headers,
    headersReceived,
  };
}

/**
 * Run the page's script as the popup itself (window.name set) or as the
 * app window, against `base`, recording what it does to the window.
 * With `control`, the page gets that AbortSignal instead of the
 * platform's timed one.
 */
function runPage(base, name, bounds, control) {
  const actions = [];
  const context = {
    URL,
    URLSearchParams,
    AbortSignal: control
      ? {
          timeout: ms => {
            control.deadlines.push(ms);
            return control.signal;
          },
        }
      : AbortSignal,
    location: {
      search: '?url=%2Fapp',
      href: base + '/media/appWindow.html?url=%2Fapp',
      origin: base,
      replace: url => actions.push('replace:' + url),
    },
    document: {},
    localStorage: {getItem: () => (bounds ? JSON.stringify(bounds) : null)},
    screenX: 0,
    screenY: 0,
    outerWidth: 800,
    outerHeight: 600,
    moveTo: () => actions.push('moveTo'),
    resizeTo: () => actions.push('resizeTo'),
    window: {
      name,
      close: () => actions.push('close'),
      open: (url, target, features) => {
        actions.push('open:' + target + ':' + features);
        return {};
      },
    },
    fetch: (url, opts) =>
      fetch(base + url, opts).then(response => {
        if (control) control.headersReceived();
        return response;
      }),
  };
  context.window.moveTo = context.moveTo;
  context.window.resizeTo = context.resizeTo;
  vm.runInNewContext(SCRIPT, context);
  return actions;
}

/** The first entry of `actions`, once the page has taken one. */
function firstAction(actions) {
  return waitFor(() => actions[0], 'the page takes an action', WATCHDOG_MS);
}

/** `promise`, or a failure once `WATCHDOG_MS` passed without it. */
function within(promise, what) {
  let timer = null;
  const watchdog = new Promise((_resolve, reject) => {
    timer = setTimeout(
      () => reject(new Error(what + ' did not happen in ' + WATCHDOG_MS + ' ms')),
      WATCHDOG_MS,
    );
  });
  return Promise.race([promise, watchdog]).finally(() => clearTimeout(timer));
}

async function main() {
  await test(
    'W6 F1: a daemon that accepts the request but never sends headers ' +
      'is taken for one without a popup once the 2 s deadline expires',
    async () => {
      let requested = null;
      const request = new Promise(resolve => {
        requested = resolve;
      });
      const daemon = await startDaemon(() => requested());
      try {
        const control = abortControl();
        const actions = runPage(daemon.base, 'kiss-app-window', null, control);
        await within(request, 'the request');
        assert.deepStrictEqual(control.deadlines, [2000], 'the deadline asked for');
        assert.deepStrictEqual(
          actions,
          [],
          'nothing happens while the daemon holds the request',
        );
        control.abort();
        const first = await firstAction(actions);
        assert.strictEqual(
          first,
          'replace:/app',
          'the popup must show the webapp in place; actions: ' +
            JSON.stringify(actions),
        );
      } finally {
        await daemon.close();
      }
    },
  );

  await test(
    'W6 F1: a daemon that sends its headers and then stalls the JSON ' +
      'body is taken for one without a popup once the deadline expires',
    async () => {
      const daemon = await startDaemon((_req, res) => {
        res.writeHead(200, {'Content-Type': 'application/json'});
        res.write('{"open":');
      });
      try {
        const control = abortControl();
        const actions = runPage(daemon.base, '', null, control);
        await within(control.headers, 'the response headers');
        assert.deepStrictEqual(control.deadlines, [2000], 'the deadline asked for');
        assert.deepStrictEqual(
          actions,
          [],
          'nothing happens while the daemon holds the body',
        );
        control.abort();
        const first = await firstAction(actions);
        assert.strictEqual(
          first,
          'open:kiss-app-window:popup,left=0,top=0,width=800,height=600',
          'the app window must open the popup; actions: ' +
            JSON.stringify(actions),
        );
        assert.deepStrictEqual(
          actions,
          [first, 'close'],
          'the app window closes itself once the popup is open',
        );
      } finally {
        await daemon.close();
      }
    },
  );

  await test(
    'W6 F1 guard: an answered "open" closes the page, an answered ' +
      '"closed" opens the popup at the saved bounds, an error answers "closed"',
    async () => {
      let reply = '{"open":true}';
      let status = 200;
      const daemon = await startDaemon((_req, res) => {
        res.writeHead(status, {'Content-Type': 'application/json'});
        res.end(reply);
      });
      // Each page gets a deadline of its own, asked for but never expired.
      async function answered(name, bounds) {
        const control = abortControl();
        const actions = runPage(daemon.base, name, bounds, control);
        await firstAction(actions);
        assert.deepStrictEqual(
          control.deadlines,
          [2000],
          'the deadline asked for',
        );
        return actions;
      }
      try {
        assert.deepStrictEqual(await answered('', null), ['close']);

        reply = '{"open":false}';
        const bounds = {left: 10, top: 20, width: 300, height: 200};
        assert.deepStrictEqual(await answered('', bounds), [
          'open:kiss-app-window:popup,left=10,top=20,width=300,height=200',
          'close',
        ]);

        reply = '{"open":false}';
        // moveTo, resizeTo and replace run in one synchronous step.
        assert.deepStrictEqual(await answered('kiss-app-window', bounds), [
          'moveTo',
          'resizeTo',
          'replace:/app',
        ]);

        status = 500;
        reply = 'boom';
        assert.deepStrictEqual(await answered('', null), [
          'open:kiss-app-window:popup,left=0,top=0,width=800,height=600',
          'close',
        ]);
      } finally {
        await daemon.close();
      }
    },
  );

  report('audit1101_W6_app_window_deadline');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  },
);
