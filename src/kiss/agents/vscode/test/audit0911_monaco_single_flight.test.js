// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) regression test for review-vscode.md #1: a timed-out
// Monaco loader flight left its script callbacks LIVE, and those callbacks
// cleared the shared `_monacoPromise` slot unconditionally.  Interleaving:
//
//   1. open code file 1 -> flight P1 starts (script S1 appended);
//   2. P1 exceeds its 10s budget -> the timeout clears the slot;
//   3. open code file 2 -> flight P2 starts (script S2 appended);
//   4. S1 (still live) reports onerror -> the STALE callback cleared the
//      slot again, wiping out P2's registration;
//   5. open code file 3 -> a THIRD loader script starts while P2 is still
//      in flight: two concurrent AMD initializations now contend for the
//      same global window.require.
//
// The fix makes every reset ownership-checked (only the flight that owns
// the slot may clear it) and retires a timed-out flight's callbacks, so
// step 4 is a no-op and step 5 reuses P2.  The test drives the real
// chat.html + api.js + main.js and controls the 10-second timer and the
// loader script's events itself (JSDOM loads no external scripts).

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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

  // Capture the Monaco loader's 10-second timeout so the test can fire it
  // deterministically; every other timer runs for real.
  const monacoTimeouts = [];
  const origSetTimeout = win.setTimeout.bind(win);
  win.setTimeout = function (fn, ms, ...args) {
    if (ms === 10000) {
      monacoTimeouts.push(fn);
      return 0x7fffffff - monacoTimeouts.length;
    }
    return origSetTimeout(fn, ms, ...args);
  };

  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
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
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=audit0911-monaco-main.js',
  );
  return {win, posted, monacoTimeouts};
}

function settle() {
  return new Promise(resolve => setTimeout(resolve, 30));
}

function loaderScripts(win) {
  return Array.from(
    win.document.head.querySelectorAll('script[src*="vs/loader.js"]'),
  );
}

function openCodeFile(win, name) {
  // fileContent arrives from the extension host as a window message.
  win.dispatchEvent(
    new win.MessageEvent('message', {
      data: {
        type: 'fileContent',
        name: name,
        path: '/ws/' + name,
        content: 'print(1)\n',
      },
    }),
  );
}

async function main() {
  const {win, monacoTimeouts} = makeWebview();
  win._testApi.endLaunch();

  // 1. First code file: flight P1 appends loader script S1.
  openCodeFile(win, 'a.py');
  await settle();
  assert.strictEqual(loaderScripts(win).length, 1, 'P1 appended S1');
  assert.strictEqual(monacoTimeouts.length, 1, 'P1 armed its 10s timeout');

  // 2. P1 times out: the single-flight slot empties (a retry is allowed).
  monacoTimeouts[0]();
  await settle();

  // 3. Second code file: flight P2 appends S2.
  openCodeFile(win, 'b.py');
  await settle();
  assert.strictEqual(loaderScripts(win).length, 2, 'P2 appended S2');
  assert.strictEqual(monacoTimeouts.length, 2, 'P2 armed its own timeout');

  // 4. The TIMED-OUT flight's script now reports its error, late.  A stale
  //    callback must not clear the slot that P2 currently owns.
  const s1 = loaderScripts(win)[0];
  s1.onerror();
  await settle();

  // 5. Third code file while P2 is still in flight: it must REUSE P2, not
  //    start a third concurrent loader.
  openCodeFile(win, 'c.py');
  await settle();
  assert.strictEqual(
    loaderScripts(win).length,
    2,
    'a stale timed-out flight cleared the newer in-flight load: a third ' +
      'concurrent loader script was started',
  );

  // 6. A timed-out flight's late onload must not run the AMD bootstrap
  //    (its callbacks are retired); only P2's own onload may.
  let amdCalls = 0;
  win.require = function () {
    amdCalls += 1;
  };
  win.require.config = function () {};
  s1.onload();
  await settle();
  assert.strictEqual(
    amdCalls,
    0,
    "the timed-out flight's retired onload still ran the AMD bootstrap",
  );

  // 7. P2 completes normally and Monaco renders: the happy path is intact.
  const created = [];
  const fakeMonaco = {
    editor: {
      create: (holder, opts) => {
        created.push({holder, opts});
        return {dispose: () => {}, updateOptions: () => {}};
      },
    },
  };
  win.require = function (_deps, onOk) {
    win.monaco = fakeMonaco;
    onOk();
  };
  win.require.config = function () {};
  const s2 = loaderScripts(win)[1];
  s2.onload();
  await settle();
  assert.ok(created.length >= 1, 'P2 still resolved and created the editor');

  // 8. The resolved flight stays cached: a fourth file adds no script.
  openCodeFile(win, 'd.py');
  await settle();
  assert.strictEqual(
    loaderScripts(win).length,
    2,
    'a resolved flight must be reused, not reloaded',
  );

  console.log('audit0911_monaco_single_flight: OK');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  },
);
