// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Audit 2026-10-12 (webapp media): the interleavings of the terminal
// tab view that terminalTabView.test.js leaves untested (the browser
// tab view's are in browserTabView.test.js).
//
// terminalTab.js
//   * dispose while xterm.js is still downloading: no Terminal is
//     built, no terminalOpen is sent, output that arrived meanwhile is
//     dropped with the view;
//   * a failed xterm.js download paints the load note; the next show
//     retries, opens the shell exactly once and replays the output
//     that queued up during the failure;
//   * a view hidden again before xterm.js arrived still opens its
//     shell but does not steal focus.
//
// xterm.js is not fetched from jsDelivr: the page gets a `fetch` that
// answers from a script (resolve / reject per URL), and the bundle text
// it returns defines Terminal / FitAddon on the page when evaluated,
// exactly as the real UMD bundles do (simplify2_harness's
// installXtermStub stand-ins, held back until then).

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {sleep, queueRaf, installXtermStub} = require('./simplify2_harness');

const MEDIA = path.join(__dirname, '..', 'media');

// A stand-in for the two UMD bundles: evaluated by terminalTab's
// evalUmd, it installs the globals ensureXterm checks for, the
// stand-ins makeTerminalPage holds back in `heldXterm`.
const FAKE_XTERM_BUNDLE =
  'this.Terminal = heldXterm.Terminal; this.FitAddon = heldXterm.FitAddon;';
const FAKE_FIT_BUNDLE = '';

/**
 * A page running terminalTab.js whose xterm.js download is scripted:
 * `fetches` collects {url, resolve, reject} per request so a test
 * decides when (and whether) each download completes; `terminals` is
 * every Terminal built once the bundle defined it.
 */
function makeTerminalPage() {
  const dom = new JSDOM('<!DOCTYPE html><body></body>', {
    runScripts: 'outside-only',
  });
  const win = dom.window;
  const fetches = [];
  win.heldXterm = {};
  const {terminals} = installXtermStub(win.heldXterm);
  win.ResizeObserver = win.heldXterm.ResizeObserver;
  queueRaf(win); // no frame ever runs
  win.AbortController = class {
    constructor() {
      this.signal = {};
    }
    abort() {}
  };
  win.fetch = url =>
    new Promise((resolve, reject) => {
      fetches.push({
        url,
        resolve: text => resolve({ok: true, text: () => Promise.resolve(text)}),
        reject,
      });
    });
  win.eval(fs.readFileSync(path.join(MEDIA, 'terminalTab.js'), 'utf8'));
  return {win, fetches, terminals};
}

function bundleFor(url) {
  if (url.endsWith('.css')) return '';
  if (url.includes('addon-fit')) return FAKE_FIT_BUNDLE;
  return FAKE_XTERM_BUNDLE;
}

/** Let the pending fetches succeed and the load chain settle. */
async function completeDownloads(page) {
  const batch = page.fetches.splice(0);
  for (const f of batch) f.resolve(bundleFor(f.url));
  await sleep(0);
}

async function failDownloads(page) {
  const batch = page.fetches.splice(0);
  for (const f of batch) f.reject(new TypeError('Failed to fetch'));
  await sleep(0);
}

function createTerminalView(page, sent) {
  const view = page.win.TerminalTabView.create('tab-1', {
    send: msg => sent.push(JSON.parse(JSON.stringify(msg))),
    workDir: '/work',
  });
  page.win.document.body.appendChild(view.el);
  return view;
}

async function testDisposeWhileXtermLoads() {
  const page = makeTerminalPage();
  const sent = [];
  const view = createTerminalView(page, sent);
  view.setVisible(true);
  assert.strictEqual(page.fetches.length, 3, 'the three bundles download');
  view.data('late output');
  view.dispose();
  await completeDownloads(page);
  assert.strictEqual(page.terminals.length, 0, 'no Terminal');
  assert.deepStrictEqual(sent, [], 'no shell asked for a disposed tab');
  // A second create on the same page reuses the loaded bundle at once.
  const sent2 = [];
  const again = createTerminalView(page, sent2);
  again.setVisible(true);
  await completeDownloads(page);
  assert.strictEqual(page.terminals.length, 1);
  assert.strictEqual(sent2.length, 1);
  assert.strictEqual(sent2[0].type, 'terminalOpen');
}

async function testFailedLoadThenRetry() {
  const page = makeTerminalPage();
  const sent = [];
  const view = createTerminalView(page, sent);
  view.setVisible(true);
  view.data('queued while loading');
  await failDownloads(page);
  const note = view.el.querySelector('.terminal-load-error');
  assert.ok(note, 'the failure is shown in the tab');
  assert.ok(note.textContent.includes('Failed to fetch'));
  assert.deepStrictEqual(sent, [], 'no shell without a terminal');

  // Switching away and back retries the download.
  view.setVisible(false);
  view.setVisible(true);
  assert.strictEqual(page.fetches.length, 3, 'the retry downloads again');
  view.setVisible(true);
  assert.strictEqual(page.fetches.length, 3, 'one download per retry');
  await completeDownloads(page);
  assert.strictEqual(view.el.querySelector('.terminal-load-error'), null);
  assert.strictEqual(page.terminals.length, 1);
  const term = page.terminals[0];
  assert.ok(term.written.startsWith('queued while loading'));
  assert.strictEqual(term.focused, 1);
  assert.deepStrictEqual(sent, [
    {type: 'terminalOpen', tab_id: 'tab-1', cols: 80, rows: 24, workDir: '/work'},
  ]);
}

async function testHiddenBeforeXtermArrives() {
  const page = makeTerminalPage();
  const sent = [];
  const view = createTerminalView(page, sent);
  view.setVisible(true);
  view.setVisible(false);
  await completeDownloads(page);
  const term = page.terminals[0];
  assert.ok(term, 'the terminal is built for the hidden tab');
  assert.strictEqual(term.focused, 0, 'a hidden tab is not focused');
  assert.strictEqual(sent.length, 1);
  assert.strictEqual(sent[0].type, 'terminalOpen');
  view.setVisible(true);
  assert.strictEqual(term.focused, 1);
  assert.strictEqual(sent.length, 1, 'a re-show opens no second shell');
}

(async () => {
  await testDisposeWhileXtermLoads();
  await testFailedLoadThenRetry();
  await testHiddenBeforeXtermArrives();
  console.log('audit1012_media_tab_views_lifecycle.test.js passed');
})().catch(err => {
  console.error(err);
  process.exit(1);
});
