// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// pdfView.js: a page whose draw fails for real (pdf.js rejects the
// render with a decoder error) used to be treated like a draw cancelled
// by a newer zoom: the rejection only cleared the task, the page box
// stayed blank (or kept the canvas of an older zoom) and the toolbar
// went on saying "Page 1 of 1".  The failure must be shown in the
// page's box, a cancelled draw must still show nothing, and a later
// draw that succeeds must replace the note.  The note replaces the
// canvas of the last successful zoom, so that zoom must be drawn
// again when the user returns to it: a cache saying it is still on
// screen would leave the note there for good.
//
// The real pdfView.js runs in jsdom; its import() of the pdf.js module
// is answered by a vm.SyntheticModule standing in for the library
// (needs --experimental-vm-modules, so the suite re-runs itself under
// that flag), whose page.render() tasks the test settles by hand.

const vm = require('vm');

if (typeof vm.SyntheticModule !== 'function') {
  const {spawnSync} = require('child_process');
  const run = spawnSync(
    process.execPath,
    [
      ...process.execArgv,
      '--experimental-vm-modules',
      '--no-warnings',
      __filename,
    ],
    {stdio: 'inherit'},
  );
  process.exit(run.status === null ? 1 : run.status);
}

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {settle, waitFor} = require('./simplify2_harness');

const MEDIA = path.join(__dirname, '..', 'media');

/** The library's page object: every render() is a task the test settles. */
function fakePage(renders) {
  return {
    getViewport: ({scale}) => ({width: 600 * scale, height: 800 * scale}),
    render() {
      const task = {};
      task.promise = new Promise((resolve, reject) => {
        task.resolve = resolve;
        task.reject = reject;
      });
      task.cancel = () => {
        const err = new Error('Rendering cancelled, page 1');
        err.name = 'RenderingCancelledException';
        task.reject(err);
      };
      renders.push(task);
      return task;
    },
  };
}

/**
 * What a loaded document keeps in the page, as the pinned pdf.js does
 * (pdfjs-dist 6.3.289, legacy/build/pdf.mjs): the fonts its FontLoader
 * adds to document.fonts and the hidden <div><svg><defs> its
 * DOMFilterFactory appends to the body.  The loading task holds both as
 * `_transport.fontLoader` and `_transport.filterFactory`, and only frees
 * them (WorkerTransport.destroy) once the worker answered "Terminate".
 */
function fakeTransport(win) {
  const doc = win.document;
  const font = {family: 'g_d' + doc.fonts.size + '_f1'};
  doc.fonts.add(font);
  const filters = doc.createElement('div');
  filters.className = 'pdf-filters';
  filters.innerHTML = '<svg width="0" height="0"><defs></defs></svg>';
  doc.body.append(filters);
  return {
    font,
    filters,
    fontLoader: {clear: () => doc.fonts.delete(font)},
    filterFactory: {destroy: () => filters.remove()},
  };
}

/**
 * A jsdom page running the real pdfView.js over a stand-in pdf.js.
 * `workers` records every Worker the viewer starts (`terminated`, and
 * `emitError` plays a worker that fails to start or dies); `tasks` the
 * library's loading tasks; `resizes` the ResizeObserver callbacks.  With
 * `opts.deadWorker` the library's load and destroy never settle, as
 * pdf.js's do over a worker that answers nothing; with `opts.hungFree`
 * the document loads and only its destroy never settles (a worker that
 * hung after delivering pages); with `opts.failedFree` its destroy
 * rejects, as pdf.js's does when the worker answers with an error.
 */
function makePage(renders, opts) {
  opts = opts || {};
  const dom = new JSDOM('<!DOCTYPE html><body></body>', {
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  const ctx = dom.getInternalVMContext();
  win.document.fonts = new Set(); // jsdom has no FontFaceSet
  win.fetch = () =>
    Promise.resolve({ok: true, blob: () => Promise.resolve(new win.Blob([]))});
  win.URL.createObjectURL = () => 'blob:fake-pdf-worker';
  const workers = [];
  win.Worker = class {
    constructor() {
      this.listeners = {};
      this.terminated = false;
      workers.push(this);
    }
    addEventListener(type, cb) {
      (this.listeners[type] = this.listeners[type] || []).push(cb);
    }
    terminate() {
      this.terminated = true;
    }
    /** The worker died: the standard asynchronous Worker `error` event. */
    emitError(err) {
      for (const cb of (this.listeners.error || []).slice()) cb(err);
    }
  };
  win.HTMLCanvasElement.prototype.getContext = () => ({});
  const resizes = [];
  win.ResizeObserver = class {
    constructor(cb) {
      resizes.push(cb);
    }
    observe() {}
    disconnect() {}
  };
  // Every page box is on screen as soon as it is observed.
  win.IntersectionObserver = class {
    constructor(cb) {
      this.cb = cb;
    }
    observe(target) {
      this.cb([{target, isIntersecting: true}]);
    }
    disconnect() {}
  };
  const page = fakePage(renders);
  const doc = {numPages: 1, getPage: () => Promise.resolve(page)};
  const tasks = [];
  const lib = {
    PDFWorker: {create: () => ({destroy() {}})},
    getDocument() {
      const task = opts.deadWorker
        ? {promise: new Promise(() => {}), destroy: () => new Promise(() => {})}
        : {
            promise: Promise.resolve(doc),
            _transport: fakeTransport(win),
            destroy() {
              if (opts.hungFree) return new Promise(() => {});
              if (opts.failedFree) return Promise.reject(new Error('no'));
              task._transport.fontLoader.clear();
              task._transport.filterFactory.destroy();
              task._transport = null;
              return Promise.resolve();
            },
          };
      tasks.push(task);
      return task;
    },
  };
  const script = new vm.Script(
    fs.readFileSync(path.join(MEDIA, 'pdfView.js'), 'utf8'),
    {
      filename: 'pdfView.js',
      importModuleDynamically: async () => {
        const mod = new vm.SyntheticModule(
          Object.keys(lib),
          function () {
            for (const key of Object.keys(lib)) this.setExport(key, lib[key]);
          },
          {context: ctx},
        );
        await mod.link(() => {});
        await mod.evaluate();
        return mod;
      },
    },
  );
  script.runInContext(ctx);
  return {win, workers, tasks, resizes};
}

/** The fonts and filter SVGs pdf.js documents have left in *win*. */
function pageResources(win) {
  return {
    fonts: [...win.document.fonts],
    filters: [...win.document.body.querySelectorAll('.pdf-filters')],
  };
}

/** Mount a viewer of an empty document into a fresh holder of *win*. */
function mount(win) {
  const holder = win.document.createElement('div');
  win.document.body.appendChild(holder);
  const viewer = win.mountPdfViewer(holder, new win.Uint8Array(0), {
    name: 'report.pdf',
  });
  return {holder, viewer};
}

/** Dispatch a keydown for *key* on *target*; whether the viewer took it. */
function pressKey(win, target, key) {
  const ev = new win.KeyboardEvent('keydown', {
    key,
    bubbles: true,
    cancelable: true,
  });
  target.dispatchEvent(ev);
  return ev.defaultPrevented;
}

function pointAt(win, target) {
  target.dispatchEvent(new win.Event('pointerdown', {bubbles: true}));
}

/**
 * A worker that fails to start answers nothing, and pdf.js's load (and
 * its destroy) never settle over it: the viewer used to say "Loading"
 * for ever, and dispose() waited for the destroy before terminating the
 * worker, so a closed tab leaked it.
 */
async function testDeadWorker() {
  const dead = makePage([], {deadWorker: true});
  const {holder, viewer} = mount(dead.win);
  await waitFor(() => dead.workers.length === 1, 'the viewer starts a worker');
  dead.workers[0].emitError({message: 'worker boot failed'});
  await waitFor(
    () => holder.querySelector('.content-binary-note'),
    'a worker that failed to start fails the load',
  );
  const note = holder.querySelector('.content-binary-note');
  assert.ok(
    note.textContent.includes('worker boot failed'),
    'the note carries the reason: ' + note.textContent,
  );
  assert.strictEqual(dead.workers[0].terminated, true, 'the worker is stopped');
  viewer.dispose();
  assert.strictEqual(holder.childElementCount, 0, 'the viewer is gone');

  // Closed while the worker is still silent: stopped at once.
  const silent = makePage([], {deadWorker: true});
  const closed = mount(silent.win);
  await waitFor(
    () => silent.workers.length === 1,
    'the viewer starts a worker',
  );
  closed.viewer.dispose();
  assert.strictEqual(
    silent.workers[0].terminated,
    true,
    'dispose must not wait for a document free the worker never answers',
  );
  assert.strictEqual(closed.holder.childElementCount, 0);

  // Closed over a worker that hung after its pages arrived (it spins
  // on a malformed stream): the document free is never answered
  // either, and the worker is stopped all the same.  pdf.js removes a
  // document's fonts and filter SVG from the page only on that answer,
  // so the viewer removes them itself -- its own, not those of the
  // other PDF open beside it.
  const hung = makePage([], {hungFree: true});
  const shown = mount(hung.win);
  const beside = mount(hung.win);
  await waitFor(
    () =>
      shown.holder.querySelector('.pdf-page') &&
      beside.holder.querySelector('.pdf-page'),
    'both pages are laid out',
  );
  const [first, second] = hung.tasks.map(task => task._transport);
  assert.deepStrictEqual(pageResources(hung.win), {
    fonts: [first.font, second.font],
    filters: [first.filters, second.filters],
  });
  shown.viewer.dispose();
  await waitFor(
    () => hung.workers[0].terminated,
    'a worker that never answers the document free is stopped anyway',
    8000,
  );
  assert.deepStrictEqual(
    pageResources(hung.win),
    {fonts: [second.font], filters: [second.filters]},
    'the closed document leaves no font or filter SVG in the page',
  );
  assert.strictEqual(hung.workers[1].terminated, false);

  // A worker that dies under a shown document never answers its free
  // either: nothing of the document stays in the page, then or when the
  // failed viewer is closed.
  const dying = makePage([]);
  const drawn = mount(dying.win);
  await waitFor(
    () => drawn.holder.querySelector('.pdf-page'),
    'the page is laid out',
  );
  assert.strictEqual(pageResources(dying.win).fonts.length, 1);
  dying.workers[0].emitError({message: 'out of memory'});
  assert.ok(drawn.holder.querySelector('.content-binary-note'));
  assert.deepStrictEqual(
    pageResources(dying.win),
    {fonts: [], filters: []},
    'a dead worker leaves no font or filter SVG in the page',
  );
  drawn.viewer.dispose();
  assert.deepStrictEqual(pageResources(dying.win), {fonts: [], filters: []});

  // A free that fails leaves the document's resources to the viewer too.
  const failing = makePage([], {failedFree: true});
  const lost = mount(failing.win);
  await waitFor(
    () => lost.holder.querySelector('.pdf-page'),
    'the page is laid out',
  );
  lost.viewer.dispose();
  await waitFor(() => failing.workers[0].terminated, 'the worker is stopped');
  assert.deepStrictEqual(
    pageResources(failing.win),
    {fonts: [], filters: []},
    'a failed free leaves no font or filter SVG in the page',
  );

  // A document pdf.js freed itself is not touched again.
  const freed = makePage([]);
  const done = mount(freed.win);
  await waitFor(
    () => done.holder.querySelector('.pdf-page'),
    'the page is laid out',
  );
  done.viewer.dispose();
  await waitFor(() => freed.workers[0].terminated, 'the worker is stopped');
  assert.strictEqual(freed.tasks[0]._transport, null);
  assert.deepStrictEqual(pageResources(freed.win), {fonts: [], filters: []});
}

/**
 * The shortcuts are the viewer's while nothing else is pointed at: a key
 * aimed at another element, or pressed after the user pointed outside
 * the viewer (the chat transcript beside it in the split layout), is
 * left alone; pointing back into the viewer, or showing it again,
 * makes the keys its own again.  A press outside the viewer while it
 * is hidden (its tab in the tab row, bringing it back) is not a press
 * away from it: the keys are its own the moment it is shown, before
 * its size is observed.
 */
async function testKeysStayWithTheirScroller() {
  const renders = [];
  const {win, resizes} = makePage(renders);
  // jsdom lays nothing out: the viewer counts as on screen.
  win.Element.prototype.getClientRects = () => [{}];
  const {holder} = mount(win);
  await waitFor(() => renders.length === 1, 'the first page is drawn');
  const output = win.document.createElement('div');
  output.textContent = 'the chat transcript';
  win.document.body.appendChild(output);
  const body = win.document.body;
  assert.strictEqual(pressKey(win, body, 'PageDown'), true, 'fresh viewer');
  assert.strictEqual(
    pressKey(win, output, 'PageDown'),
    false,
    'aimed elsewhere',
  );
  pointAt(win, output);
  assert.strictEqual(pressKey(win, body, 'PageDown'), false, 'pointed away');
  assert.strictEqual(pressKey(win, body, 'End'), false, 'pointed away');
  pointAt(win, holder.querySelector('.pdf-scroller'));
  assert.strictEqual(pressKey(win, body, 'PageDown'), true, 'pointed back');
  pointAt(win, output);
  assert.strictEqual(pressKey(win, body, 'Home'), false, 'pointed away');
  // Hidden (another tab shown) and shown again: the keys are its own.
  const scroller = holder.querySelector('.pdf-scroller');
  let width = 0;
  Object.defineProperty(scroller, 'clientWidth', {get: () => width});
  assert.strictEqual(resizes.length, 1, 'the viewer watches its size');
  resizes[0]();
  width = 600;
  resizes[0]();
  assert.strictEqual(pressKey(win, body, 'PageDown'), true, 'shown again');
  resizes[0](); // resized while shown: not a re-show
  pointAt(win, output);
  resizes[0]();
  assert.strictEqual(pressKey(win, body, 'PageDown'), false, 'still away');
  // Pointed away, then hidden and its tab pressed (outside the viewer)
  // to bring it back: the keys are its own the moment it is shown,
  // before its size is observed.
  let onScreen = false;
  win.Element.prototype.getClientRects = () => (onScreen ? [{}] : []);
  pointAt(win, output); // hidden: not pointing away from it
  onScreen = true;
  assert.strictEqual(pressKey(win, body, 'PageDown'), true, 'pressed back while hidden');
  pointAt(win, output); // on screen: pointing away
  assert.strictEqual(pressKey(win, body, 'PageDown'), false, 'pointed away again');
}

async function main() {
  const renders = [];
  const {win} = makePage(renders);
  const {holder, viewer} = mount(win);
  await waitFor(() => renders.length === 1, 'the first page is drawn');
  const box = holder.querySelector('.pdf-page');
  assert.ok(box, 'one page box');
  const status = holder.querySelector('.pdf-status');
  assert.ok(status.textContent.endsWith('of 1'), status.textContent);

  // A zoom cancels the draw in flight: nothing to show for it.
  holder.querySelector('.pdf-zoom-in').click();
  await waitFor(() => renders.length === 2, 'the zoom draws again');
  await settle();
  assert.strictEqual(box.childElementCount, 0, 'a cancelled draw shows nothing');

  // The draw at the new scale fails for real: the page says so.
  renders[1].reject(new Error('Image decoder failed'));
  await waitFor(
    () => box.querySelector('.content-binary-note'),
    'a failed draw is reported in the page box',
  );
  const note = box.querySelector('.content-binary-note');
  assert.ok(
    note.textContent.includes('Image decoder failed'),
    'the note carries the reason: ' + note.textContent,
  );
  assert.ok(note.textContent.includes('page 1'), note.textContent);
  assert.strictEqual(box.querySelectorAll('canvas').length, 0);
  assert.ok(status.textContent.endsWith('of 1'), 'the controls stay');

  // The next draw (another zoom) succeeds: the canvas replaces the note.
  holder.querySelector('.pdf-zoom-in').click();
  await waitFor(() => renders.length === 3, 'the next zoom draws again');
  renders[2].resolve();
  await waitFor(
    () => box.querySelector('canvas'),
    'a later successful draw shows its canvas',
  );
  assert.strictEqual(box.childElementCount, 1, 'the note is gone');

  // The failed draw of a further zoom discards that canvas: zooming
  // back to its scale must draw the page again, not trust the stale
  // cache and leave the note where the canvas was.
  const zoomLevel = holder.querySelector('.pdf-zoom-level');
  const goodZoom = zoomLevel.textContent;
  holder.querySelector('.pdf-zoom-in').click();
  await waitFor(() => renders.length === 4, 'the zoom draws again');
  renders[3].reject(new Error('Image decoder failed'));
  await waitFor(
    () => box.querySelector('.content-binary-note'),
    'the failed zoom is reported',
  );
  holder.querySelector('.pdf-zoom-out').click();
  assert.strictEqual(zoomLevel.textContent, goodZoom, 'back at the good zoom');
  await waitFor(
    () => renders.length === 5,
    'REGRESSION: returning to the last successful zoom starts no draw ' +
      '(its canvas is gone but its scale is still cached as rendered)',
  );
  renders[4].resolve();
  await waitFor(
    () => box.querySelector('canvas'),
    'the page is drawn again at the good zoom',
  );
  assert.strictEqual(box.childElementCount, 1, 'the note is gone again');

  // A failure of a draw that dispose cancelled is not reported.
  holder.querySelector('.pdf-zoom-in').click();
  await waitFor(() => renders.length === 6, 'one more draw');
  viewer.dispose();
  await settle();
  assert.strictEqual(holder.childElementCount, 0, 'the viewer is gone');
  assert.strictEqual(
    box.querySelector('.content-binary-note'),
    null,
    'a draw cancelled by dispose reports no failure',
  );
  await testDeadWorker();
  await testKeysStayWithTheirScroller();
  console.log('pdfViewRenderFailure.test.js passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
