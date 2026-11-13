// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests for the speech-model load lifecycle: voice.js must survive a
// model load that never settles, and vosk.js must not leak the Web Worker
// of a model that failed to load.
//
// Defects being reproduced (each test failed against the pre-fix code):
//
//  CAND-4  vosk.js's createModel settles only when its worker posts a
//          'load' message. A worker that fails to boot at all (its blob:
//          URL blocked by CSP, a top-level script/WASM crash) fires only
//          a Worker 'error' event that nobody listens for, so createModel
//          never settles, startBrowserPipeline never reaches its settle
//          handler, and `busy` stays true for the page lifetime: the mic
//          button is wedged on 'loading' and every later re-enable defers
//          to `busy` and merely repaints 'loading'. voice.js must bound
//          the load with a timeout that paints the error state and clears
//          `busy` so a re-enable can try again.
//
//  CAND-5  When the worker boots but the model fails to load (bad or
//          unreachable modelUrl: the worker posts 'load' with
//          result=false), createModel rejected and dropped the Model
//          instance without terminate(): the live Web Worker leaked, one
//          more per retry. The real media/vosk.js is exercised for this,
//          with only the Worker/createObjectURL browser primitives
//          stubbed (jsdom has no Web Workers).
//
//  W5 F1   (audit 2026-11-01) The vendored worker reports a failed model
//          load (WASM init, IndexedDB, a model download answering 404,
//          extraction, Kaldi model construction) as the protocol message
//          {event:'error', error:<message>}, never as {event:'load',
//          result:false}.  createModel listened only for 'load' and the
//          native Worker 'error' event, so a real load failure left it
//          pending for ever with its Worker alive; voice.js then timed
//          out and, its page-wide modelPromise being forgotten only on
//          rejection, every later enable reused the dead promise: no new
//          download until the page was reloaded.
//
// The CAND-4 harness mimics voiceBrowserPipelineLifecycle.test.js: real
// voice.js + main.js in jsdom, with the embedder capabilities (Vosk,
// AudioContext, getUserMedia) stubbed the way that suite stubs them.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {
  makePage: makeChatPage,
  stubRaf,
  settle,
  waitFor,
  MEDIA,
} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {micBtn, makeVoskWindow, VOICE_CFG} = require('./voice_page_helpers');

const {test, report} = makeRunner();

const MODEL_URL = 'https://models.example.com/m.tar.gz';
const LOAD_ERROR = 'HTTP error! status: 404';

/** A loaded model.  No test grants the microphone, so the pipeline
 *  never builds a recognizer from it. */
function fakeModel() {
  return {};
}

// voice.js bounds every model load with setTimeout(MODEL_LOAD_TIMEOUT_MS).
// The tests fire that timer by hand (page.fireLoadTimeout) instead of
// sleeping past a real one, so the "still loading" and "timed out"
// assertions cannot race the wall clock when run-all.js runs suites
// concurrently.  Any other delay (the fake worker's own failure, the
// pipeline's own timers) keeps the real setTimeout.  Between two steps a
// test waits one timer turn (settle): the pipeline and createModel move
// on promise chains alone.
const LOAD_TIMEOUT_MS = 60;

/**
 * A Worker stand-in for jsdom (which has none): records postMessage
 * traffic and terminate() calls and lets the test play the worker's side
 * of the vosk protocol.  `failure` makes every instance fail by itself:
 * `{bootError}` raises the standard asynchronous `error` event shortly
 * after construction without ever posting a load message (a script/WASM
 * that fails to boot); `{loadError}` answers the {action:'load'} message
 * with the protocol message {event:'error', error} (what the embedded
 * RecognizerWorker posts when load() rejects) and never with a 'load'.
 */
function fakeWorkerClass(workers, failure) {
  const {bootError, loadError} = failure || {};
  return class FakeWorker {
    constructor() {
      this.listeners = new Map();
      this.posted = [];
      this.terminated = false;
      workers.push(this);
      if (bootError) setTimeout(() => this.emitError({message: bootError}), 5);
    }
    addEventListener(type, cb) {
      const list = this.listeners.get(type) || [];
      list.push(cb);
      this.listeners.set(type, list);
    }
    removeEventListener(type, cb) {
      const list = this.listeners.get(type) || [];
      const i = list.indexOf(cb);
      if (i >= 0) list.splice(i, 1);
    }
    postMessage(msg) {
      this.posted.push(msg);
      if (loadError && msg && msg.action === 'load') {
        setTimeout(() => this.emit({event: 'error', error: loadError}), 5);
      }
    }
    terminate() {
      this.terminated = true;
    }
    /** Play the worker: deliver `data` to the page-side Model. */
    emit(data) {
      for (const cb of (this.listeners.get('message') || []).slice())
        cb({data});
    }
    /** Boot failure: the standard asynchronous Worker `error` event. */
    emitError(err) {
      for (const cb of (this.listeners.get('error') || []).slice()) cb(err);
    }
    errorListenerCount() {
      return (this.listeners.get('error') || []).length;
    }
  };
}

/**
 * A browser-mode page whose window.Vosk.createModel returns promises the
 * test settles (or never settles) by hand, counting every call.  With
 * `realVosk` the real vendored vosk.js supplies window.Vosk instead, over
 * fake Workers (`bootWorkers`) that fail as `workerFailure` says
 * (fakeWorkerClass's `failure`; without one they never report at all).
 */
function makePage(opts) {
  const options = opts || {};
  const mic = {calls: []};
  const vosk = {createModelCalls: 0, settlers: []};
  const bootWorkers = [];
  const loadTimers = [];
  const {win} = makeChatPage({
    beforeScripts: win =>
      installVoiceStubs(win, options, mic, vosk, bootWorkers, loadTimers),
    extraScripts: ['voice.js'],
  });
  // voice.js only reads window.Vosk lazily (loadVosk at mic click), so
  // the real bundle may follow it.
  if (options.realVosk) {
    win.eval(fs.readFileSync(path.join(MEDIA, 'vosk.js'), 'utf8'));
  }
  /** Fire every armed, uncleared model-load timeout, as the clock would. */
  function fireLoadTimeout() {
    const due = loadTimers.splice(0);
    assert.ok(due.length > 0, 'a model-load timeout must be armed');
    for (const t of due) t.cb();
  }
  return {win, mic, vosk, bootWorkers, fireLoadTimeout};
}

function captureLoadTimeouts(win, loadTimers) {
  const realSetTimeout = win.setTimeout;
  const realClearTimeout = win.clearTimeout;
  win.setTimeout = function (cb, ms, ...args) {
    if (ms !== LOAD_TIMEOUT_MS) return realSetTimeout.call(win, cb, ms, ...args);
    const timer = {cb};
    loadTimers.push(timer);
    return timer;
  };
  win.clearTimeout = function (id) {
    const i = loadTimers.indexOf(id);
    if (i >= 0) loadTimers.splice(i, 1);
    else realClearTimeout.call(win, id);
  };
}

function installVoiceStubs(win, options, mic, vosk, bootWorkers, loadTimers) {
  stubRaf(win);
  captureLoadTimeouts(win, loadTimers);

  Object.defineProperty(win.navigator, 'mediaDevices', {
    configurable: true,
    value: {
      // Counted, never granted: the request stays pending.
      getUserMedia: () => {
        mic.calls.push('getUserMedia');
        return new Promise(() => {});
      },
      enumerateDevices: () => Promise.resolve([]),
    },
  });

  if (options.realVosk) {
    // Integrated mode: the REAL vendored vosk.js is loaded below and
    // window.Vosk comes from it; its Workers fail to boot (the leak the
    // boot-error path exists for), fail the model load (the protocol
    // error) or, without a `workerFailure`, never report at all (the
    // leak the give-up cancel exists for).
    win.Worker = fakeWorkerClass(bootWorkers, options.workerFailure);
    if (typeof win.URL.createObjectURL !== 'function') {
      win.URL.createObjectURL = () => 'blob:fake-vosk-worker';
    }
  } else {
    win.Vosk = {
      createModel: () => {
        vosk.createModelCalls++;
        const settler = {cancelled: false};
        const load = new Promise((resolve, reject) => {
          settler.resolve = resolve;
          settler.reject = reject;
        });
        // The engine's give-up handle (vosk.js createModel.cancel): the
        // real one terminates the worker, then rejects the load.
        load.cancel = () => {
          settler.cancelled = true;
          settler.reject(new Error('Gave up loading the speech model'));
        };
        vosk.settlers.push(settler);
        return load;
      },
    };
  }

  win.__VOICE__ = Object.assign({}, VOICE_CFG, options.cfg || {});
  win.localStorage.setItem('kissVoiceEnabled', '0');
}

// ---------------------------------------------------------------------
// CAND-5 harness: the REAL media/vosk.js in a bare window. jsdom has no
// Web Workers, so the Worker handed to vosk's WorkerFactory is a fake
// that records postMessage traffic and terminate() calls and lets the
// test play the worker's side of the protocol.
// ---------------------------------------------------------------------

/** True when the worker was told to die: DOM terminate() or the vosk
 *  protocol's {action:'terminate'} message, which a worker whose model
 *  loaded handles by freeing the Kaldi objects and closing itself.  The
 *  "stays alive" checks rule out both; a worker whose load failed or
 *  never finished is reclaimed by terminate() alone, so the "is
 *  reclaimed" checks read `terminated`. */
function workerReclaimed(worker) {
  return (
    worker.terminated || worker.posted.some(m => m && m.action === 'terminate')
  );
}

/** The live list of 'error' listeners on the page-side Models of `win`
 *  (createModel's one-shot protocol-error listener). */
function modelErrorListeners(win) {
  const listeners = [];
  const realAdd = win.EventTarget.prototype.addEventListener;
  const realRemove = win.EventTarget.prototype.removeEventListener;
  win.EventTarget.prototype.addEventListener = function (type, cb, opts) {
    if (type === 'error' && this instanceof win.Vosk.Model) listeners.push(cb);
    return realAdd.call(this, type, cb, opts);
  };
  win.EventTarget.prototype.removeEventListener = function (type, cb, opts) {
    if (type === 'error' && this instanceof win.Vosk.Model) {
      const i = listeners.indexOf(cb);
      if (i >= 0) listeners.splice(i, 1);
    }
    return realRemove.call(this, type, cb, opts);
  };
  return listeners;
}

async function main() {
  await test(
    'CAND-4: a model load that never settles times out into the error ' +
      'state instead of wedging voice on loading forever',
    async () => {
      const page = makePage({cfg: {modelLoadTimeoutMs: LOAD_TIMEOUT_MS}});
      const {win, vosk, fireLoadTimeout} = page;
      micBtn(win).click(); // browser mode: pipeline starts directly
      await settle();
      assert.strictEqual(vosk.createModelCalls, 1);
      assert.ok(
        micBtn(win).classList.contains('voice-loading'),
        `must be loading while the model loads: ${micBtn(win).className}`,
      );
      // The worker never posts 'load': createModel never settles.
      fireLoadTimeout();
      await settle();
      const btn = micBtn(win);
      assert.ok(
        btn.classList.contains('voice-error'),
        `the timed-out load must paint the error state: ${btn.className}`,
      );
      assert.strictEqual(
        win.localStorage.getItem('kissVoiceEnabled'),
        '0',
        'the failed attempt must clear the persisted intent',
      );
      win.close();
    },
  );

  await test(
    'CAND-4: after the timeout, busy is cleared so re-enabling is honoured; ' +
      'it waits for the download already in flight instead of starting a ' +
      'second one',
    async () => {
      const page = makePage({cfg: {modelLoadTimeoutMs: LOAD_TIMEOUT_MS}});
      const {win, mic, vosk, fireLoadTimeout} = page;
      micBtn(win).click();
      await settle();
      fireLoadTimeout(); // load timed out; the error turned the mic off
      await settle();
      assert.ok(micBtn(win).classList.contains('voice-error'));
      micBtn(win).click(); // the user tries again
      await settle();
      assert.strictEqual(
        vosk.createModelCalls,
        1,
        'a model download that is still running is shared by the retry, ' +
          'not duplicated',
      );
      assert.ok(
        micBtn(win).classList.contains('voice-loading'),
        `the retry must be visibly loading: ${micBtn(win).className}`,
      );
      // The slow download completes: the retry goes on to the microphone.
      vosk.settlers[0].resolve(fakeModel());
      await settle();
      assert.strictEqual(
        mic.calls.length,
        1,
        'the retry must open the microphone once the shared load arrives',
      );
      win.close();
    },
  );

  await test(
    'CAND-4: a model that arrives after its timeout is kept for the next ' +
      'enable, which uses it without another download',
    async () => {
      const page = makePage({cfg: {modelLoadTimeoutMs: LOAD_TIMEOUT_MS}});
      const {win, mic, vosk, fireLoadTimeout} = page;
      micBtn(win).click();
      await settle();
      fireLoadTimeout();
      await settle();
      assert.ok(micBtn(win).classList.contains('voice-error'));
      vosk.settlers[0].resolve(fakeModel());
      await settle();
      assert.strictEqual(
        mic.calls.length,
        0,
        'a timed-out attempt must never go on to open the microphone',
      );
      assert.ok(
        micBtn(win).classList.contains('voice-error'),
        `the late arrival must not repaint the UI: ${micBtn(win).className}`,
      );
      micBtn(win).click();
      await settle();
      assert.strictEqual(vosk.createModelCalls, 1, 'no second download');
      assert.strictEqual(mic.calls.length, 1, 'the kept model is used');
      win.close();
    },
  );

  await test(
    'CAND-4: a shared load that fails after the timeout is forgotten, so ' +
      'the next enable starts a fresh one',
    async () => {
      const page = makePage({cfg: {modelLoadTimeoutMs: LOAD_TIMEOUT_MS}});
      const {win, vosk, fireLoadTimeout} = page;
      micBtn(win).click();
      await settle();
      fireLoadTimeout();
      await settle();
      vosk.settlers[0].reject(new Error('model download failed'));
      await settle();
      assert.ok(
        micBtn(win).classList.contains('voice-error'),
        `the late failure must not repaint the UI: ${micBtn(win).className}`,
      );
      micBtn(win).click();
      await settle();
      assert.strictEqual(
        vosk.createModelCalls,
        2,
        'a failed download is not reused; the retry downloads again',
      );
      win.close();
    },
  );

  await test(
    'CAND-4: a load that two attempts in a row timed out on is given up as ' +
      'stalled: the engine cancels it and the next enable downloads afresh',
    async () => {
      const page = makePage({cfg: {modelLoadTimeoutMs: LOAD_TIMEOUT_MS}});
      const {win, mic, vosk, fireLoadTimeout} = page;
      micBtn(win).click();
      await settle();
      fireLoadTimeout();
      await settle();
      assert.ok(!vosk.settlers[0].cancelled, 'the first timeout keeps sharing');
      micBtn(win).click(); // the retry shares the load still in flight
      await settle();
      assert.strictEqual(vosk.createModelCalls, 1);
      fireLoadTimeout(); // ... which never settles either
      await settle();
      assert.ok(micBtn(win).classList.contains('voice-error'));
      assert.ok(vosk.settlers[0].cancelled, 'the second timeout cancels it');
      micBtn(win).click();
      await settle();
      assert.strictEqual(
        vosk.createModelCalls,
        2,
        'the third attempt must not wait on the stalled load again',
      );
      assert.ok(
        micBtn(win).classList.contains('voice-loading'),
        `the fresh load must be visibly loading: ${micBtn(win).className}`,
      );
      vosk.settlers[1].resolve(fakeModel());
      await settle();
      assert.strictEqual(mic.calls.length, 1, 'the fresh model opens the mic');
      assert.ok(!vosk.settlers[1].cancelled, 'the fresh load is never cancelled');
      win.close();
    },
  );

  await test(
    'W10/F1 integrated: real vosk.js+voice.js — a worker that never reports ' +
      'is terminated when its load is given up, so retries never pile up ' +
      'live workers',
    async () => {
      const page = makePage({
        realVosk: true,
        cfg: {modelLoadTimeoutMs: LOAD_TIMEOUT_MS},
      });
      const {win, bootWorkers, fireLoadTimeout} = page;
      for (let attempt = 1; attempt <= 6; attempt++) {
        micBtn(win).click();
        await settle();
        fireLoadTimeout();
        await settle();
        assert.ok(micBtn(win).classList.contains('voice-error'));
        const live = bootWorkers.filter(w => !w.terminated);
        assert.ok(
          live.length <= 1,
          `attempt ${attempt}: ${live.length} live workers of ${bootWorkers.length}`,
        );
        assert.strictEqual(
          live.length,
          attempt % 2,
          `attempt ${attempt}: the given-up worker must be terminated`,
        );
      }
      assert.strictEqual(bootWorkers.length, 3, 'one fresh load per give-up');
      assert.ok(bootWorkers.every(w => w.terminated), 'all hard-terminated');
      win.close();
    },
  );

  await test(
    'CAND-4: a load failure reported before the timeout still paints the ' +
      'error and clears busy (regression guard)',
    async () => {
      const page = makePage({cfg: {modelLoadTimeoutMs: 5000}});
      const {win, vosk} = page;
      micBtn(win).click();
      await settle();
      vosk.settlers[0].reject(new Error('model download failed'));
      await settle();
      const btn = micBtn(win);
      assert.ok(
        btn.classList.contains('voice-error'),
        `a real load failure must stay visible: ${btn.className}`,
      );
      assert.ok(
        btn.getAttribute('data-tooltip').includes('model download failed'),
        `the failure reason must be shown: ${btn.getAttribute('data-tooltip')}`,
      );
      micBtn(win).click();
      await settle();
      assert.strictEqual(vosk.createModelCalls, 2, 'retry must be possible');
      win.close();
    },
  );

  await test(
    'CAND-4: a bare rejection (the vendored reject() carries no error) ' +
      'still paints a readable reason',
    async () => {
      const page = makePage({cfg: {modelLoadTimeoutMs: 5000}});
      const {win, vosk} = page;
      micBtn(win).click();
      await settle();
      vosk.settlers[0].reject(undefined);
      await settle();
      const btn = micBtn(win);
      assert.ok(
        btn.classList.contains('voice-error'),
        `the failure must be visible: ${btn.className}`,
      );
      assert.ok(
        btn
          .getAttribute('data-tooltip')
          .includes('failed to load speech model'),
        `a message must be supplied: ${btn.getAttribute('data-tooltip')}`,
      );
      win.close();
    },
  );

  await test('CAND-5: a model that fails to load terminates its spawned worker', async () => {
    const {win, workers} = makeVoskWindow(fakeWorkerClass);
    let rejected = 0;
    const p = win.Vosk.createModel(MODEL_URL);
    p.catch(() => {
      rejected++;
    });
    assert.strictEqual(workers.length, 1, 'the model must spawn a worker');
    const worker = workers[0];
    assert.ok(
      worker.posted.some(m => m && m.action === 'load'),
      'the model must have asked its worker to load: ' +
        JSON.stringify(worker.posted),
    );
    // The worker booted, but the model failed to load (bad modelUrl).
    worker.emit({event: 'load', result: false});
    await settle();
    assert.strictEqual(rejected, 1, 'createModel must reject');
    // Hard-terminated: the worker's own {action:'terminate'} handler
    // deletes the model before close(), and throws on one never built.
    assert.ok(
      worker.terminated,
      'the failed model must hard-terminate its worker: ' +
        JSON.stringify(worker.posted),
    );
    win.close();
  });

  await test('W10/F1: createModel(...).cancel() terminates a worker that never reports and rejects once; after the load it is a no-op', async () => {
    const {win, workers} = makeVoskWindow(fakeWorkerClass);
    let rejection = null;
    const load = win.Vosk.createModel(MODEL_URL);
    load.catch(e => {
      rejection = e;
    });
    load.cancel();
    load.cancel();
    await settle();
    assert.ok(workers[0].terminated, 'the silent worker is hard-terminated');
    assert.match(rejection && rejection.message, /Gave up loading the speech model/);
    const loaded = win.Vosk.createModel(MODEL_URL);
    let model = null;
    loaded.then(m => {
      model = m;
    });
    workers[1].emit({event: 'load', result: true});
    // Not awaited: a load that never settled holds no handle, so a bare
    // await of it would let node exit 0 with the later tests unrun.
    await settle();
    assert.ok(model, 'createModel must resolve once its worker reports');
    loaded.cancel();
    await settle();
    assert.ok(
      !workerReclaimed(workers[1]),
      'cancel after the load leaves it alone',
    );
    assert.ok(model.ready, 'the loaded model stays ready');
    win.close();
  });

  await test('CAND-5: every retry frees its own worker — no leak per attempt', async () => {
    const {win, workers} = makeVoskWindow(fakeWorkerClass);
    for (let attempt = 0; attempt < 3; attempt++) {
      let rejected = 0;
      const p = win.Vosk.createModel(MODEL_URL);
      p.catch(() => {
        rejected++;
      });
      assert.strictEqual(workers.length, attempt + 1);
      workers[attempt].emit({event: 'load', result: false});
      await settle();
      assert.strictEqual(rejected, 1, `attempt ${attempt} must reject`);
    }
    const leaked = workers.filter(w => !w.terminated);
    assert.strictEqual(
      leaked.length,
      0,
      `${leaked.length} of ${workers.length} workers leaked`,
    );
    win.close();
  });

  await test(
    'CAND-6: a Worker that errors before ever posting load is hard-' +
      'terminated and createModel rejects',
    async () => {
      const {win, workers} = makeVoskWindow(fakeWorkerClass);
      let rejection = null;
      const p = win.Vosk.createModel(MODEL_URL);
      p.catch(e => {
        rejection = e;
      });
      assert.strictEqual(workers.length, 1, 'the model must spawn a worker');
      const worker = workers[0];
      assert.ok(
        worker.errorListenerCount() > 0,
        'createModel must listen for Worker boot errors',
      );
      // The Worker constructed, then its script failed to boot: it
      // emits `error` and never posts a load message.  It cannot
      // process the protocol {action:'terminate'} message either, so
      // only the real Worker.terminate() can reclaim it.
      worker.emitError({message: 'worker boot failed'});
      await settle();
      assert.ok(rejection, 'createModel must reject on a Worker boot error');
      assert.ok(
        String(rejection.message).includes('worker boot failed'),
        'the rejection carries the boot error detail: ' + rejection.message,
      );
      assert.strictEqual(
        worker.terminated,
        true,
        'the boot-failed Worker must be reclaimed with Worker.terminate()',
      );
      assert.strictEqual(
        worker.errorListenerCount(),
        0,
        'the one-shot boot-error listener is removed on settlement',
      );
      win.close();
    },
  );

  await test('CAND-6: every boot-failure retry frees its own worker — no leak', async () => {
    const {win, workers} = makeVoskWindow(fakeWorkerClass);
    for (let attempt = 0; attempt < 3; attempt++) {
      let rejected = 0;
      const p = win.Vosk.createModel(MODEL_URL);
      p.catch(() => {
        rejected++;
      });
      assert.strictEqual(workers.length, attempt + 1);
      workers[attempt].emitError({message: 'boot failed'});
      await settle();
      assert.strictEqual(rejected, 1, `attempt ${attempt} must reject`);
    }
    const leaked = workers.filter(w => !w.terminated);
    assert.strictEqual(
      leaked.length,
      0,
      `${leaked.length} of ${workers.length} boot-failed workers leaked`,
    );
    win.close();
  });

  await test(
    'CAND-6: a Worker error AFTER a successful load does not kill the ' +
      'live model (the boot listener is one-shot)',
    async () => {
      const {win, workers} = makeVoskWindow(fakeWorkerClass);
      let model = null;
      win.Vosk.createModel(MODEL_URL).then(m => {
        model = m;
      });
      workers[0].emit({event: 'load', result: true});
      await settle();
      assert.ok(model, 'createModel must resolve');
      workers[0].emitError({message: 'late runtime error'});
      await settle();
      assert.ok(
        !workerReclaimed(workers[0]),
        'a runtime error after load must not terminate the live worker',
      );
      assert.ok(model.ready, 'the model stays ready');
      win.close();
    },
  );

  await test(
    'CAND-5: a successful load resolves with a live model and does not ' +
      'terminate its worker (regression guard)',
    async () => {
      const {win, workers} = makeVoskWindow(fakeWorkerClass);
      let model = null;
      const p = win.Vosk.createModel(MODEL_URL);
      p.then(m => {
        model = m;
      });
      workers[0].emit({event: 'load', result: true});
      await settle();
      assert.ok(model, 'createModel must resolve with the model');
      assert.ok(model.ready, 'the resolved model must be ready');
      assert.ok(
        !workerReclaimed(workers[0]),
        'a healthy worker must not be terminated: ' +
          JSON.stringify(workers[0].posted),
      );
      win.close();
    },
  );

  await test(
    "W5 F1: the worker's {event:'error'} load failure rejects createModel " +
      'with the reason, hard-terminates the Worker and drops the listeners',
    async () => {
      const {win, workers} = makeVoskWindow(fakeWorkerClass);
      const modelListeners = modelErrorListeners(win);
      let rejection = null;
      win.Vosk.createModel(MODEL_URL).catch(e => {
        rejection = e;
      });
      assert.strictEqual(workers.length, 1, 'the model must spawn a worker');
      const worker = workers[0];
      assert.ok(
        worker.posted.some(m => m && m.action === 'load'),
        'the model must have asked its worker to load',
      );
      assert.strictEqual(
        modelListeners.length,
        1,
        'createModel must listen for the protocol error while loading',
      );
      // The real worker: LoadVosk()/IDBFS/downloadAndExtract rejected, so
      // RecognizerWorker posts the protocol error and no load message.
      worker.emit({event: 'error', error: LOAD_ERROR});
      await settle();
      assert.ok(rejection, 'createModel must reject on the protocol error');
      assert.ok(
        String(rejection.message).includes(LOAD_ERROR),
        'the rejection carries the load error detail: ' + rejection.message,
      );
      assert.ok(
        String(rejection.message).includes(MODEL_URL),
        'the rejection names the model: ' + rejection.message,
      );
      // The worker never built its Kaldi model, so the protocol
      // {action:'terminate'} would throw in its terminate handler before
      // close(): only Worker.terminate() reclaims it.
      assert.strictEqual(
        worker.terminated,
        true,
        'the load-failed Worker must be reclaimed with Worker.terminate()',
      );
      assert.strictEqual(
        worker.errorListenerCount(),
        0,
        'the one-shot boot-error listener is removed on settlement',
      );
      assert.strictEqual(
        modelListeners.length,
        0,
        'the one-shot protocol-error listener is removed on settlement',
      );
      win.close();
    },
  );

  await test(
    'W5 F1: every load-failure retry frees its own Worker, and a model ' +
      'that loaded keeps its Worker through a later protocol error',
    async () => {
      const {win, workers} = makeVoskWindow(fakeWorkerClass);
      const modelListeners = modelErrorListeners(win);
      for (let attempt = 0; attempt < 3; attempt++) {
        let rejected = 0;
        win.Vosk.createModel(MODEL_URL).catch(() => {
          rejected++;
        });
        assert.strictEqual(workers.length, attempt + 1);
        workers[attempt].emit({event: 'error', error: LOAD_ERROR});
        await settle();
        assert.strictEqual(rejected, 1, `attempt ${attempt} must reject`);
      }
      const leaked = workers.filter(w => !w.terminated);
      assert.strictEqual(
        leaked.length,
        0,
        `${leaked.length} of ${workers.length} load-failed workers leaked`,
      );
      // A successful load: the protocol-error listener goes away with
      // the boot listener, so a runtime error (an unknown message, a
      // recognizer the worker no longer knows) cannot kill the model.
      let model = null;
      win.Vosk.createModel(MODEL_URL).then(m => {
        model = m;
      });
      const live = workers[3];
      live.emit({event: 'load', result: true});
      await settle();
      assert.ok(model, 'createModel must resolve');
      assert.strictEqual(modelListeners.length, 0, 'listener dropped on load');
      live.emit({event: 'error', error: 'Unknown message {}'});
      live.emit({event: 'error', recognizerId: 'gone', error: 'no such'});
      await settle();
      assert.ok(
        !workerReclaimed(live),
        'a late error must not kill the live model',
      );
      assert.ok(model.ready, 'the model stays ready');
      win.close();
    },
  );

  await test(
    'W5 F1 integrated: real main.js+voice.js+vosk.js — a 404 model ' +
      'download paints its reason, and the retry downloads afresh',
    async () => {
      const {win, bootWorkers} = makePage({
        realVosk: true,
        workerFailure: {loadError: LOAD_ERROR},
        cfg: {modelLoadTimeoutMs: 5000},
      });
      micBtn(win).click();
      await waitFor(
        () => micBtn(win).classList.contains('voice-error'),
        'the mic button must recover to the error state',
      );
      const tooltip = micBtn(win).getAttribute('data-tooltip');
      assert.ok(
        tooltip.includes(LOAD_ERROR),
        `the download failure must be shown, not a timeout: ${tooltip}`,
      );
      assert.strictEqual(bootWorkers.length, 1, 'one Worker for the first try');
      assert.strictEqual(
        bootWorkers[0].terminated,
        true,
        'and it is reclaimed',
      );
      // The outage is over as far as voice.js can know: the retry must
      // start a fresh download instead of reusing the dead promise.
      micBtn(win).click();
      await waitFor(
        () => bootWorkers.length === 2 && bootWorkers[1].terminated,
        'the retry must create a second Worker and reclaim it in turn',
      );
      assert.ok(
        micBtn(win).classList.contains('voice-error'),
        `the retry fails visibly again: ${micBtn(win).className}`,
      );
      win.close();
    },
  );

  await test(
    'CAND-6 integrated: real main.js+vosk.js+voice.js — a Worker boot ' +
      'error recovers the mic UI AND terminates the Worker before retry',
    async () => {
      const page = makePage({
        realVosk: true,
        workerFailure: {bootError: 'worker boot failed'},
        cfg: {modelLoadTimeoutMs: 5000},
      });
      const {win, bootWorkers} = page;
      micBtn(win).click();
      await waitFor(
        () => micBtn(win).classList.contains('voice-error'),
        'the mic button must recover to the error state',
      );
      assert.strictEqual(
        bootWorkers.length,
        1,
        'the first model attempt created one Worker',
      );
      assert.strictEqual(
        bootWorkers[0].terminated,
        true,
        'the boot-failed Worker is terminated, not leaked',
      );
      // Retry: a fresh Worker is created and reclaimed in turn.
      micBtn(win).click();
      await waitFor(
        () => bootWorkers.length === 2 && bootWorkers[1].terminated,
        'the retry must create a second Worker and reclaim it',
      );
      const leaked = bootWorkers.filter(w => !w.terminated);
      assert.strictEqual(
        leaked.length,
        0,
        `${leaked.length} boot-failed Workers leaked across retries`,
      );
      win.close();
    },
  );

  report('conc2026_vosk_model_lifecycle');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  },
);
