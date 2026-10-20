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
// The CAND-4 harness mimics voiceBrowserPipelineLifecycle.test.js: real
// voice.js + main.js in jsdom, with the embedder capabilities (Vosk,
// AudioContext, getUserMedia) stubbed the way that suite stubs them.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {
  makePage: makeChatPage,
  stubRaf,
  sleep,
  MEDIA,
} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');

const {test, report} = makeRunner();

const FALLBACK_CFG = {
  voskSrc: '/media/vosk.js',
  modelUrl: 'https://models.example.com/model.tar.gz',
  ackAudioUrl: '/media/working-on-it.mp3',
  nonce: 'test-nonce',
};

/** A KaldiRecognizer with the surface the pipeline uses. */
class FakeRecognizer {
  setWords() {}
  on() {}
  acceptWaveform() {}
  retrieveFinalResult() {}
  remove() {}
}

/** A loaded model with the surface the pipeline uses. */
function fakeModel() {
  return {KaldiRecognizer: FakeRecognizer, terminate() {}};
}

/**
 * A browser-mode page whose window.Vosk.createModel returns promises the
 * test settles (or never settles) by hand, counting every call.
 */
function makePage(opts) {
  const options = opts || {};
  const mic = {calls: []};
  const vosk = {createModelCalls: 0, settlers: []};
  const bootWorkers = [];
  const {win, posted} = makeChatPage({
    beforeScripts: win =>
      installVoiceStubs(win, options, mic, vosk, bootWorkers),
    extraScripts: ['voice.js'],
  });
  // voice.js only reads window.Vosk lazily (loadVosk at mic click), so
  // the real bundle may follow it.
  if (options.realVosk) {
    win.eval(fs.readFileSync(path.join(MEDIA, 'vosk.js'), 'utf8'));
  }
  return {win, posted, mic, vosk, bootWorkers};
}

function installVoiceStubs(win, options, mic, vosk, bootWorkers) {
  stubRaf(win);
  win.Audio = function (src) {
    return {src, play: () => Promise.resolve()};
  };

  Object.defineProperty(win.navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia: () => {
        return new Promise(resolve => {
          mic.calls.push({
            grant: () =>
              resolve({
                getTracks: () => [{stop: () => {}}],
                getAudioTracks: () => [],
              }),
          });
        });
      },
      enumerateDevices: () => Promise.resolve([]),
    },
  });

  if (options.realVosk) {
    // Integrated mode: the REAL vendored vosk.js is loaded below and
    // window.Vosk comes from it.  jsdom has no Web Workers, so this
    // fake stands in: it constructs fine, then its script fails to
    // boot and it raises the standard asynchronous `error` event
    // without ever posting a load message (the leak the boot-error
    // path exists for).
    win.Worker = class BootFailingWorker {
      constructor(url) {
        this.url = url;
        this.listeners = new Map();
        this.posted = [];
        this.terminated = false;
        bootWorkers.push(this);
        setTimeout(() => {
          for (const cb of (this.listeners.get('error') || []).slice())
            cb({message: 'worker boot failed'});
        }, 5);
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
      }
      terminate() {
        this.terminated = true;
      }
    };
    if (typeof win.URL.createObjectURL !== 'function') {
      win.URL.createObjectURL = () => 'blob:fake-vosk-worker';
    }
  } else {
    win.Vosk = {
      createModel: () => {
        vosk.createModelCalls++;
        return new Promise((resolve, reject) => {
          vosk.settlers.push({resolve, reject});
        });
      },
    };
  }
  win.AudioContext = class FakeAudioContext {
    constructor() {
      this.sampleRate = 16000;
      this.state = 'running';
      this.destination = {};
    }
    resume() {
      return Promise.resolve();
    }
    close() {
      return Promise.resolve();
    }
    createMediaStreamSource() {
      return {connect: () => {}, disconnect: () => {}};
    }
    createScriptProcessor() {
      return {connect: () => {}, disconnect: () => {}, onaudioprocess: null};
    }
  };

  win.__VOICE__ = Object.assign({}, FALLBACK_CFG, options.cfg || {});
  win.localStorage.setItem('kissVoiceEnabled', '0');
}

function micBtn(win) {
  return win.document.getElementById('voice-btn');
}

// ---------------------------------------------------------------------
// CAND-5 harness: the REAL media/vosk.js in a bare window. jsdom has no
// Web Workers, so the Worker handed to vosk's WorkerFactory is a fake
// that records postMessage traffic and terminate() calls and lets the
// test play the worker's side of the protocol.
// ---------------------------------------------------------------------

function makeVoskWindow() {
  const dom = new JSDOM('<!doctype html><html><body></body></html>', {
    runScripts: 'dangerously',
    url: 'https://localhost/',
  });
  const win = dom.window;
  const workers = [];
  win.Worker = class FakeWorker {
    constructor(url) {
      this.url = url;
      this.listeners = new Map();
      this.posted = [];
      this.terminated = false;
      workers.push(this);
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
  if (typeof win.URL.createObjectURL !== 'function') {
    win.URL.createObjectURL = () => 'blob:fake-worker-url';
  }
  win.eval(fs.readFileSync(path.join(MEDIA, 'vosk.js'), 'utf8'));
  assert.ok(win.Vosk && win.Vosk.createModel, 'vosk.js must export Vosk');
  return {win, workers};
}

/** True when the worker was told to die: DOM terminate() or the vosk
 *  protocol's {action:'terminate'} message (its worker handles that by
 *  freeing the Kaldi objects and closing itself). */
function workerReclaimed(worker) {
  return (
    worker.terminated || worker.posted.some(m => m && m.action === 'terminate')
  );
}

async function main() {
  await test(
    'CAND-4: a model load that never settles times out into the error ' +
      'state instead of wedging voice on loading forever',
    async () => {
      const page = makePage({cfg: {modelLoadTimeoutMs: 60}});
      const {win, vosk} = page;
      micBtn(win).click(); // browser mode: pipeline starts directly
      await sleep(20);
      assert.strictEqual(vosk.createModelCalls, 1);
      assert.ok(
        micBtn(win).classList.contains('voice-loading'),
        `must be loading while the model loads: ${micBtn(win).className}`,
      );
      // The worker never posts 'load': createModel never settles.
      await sleep(150);
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
      const page = makePage({cfg: {modelLoadTimeoutMs: 60}});
      const {win, mic, vosk} = page;
      micBtn(win).click();
      await sleep(150); // load timed out; the error turned the mic off
      assert.ok(micBtn(win).classList.contains('voice-error'));
      micBtn(win).click(); // the user tries again
      await sleep(20);
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
      await sleep(20);
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
      const page = makePage({cfg: {modelLoadTimeoutMs: 60}});
      const {win, mic, vosk} = page;
      micBtn(win).click();
      await sleep(150); // timed out
      assert.ok(micBtn(win).classList.contains('voice-error'));
      vosk.settlers[0].resolve(fakeModel());
      await sleep(20);
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
      await sleep(20);
      assert.strictEqual(vosk.createModelCalls, 1, 'no second download');
      assert.strictEqual(mic.calls.length, 1, 'the kept model is used');
      win.close();
    },
  );

  await test(
    'CAND-4: a shared load that fails after the timeout is forgotten, so ' +
      'the next enable starts a fresh one',
    async () => {
      const page = makePage({cfg: {modelLoadTimeoutMs: 60}});
      const {win, vosk} = page;
      micBtn(win).click();
      await sleep(150); // timed out
      vosk.settlers[0].reject(new Error('model download failed'));
      await sleep(20);
      assert.ok(
        micBtn(win).classList.contains('voice-error'),
        `the late failure must not repaint the UI: ${micBtn(win).className}`,
      );
      micBtn(win).click();
      await sleep(20);
      assert.strictEqual(
        vosk.createModelCalls,
        2,
        'a failed download is not reused; the retry downloads again',
      );
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
      await sleep(20);
      vosk.settlers[0].reject(new Error('model download failed'));
      await sleep(20);
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
      await sleep(20);
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
      await sleep(20);
      vosk.settlers[0].reject(undefined);
      await sleep(20);
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
    const {win, workers} = makeVoskWindow();
    let rejected = 0;
    const p = win.Vosk.createModel('https://models.example.com/m.tar.gz');
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
    await sleep(20);
    assert.strictEqual(rejected, 1, 'createModel must reject');
    assert.ok(
      workerReclaimed(worker),
      'the failed model must terminate its worker: ' +
        JSON.stringify(worker.posted),
    );
    win.close();
  });

  await test('CAND-5: every retry frees its own worker — no leak per attempt', async () => {
    const {win, workers} = makeVoskWindow();
    for (let attempt = 0; attempt < 3; attempt++) {
      let rejected = 0;
      const p = win.Vosk.createModel('https://models.example.com/m.tar.gz');
      p.catch(() => {
        rejected++;
      });
      assert.strictEqual(workers.length, attempt + 1);
      workers[attempt].emit({event: 'load', result: false});
      await sleep(20);
      assert.strictEqual(rejected, 1, `attempt ${attempt} must reject`);
    }
    const leaked = workers.filter(w => !workerReclaimed(w));
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
      const {win, workers} = makeVoskWindow();
      let rejection = null;
      const p = win.Vosk.createModel('https://models.example.com/m.tar.gz');
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
      await sleep(20);
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
      // A late load message from a half-dead worker must stay ignored.
      let resolvedLate = false;
      p.then(() => {
        resolvedLate = true;
      }).catch(() => {});
      worker.emit({event: 'load', result: true});
      await sleep(20);
      assert.ok(!resolvedLate, 'a late load cannot resurrect the rejection');
      win.close();
    },
  );

  await test('CAND-6: every boot-failure retry frees its own worker — no leak', async () => {
    const {win, workers} = makeVoskWindow();
    for (let attempt = 0; attempt < 3; attempt++) {
      let rejected = 0;
      const p = win.Vosk.createModel('https://models.example.com/m.tar.gz');
      p.catch(() => {
        rejected++;
      });
      assert.strictEqual(workers.length, attempt + 1);
      workers[attempt].emitError({message: 'boot failed'});
      await sleep(20);
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
      const {win, workers} = makeVoskWindow();
      let model = null;
      win.Vosk.createModel('https://models.example.com/m.tar.gz').then(m => {
        model = m;
      });
      workers[0].emit({event: 'load', result: true});
      await sleep(20);
      assert.ok(model, 'createModel must resolve');
      workers[0].emitError({message: 'late runtime error'});
      await sleep(20);
      assert.ok(
        !workers[0].terminated,
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
      const {win, workers} = makeVoskWindow();
      let model = null;
      const p = win.Vosk.createModel('https://models.example.com/m.tar.gz');
      p.then(m => {
        model = m;
      });
      workers[0].emit({event: 'load', result: true});
      await sleep(20);
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
    'CAND-6 integrated: real main.js+vosk.js+voice.js — a Worker boot ' +
      'error recovers the mic UI AND terminates the Worker before retry',
    async () => {
      const page = makePage({
        realVosk: true,
        cfg: {modelLoadTimeoutMs: 5000},
      });
      const {win, bootWorkers} = page;
      micBtn(win).click();
      await sleep(150);
      assert.strictEqual(
        bootWorkers.length,
        1,
        'the first model attempt created one Worker',
      );
      assert.ok(
        micBtn(win).classList.contains('voice-error'),
        'the mic button recovers to the error state: ' + micBtn(win).className,
      );
      assert.strictEqual(
        bootWorkers[0].terminated,
        true,
        'the boot-failed Worker is terminated, not leaked',
      );
      // Retry: a fresh Worker is created and reclaimed in turn.
      micBtn(win).click();
      await sleep(150);
      assert.strictEqual(
        bootWorkers.length,
        2,
        'the retry created a second Worker',
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
