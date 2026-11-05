// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E regression for audit 2026-11-01 W5 F1: the vendored vosk.js worker
// reports a failed model load (WASM init, IndexedDB, the model download
// answering 404, extraction, Kaldi model construction) as the protocol
// message {event:'error', error:<message>} and never as {event:'load',
// result:false}.  createModel listened only for 'load' and for the
// native Worker 'error' event, so a real load failure left createModel
// pending for ever with its Worker alive; voice.js then timed out, and
// because its page-wide modelPromise is only forgotten on rejection,
// every later enable reused the same dead promise: no new download until
// the page was reloaded.
//
// Both tests failed against the pre-fix vosk.js: the bare one because
// createModel never settled and the Worker was never terminated, the
// integrated one because the retry never called createModel again.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {
  makePage: makeChatPage,
  stubRaf,
  sleep,
  waitFor,
  MEDIA,
} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {micBtn, makeVoskWindow} = require('./voice_page_helpers');

const {test, report} = makeRunner();

const MODEL_URL = 'https://models.example.com/model.tar.gz';
const LOAD_ERROR = 'HTTP error! status: 404';

const FALLBACK_CFG = {
  voskSrc: '/media/vosk.js',
  modelUrl: MODEL_URL,
  ackAudioUrl: '/media/working-on-it.mp3',
  nonce: 'test-nonce',
};

/**
 * A Worker stand-in for jsdom (which has none) that plays the real
 * worker's load-failure protocol: on the {action:'load'} message it
 * posts {event:'error', error} (what the embedded RecognizerWorker does
 * when load() rejects) and never a 'load' message.  With `loadError`
 * null the test plays the worker by hand through emit().
 */
function fakeWorkerClass(workers, loadError) {
  return class FakeWorker {
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
    errorListenerCount() {
      return (this.listeners.get('error') || []).length;
    }
  };
}

/** The REAL media/vosk.js in a bare window with a hand-played Worker. */
/** Count the 'error' listeners registered on the page-side Model. */
function modelErrorListeners(win) {
  const counts = [];
  const realAdd = win.EventTarget.prototype.addEventListener;
  const realRemove = win.EventTarget.prototype.removeEventListener;
  win.EventTarget.prototype.addEventListener = function (type, cb, opts) {
    if (type === 'error' && this instanceof win.Vosk.Model) counts.push(cb);
    return realAdd.call(this, type, cb, opts);
  };
  win.EventTarget.prototype.removeEventListener = function (type, cb, opts) {
    if (type === 'error' && this instanceof win.Vosk.Model) {
      const i = counts.indexOf(cb);
      if (i >= 0) counts.splice(i, 1);
    }
    return realRemove.call(this, type, cb, opts);
  };
  return counts;
}

/**
 * A browser-mode chat page running the REAL main.js + voice.js +
 * vosk.js, whose Workers answer every model load with the protocol
 * error message.
 */
function makePage(cfg) {
  const workers = [];
  const {win, posted} = makeChatPage({
    beforeScripts: win => installVoiceStubs(win, cfg, workers),
    extraScripts: ['voice.js'],
  });
  win.eval(fs.readFileSync(path.join(MEDIA, 'vosk.js'), 'utf8'));
  return {win, posted, workers};
}

function installVoiceStubs(win, cfg, workers) {
  stubRaf(win);
  win.Audio = function (src) {
    return {src, play: () => Promise.resolve()};
  };
  Object.defineProperty(win.navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia: () => new Promise(() => {}),
      enumerateDevices: () => Promise.resolve([]),
    },
  });
  win.Worker = fakeWorkerClass(workers, LOAD_ERROR);
  if (typeof win.URL.createObjectURL !== 'function') {
    win.URL.createObjectURL = () => 'blob:fake-vosk-worker';
  }
  win.__VOICE__ = Object.assign({}, FALLBACK_CFG, cfg || {});
  win.localStorage.setItem('kissVoiceEnabled', '0');
}

async function main() {
  await test(
    "W5 F1: the worker's {event:'error'} load failure rejects createModel " +
      'with the reason, hard-terminates the Worker and drops the listeners',
    async () => {
      const {win, workers} = makeVoskWindow(fakeWorkerClass);
      const modelListeners = modelErrorListeners(win);
      let rejection = null;
      const p = win.Vosk.createModel(MODEL_URL);
      p.catch(e => {
        rejection = e;
      });
      assert.strictEqual(workers.length, 1, 'the model must spawn a worker');
      const worker = workers[0];
      assert.ok(
        worker.posted.some(m => m && m.action === 'load'),
        'the model must have asked its worker to load',
      );
      // The real worker: LoadVosk()/IDBFS/downloadAndExtract rejected, so
      // RecognizerWorker posts the protocol error and no load message.
      worker.emit({event: 'error', error: LOAD_ERROR});
      await sleep(20);
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
      // A late load message from the dead worker must stay ignored.
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
        await sleep(20);
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
      await sleep(20);
      assert.ok(model, 'createModel must resolve');
      assert.strictEqual(modelListeners.length, 0, 'listener dropped on load');
      live.emit({event: 'error', error: 'Unknown message {}'});
      live.emit({event: 'error', recognizerId: 'gone', error: 'no such'});
      await sleep(20);
      assert.ok(!live.terminated, 'a late error must not kill the live model');
      assert.ok(model.ready, 'the model stays ready');
      win.close();
    },
  );

  await test(
    'W5 F1 integrated: real main.js+voice.js+vosk.js — a 404 model ' +
      'download paints its reason, and the retry downloads afresh',
    async () => {
      const {win, workers} = makePage({modelLoadTimeoutMs: 5000});
      micBtn(win).click();
      await waitFor(
        () => micBtn(win).classList.contains('voice-error'),
        'the mic button must recover to the error state',
        2000,
      );
      const btn = micBtn(win);
      assert.ok(
        btn.getAttribute('data-tooltip').includes(LOAD_ERROR),
        `the download failure must be shown, not a timeout: ${btn.getAttribute(
          'data-tooltip',
        )}`,
      );
      assert.strictEqual(workers.length, 1, 'one Worker for the first try');
      assert.strictEqual(workers[0].terminated, true, 'and it is reclaimed');
      // The outage is over as far as voice.js can know: the retry must
      // start a fresh download instead of reusing the dead promise.
      micBtn(win).click();
      await waitFor(
        () => workers.length === 2 && workers[1].terminated,
        'the retry must create a second Worker and reclaim it in turn',
        2000,
      );
      assert.ok(
        micBtn(win).classList.contains('voice-error'),
        `the retry fails visibly again: ${micBtn(win).className}`,
      );
      win.close();
    },
  );

  report('audit1101_W5_vosk_load_error');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  },
);
