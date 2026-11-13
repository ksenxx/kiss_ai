// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// voice.js: `new model.KaldiRecognizer(...)` only posts {action:'create'}
// to the speech worker; the worker answers a failure (Kaldi could not
// build the recognizer: out of memory, a bad grammar) later, as the
// protocol message {event:'error', recognizerId, error}, which vosk.js
// routes to that recognizer's 'error' event.  The pipeline listened for
// 'result' and 'partialresult' only, so such a failure left the
// microphone open and the button saying "listening" while nothing could
// ever wake.  It must come down like a lost microphone does: mic off,
// the reason on the button, the next click retries on the same model.
//
// The real main.js + voice.js + vosk.js run over the stubbed voice
// surface (voice_page_helpers); the Worker is played by hand.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {makePage, waitFor, MEDIA} = require('./simplify2_harness');
const {
  VOICE_CFG,
  voicePageState,
  stubVoiceSurface,
  micBtn,
  settle,
} = require('./voice_page_helpers');

/**
 * A Worker stand-in for jsdom (which has none): the model loads, and a
 * recognizer create is silence (success) until the test answers it
 * with `failRecognizer(id)`, the worker's creation failure.  A real
 * worker answers after the page has gone live on the new recognizer;
 * an answer queued at the create would make the listening state last
 * one event-loop turn, which no poll can be trusted to see.  Messages
 * are delivered on Node's own loop so the page's recorded timers stay
 * out of it.
 */
function fakeWorkerClass(workers) {
  return class FakeWorker {
    constructor() {
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
      if (msg && msg.action === 'load') {
        this.emit({event: 'load', result: true});
      }
    }
    failRecognizer(recognizerId) {
      this.emit({
        event: 'error',
        recognizerId,
        error:
          'Recognizer (id: ' +
          recognizerId +
          '): Could not be created due to: out of memory',
      });
    }
    terminate() {
      this.terminated = true;
    }
    emit(data) {
      setImmediate(() => {
        for (const cb of (this.listeners.get('message') || []).slice())
          cb({data});
      });
    }
    /** The worker died: the standard asynchronous Worker `error` event. */
    emitError(err) {
      setImmediate(() => {
        for (const cb of (this.listeners.get('error') || []).slice()) cb(err);
      });
    }
  };
}

function makeVoicePage(options) {
  const workers = [];
  const page = voicePageState(
    Object.assign({autoGrant: true, noVosk: true}, options),
  );
  const {win} = makePage({
    beforeScripts: win => {
      stubVoiceSurface(win, page);
      win.Worker = fakeWorkerClass(workers);
      if (typeof win.URL.createObjectURL !== 'function') {
        win.URL.createObjectURL = () => 'blob:fake-vosk-worker';
      }
      win.__VOICE__ = Object.assign({}, VOICE_CFG);
      win.localStorage.setItem('kissVoiceEnabled', '0');
    },
    extraScripts: ['voice.js'],
  });
  win.eval(fs.readFileSync(path.join(MEDIA, 'vosk.js'), 'utf8'));
  return Object.assign(page, {win, workers});
}

async function main() {
  const page = makeVoicePage();
  const {win, log, state, workers} = page;
  const btn = micBtn(win);
  btn.click();
  await waitFor(
    () => btn.classList.contains('voice-listening'),
    'the pipeline comes up on the loaded model',
  );
  assert.strictEqual(workers.length, 1, 'one speech worker');
  assert.strictEqual(state.tracks.length, 1, 'the microphone was granted');
  const created = workers[0].posted.filter(m => m.action === 'create');
  assert.strictEqual(
    created.length,
    2,
    'the wake recognizer and its grammar-free check were asked for',
  );

  // The worker's creation failure arrives: no recognizer exists.
  workers[0].failRecognizer(created[0].recognizerId);
  await waitFor(
    () => btn.classList.contains('voice-error'),
    'the failed recognizer must take the pipeline down',
  );
  assert.ok(!btn.classList.contains('voice-listening'));
  assert.strictEqual(btn.getAttribute('aria-pressed'), 'false');
  assert.ok(
    /out of memory/.test(btn.getAttribute('data-tooltip') || ''),
    'the reason is on the button: ' + btn.getAttribute('data-tooltip'),
  );
  assert.deepStrictEqual(
    log,
    ['ctx-open', 'track-stopped', 'ctx-closed'],
    'the microphone and the audio graph are released',
  );
  assert.strictEqual(win.localStorage.getItem('kissVoiceEnabled'), '0');
  assert.ok(!workers[0].terminated, 'the loaded model is kept for a retry');

  // The next click retries on the same model; this time Kaldi obliges.
  btn.click();
  await waitFor(
    () => btn.classList.contains('voice-listening'),
    'the retry comes back live',
  );
  await settle();
  assert.strictEqual(workers.length, 1, 'no second model download');
  assert.strictEqual(
    workers[0].posted.filter(m => m.action === 'create').length,
    4,
    'a fresh recognizer pair was asked for',
  );
  assert.strictEqual(state.tracks.length, 2, 'a fresh microphone request');
  assert.ok(btn.classList.contains('voice-listening'), btn.className);
  win.close();

  // A worker that dies after its load (an uncaught error in the worker:
  // it decodes nothing any more) takes the live pipeline down the same
  // way, and the dead model is not kept: the next click loads a fresh one.
  const crashed = makeVoicePage();
  const crashedBtn = micBtn(crashed.win);
  crashedBtn.click();
  await waitFor(
    () => crashedBtn.classList.contains('voice-listening'),
    'the pipeline comes up on the loaded model',
  );
  crashed.workers[0].emitError({message: 'the speech worker crashed'});
  await waitFor(
    () => crashedBtn.classList.contains('voice-error'),
    'a worker that died must take the pipeline down',
  );
  assert.ok(!crashedBtn.classList.contains('voice-listening'));
  assert.strictEqual(crashedBtn.getAttribute('aria-pressed'), 'false');
  assert.ok(
    /worker crashed/.test(crashedBtn.getAttribute('data-tooltip') || ''),
    'the reason is on the button: ' + crashedBtn.getAttribute('data-tooltip'),
  );
  assert.deepStrictEqual(
    crashed.log,
    ['ctx-open', 'track-stopped', 'ctx-closed'],
    'the microphone and the audio graph are released',
  );
  assert.ok(crashed.workers[0].terminated, 'the dead worker is reclaimed');
  crashedBtn.click();
  await waitFor(
    () => crashedBtn.classList.contains('voice-listening'),
    'the retry comes back live on a fresh model',
  );
  assert.strictEqual(crashed.workers.length, 2, 'a fresh model was loaded');
  assert.ok(!crashed.workers[1].terminated, 'the fresh worker is live');
  crashed.win.close();

  // A worker that dies while a start is still in flight (the microphone
  // prompt is unanswered) fails that start, and the error stays on the
  // button when the superseded start settles on the late grant.
  const starting = makeVoicePage({autoGrant: false});
  const startingBtn = micBtn(starting.win);
  startingBtn.click();
  await waitFor(
    () => starting.mic.calls.length === 1,
    'the microphone is asked for',
  );
  starting.workers[0].emitError({message: 'the speech worker crashed'});
  await waitFor(
    () => startingBtn.classList.contains('voice-error'),
    'a worker that died must fail the start',
  );
  starting.mic.calls[0].grant();
  await settle();
  await settle();
  assert.ok(
    startingBtn.classList.contains('voice-error') &&
      /worker crashed/.test(startingBtn.getAttribute('data-tooltip') || ''),
    'the error must outlive the start it failed: ' + startingBtn.className,
  );
  assert.deepStrictEqual(
    starting.log,
    ['track-stopped'],
    'the late microphone is released and no audio graph is opened',
  );
  starting.win.close();
  console.log('voiceRecognizerError.test.js passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
