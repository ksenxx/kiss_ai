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
  voicePageState,
  stubVoiceSurface,
  micBtn,
  settle,
} = require('./voice_page_helpers');

const VOICE_CFG = {
  voskSrc: '/media/vosk.js',
  modelUrl: 'https://models.example.com/model.tar.gz',
  ackAudioUrl: '/media/working-on-it.mp3',
  nonce: 'test-nonce',
};

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
      this.listeners = [];
      this.posted = [];
      this.terminated = false;
      workers.push(this);
    }
    addEventListener(type, cb) {
      if (type === 'message') this.listeners.push(cb);
    }
    removeEventListener() {}
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
        for (const cb of this.listeners.slice()) cb({data});
      });
    }
  };
}

function makeVoicePage() {
  const workers = [];
  const page = voicePageState({autoGrant: true, noVosk: true});
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
  assert.strictEqual(created.length, 1, 'the wake recognizer was asked for');

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
    2,
    'a fresh recognizer was asked for',
  );
  assert.strictEqual(state.tracks.length, 2, 'a fresh microphone request');
  assert.ok(btn.classList.contains('voice-listening'), btn.className);
  win.close();
  console.log('voiceRecognizerError.test.js passed');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
