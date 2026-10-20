// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// A chat.html page running the real main.js + voice.js over a scriptable
// embedder surface.  jsdom has no speech engine, audio graph or
// microphone, so this module supplies them: a microphone the test grants
// or refuses, a fake Vosk model whose recognizers hand their result
// handlers to the test, audio processor nodes the test feeds, a clock
// for the wake cooldown and a ledger of the page's safety timers.

const assert = require('assert');
const {makePage, sleep, stubRaf} = require('./simplify2_harness');

const VOICE_CFG = {
  voskSrc: '/media/vosk.js',
  modelUrl: 'https://models.example.com/model.tar.gz',
  ackAudioUrl: '/media/working-on-it.mp3',
};

/**
 * A fake vosk Model: KaldiRecognizer instances register themselves in
 * `model.recognizers` and deregister on remove(), mirroring the real
 * vosk.js Model (registerRecognizer / unregisterRecognizer).  Every
 * recognizer ever built is also appended to `state.recognizers`.
 */
function makeFakeModel(state) {
  const model = {recognizers: new Map()};
  let nextId = 1;
  model.KaldiRecognizer = class FakeRecognizer {
    constructor(sampleRate, grammar) {
      this.id = 'rec-' + nextId++;
      this.sampleRate = sampleRate;
      this.grammar = grammar;
      this.handlers = {};
      this.removed = false;
      model.recognizers.set(this.id, this);
      state.recognizers.push(this);
    }
    setWords() {}
    on(event, cb) {
      this.handlers[event] = cb;
    }
    acceptWaveform() {}
    retrieveFinalResult() {}
    remove() {
      this.removed = true;
      model.recognizers.delete(this.id);
    }
  };
  return model;
}

/**
 * Install the voice pipeline's embedder surface on *win* before the page
 * scripts run: the clock, a silent `Audio`, the scriptable microphone,
 * the fake Vosk model, the audio context and the timer ledger.
 */
function stubVoiceSurface(win, page) {
  const {state, mic, clock, model, timers} = page;
  stubRaf(win);
  // triggerWake enforces a 2 s cooldown between wakes via Date.now(), so
  // tests advance the clock instead of sleeping.
  win.Date.now = () => clock.now;
  win.Audio = function (src) {
    return {src, play: () => Promise.resolve()};
  };
  Object.defineProperty(win.navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia: constraints =>
        new Promise((resolve, reject) => {
          mic.calls.push({
            constraints,
            grant: () =>
              resolve({
                getTracks: () => [
                  {
                    stop: () => {
                      state.tracksStopped++;
                    },
                  },
                ],
                getAudioTracks: () => [],
              }),
            refuse: message => {
              const err = new Error(message || 'Permission denied');
              err.name = 'NotAllowedError';
              reject(err);
            },
          });
        }),
      enumerateDevices: () => Promise.resolve([]),
    },
  });
  win.Vosk = {createModel: () => Promise.resolve(model)};
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
      const node = {
        connect: () => {},
        disconnect: () => {},
        onaudioprocess: null,
      };
      state.processors.push(node);
      return node;
    }
  };
  // The page's safety timers (the 45 s wake flash, the 60 s transcribing
  // flash) are recorded, still armed, so a test can fire one without
  // waiting (see fireTimer).
  const realSetTimeout = win.setTimeout;
  const realClearTimeout = win.clearTimeout;
  win.setTimeout = function (cb, ms) {
    const id = realSetTimeout(cb, ms);
    timers.push({id, cb, ms, cleared: false});
    return id;
  };
  win.clearTimeout = function (id) {
    for (const t of timers) if (t.id === id) t.cleared = true;
    realClearTimeout(id);
  };
}

/**
 * Build a chat page running the real main.js + voice.js over the stubbed
 * voice surface.
 *
 * @param {{enabled?: boolean, debug?: boolean, sourceUrl?: *}} [opts]
 *   `enabled` remembers the mic ON (`kissVoiceEnabled=1`, the pipeline
 *   starts on load; default off, so a test clicks the mic button);
 *   `debug` sets `kissVoiceDebug=1` (a free-text recognizer runs next to
 *   the wake one); `sourceUrl` is passed to simplify2_harness.makePage.
 * @returns {{win: object, posted: Array<object>, getState: Function,
 *   state: {recognizers: Array, processors: Array, tracksStopped: number},
 *   mic: {calls: Array<{constraints: *, grant: Function, refuse: Function}>},
 *   clock: {now: number}, model: object, timers: Array<object>}}
 *   `state.recognizers` / `state.processors` list every recognizer and
 *   audio processor node the pipeline ever built; `mic.calls` every
 *   getUserMedia request, settled by the test; `clock.now` is what the
 *   page's Date.now() returns; `model` is the fake Vosk model
 *   (`model.recognizers` maps the ids of the live recognizers); `timers`
 *   is the ledger of the page's setTimeout calls.
 */
function makeVoicePage(opts) {
  const options = opts || {};
  const state = {recognizers: [], processors: [], tracksStopped: 0};
  const page = {
    state,
    mic: {calls: []},
    clock: {now: 1000000},
    model: makeFakeModel(state),
    timers: [],
  };
  function beforeScripts(win) {
    stubVoiceSurface(win, page);
    win.__VOICE__ = Object.assign({}, VOICE_CFG);
    win.localStorage.setItem('kissVoiceEnabled', options.enabled ? '1' : '0');
    if (options.debug) win.localStorage.setItem('kissVoiceDebug', '1');
  }
  const {win, posted, getState} = makePage({
    beforeScripts,
    extraScripts: ['voice.js'],
    sourceUrl: options.sourceUrl,
  });
  return Object.assign({win, posted, getState}, page);
}

/** The mic trigger button. */
function micBtn(win) {
  return win.document.getElementById('voice-btn');
}

/**
 * Grant the pipeline its microphone and wait until it is listening.  A
 * page built with `enabled: true` starts the pipeline on load; otherwise
 * the test clicks the mic button first (see startPipeline).
 */
async function startMic(page) {
  await sleep(20);
  page.mic.calls[page.mic.calls.length - 1].grant();
  await sleep(20);
  assert.ok(
    micBtn(page.win).classList.contains('voice-listening'),
    `pipeline must be live: ${micBtn(page.win).className}`,
  );
}

/** Turn the mic on with its button, then grant the microphone. */
async function startPipeline(page) {
  micBtn(page.win).click();
  await startMic(page);
}

/** One onaudioprocess block: 4096 samples at 16 kHz = 256 ms. */
function audioBlock(loud) {
  const samples = new Float32Array(4096);
  if (loud) samples.fill(0.5);
  return {
    inputBuffer: {
      getChannelData: () => samples,
      sampleRate: 16000,
      numberOfChannels: 1,
    },
  };
}

/** Feed `blocks` audio blocks, loud or silent, to a processor node. */
function feed(proc, loud, blocks) {
  for (let i = 0; i < blocks; i++) proc.onaudioprocess(audioBlock(loud));
}

/** The audio processor node of the current pipeline. */
function lastProcessor(page) {
  return page.state.processors[page.state.processors.length - 1];
}

/** The recognizer the current pipeline built last. */
function lastRecognizer(page) {
  return page.state.recognizers[page.state.recognizers.length - 1];
}

/** Fire the wake word through the pipeline's own result handler. */
function fireWake(page) {
  page.clock.now += 3000; // clear the 2 s wake cooldown
  lastRecognizer(page).handlers.result({result: {text: 'abracadabra'}});
}

/**
 * Clear the awaitingFlush latch fireWake armed: the real worker answers
 * retrieveFinalResult with one result message that voice.js swallows.
 */
function flushWake(page) {
  lastRecognizer(page).handlers.result({result: {text: ''}});
}

/**
 * Say the wake word and replay the worker's flush: the round begins,
 * owned by the conversation on screen.
 */
function wake(page) {
  fireWake(page);
  flushWake(page);
  assert.ok(
    micBtn(page.win).classList.contains('voice-triggered'),
    'the wake must have fired',
  );
}

/**
 * Speak for half a second and fall silent for two: the capture ends and
 * its audio is posted to the daemon for transcription.
 */
function speak(page) {
  const proc = lastProcessor(page);
  const before = page.posted.filter(m => m.type === 'voiceTranscribe').length;
  feed(proc, true, 2);
  feed(proc, false, 8);
  assert.strictEqual(
    page.posted.filter(m => m.type === 'voiceTranscribe').length,
    before + 1,
    'the round must have posted its audio',
  );
}

/** Fire the page's one pending safety timer of `ms` milliseconds. */
function fireTimer(page, ms) {
  const live = page.timers.filter(t => !t.cleared && t.ms === ms);
  assert.strictEqual(live.length, 1, `one live ${ms} ms timer is armed`);
  page.win.clearTimeout(live[0].id);
  live[0].cb();
}

module.exports = {
  VOICE_CFG,
  makeVoicePage,
  micBtn,
  startMic,
  startPipeline,
  audioBlock,
  feed,
  lastProcessor,
  lastRecognizer,
  fireWake,
  flushWake,
  wake,
  speak,
  fireTimer,
};
