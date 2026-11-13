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
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {
  makePage,
  send,
  settle,
  stubRaf,
  captureTimers,
  MEDIA,
} = require('./simplify2_harness');

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
    remove() {
      this.removed = true;
      model.recognizers.delete(this.id);
    }
  };
  return model;
}

/**
 * An audio track of a granted microphone: a `win.EventTarget` whose
 * `ended` the test fires with `end()` (the browser does so when the
 * device is unplugged or the permission revoked); `stop()` is what the
 * page calls to release it.
 */
function makeTrack(win, state, log) {
  const track = new win.EventTarget();
  track.kind = 'audio';
  track.label = 'Test mic';
  track.readyState = 'live';
  track.getSettings = () => ({});
  track.stop = () => {
    track.readyState = 'ended';
    log.push('track-stopped');
  };
  track.end = () => {
    track.readyState = 'ended';
    track.dispatchEvent(new win.Event('ended'));
  };
  state.tracks.push(track);
  return track;
}

/**
 * The record a voice page's stubs write to (see stubVoiceSurface) and
 * the knobs they read.
 *
 * @param {{noVosk?: boolean, pendingModel?: boolean, autoGrant?: boolean}}
 *   [opts] `noVosk` leaves `win.Vosk` undefined (the page must inject the
 *   vosk bundle itself); `pendingModel` holds `createModel` until the test
 *   calls `page.resolveModel()`; `autoGrant` grants every microphone
 *   request at once instead of waiting for the test's `grant()`.
 * @returns {{state: {recognizers: Array, processors: Array, tracks: Array},
 *   mic: {calls: Array}, clock: {now: number},
 *   model: object, timers: Array, log: Array<string>, options: object,
 *   resolveModel: ?Function}} `state.tracks` lists every microphone
 *   track granted (see makeTrack); `log` lists the audio-graph and
 *   microphone events in order (`ctx-open`, `ctx-closed`,
 *   `track-stopped`).
 */
function voicePageState(opts) {
  const state = {recognizers: [], processors: [], tracks: []};
  return {
    state,
    mic: {calls: []},
    clock: {now: 1000000},
    model: makeFakeModel(state),
    timers: [],
    log: [],
    options: opts || {},
    resolveModel: null,
  };
}

/**
 * Install the voice pipeline's embedder surface on *win* before the page
 * scripts run: the clock, a silent `Audio`, the scriptable microphone,
 * the fake Vosk model, the audio context and the timer ledger.  *page*
 * is a voicePageState record.
 */
function stubVoiceSurface(win, page) {
  const {state, mic, clock, model, timers, log, options} = page;
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
          const call = {
            constraints,
            grant: () => {
              const track = makeTrack(win, state, log);
              resolve({
                getTracks: () => [track],
                getAudioTracks: () => [track],
              });
            },
            refuse: (message, name) => {
              const err = new Error(message || 'Permission denied');
              err.name = name || 'NotAllowedError';
              reject(err);
            },
          };
          mic.calls.push(call);
          if (options.autoGrant) call.grant();
        }),
      enumerateDevices: () => Promise.resolve([]),
    },
  });
  if (!options.noVosk) {
    win.Vosk = {
      createModel: () =>
        new Promise(resolve => {
          page.resolveModel = () => resolve(model);
          if (!options.pendingModel) page.resolveModel();
        }),
    };
  }
  win.AudioContext = class FakeAudioContext {
    constructor() {
      this.sampleRate = 16000;
      this.state = 'running';
      this.destination = {};
      log.push('ctx-open');
    }
    close() {
      log.push('ctx-closed');
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
  captureTimers(win, timers);
}

/**
 * Build a chat page running the real main.js + voice.js over the stubbed
 * voice surface.
 *
 * @param {{enabled?: boolean, debug?: boolean, sourceUrl?: *,
 *   noVosk?: boolean, pendingModel?: boolean, autoGrant?: boolean,
 *   beforeScripts?: Function}} [opts]
 *   `enabled` remembers the mic ON (`kissVoiceEnabled=1`, the pipeline
 *   starts on load; default off, so a test clicks the mic button);
 *   `debug` sets `kissVoiceDebug=1` (a free-text recognizer runs next to
 *   the wake one); `sourceUrl` is passed to simplify2_harness.makePage;
 *   `noVosk` / `pendingModel` / `autoGrant` as for voicePageState;
 *   `beforeScripts(win)` runs after the voice surface is installed and
 *   before the page scripts, so a suite's own stubs and stored settings
 *   win.
 * @returns {{win: object, posted: Array<object>, getState: Function,
 *   state: {recognizers: Array, processors: Array, tracks: Array},
 *   mic: {calls: Array<{constraints: *, grant: Function, refuse: Function}>},
 *   clock: {now: number}, model: object, timers: Array<object>,
 *   log: Array<string>, resolveModel: ?Function}}
 *   `state.recognizers` / `state.processors` list every recognizer and
 *   audio processor node the pipeline ever built; `mic.calls` every
 *   getUserMedia request, settled by the test (`refuse(message, name)`
 *   fails it with an error named `name`, NotAllowedError by default);
 *   `clock.now` is what the page's Date.now() returns; `model` is the
 *   fake Vosk model (`model.recognizers` maps the ids of the live
 *   recognizers); `timers` is the ledger of the page's setTimeout calls;
 *   `log` and `resolveModel` as for voicePageState.
 */
function makeVoicePage(opts) {
  const options = opts || {};
  const page = voicePageState(options);
  function beforeScripts(win) {
    stubVoiceSurface(win, page);
    win.__VOICE__ = Object.assign({}, VOICE_CFG);
    win.localStorage.setItem('kissVoiceEnabled', options.enabled ? '1' : '0');
    if (options.debug) win.localStorage.setItem('kissVoiceDebug', '1');
    if (options.beforeScripts) options.beforeScripts(win);
  }
  const {win, posted, getState} = makePage({
    beforeScripts,
    extraScripts: ['voice.js'],
    sourceUrl: options.sourceUrl,
  });
  // The stubs write to `page` itself (resolveModel), so it is returned,
  // not copied.
  return Object.assign(page, {win, posted, getState});
}

/** The mic trigger button. */
function micBtn(win) {
  return win.document.getElementById('voice-btn');
}

/**
 * Grant the pipeline its microphone and wait until it is listening.  A
 * page built with `enabled: true` starts the pipeline on load; otherwise
 * the test clicks the mic button first (see startPipeline).  On an
 * `autoGrant` page the request is already granted, so this only waits.
 */
async function startMic(page) {
  await settle();
  if (!page.options.autoGrant) {
    page.mic.calls[page.mic.calls.length - 1].grant();
  }
  await settle();
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

/**
 * Feed `blocks` onaudioprocess blocks (4096 samples at 16 kHz = 256 ms
 * each), loud or silent, to a processor node; returns the samples the
 * node wrote to its output.
 */
function feed(proc, loud, blocks) {
  const samples = new Float32Array(4096);
  if (loud) samples.fill(0.5);
  const inputBuffer = {
    getChannelData: () => samples,
    sampleRate: 16000,
    numberOfChannels: 1,
  };
  const output = new Float32Array(4096);
  const outputBuffer = {getChannelData: () => output};
  for (let i = 0; i < blocks; i++) proc.onaudioprocess({inputBuffer, outputBuffer});
  return output;
}

/** The audio processor node of the current pipeline. */
function lastProcessor(page) {
  return page.state.processors[page.state.processors.length - 1];
}

/** The newest recognizer whose grammar is (or is not) set. */
function newestRecognizer(page, withGrammar) {
  const recs = page.state.recognizers.filter(r => !!r.grammar === withGrammar);
  return recs[recs.length - 1];
}

/** The wake-word (grammar) recognizer the current pipeline built last. */
function lastRecognizer(page) {
  return newestRecognizer(page, true);
}

/** The grammar-free recognizer the current pipeline built last. */
function lastFreeRecognizer(page) {
  return newestRecognizer(page, false);
}

/**
 * Fire the wake word through the pipeline's own result handlers: the
 * grammar recognizer's final `text`, then the grammar-free one's, which
 * must agree before the page wakes.
 */
function fireWake(page, text = 'hey sore car') {
  page.clock.now += 3000; // clear the 2 s wake cooldown
  lastRecognizer(page).handlers.result({result: {text}});
  lastFreeRecognizer(page).handlers.result({result: {text: 'hey soccer'}});
}

/**
 * Say the wake word: the round begins, owned by the conversation on
 * screen.
 */
function wake(page) {
  fireWake(page);
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

/**
 * Say the wake word and speak: the round's audio is posted for
 * transcription.  Returns the tag the page posted with it, which the
 * daemon echoes on the round's transcript.
 */
function speakRound(page) {
  wake(page);
  speak(page);
  const sent = page.posted.filter(m => m.type === 'voiceTranscribe');
  return sent[sent.length - 1].round;
}

/**
 * Dictate `text`: speak a round, then deliver the daemon's transcript of
 * it, tagged as the daemon tags it.  `extra` adds its `speaker` /
 * `language` fields.
 */
function dictate(page, text, extra) {
  const round = speakRound(page);
  send(page.win, Object.assign({type: 'voiceSpeech', text, round}, extra));
}

/** Fire the page's one pending safety timer of `ms` milliseconds. */
function fireTimer(page, ms) {
  const live = page.timers.filter(
    t => !t.cleared && !t.fired && t.ms === ms,
  );
  assert.strictEqual(live.length, 1, `one live ${ms} ms timer is armed`);
  page.win.clearTimeout(live[0].id);
  live[0].fired = true;
  live[0].cb(...live[0].args);
}

/**
 * A bare jsdom window running only media/voice.js: the mic button and
 * the task textarea, `window.__VOICE__ = cfg`, then `beforeScript(win)`
 * for the suite's stubs (fake mediaDevices, Vosk, timers, listeners)
 * before the script is injected.  Returns the window.
 */
function makeBareVoiceWindow(cfg, beforeScript) {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body>' +
      '<button id="voice-btn" class="toggle-btn"></button>' +
      '<textarea id="task-input"></textarea>' +
      '</body></html>',
    {runScripts: 'dangerously', url: 'https://localhost/'},
  );
  const win = dom.window;
  win.__VOICE__ = cfg;
  if (beforeScript) beforeScript(win);
  const script = win.document.createElement('script');
  script.textContent = fs.readFileSync(path.join(MEDIA, 'voice.js'), 'utf-8');
  win.document.body.appendChild(script);
  return win;
}

/**
 * A bare voice window (see makeBareVoiceWindow) over the stubbed voice
 * surface, as a page record for the helpers above.  No main.js forwards
 * what voice.js posts to the daemon here, so `posted` collects its
 * `kiss-voice-post` events.  `cfg` overrides VOICE_CFG; `beforeScript(win)`
 * runs after the surface is installed, so a suite's own stubs win.
 */
function makeBareVoicePage(cfg, beforeScript) {
  const page = voicePageState();
  page.posted = [];
  page.win = makeBareVoiceWindow(Object.assign({}, VOICE_CFG, cfg), win => {
    stubVoiceSurface(win, page);
    win.addEventListener('kiss-voice-post', event =>
      page.posted.push(event.detail),
    );
    if (beforeScript) beforeScript(win);
  });
  return page;
}

/**
 * A bare jsdom window running only media/vosk.js, its Worker the fake
 * `workerFactory(workers, null)` returns (`workers` collects the workers
 * it makes), with a `URL.createObjectURL` where jsdom has none.  Returns
 * `{win, workers}`.
 */
function makeVoskWindow(workerFactory) {
  const dom = new JSDOM('<!doctype html><html><body></body></html>', {
    runScripts: 'dangerously',
    url: 'https://localhost/',
  });
  const win = dom.window;
  const workers = [];
  win.Worker = workerFactory(workers, null);
  if (typeof win.URL.createObjectURL !== 'function') {
    win.URL.createObjectURL = () => 'blob:fake-worker-url';
  }
  win.eval(fs.readFileSync(path.join(MEDIA, 'vosk.js'), 'utf8'));
  assert.ok(win.Vosk && win.Vosk.createModel, 'vosk.js must export Vosk');
  return {win, workers};
}

module.exports = {
  VOICE_CFG,
  voicePageState,
  stubVoiceSurface,
  makeVoicePage,
  makeBareVoiceWindow,
  makeBareVoicePage,
  makeVoskWindow,
  micBtn,
  settle,
  startMic,
  startPipeline,
  feed,
  lastProcessor,
  lastRecognizer,
  lastFreeRecognizer,
  fireWake,
  wake,
  speak,
  speakRound,
  dictate,
  fireTimer,
};
