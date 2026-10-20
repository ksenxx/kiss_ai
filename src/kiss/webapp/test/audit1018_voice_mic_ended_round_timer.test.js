// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// voice.js: the microphone goes away (track `ended`) while a round's
// audio is posted and its transcript still pending.  onMicEnded keeps
// that round's owner (its transcript still stands) but called
// flash(null) unconditionally, which cleared the 60 s "transcribing"
// timer that retires a round whose reply never comes.  With the timer
// gone the owner sat in the queue for good, and the NEXT round's
// transcript (dictated after the mic came back, in another chat) was
// paired with the stale owner and submitted into the wrong chat.
//
// Harness as conc2026_voice_rounds.test.js (scriptable microphone,
// recognizers and audio processors, recorded safety timers) with a
// track whose `ended` the test fires (audit1016_voice_mic_track_ended).

const assert = require('assert');
const {
  makePage: makeSharedPage,
  send,
  sleep,
  stubRaf,
} = require('./simplify2_harness');

function makePage() {
  const clock = {now: 1000000};
  const state = {recognizers: [], processors: [], tracks: []};
  const timers = [];
  const {win, posted} = makeSharedPage({
    beforeScripts: win => stubVoice(win, clock, state, timers),
    extraScripts: ['voice.js'],
  });
  return {win, posted, state, clock, timers};
}

function stubVoice(win, clock, state, timers) {
  stubRaf(win);
  win.Audio = function (src) {
    return {src, play: () => Promise.resolve()};
  };
  win.Date.now = () => clock.now;

  Object.defineProperty(win.navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia: () => {
        const track = new win.EventTarget();
        track.kind = 'audio';
        track.readyState = 'live';
        track.getSettings = () => ({});
        track.stop = () => {};
        track.end = () => track.dispatchEvent(new win.Event('ended'));
        state.tracks.push(track);
        return Promise.resolve({
          getTracks: () => [track],
          getAudioTracks: () => [track],
        });
      },
      enumerateDevices: () => Promise.resolve([]),
    },
  });
  const model = {recognizers: new Map()};
  model.KaldiRecognizer = class FakeRecognizer {
    constructor() {
      this.id = 'rec-' + (state.recognizers.length + 1);
      this.handlers = {};
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
      model.recognizers.delete(this.id);
    }
  };
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
  // flash) are recorded so the test can fire one without waiting.
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
  win.__VOICE__ = {
    voskSrc: '/media/vosk.js',
    modelUrl: 'https://models.example.com/model.tar.gz',
    ackAudioUrl: '/media/working-on-it.mp3',
  };
  win.localStorage.setItem('kissVoiceEnabled', '0');
}

const micBtn = win => win.document.getElementById('voice-btn');
const last = list => list[list.length - 1];
const liveTimers = (page, ms) =>
  page.timers.filter(t => !t.cleared && t.ms === ms);

async function startPipeline(page) {
  micBtn(page.win).click();
  await sleep(20);
  assert.ok(
    micBtn(page.win).classList.contains('voice-listening'),
    'pipeline is live',
  );
}

/** One onaudioprocess block: 4096 samples at 16 kHz = 256 ms. */
function feed(proc, loud, blocks) {
  const samples = new Float32Array(4096);
  if (loud) samples.fill(0.5);
  const block = {
    inputBuffer: {
      getChannelData: () => samples,
      sampleRate: 16000,
      numberOfChannels: 1,
    },
  };
  for (let i = 0; i < blocks; i++) proc.onaudioprocess(block);
}

/** Wake, speak, fall silent: the round's audio is posted. */
function speakRound(page) {
  page.clock.now += 3000; // clear the 2 s wake cooldown
  last(page.state.recognizers).handlers.result({result: {text: 'abracadabra'}});
  feed(last(page.state.processors), true, 2);
  feed(last(page.state.processors), false, 8);
}

async function main() {
  const page = makePage();
  const {win, posted, state} = page;
  const api = win._testApi;
  const first = api.getActiveTabId();
  api.createNewTab();
  const second = api.getActiveTabId();
  api.switchToTab(first); // chat A on screen

  await startPipeline(page);
  speakRound(page); // round A: audio posted, transcript pending
  assert.strictEqual(
    posted.filter(m => m.type === 'voiceTranscribe').length,
    1,
  );
  assert.strictEqual(
    liveTimers(page, 60000).length,
    1,
    'the 60 s transcribing timer is armed',
  );

  // The microphone goes away while round A transcribes.
  state.tracks[0].end();
  await sleep(20);
  assert.ok(
    micBtn(win).classList.contains('voice-error'),
    'the trigger is off with the reason',
  );
  assert.ok(
    micBtn(win).classList.contains('voice-transcribing'),
    'the round still awaiting its transcript keeps the transcribing flash: ' +
      micBtn(win).className,
  );
  const live = liveTimers(page, 60000);
  assert.strictEqual(
    live.length,
    1,
    'REGRESSION: the mic loss must keep (or re-arm) the 60 s timer that retires the round ' +
      'if its transcript never comes, got ' +
      live.length,
  );

  // Round A's reply never comes: the timer retires it.
  win.clearTimeout(live[0].id);
  live[0].cb();
  assert.ok(
    !micBtn(win).classList.contains('voice-transcribing'),
    'the flash is gone',
  );

  // The mic comes back; the user dictates in chat B.
  api.switchToTab(second);
  await startPipeline(page);
  speakRound(page); // round B
  assert.strictEqual(
    posted.filter(m => m.type === 'voiceTranscribe').length,
    2,
  );
  posted.length = 0;
  send(win, {type: 'voiceSpeech', text: 'words spoken in chat B QB1'});
  assert.deepStrictEqual(
    posted
      .filter(m => m.type === 'submit')
      .map(m => ({onB: m.tabId === second, prompt: m.prompt})),
    [{onB: true, prompt: 'words spoken in chat B QB1'}],
    "round B's words must reach chat B, not the chat of the round the mic loss stranded: " +
      JSON.stringify(posted),
  );
  win.close();
  console.log('  ok - a mic loss keeps the timer that retires a posted round');
  console.log('audit1018_voice_mic_ended_round_timer: all passed');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
