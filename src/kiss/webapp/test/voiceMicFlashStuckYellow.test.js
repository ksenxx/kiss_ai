// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The mic button's yellow "transcribing" flash is driven by the in-page
// pipeline: a captured round posts its audio and arms a 60 s safety
// timer; the daemon's voiceSpeech reply clears the flash. When the reply
// is lost the timer must clear the stale flash AND reset the round
// counter, otherwise every later utterance leaves the button blinking.
//
// jsdom has no speech engine, audio graph or microphone, so the harness
// supplies the minimal window.Vosk / AudioContext / mediaDevices the page
// expects and feeds real audio blocks through the real capture logic.

const assert = require('assert');
const {makeRunner} = require('./ui_antipattern_harness');
const {send} = require('./simplify2_harness');
const {
  micBtn,
  audioBlock,
  makeBareVoiceWindow,
} = require('./voice_page_helpers');
const {test, report} = makeRunner();

const CFG = {
  voskSrc: '/media/vosk.js',
  modelUrl: 'https://models.example.com/model.tar.gz',
};

class FakeRecognizer {
  constructor() {
    this.handlers = {};
  }
  setWords() {}
  on(event, cb) {
    this.handlers[event] = cb;
  }
  acceptWaveform() {}
  retrieveFinalResult() {}
  remove() {}
}

function makeWindow() {
  const page = {
    timers: [],
    clock: {now: 1000000},
    state: {recognizers: [], processors: []},
    posted: [],
  };
  page.win = makeBareVoiceWindow(CFG, win => stubVoiceSurface(win, page));
  return page;
}

/**
 * Give `win` the microphone, Vosk model and audio graph the pipeline
 * expects, recording recognizers, processor nodes and posts in `page`.
 * The page's timers are recorded too, so a test can fire the flash
 * safety timeout instead of waiting a minute.
 */
function stubVoiceSurface(win, page) {
  const {timers, clock, state, posted} = page;
  win.setTimeout = (fn, ms) => {
    timers.push({fn, ms, cleared: false});
    return timers.length;
  };
  win.clearTimeout = id => {
    if (typeof id === 'number' && timers[id - 1]) {
      timers[id - 1].cleared = true;
    }
  };

  win.Date.now = () => clock.now;

  Object.defineProperty(win.navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia: () =>
        Promise.resolve({
          getTracks: () => [{stop: () => {}}],
          getAudioTracks: () => [],
        }),
      enumerateDevices: () => Promise.resolve([]),
    },
  });
  win.Vosk = {
    createModel: () =>
      Promise.resolve({
        KaldiRecognizer: class extends FakeRecognizer {
          constructor() {
            super();
            state.recognizers.push(this);
          }
        },
      }),
  };
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

  win.addEventListener('kiss-voice-post', e => posted.push(e.detail));
}

function tick() {
  return new Promise(resolve => setTimeout(resolve, 20));
}

/** Turn the mic on and let the (instantly granted) pipeline come up. */
async function startMic(page) {
  micBtn(page.win).click();
  await tick();
  assert.ok(
    micBtn(page.win).classList.contains('voice-listening'),
    `pipeline must be live: ${micBtn(page.win).className}`,
  );
}

// Say the wake word, speak for half a second and fall silent for two:
// the round's audio is posted and the yellow transcribing flash is on.
function speakRound(page) {
  page.clock.now += 10000; // clear the 2 s wake cooldown
  const rec = page.state.recognizers[page.state.recognizers.length - 1];
  rec.handlers.result({result: {text: 'abracadabra'}});
  const proc = page.state.processors[page.state.processors.length - 1];
  for (let i = 0; i < 2; i++) proc.onaudioprocess(audioBlock(true));
  for (let i = 0; i < 8; i++) proc.onaudioprocess(audioBlock(false));
  assert.ok(
    micBtn(page.win).classList.contains('voice-transcribing'),
    `the round must show the yellow transcribing flash: ${micBtn(page.win).className}`,
  );
}

function firePendingTimer(timers) {
  for (let i = timers.length - 1; i >= 0; i--) {
    if (!timers[i].cleared) {
      timers[i].cleared = true;
      timers[i].fn();
      return timers[i].ms;
    }
  }
  throw new Error('no pending timer to fire');
}

async function main() {
  await test(
    'speech after a lost terminal event must not leave the yellow ' +
      'flash blinking (safety timeout self-heals the round counter)',
    async () => {
      const page = makeWindow();
      const {win, timers, posted} = page;
      const btn = micBtn(win);
      const inp = win.document.getElementById('task-input');

      await startMic(page);
      speakRound(page); // round A: audio posted, reply never comes
      assert.strictEqual(
        posted.filter(m => m.type === 'voiceTranscribe').length,
        1,
      );

      const ms = firePendingTimer(timers);
      assert.strictEqual(ms, 60000, 'yellow safety timer must be 60s');
      assert.ok(
        !btn.classList.contains('voice-transcribing'),
        'the safety timeout must clear the stale yellow flash',
      );

      speakRound(page); // round B
      send(win, {type: 'voiceSpeech', text: 'hello world'});

      assert.ok(
        !btn.classList.contains('voice-transcribing'),
        'yellow flash must clear once the spoken text was delivered',
      );
      assert.ok(!btn.classList.contains('voice-triggered'));
      assert.strictEqual(inp.value, 'hello world');
    },
  );

  await test('every later utterance also clears the flash after the self-heal', async () => {
    const page = makeWindow();
    const {win, timers} = page;
    const btn = micBtn(win);

    await startMic(page);
    speakRound(page);
    firePendingTimer(timers);

    for (let round = 1; round <= 3; round++) {
      speakRound(page);
      send(win, {type: 'voiceSpeech', text: `part${round}`});
      assert.ok(
        !btn.classList.contains('voice-transcribing'),
        `round ${round} must clear the yellow flash`,
      );
    }
  });

  await test('overlapping rounds still keep the flash for the newer round', async () => {
    const page = makeWindow();
    const {win} = page;
    const btn = micBtn(win);

    await startMic(page);
    speakRound(page);
    speakRound(page);

    send(win, {type: 'voiceSpeech', text: 'first'});
    assert.ok(
      btn.classList.contains('voice-transcribing'),
      'the newer in-flight round still owns the yellow flash',
    );

    send(win, {type: 'voiceSpeech', text: 'second'});
    assert.ok(!btn.classList.contains('voice-transcribing'));
  });

  report('voiceMicFlashStuckYellow');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
