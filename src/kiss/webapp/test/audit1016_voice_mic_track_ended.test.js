// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// voice.js: the microphone can go away under a live pipeline (the
// headset is unplugged, the OS or another app takes the device, the
// site's permission is revoked).  The browser then fires `ended` on the
// audio track and delivers no more audio.  Before the fix nothing
// listened for that: the button kept saying "listening", every later
// click deferred to a pipeline that was dead, and the audio context and
// recogniser stayed allocated for the page lifetime.  The pipeline must
// come down like a failed start does: the mic turns off, the error
// explains why, and the next click asks for the microphone again.
//
// Same harness as voiceBrowserPipelineLifecycle.test.js: jsdom has no
// speech engine, audio graph or microphone, so the page's embedder
// surface (window.Vosk / AudioContext / mediaDevices) is supplied here;
// the real voice.js, main.js and DOM run.

const assert = require('assert');
const {makePage: makeChatPage, sleep, stubRaf} = require('./simplify2_harness');
const {micBtn} = require('./voice_page_helpers');

const recognizers = [];

class FakeRecognizer {
  constructor() {
    this._handlers = {};
    recognizers.push(this);
  }
  setWords() {}
  on(event, cb) {
    this._handlers[event] = cb;
  }
  acceptWaveform() {}
  retrieveFinalResult() {}
  remove() {}
}

function makeAudioContextClass(log) {
  return class FakeAudioContext {
    constructor() {
      this.sampleRate = 16000;
      this.state = 'running';
      this.destination = {};
      log.push('ctx-open');
    }
    resume() {
      return Promise.resolve();
    }
    close() {
      log.push('ctx-closed');
      return Promise.resolve();
    }
    createMediaStreamSource() {
      return {connect: () => {}, disconnect: () => {}};
    }
    createScriptProcessor() {
      return {connect: () => {}, disconnect: () => {}, onaudioprocess: null};
    }
  };
}

/** An audio track whose `ended` event the test can fire. */
function makeTrack(win, log) {
  const track = new win.EventTarget();
  track.kind = 'audio';
  track.label = 'Test mic';
  track.readyState = 'live';
  track.onended = null;
  track.getSettings = () => ({});
  track.stop = () => {
    track.readyState = 'ended';
    log.push('track-stopped');
  };
  track.end = () => {
    track.readyState = 'ended';
    const ev = new win.Event('ended');
    track.dispatchEvent(ev);
    if (typeof track.onended === 'function') track.onended(ev);
  };
  return track;
}

function makePage() {
  const log = [];
  const tracks = [];
  function stubVoiceSurface(win) {
    stubRaf(win);
    win.Audio = function (src) {
      return {src, play: () => Promise.resolve()};
    };
    Object.defineProperty(win.navigator, 'mediaDevices', {
      configurable: true,
      value: {
        getUserMedia: () => {
          const track = makeTrack(win, log);
          tracks.push(track);
          return Promise.resolve({
            getTracks: () => [track],
            getAudioTracks: () => [track],
          });
        },
        enumerateDevices: () => Promise.resolve([]),
      },
    });
    win.Vosk = {
      createModel: () => Promise.resolve({KaldiRecognizer: FakeRecognizer}),
    };
    win.AudioContext = makeAudioContextClass(log);
    win.__VOICE__ = {
      voskSrc: '/media/vosk.js',
      modelUrl: 'https://models.example.com/model.tar.gz',
      ackAudioUrl: '/media/working-on-it.mp3',
    };
    win.localStorage.setItem('kissVoiceEnabled', '0');
  }
  const {win, posted} = makeChatPage({
    beforeScripts: stubVoiceSurface,
    extraScripts: ['voice.js'],
  });
  return {win, log, tracks, posted};
}

async function main() {
  const {win, log, tracks} = makePage();
  const btn = micBtn(win);
  btn.click();
  await sleep(20);
  assert.ok(btn.classList.contains('voice-listening'), 'pipeline is live');
  assert.strictEqual(tracks.length, 1);
  assert.deepStrictEqual(log, ['ctx-open']);

  // The microphone goes away under the live pipeline.
  tracks[0].end();
  await sleep(20);
  assert.ok(
    !btn.classList.contains('voice-listening'),
    'a pipeline whose microphone ended must not claim to listen',
  );
  assert.ok(
    btn.classList.contains('voice-error'),
    'the button shows the error state, got ' + btn.className,
  );
  assert.ok(
    /microphone/i.test(btn.getAttribute('data-tooltip') || ''),
    'the tooltip explains the microphone went away: ' +
      btn.getAttribute('data-tooltip'),
  );
  assert.deepStrictEqual(
    log,
    ['ctx-open', 'track-stopped', 'ctx-closed'],
    'the audio graph is released',
  );
  assert.strictEqual(
    win.localStorage.getItem('kissVoiceEnabled'),
    '0',
    'the trigger is remembered as off, like after a failed start',
  );

  // The next click asks for the microphone again and comes back live.
  btn.click();
  await sleep(20);
  assert.strictEqual(tracks.length, 2, 'a fresh microphone request');
  assert.ok(btn.classList.contains('voice-listening'), 'listening again');
  assert.deepStrictEqual(log, [
    'ctx-open',
    'track-stopped',
    'ctx-closed',
    'ctx-open',
  ]);

  // Audit 2026-10-16 (webapp-other F1): the microphone goes away in the
  // middle of a wake-word capture.  The "Listening ..." overlay and the
  // red flash the wake turned on (for 45 s) must come down with the
  // pipeline, as they do when the user turns the mic off.
  const wrap = win.document.getElementById('input-text-wrap');
  recognizers[recognizers.length - 1]._handlers.result({
    result: {text: 'abracadabra'},
  });
  assert.ok(wrap.classList.contains('listening'), 'overlay on after wake');
  assert.ok(btn.classList.contains('voice-triggered'), 'flash on after wake');
  tracks[1].end();
  await sleep(20);
  assert.ok(btn.classList.contains('voice-error'), 'error state shown');
  assert.ok(
    !wrap.classList.contains('listening'),
    'the Listening overlay is gone with the microphone',
  );
  assert.ok(
    !btn.classList.contains('voice-triggered'),
    'the wake flash is gone with the microphone: ' + btn.className,
  );
  btn.click();
  await sleep(20);
  assert.strictEqual(tracks.length, 3, 'a fresh microphone request');
  assert.ok(btn.classList.contains('voice-listening'), 'listening again');

  // An `ended` from the track of a pipeline that was already torn down
  // (the user turned the mic off first) changes nothing.
  btn.click();
  await sleep(20);
  assert.ok(btn.classList.contains('voice-off'));
  const after = log.slice();
  tracks[2].end();
  await sleep(20);
  assert.ok(btn.classList.contains('voice-off'), 'still off');
  assert.deepStrictEqual(log, after, 'nothing torn down twice');

  console.log(
    '  ok - a microphone that ends under the live pipeline turns the trigger off with an error',
  );
  console.log('audit1016_voice_mic_track_ended: all passed');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
