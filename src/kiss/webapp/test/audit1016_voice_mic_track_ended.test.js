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
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

class FakeRecognizer {
  constructor() {
    this._handlers = {};
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
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };
  const posted = [];
  win.acquireTransport = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.Audio = function (src) {
    return {src, play: () => Promise.resolve()};
  };
  const log = [];
  const tracks = [];
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
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'voice.js'), 'utf8'));
  return {win, log, tracks, posted};
}

function micBtn(win) {
  return win.document.getElementById('voice-btn');
}

function tick(ms) {
  return new Promise(resolve => setTimeout(resolve, ms || 20));
}

async function main() {
  const {win, log, tracks} = makePage();
  const btn = micBtn(win);
  btn.click();
  await tick();
  assert.ok(btn.classList.contains('voice-listening'), 'pipeline is live');
  assert.strictEqual(tracks.length, 1);
  assert.deepStrictEqual(log, ['ctx-open']);

  // The microphone goes away under the live pipeline.
  tracks[0].end();
  await tick();
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
    log, ['ctx-open', 'track-stopped', 'ctx-closed'],
    'the audio graph is released',
  );
  assert.strictEqual(
    win.localStorage.getItem('kissVoiceEnabled'), '0',
    'the trigger is remembered as off, like after a failed start',
  );

  // The next click asks for the microphone again and comes back live.
  btn.click();
  await tick();
  assert.strictEqual(tracks.length, 2, 'a fresh microphone request');
  assert.ok(btn.classList.contains('voice-listening'), 'listening again');
  assert.deepStrictEqual(
    log, ['ctx-open', 'track-stopped', 'ctx-closed', 'ctx-open'],
  );

  // An `ended` from the track of a pipeline that was already torn down
  // (the user turned the mic off first) changes nothing.
  btn.click();
  await tick();
  assert.ok(btn.classList.contains('voice-off'));
  const after = log.slice();
  tracks[1].end();
  await tick();
  assert.ok(btn.classList.contains('voice-off'), 'still off');
  assert.deepStrictEqual(log, after, 'nothing torn down twice');

  console.log('  ok - a microphone that ends under the live pipeline turns the trigger off with an error');
  console.log('audit1016_voice_mic_track_ended: all passed');
}

main().catch(err => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
