// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E page tests for the async lifecycle of the in-page (browser-mic)
// voice pipeline in voice.js: cancellation while a start is in flight,
// toggle races around the permission probe, and suspend/resume when a
// webview page is hidden.
//
// Defects being reproduced (each test failed against the pre-fix code):
//  1. Toggling OFF while the pipeline was starting (engine/model still
//     loading) left the button stuck in the loading state, and a pending
//     second getUserMedia could still open the microphone.
//  2. A pipeline failure that settled AFTER the user toggled off painted
//     a stale red error over the user's explicit "off".
//  3. ON→OFF→ON bursts weren't tied to the probe that started them: a
//     stale probe could post a duplicate voiceToggle(true) or start the
//     host listener and the in-page pipeline together.
//  4. A granted browser microphone kept recording after the webview page
//     was hidden (VS Code retains hidden webviews, so page scripts — and
//     open microphones — keep running unseen).
//
// jsdom provides no speech engine, audio graph, or microphone, so the
// harness supplies the minimal window.Vosk / AudioContext / mediaDevices
// capabilities the page expects from its embedder — the same approach the
// suite's other page tests take for getUserMedia. Everything else (the
// real voice.js, main.js, DOM, message plumbing) is exercised for real.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

const FALLBACK_CFG = {
  voskSrc: '/media/vosk.js',
  modelUrl: 'https://models.example.com/model.tar.gz',
  ackAudioUrl: '/media/working-on-it.mp3',
  nonce: 'test-nonce',
};

/** A KaldiRecognizer with the surface the pipeline uses. */
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

/** The model object window.Vosk.createModel resolves with. */
function fakeModel() {
  return {KaldiRecognizer: FakeRecognizer};
}

/** A minimal AudioContext whose close() is observable. */
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

/**
 * Build a webview/browser page with a scriptable microphone: each
 * getUserMedia call is recorded and settled by the test through the
 * returned `mic.calls` list.
 */
function makePage(opts) {
  const options = opts || {};
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
  win.acquireVsCodeApi = function () {
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
  const mic = {calls: []};
  Object.defineProperty(win.navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia: constraints => {
        return new Promise((resolve, reject) => {
          mic.calls.push({
            constraints,
            grant: () =>
              resolve({
                getTracks: () => [
                  {
                    stop: () => log.push('track-stopped'),
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
        });
      },
      enumerateDevices: () => Promise.resolve([]),
    },
  });

  win.Vosk = {
    createModel: () => {
      if (options.pendingModel) {
        return new Promise((resolve, reject) => {
          mic.modelResolve = () => resolve(fakeModel());
          mic.modelReject = reject;
        });
      }
      return Promise.resolve(fakeModel());
    },
  };
  win.AudioContext = makeAudioContextClass(log);

  win.__VOICE__ = Object.assign(
    {mode: options.mode || 'webview'},
    FALLBACK_CFG,
  );
  win.localStorage.setItem('kissVoiceEnabled', '0');
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'voice.js'), 'utf8'));
  return {win, posted, log, mic};
}

function micBtn(win) {
  return win.document.getElementById('voice-btn');
}

function onToggles(posted) {
  return posted.filter(m => m.type === 'voiceToggle' && m.enabled);
}

function setPageVisibility(win, state) {
  Object.defineProperty(win.document, 'visibilityState', {
    configurable: true,
    get: () => state,
  });
  Object.defineProperty(win.document, 'hidden', {
    configurable: true,
    get: () => state === 'hidden',
  });
  win.document.dispatchEvent(new win.Event('visibilitychange'));
}

function tick(ms) {
  return new Promise(resolve => setTimeout(resolve, ms || 20));
}

let passed = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.stack || e.message}`);
  }
}

async function main() {
  await test(
    'toggling off while the engine loads settles at off, never stuck loading, mic never opened',
    async () => {
      const {win, mic, log} = makePage({pendingModel: true});
      micBtn(win).click(); // on: permission probe pending
      await tick();
      mic.calls[0].grant(); // probe granted -> pipeline starts, model pending
      await tick();
      assert.ok(
        micBtn(win).classList.contains('voice-loading'),
        `pipeline should be loading: ${micBtn(win).className}`,
      );
      micBtn(win).click(); // off while the model is still loading
      assert.ok(
        micBtn(win).classList.contains('voice-off'),
        `off click must paint off immediately: ${micBtn(win).className}`,
      );
      mic.modelResolve(); // the superseded start now settles
      await tick();
      assert.ok(
        micBtn(win).classList.contains('voice-off'),
        `must stay off after the stale start settles: ${micBtn(win).className}`,
      );
      assert.strictEqual(
        mic.calls.length,
        1,
        'the superseded pipeline must never request the microphone',
      );
      assert.strictEqual(log.filter(e => e === 'ctx-open').length, 0);
    },
  );

  await test(
    'a microphone granted after the off click is released immediately',
    async () => {
      const {win, mic, log} = makePage();
      micBtn(win).click();
      await tick();
      mic.calls[0].grant(); // probe
      await tick(); // pipeline start: model resolves, getUserMedia pending
      assert.strictEqual(mic.calls.length, 2, 'pipeline must request the mic');
      micBtn(win).click(); // off while getUserMedia is pending
      mic.calls[1].grant(); // the embedder grants it anyway
      await tick();
      assert.ok(
        log.includes('track-stopped'),
        `the late-granted stream must be stopped: ${JSON.stringify(log)}`,
      );
      assert.strictEqual(log.filter(e => e === 'ctx-open').length, 0);
      assert.ok(micBtn(win).classList.contains('voice-off'));
    },
  );

  await test(
    'a failure settling after the off click paints off, not a stale red error',
    async () => {
      const {win, mic} = makePage();
      micBtn(win).click();
      await tick();
      mic.calls[0].grant();
      await tick();
      micBtn(win).click(); // off while the pipeline getUserMedia is pending
      mic.calls[1].refuse('Requested device not found');
      await tick();
      const btn = micBtn(win);
      assert.ok(
        !btn.classList.contains('voice-error'),
        `a superseded failure must not paint an error: ${btn.className}`,
      );
      assert.ok(btn.classList.contains('voice-off'));
    },
  );

  await test(
    'a failure of an attempt the user still wants keeps the red error',
    async () => {
      const {win, mic} = makePage();
      micBtn(win).click();
      await tick();
      mic.calls[0].grant();
      await tick();
      mic.calls[1].refuse('Requested device not found');
      await tick();
      const btn = micBtn(win);
      assert.ok(
        btn.classList.contains('voice-error'),
        `a real, current failure must stay visible: ${btn.className}`,
      );
      assert.strictEqual(
        win.localStorage.getItem('kissVoiceEnabled'),
        '0',
        'the failed attempt must clear the persisted intent',
      );
    },
  );

  await test(
    'ON-OFF-ON: a stale granted probe stands down; only the live probe answers',
    async () => {
      const {win, mic, posted} = makePage();
      micBtn(win).click(); // on: probe 1 pending
      micBtn(win).click(); // off
      micBtn(win).click(); // on again: probe 2 pending
      assert.strictEqual(mic.calls.length, 2);
      mic.calls[0].grant(); // stale probe says "granted"
      await tick();
      assert.strictEqual(
        mic.calls.length,
        2,
        'the stale grant must not start the in-page pipeline',
      );
      mic.calls[1].refuse(); // the live probe is refused (VS Code webview)
      await tick();
      assert.strictEqual(
        onToggles(posted).length,
        1,
        `exactly one voiceToggle(true), from the live probe: ` +
          JSON.stringify(posted.filter(m => m.type === 'voiceToggle')),
      );
      assert.strictEqual(
        mic.calls.length,
        2,
        'the host listener and the in-page pipeline must never run together',
      );
    },
  );

  await test(
    'ON-OFF-ON: a stale refused probe posts nothing; the live granted probe starts the pipeline',
    async () => {
      const {win, mic, posted} = makePage();
      micBtn(win).click(); // on: probe 1 pending
      micBtn(win).click(); // off
      micBtn(win).click(); // on again: probe 2 pending
      mic.calls[0].refuse(); // stale probe refused
      await tick();
      assert.strictEqual(
        onToggles(posted).length,
        0,
        'a stale refusal must not start the host listener',
      );
      mic.calls[1].grant(); // live probe granted -> pipeline
      await tick();
      assert.strictEqual(mic.calls.length, 3, 'pipeline must request the mic');
      mic.calls[2].grant();
      await tick();
      assert.ok(
        micBtn(win).classList.contains('voice-listening'),
        `pipeline must be live: ${micBtn(win).className}`,
      );
      assert.strictEqual(onToggles(posted).length, 0);
    },
  );

  await test(
    'hiding the webview releases the browser microphone; showing it resumes',
    async () => {
      const {win, mic, log} = makePage();
      micBtn(win).click();
      await tick();
      mic.calls[0].grant(); // probe
      await tick();
      mic.calls[1].grant(); // pipeline mic
      await tick();
      assert.ok(micBtn(win).classList.contains('voice-listening'));

      setPageVisibility(win, 'hidden');
      assert.ok(
        log.includes('track-stopped'),
        `hiding the page must stop the microphone: ${JSON.stringify(log)}`,
      );
      assert.ok(
        log.includes('ctx-closed'),
        'hiding the page must close the audio graph',
      );
      assert.strictEqual(
        win.localStorage.getItem('kissVoiceEnabled'),
        '1',
        'a hide-suspend must not clear the persisted intent',
      );

      setPageVisibility(win, 'visible');
      await tick();
      assert.strictEqual(
        mic.calls.length,
        3,
        'showing the page must restart the pipeline',
      );
      mic.calls[2].grant();
      await tick();
      assert.ok(
        micBtn(win).classList.contains('voice-listening'),
        `pipeline must be live again: ${micBtn(win).className}`,
      );
    },
  );

  await test(
    'a probe granted while the page is hidden defers the microphone to the next show',
    async () => {
      const {win, mic} = makePage();
      micBtn(win).click(); // probe pending
      setPageVisibility(win, 'hidden'); // user hides the panel mid-probe
      mic.calls[0].grant();
      await tick();
      assert.strictEqual(
        mic.calls.length,
        1,
        'no microphone may be opened for a hidden page',
      );
      setPageVisibility(win, 'visible');
      await tick();
      assert.strictEqual(
        mic.calls.length,
        2,
        'the deferred pipeline must start on show',
      );
      mic.calls[1].grant();
      await tick();
      assert.ok(micBtn(win).classList.contains('voice-listening'));
    },
  );

  await test(
    'OFF then ON during an in-flight start restarts the pipeline exactly once',
    async () => {
      const {win, mic} = makePage({pendingModel: true});
      micBtn(win).click();
      await tick();
      mic.calls[0].grant(); // probe -> pipeline start, model pending
      await tick();
      micBtn(win).click(); // off during the load
      micBtn(win).click(); // on again during the same load
      assert.ok(micBtn(win).classList.contains('voice-loading'));
      mic.modelResolve(); // the stale start settles and hands over
      await tick();
      assert.strictEqual(
        mic.calls.length,
        2,
        'exactly one restarted pipeline may request the mic',
      );
      mic.calls[1].grant();
      await tick();
      assert.ok(
        micBtn(win).classList.contains('voice-listening'),
        `the re-enabled pipeline must end live: ${micBtn(win).className}`,
      );
    },
  );

  await test(
    'browser mode (remote webapp): a hidden tab keeps listening for its wake word',
    async () => {
      const {win, mic, log} = makePage({mode: 'browser'});
      micBtn(win).click(); // browser mode: no probe, pipeline directly
      await tick();
      mic.calls[0].grant();
      await tick();
      assert.ok(micBtn(win).classList.contains('voice-listening'));
      setPageVisibility(win, 'hidden');
      assert.ok(
        !log.includes('track-stopped'),
        'a backgrounded remote-webapp tab must keep its microphone',
      );
      assert.ok(micBtn(win).classList.contains('voice-listening'));
    },
  );

  console.log(`\n${passed} passed, ${failures.length} failed`);
  if (failures.length) process.exit(1);
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  },
);
