// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E page tests for unkeyed round-owner hygiene in the in-page (browser
// mic) voice pipeline of voice.js, and for the debug-mode free recognizer.
//
// Defects being reproduced (each test failed against the pre-fix code):
//
//  CAND-1  Hiding the webview mid-capture (stopBrowserPipeline) threw the
//          active capture away but left its just-begun unkeyed round in
//          unkeyedOwners. No voiceTranscribe was ever posted for it, so no
//          transcript could ever retire it, and the NEXT unkeyed
//          transcript was paired with the stale owner: words spoken later
//          on tab B were charged to pre-hide tab A and voiceDropped
//          (valid dictation silently lost) or typed into the wrong task.
//
//  CAND-2  finishCapture's no-audio branch called retireRound(null),
//          which shifts the OLDEST unkeyed round. Rounds overlap by
//          design (wake B can begin while round A's audio is still being
//          transcribed), and the round that produced no audio is the one
//          that JUST began — the NEWEST. Retiring the oldest consumed
//          round A's owner, so A's transcript was then paired with owner
//          B: one tab's words submitted into another tab's conversation.
//
//  CAND-3  With kissVoiceDebug=1, startBrowserPipeline created a second,
//          grammar-free KaldiRecognizer for logging, held only by a
//          closure local. stopBrowserPipeline removed the module-level
//          `recognizer` but never the free one, leaking one worker-side
//          recognizer (a model.recognizers map entry) per pipeline start.
//
// jsdom provides no speech engine, audio graph, or microphone, so the
// harness supplies the minimal window.Vosk / AudioContext / mediaDevices
// capabilities the page expects from its embedder — the same approach as
// voiceBrowserPipelineLifecycle.test.js. The fake KaldiRecognizer keeps a
// model.recognizers map exactly like the real vosk.js Model does, and the
// fake ScriptProcessorNode hands its onaudioprocess callback to the test
// so real audio blocks drive the real feedCapture/finishCapture logic.
// Everything else (the real voice.js, main.js, tabs, message plumbing) is
// exercised for real.

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

/**
 * A fake vosk Model: KaldiRecognizer instances register themselves in
 * model.recognizers and deregister on remove(), mirroring the real
 * vosk.js Model (registerRecognizer / unregisterRecognizer).
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
 * Build a page running the real main.js + voice.js with a scriptable
 * microphone, clock, recognizer set, and audio processor nodes.
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

  // Controllable clock: triggerWake enforces a 2 s cooldown between wakes
  // via Date.now(), so tests advance time instead of sleeping.
  const clock = {now: 1000000};
  win.Date.now = () => clock.now;

  const state = {recognizers: [], processors: [], tracksStopped: 0};
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
        });
      },
      enumerateDevices: () => Promise.resolve([]),
    },
  });

  const model = makeFakeModel(state);
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

  win.__VOICE__ = Object.assign(
    {mode: options.mode || 'webview'},
    FALLBACK_CFG,
  );
  win.localStorage.setItem('kissVoiceEnabled', '0');
  if (options.debug) win.localStorage.setItem('kissVoiceDebug', '1');
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'voice.js'), 'utf8'));
  return {win, posted, state, mic, clock, model};
}

function micBtn(win) {
  return win.document.getElementById('voice-btn');
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function twoTabs(win) {
  const api = win._testApi;
  assert.ok(api, '_testApi must be exposed by main.js');
  const first = api.getActiveTabId();
  api.createNewTab();
  const second = api.getActiveTabId();
  assert.ok(second && second !== first, 'a fresh second tab must be active');
  return {api, first, second};
}

function clickTab(win, tabId) {
  const el = win.document.querySelector(
    `.chat-tab[data-tab-id=${JSON.stringify(tabId)}]`,
  );
  assert.ok(el, `tab ${tabId} must exist in the tab bar`);
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function setPageVisibility(win, stateName) {
  Object.defineProperty(win.document, 'visibilityState', {
    configurable: true,
    get: () => stateName,
  });
  Object.defineProperty(win.document, 'hidden', {
    configurable: true,
    get: () => stateName === 'hidden',
  });
  win.document.dispatchEvent(new win.Event('visibilitychange'));
}

function tick(ms) {
  return new Promise(resolve => setTimeout(resolve, ms || 20));
}

/** Turn the mic on and settle the webview probe + pipeline mic grants. */
async function startPipeline(page) {
  micBtn(page.win).click();
  await tick();
  page.mic.calls[page.mic.calls.length - 1].grant(); // permission probe
  await tick();
  page.mic.calls[page.mic.calls.length - 1].grant(); // pipeline microphone
  await tick();
  assert.ok(
    micBtn(page.win).classList.contains('voice-listening'),
    `pipeline must be live: ${micBtn(page.win).className}`,
  );
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

function feed(proc, loud, blocks) {
  for (let i = 0; i < blocks; i++) proc.onaudioprocess(audioBlock(loud));
}

/** Latest live recognizer / processor of the current pipeline. */
function lastRecognizer(page) {
  return page.state.recognizers[page.state.recognizers.length - 1];
}

function lastProcessor(page) {
  return page.state.processors[page.state.processors.length - 1];
}

/** Fire the wake word through the pipeline's own result handler. */
function fireWake(page) {
  page.clock.now += 3000; // clear the 2 s wake cooldown
  lastRecognizer(page).handlers.result({result: {text: 'sorcar'}});
}

/**
 * Clear the awaitingFlush latch fireWake armed: the real worker answers
 * retrieveFinalResult with one result message that voice.js swallows.
 */
function flushWake(page) {
  lastRecognizer(page).handlers.result({result: {text: ''}});
}

function submits(posted) {
  return posted.filter(m => m.type === 'submit');
}

function droppedTo(posted, tabId) {
  return posted.filter(m => m.type === 'voiceDropped' && m.tabId === tabId);
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
    'CAND-1: hide-suspend mid-capture retires the just-begun round, so the ' +
      'next transcript pairs with the round that spoke it',
    async () => {
      const page = makePage();
      const {win, posted} = page;
      const {first, second} = twoTabs(win);
      clickTab(win, first); // tab A on screen

      await startPipeline(page);
      fireWake(page); // capture begins: unkeyed round owned by tab A
      assert.ok(
        micBtn(win).classList.contains('voice-triggered'),
        'the wake must have fired',
      );

      // The user hides the panel mid-capture: the capture's audio dies with
      // the pipeline (no voiceTranscribe was posted), so its round must die
      // with it too.
      setPageVisibility(win, 'hidden');
      assert.strictEqual(
        posted.filter(m => m.type === 'voiceTranscribe').length,
        0,
        'the abandoned capture must not have posted audio',
      );

      // Resume, move to tab B, and speak a full round there.
      setPageVisibility(win, 'visible');
      await tick();
      page.mic.calls[page.mic.calls.length - 1].grant();
      await tick();
      clickTab(win, second); // tab B on screen
      fireWake(page); // round B begins on tab B
      feed(lastProcessor(page), true, 2); // speech...
      feed(lastProcessor(page), false, 8); // ...then 2 s of silence
      const transcribes = posted.filter(m => m.type === 'voiceTranscribe');
      assert.strictEqual(
        transcribes.length,
        1,
        'round B must have posted its audio: ' + JSON.stringify(posted),
      );

      // The server answers round B (unkeyed, as the browser pipeline posts
      // it). Its words belong to tab B, the tab on screen.
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'round B words QC1'});
      assert.strictEqual(
        droppedTo(posted, first).length,
        0,
        "round B's words must not be charged to pre-hide tab A: " +
          JSON.stringify(posted),
      );
      const subs = submits(posted);
      assert.strictEqual(
        subs.length,
        1,
        "round B's words belong to the tab that spoke them: " +
          JSON.stringify(posted),
      );
      assert.ok(subs[0].prompt.includes('round B words QC1'));
      win.close();
    },
  );

  await test(
    'CAND-1: after a hide mid-capture, an unsolicited host transcript is ' +
      'still delivered instead of being eaten by the stale round',
    async () => {
      const page = makePage();
      const {win, posted} = page;
      const {first, second} = twoTabs(win);
      clickTab(win, first);

      await startPipeline(page);
      fireWake(page); // capture begins on tab A
      setPageVisibility(win, 'hidden'); // capture abandoned
      setPageVisibility(win, 'visible');
      await tick();
      page.mic.calls[page.mic.calls.length - 1].grant();
      await tick();
      clickTab(win, second);

      // The extension host transcribes on its own too: a transcript that
      // was never tied to any round this page saw must still arrive. A
      // leftover entry for the abandoned capture would misattribute it.
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'host side words QC2'});
      assert.strictEqual(
        droppedTo(posted, first).length,
        0,
        'no words may be charged to the abandoned round: ' +
          JSON.stringify(posted),
      );
      assert.strictEqual(
        submits(posted).length,
        1,
        'an unsolicited host transcript must still be delivered: ' +
          JSON.stringify(posted),
      );
      win.close();
    },
  );

  await test(
    'CAND-2: a wake that captures no audio retires ITS OWN round, not the ' +
      'oldest one still awaiting its transcript',
    async () => {
      const page = makePage();
      const {win, posted} = page;
      const {first, second} = twoTabs(win);
      clickTab(win, first); // tab A on screen

      await startPipeline(page);
      const proc = lastProcessor(page);

      // Round A: wake, speech, 2 s silence -> audio posted, transcript
      // pending. Its owner (tab A) must stay queued until the reply.
      fireWake(page);
      feed(proc, true, 2);
      feed(proc, false, 8);
      assert.strictEqual(
        posted.filter(m => m.type === 'voiceTranscribe').length,
        1,
        "round A's audio must be in flight",
      );

      // Round B begins on tab B while A transcribes, then the user says
      // nothing for 5 s: the no-audio branch must retire round B itself.
      clickTab(win, second);
      flushWake(page);
      fireWake(page);
      feed(proc, false, 20); // 5.12 s of silence, no speech ever started
      assert.strictEqual(
        posted.filter(m => m.type === 'voiceTranscribe').length,
        1,
        'the silent round must not post audio',
      );

      // Round A's transcript arrives (the server replies unkeyed). Its
      // words were spoken on tab A; tab B is on screen, so they must be
      // handed back naming tab A — never submitted into tab B.
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'round A words QC3'});
      assert.strictEqual(
        submits(posted).length,
        0,
        "round A's words must not be typed into tab B's conversation: " +
          JSON.stringify(posted),
      );
      const dropped = droppedTo(posted, first);
      assert.ok(
        dropped.some(m => m.text.includes('round A words QC3')),
        'the words must be handed back naming the tab that spoke them: ' +
          JSON.stringify(posted),
      );
      win.close();
    },
  );

  await test(
    'CAND-2: a rounds reset mid-capture leaves the silent finish with ' +
      'nothing to retire and the accounting sound',
    async () => {
      const page = makePage();
      const {win, posted} = page;
      const {first} = twoTabs(win);
      clickTab(win, first);

      await startPipeline(page);
      fireWake(page); // capture begins: one unkeyed round
      // The host reports a not-listening status mid-capture; voice.js
      // resets every outstanding round. The capture itself lives on and
      // ends in the no-audio branch, which now has nothing of its own
      // left to retire — and must not eat into anything else.
      send(win, {type: 'voiceState', listening: false});
      feed(lastProcessor(page), false, 20); // 5.12 s: silent finish

      // The reset moved the capture's round into the cancelled-unkeyed
      // credit, so the next unkeyed transcript fails CLOSED against it
      // (by design: a cancelled round's words may not be typed anywhere).
      // The silent finish must not have consumed that credit, nor pushed
      // any counter negative.
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'cancelled words QC4'});
      assert.strictEqual(
        submits(posted).length,
        0,
        "a cancelled round's transcript must fail closed: " +
          JSON.stringify(posted),
      );

      // With the credit spent, an unsolicited host transcript flows again:
      // the accounting is exactly one round behind, never stale forever.
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'after reset words QC5'});
      assert.strictEqual(
        submits(posted).length,
        1,
        'an unsolicited host transcript must still be delivered: ' +
          JSON.stringify(posted),
      );
      win.close();
    },
  );

  await test(
    'CAND-3: stopping the pipeline removes the debug free recognizer too',
    async () => {
      const page = makePage({mode: 'browser', debug: true});
      const {win, state, model} = page;

      micBtn(win).click(); // browser mode: no probe, pipeline directly
      await tick();
      page.mic.calls[0].grant();
      await tick();
      assert.ok(micBtn(win).classList.contains('voice-listening'));
      assert.strictEqual(
        state.recognizers.length,
        2,
        'debug mode runs a wake recognizer plus a free-text one',
      );
      assert.strictEqual(model.recognizers.size, 2);

      micBtn(win).click(); // off: stopBrowserPipeline
      assert.strictEqual(
        model.recognizers.size,
        0,
        'every recognizer the pipeline registered must be removed: ' +
          JSON.stringify(Array.from(model.recognizers.keys())),
      );
      for (const rec of state.recognizers) {
        assert.ok(
          rec.removed,
          `recognizer ${rec.id} (grammar: ${rec.grammar || 'free'}) leaked`,
        );
      }
      win.close();
    },
  );

  await test(
    'CAND-3: repeated debug-mode pipeline restarts leave no recognizer behind',
    async () => {
      const page = makePage({debug: true});
      const {win, state, model} = page;
      await startPipeline(page);

      // Hide/show park and restart the pipeline; each cycle creates a
      // fresh recognizer pair and must free the previous one entirely.
      setPageVisibility(win, 'hidden');
      setPageVisibility(win, 'visible');
      await tick();
      page.mic.calls[page.mic.calls.length - 1].grant();
      await tick();
      assert.ok(micBtn(win).classList.contains('voice-listening'));
      assert.strictEqual(
        model.recognizers.size,
        2,
        'only the live pipeline pair may be registered after a restart: ' +
          JSON.stringify(Array.from(model.recognizers.keys())),
      );
      assert.strictEqual(state.recognizers.length, 4);

      micBtn(win).click(); // off
      assert.strictEqual(
        model.recognizers.size,
        0,
        'stopping must free the last pair as well',
      );
      win.close();
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
