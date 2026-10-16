// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E page tests for round-owner hygiene in the in-page (browser mic)
// voice pipeline of voice.js, and for the debug-mode free recognizer.
//
// Defects being reproduced (each test failed against the pre-fix code):
//
//  CAND-1  Stopping the pipeline mid-capture (stopBrowserPipeline) threw
//          the active capture away but left its just-begun round in
//          roundOwners. No voiceTranscribe was ever posted for it, so no
//          transcript could ever retire it, and the NEXT transcript was
//          paired with the stale owner: words spoken later on tab B were
//          charged to tab A and silently dropped or typed into the wrong
//          task.
//
//  CAND-2  finishCapture's no-audio branch retired the OLDEST round.
//          Rounds overlap by design (wake B can begin while round A's
//          audio is still being transcribed), and the round that produced
//          no audio is the one that JUST began — the NEWEST. Retiring the
//          oldest consumed round A's owner, so A's transcript was then
//          paired with owner B: one tab's words submitted into another
//          tab's conversation.
//
//  CAND-3  With kissVoiceDebug=1, startBrowserPipeline created a second,
//          grammar-free KaldiRecognizer for logging, held only by a
//          closure local. stopBrowserPipeline removed the module-level
//          `recognizer` but never the free one, leaking one worker-side
//          recognizer (a model.recognizers map entry) per pipeline start.
//
//  REV-1   The transcribe-flash safety timer FORGOT the outstanding rounds
//          (no cancelled credit), so a reply arriving after it had fired,
//          with no newer round to claim it, retired as "no round at all"
//          and was typed into whatever chat was on screen: words spoken
//          in chat A submitted into chat B.
//
//  REV-2   retireRound charged the cancelled credit BEFORE the live
//          rounds. A reply lost with its socket connection has no
//          transcript on its way, so the credit a later mic-off turned it
//          into ate the next valid dictation and left that round's owner
//          queued against the one after. The page cannot tell a cancelled
//          round whose reply is still coming from one whose reply was
//          lost, so a live round must always be answered first.
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

const VOICE_CFG = {
  voskSrc: '/media/vosk.js',
  modelUrl: 'https://models.example.com/model.tar.gz',
  ackAudioUrl: '/media/working-on-it.mp3',
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

  // The page's safety timers (the 45 s wake flash, the 60 s transcribing
  // flash) are recorded so a test can fire one without waiting.
  const timers = [];
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

  win.__VOICE__ = Object.assign({}, VOICE_CFG);
  win.localStorage.setItem('kissVoiceEnabled', '0');
  if (options.debug) win.localStorage.setItem('kissVoiceDebug', '1');
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'voice.js'), 'utf8'));
  return {win, posted, state, mic, clock, model, timers};
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

// Put chat *tabId* on screen: the Chats-panel pick (there is no row of
// chat tabs any more), which retires nothing.
function clickTab(win, tabId) {
  const api = win._testApi;
  assert.ok(
    api.openTabs().some(t => t.id === tabId),
    `tab ${tabId} must be open`,
  );
  api.switchToTab(tabId);
  assert.strictEqual(api.getActiveTabId(), tabId, `tab ${tabId} is on screen`);
}

/** Fire the page's pending safety timer of `ms` milliseconds. */
function fireTimer(page, ms) {
  const live = page.timers.filter(t => !t.cleared && t.ms === ms);
  assert.strictEqual(live.length, 1, `one live ${ms} ms timer is armed`);
  page.win.clearTimeout(live[0].id);
  live[0].cb();
}

function tick(ms) {
  return new Promise(resolve => setTimeout(resolve, ms || 20));
}

/** Turn the mic on and grant the pipeline its microphone. */
async function startPipeline(page) {
  micBtn(page.win).click();
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
  lastRecognizer(page).handlers.result({result: {text: 'hey sorcar'}});
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
    'CAND-1: turning the mic off mid-capture retires the just-begun round, ' +
      'so the next transcript pairs with the round that spoke it',
    async () => {
      const page = makePage();
      const {win, posted} = page;
      const {first, second} = twoTabs(win);
      clickTab(win, first); // tab A on screen

      await startPipeline(page);
      fireWake(page); // capture begins: round owned by tab A
      assert.ok(
        micBtn(win).classList.contains('voice-triggered'),
        'the wake must have fired',
      );

      // The user turns the mic off mid-capture: the capture's audio dies
      // with the pipeline (no voiceTranscribe was posted), so its round
      // must die with it too.
      micBtn(win).click();
      assert.strictEqual(
        posted.filter(m => m.type === 'voiceTranscribe').length,
        0,
        'the abandoned capture must not have posted audio',
      );

      // Mic back on, move to tab B, and speak a full round there.
      await startPipeline(page);
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

      // The daemon answers round B. Its words belong to tab B, the tab on
      // screen — never to the abandoned round's tab A.
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'round B words QC1'});
      const subs = submits(posted);
      assert.strictEqual(
        subs.length,
        1,
        "round B's words belong to the tab that spoke them: " +
          JSON.stringify(posted),
      );
      assert.ok(subs[0].prompt.includes('round B words QC1'));
      assert.strictEqual(subs[0].tabId, second);
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

      // Round A's transcript arrives. Its words were spoken on tab A; tab
      // B is on screen, so they are dropped — never submitted into tab B.
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'round A words QC3'});
      assert.strictEqual(
        submits(posted).length,
        0,
        "round A's words must not be typed into tab B's conversation: " +
          JSON.stringify(posted),
      );
      assert.ok(
        !win.document.getElementById('task-input').value.includes('QC3'),
        "round A's words must not be drafted into tab B's input either",
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
      fireWake(page); // capture begins: one round
      // The 45 s wake-flash safety timer fires mid-capture (the audio clock
      // stalled); voice.js cancels every outstanding round. The capture
      // itself lives on and ends in the no-audio branch, which now has
      // nothing of its own left to retire — and must not eat into
      // anything else.
      fireTimer(page, 45000);
      assert.ok(
        !micBtn(win).classList.contains('voice-triggered'),
        'the safety timer clears the wake flash',
      );
      feed(lastProcessor(page), false, 20); // 5.12 s: silent finish

      // The cancellation moved the capture's round into the cancelled
      // credit, so the next transcript fails CLOSED against it (by
      // design: a cancelled round's words may not be typed anywhere).
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

      // With the credit spent, an unsolicited transcript flows again: the
      // accounting is exactly one round behind, never stale forever.
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'after reset words QC5'});
      assert.strictEqual(
        submits(posted).length,
        1,
        'an unsolicited transcript must still be delivered: ' +
          JSON.stringify(posted),
      );
      win.close();
    },
  );

  await test(
    'REV-1: a reply arriving after the transcribe window ran out is refused, ' +
      'never typed into the chat now on screen',
    async () => {
      const page = makePage();
      const {win, posted} = page;
      const {first, second} = twoTabs(win);
      clickTab(win, first); // chat A on screen

      await startPipeline(page);
      const proc = lastProcessor(page);
      fireWake(page); // round A begins on chat A
      feed(proc, true, 2);
      feed(proc, false, 8); // audio posted, transcript pending
      assert.strictEqual(
        posted.filter(m => m.type === 'voiceTranscribe').length,
        1,
        "round A's audio must be in flight",
      );

      // The user moves to chat B; the daemon is slow and the 60 s
      // transcribing flash runs out before its reply lands.
      clickTab(win, second);
      fireTimer(page, 60000);
      assert.ok(
        !micBtn(win).classList.contains('voice-transcribing'),
        'the safety timer clears the transcribing flash',
      );

      // Round A's delayed reply arrives with no newer round to claim it.
      // Its words were spoken in chat A: they must not be submitted into
      // (or drafted for) chat B.
      posted.length = 0;
      send(win, {type: 'voiceSpeech', text: 'words spoken in chat A QR1'});
      assert.deepStrictEqual(
        submits(posted).map(m => ({onB: m.tabId === second, prompt: m.prompt})),
        [],
        "a cancelled round's late words must not be submitted: " +
          JSON.stringify(posted),
      );
      assert.ok(
        !win.document.getElementById('task-input').value.includes('QR1'),
        "nor drafted into chat B's input",
      );
      win.close();
    },
  );

  await test(
    'REV-2: a reply lost with the connection, then a mic restart, never ' +
      'eats the next valid dictation',
    async () => {
      const page = makePage();
      const {win, posted} = page;
      const {first, second} = twoTabs(win);
      clickTab(win, first); // chat A on screen

      await startPipeline(page);
      fireWake(page); // round A begins on chat A
      feed(lastProcessor(page), true, 2);
      feed(lastProcessor(page), false, 8); // audio posted
      assert.strictEqual(
        posted.filter(m => m.type === 'voiceTranscribe').length,
        1,
        "round A's audio must be in flight",
      );

      // The socket drops while the daemon transcribes: its reply to
      // round A is lost for good. The user turns the mic off and on
      // again, the page reconnects, and they dictate in chat B.
      send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
      micBtn(win).click(); // off: round A is cancelled
      send(win, {type: 'daemonStatus', connected: true});
      await startPipeline(page);
      clickTab(win, second);
      fireWake(page); // round B begins on chat B
      feed(lastProcessor(page), true, 2);
      feed(lastProcessor(page), false, 8);
      assert.strictEqual(
        posted.filter(m => m.type === 'voiceTranscribe').length,
        2,
        "round B's audio must have been posted",
      );

      // Only round B's reply ever arrives. It belongs to the live round,
      // never to the credit the lost round left behind.
      posted.length = 0;
      send(win, {
        type: 'voiceSpeech',
        text: 'valid words spoken in chat B QR2',
      });
      assert.deepStrictEqual(
        submits(posted).map(m => ({onB: m.tabId === second, prompt: m.prompt})),
        [{onB: true, prompt: 'valid words spoken in chat B QR2'}],
        "round B's words must reach the chat that spoke them: " +
          JSON.stringify(posted),
      );
      win.close();
    },
  );

  await test('CAND-3: stopping the pipeline removes the debug free recognizer too', async () => {
    const page = makePage({debug: true});
    const {win, state, model} = page;

    await startPipeline(page);
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
  });

  await test('CAND-3: repeated debug-mode pipeline restarts leave no recognizer behind', async () => {
    const page = makePage({debug: true});
    const {win, state, model} = page;
    await startPipeline(page);

    // Off and on again restart the pipeline; each cycle creates a fresh
    // recognizer pair and must free the previous one entirely.
    micBtn(win).click(); // off
    await startPipeline(page);
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
  });

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
