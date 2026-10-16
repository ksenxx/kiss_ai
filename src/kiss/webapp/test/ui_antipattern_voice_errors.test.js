// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// UI anti-pattern fix A9 / A3 for the voice trigger (media/voice.js): a
// microphone / audio / server failure is no longer visible only as a
// hover tooltip on a button hidden inside the collapsed "More actions"
// menu.  It also raises a sticky, dismissible toast in the page's
// notification container, worded in plain language with a remedy, and a
// later successful start closes that toast again.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');
const {waitFor} = require('./simplify2_harness');

const MEDIA = path.join(__dirname, '..', 'media');

// Every page a test builds; closed after the test so no voice or toast
// timer keeps node alive.
const windows = [];

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
  if (options.mediaDevices) {
    Object.defineProperty(win.navigator, 'mediaDevices', {
      value: options.mediaDevices,
      configurable: true,
    });
  }
  if (options.vosk) win.Vosk = options.vosk;
  win.AudioContext = FakeAudioContext;
  win.__VOICE__ = Object.assign({}, options.voiceCfg);
  win.localStorage.setItem('kissVoiceEnabled', '0');
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'voice.js'), 'utf8'));
  windows.push(win);
  return {win, posted};
}

const BROWSER_CFG = {
  voskSrc: '/media/vosk.js',
  modelUrl: 'https://models.example.com/model.tar.gz',
  ackAudioUrl: '/media/working-on-it.mp3',
};

/** A KaldiRecognizer with the surface the pipeline uses. */
class FakeRecognizer {
  setWords() {}
  on() {}
  acceptWaveform() {}
  retrieveFinalResult() {}
  remove() {}
}

/** A speech engine whose model loads at once, so getUserMedia is reached. */
const FAKE_VOSK = {
  createModel: () =>
    Promise.resolve({KaldiRecognizer: FakeRecognizer, terminate() {}}),
};

/** A minimal AudioContext for a granted microphone. */
class FakeAudioContext {
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
    return {connect: () => {}, disconnect: () => {}, onaudioprocess: null};
  }
}

function denyingMediaDevices(name, message) {
  return {
    getUserMedia: () => {
      const err = new Error(message);
      err.name = name;
      return Promise.reject(err);
    },
  };
}

/**
 * A microphone that fails each request with the next message in
 * `messages`, then grants every later request.
 */
function flakyMediaDevices(messages) {
  const pending = messages.slice();
  return {
    getUserMedia: () => {
      if (pending.length) {
        const err = new Error(pending.shift());
        err.name = 'AbortError';
        return Promise.reject(err);
      }
      return Promise.resolve({
        getTracks: () => [{stop: () => {}}],
        getAudioTracks: () => [],
      });
    },
    enumerateDevices: () => Promise.resolve([]),
  };
}

function micBtn(win) {
  return win.document.getElementById('voice-btn');
}

function toast(win) {
  return win.document.querySelector(
    '.kiss-notification[data-notification-id="voice-trigger-error"]',
  );
}

/**
 * Wait for the voice trigger's failure to land: voice.js raises the
 * toast and paints the button's state in one synchronous step, so once
 * the toast exists the class, tooltip and aria-pressed are set too.
 */
function waitForToast(win) {
  return waitFor(() => toast(win), 'no voice error toast appeared');
}

let passed = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  ok - ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  not ok - ${name}`);
  } finally {
    while (windows.length) windows.pop().close();
  }
}

async function main() {
  await test('a denied microphone raises a sticky error toast with a remedy', async () => {
    const {win} = makePage({
      voiceCfg: BROWSER_CFG,
      vosk: FAKE_VOSK,
      mediaDevices: denyingMediaDevices('NotAllowedError', 'Permission denied'),
    });
    assert.strictEqual(toast(win), null, 'no toast before the failure');
    micBtn(win).click();
    await waitForToast(win);
    assert.ok(
      micBtn(win).classList.contains('voice-error'),
      micBtn(win).className,
    );
    const t = toast(win);
    assert.ok(t, 'a visible notification must be raised');
    assert.ok(t.classList.contains('kiss-notification-error'), t.className);
    assert.strictEqual(
      t.dataset.notificationSticky,
      'true',
      'errors never auto-dismiss',
    );
    const text = t.querySelector('.kiss-notification-message').textContent;
    assert.strictEqual(
      text,
      'Microphone access was denied. Allow the microphone for this site in ' +
        'the browser settings, then turn the voice trigger on again.',
    );
    assert.ok(!/Permission denied/.test(text), 'no raw error jargon');
    // The tooltip carries the same plain-language text.
    assert.ok(
      /Microphone access was denied/.test(
        micBtn(win).getAttribute('data-tooltip'),
      ),
    );
    // The toast is dismissible.
    t.querySelector('.kiss-notification-close').click();
    assert.strictEqual(toast(win), null, 'the close button removes the toast');
  });

  await test('a missing microphone gets its own remedy', async () => {
    const {win} = makePage({
      voiceCfg: BROWSER_CFG,
      vosk: FAKE_VOSK,
      mediaDevices: denyingMediaDevices(
        'NotFoundError',
        'Requested device not found',
      ),
    });
    micBtn(win).click();
    await waitForToast(win);
    const text = toast(win).querySelector(
      '.kiss-notification-message',
    ).textContent;
    assert.ok(
      /No microphone was found\. Connect a microphone/.test(text),
      text,
    );
  });

  await test('an unknown failure names the cause and what to do next', async () => {
    const {win} = makePage({
      voiceCfg: BROWSER_CFG,
      vosk: FAKE_VOSK,
      mediaDevices: denyingMediaDevices('AbortError', 'Starting audio failed'),
    });
    micBtn(win).click();
    await waitForToast(win);
    const text = toast(win).querySelector(
      '.kiss-notification-message',
    ).textContent;
    assert.strictEqual(
      text,
      'The voice trigger could not start (Starting audio failed). Check the ' +
        'microphone, then turn the voice trigger on again.',
    );
  });

  await test('a repeated failure updates the same toast, and a later success closes it', async () => {
    const {win} = makePage({
      voiceCfg: BROWSER_CFG,
      vosk: FAKE_VOSK,
      mediaDevices: flakyMediaDevices([
        'microphone busy (code 1)',
        'microphone busy (code 2)',
      ]),
    });
    micBtn(win).click();
    await waitForToast(win);
    const t = toast(win);
    assert.ok(t, 'the failure must be visible');
    assert.ok(t.classList.contains('kiss-notification-error'));
    assert.ok(/microphone busy \(code 1\)/.test(t.textContent));
    assert.ok(/turn the voice trigger on again/.test(t.textContent));
    assert.strictEqual(micBtn(win).getAttribute('aria-pressed'), 'false');
    // A repeated failure updates the same toast instead of stacking.
    micBtn(win).click();
    await waitFor(
      () => /code 2/.test(toast(win).textContent),
      'the toast was not updated',
    );
    assert.strictEqual(
      win.document.querySelectorAll('.kiss-notification').length,
      1,
      'one toast, updated in place',
    );
    assert.ok(/code 2/.test(toast(win).textContent));
    assert.strictEqual(micBtn(win).getAttribute('aria-pressed'), 'false');
    // The third attempt gets the microphone: the stale error goes away.
    micBtn(win).click();
    await waitFor(
      () => micBtn(win).classList.contains('voice-listening'),
      'the voice trigger never started listening',
    );
    assert.ok(
      micBtn(win).classList.contains('voice-listening'),
      micBtn(win).className,
    );
    assert.strictEqual(
      toast(win),
      null,
      'a working voice trigger clears the stale error',
    );
    assert.strictEqual(
      micBtn(win).getAttribute('aria-pressed'),
      'true',
      'toggle state is exposed',
    );
  });
}

main().then(() => {
  console.log(`\n${passed} passed, ${failures.length} failed`);
  for (const f of failures) {
    console.error(`\n${f.name}\n${f.error && f.error.stack}`);
  }
  if (failures.length) process.exit(1);
});
