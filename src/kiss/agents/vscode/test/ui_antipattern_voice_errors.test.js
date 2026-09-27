// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// UI anti-pattern fix A9 / A3 for the voice trigger (media/voice.js): a
// microphone / audio / server failure is no longer visible only as a
// hover tooltip on a button hidden inside the collapsed "More actions"
// menu.  It also raises a sticky, dismissible toast in the webview's
// notification container, worded in plain language with a remedy, and a
// later successful start closes that toast again.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview(opts) {
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
  if (options.mediaDevices) {
    Object.defineProperty(win.navigator, 'mediaDevices', {
      value: options.mediaDevices,
      configurable: true,
    });
  }
  if (options.vosk) win.Vosk = options.vosk;
  win.__VOICE__ = Object.assign({mode: 'browser'}, options.voiceCfg);
  win.localStorage.setItem('kissVoiceEnabled', '0');
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'voice.js'), 'utf8'));
  return {win, posted};
}

const BROWSER_CFG = {
  voskSrc: '/media/vosk.js',
  modelUrl: 'https://models.example.com/model.tar.gz',
  ackAudioUrl: '/media/working-on-it.mp3',
};

/** A speech engine whose model loads at once, so getUserMedia is reached. */
const FAKE_VOSK = {
  createModel: () => Promise.resolve({terminate() {}}),
};

function denyingMediaDevices(name, message) {
  return {
    getUserMedia: () => {
      const err = new Error(message);
      err.name = name;
      return Promise.reject(err);
    },
  };
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function micBtn(win) {
  return win.document.getElementById('voice-btn');
}

function toast(win) {
  return win.document.querySelector(
    '.kiss-notification[data-notification-id="voice-trigger-error"]',
  );
}

function tick() {
  return new Promise(resolve => setTimeout(resolve, 30));
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
  }
}

async function main() {
  await test('a denied microphone raises a sticky error toast with a remedy', async () => {
    const {win} = makeWebview({
      voiceCfg: BROWSER_CFG,
      vosk: FAKE_VOSK,
      mediaDevices: denyingMediaDevices('NotAllowedError', 'Permission denied'),
    });
    assert.strictEqual(toast(win), null, 'no toast before the failure');
    micBtn(win).click();
    await tick();
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
    const {win} = makeWebview({
      voiceCfg: BROWSER_CFG,
      vosk: FAKE_VOSK,
      mediaDevices: denyingMediaDevices(
        'NotFoundError',
        'Requested device not found',
      ),
    });
    micBtn(win).click();
    await tick();
    const text = toast(win).querySelector(
      '.kiss-notification-message',
    ).textContent;
    assert.ok(
      /No microphone was found\. Connect a microphone/.test(text),
      text,
    );
  });

  await test('an unknown failure names the cause and what to do next', async () => {
    const {win} = makeWebview({
      voiceCfg: BROWSER_CFG,
      vosk: FAKE_VOSK,
      mediaDevices: denyingMediaDevices('AbortError', 'Starting audio failed'),
    });
    micBtn(win).click();
    await tick();
    const text = toast(win).querySelector(
      '.kiss-notification-message',
    ).textContent;
    assert.strictEqual(
      text,
      'The voice trigger could not start (Starting audio failed). Check the ' +
        'microphone, then turn the voice trigger on again.',
    );
  });

  await test('a server-side capture error toasts, and a later success closes the toast', async () => {
    const {win} = makeWebview({voiceCfg: {}});
    send(win, {
      type: 'voiceState',
      listening: false,
      error: 'voice listener exited (code 1)',
    });
    const t = toast(win);
    assert.ok(t, 'server error must be visible');
    assert.ok(t.classList.contains('kiss-notification-error'));
    assert.ok(/voice listener exited \(code 1\)/.test(t.textContent));
    assert.ok(/turn the voice trigger on again/.test(t.textContent));
    // A repeated failure updates the same toast instead of stacking.
    send(win, {
      type: 'voiceState',
      listening: false,
      error: 'voice listener exited (code 2)',
    });
    assert.strictEqual(
      win.document.querySelectorAll('.kiss-notification').length,
      1,
      'one toast, updated in place',
    );
    assert.ok(/code 2/.test(toast(win).textContent));
    assert.strictEqual(micBtn(win).getAttribute('aria-pressed'), 'false');
    send(win, {type: 'voiceState', listening: true});
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

  await test('the calm "no microphone on the host" state is a sticky warning, not an error', async () => {
    const {win} = makeWebview({voiceCfg: {}});
    send(win, {type: 'voiceState', listening: false, hostMicUnavailable: true});
    const t = toast(win);
    assert.ok(t, 'the unavailable state must be visible');
    assert.ok(t.classList.contains('kiss-notification-warning'), t.className);
    assert.strictEqual(t.dataset.notificationSticky, 'true');
    assert.ok(/remote web app/i.test(t.textContent));
  });
}

main().then(() => {
  console.log(`\n${passed} passed, ${failures.length} failed`);
  for (const f of failures) {
    console.error(`\n${f.name}\n${f.error && f.error.stack}`);
  }
  if (failures.length) process.exit(1);
});
