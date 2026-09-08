// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E page tests for the in-page (browser-mic) capture fallback and the
// calm mic-less-host state in voice.js (webview mode).
//
// Scenario: the KISS daemon / extension host runs on a machine without a
// microphone. The mic button must never paint a raw listener failure
// ("Voice trigger error: ... OSError: PortAudio library not found").
// Instead:
//  - clicking the mic first PROBES in-page capture (getUserMedia): when
//    the embedder grants it, the page records with the BROWSER's mic and
//    the host listener is never asked for (no voiceToggle);
//  - when the embedder refuses (today's VS Code webviews reject with
//    NotAllowedError), the host listener is tried exactly as before;
//  - a host reply of `voiceState {hostMicUnavailable:true}` puts the
//    button into a calm dimmed "unavailable" state, not the red error;
//  - pages without the fallback wiring (no voskSrc/modelUrl — older
//    hosts, plain dictation surfaces) keep the synchronous host-toggle
//    path.

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

  // Every constructed in-page ack player, so tests can tell in-page audio
  // from the host voiceAck route.
  const audios = [];
  win.Audio = function (src) {
    const player = {src, play: () => Promise.resolve()};
    audios.push(player);
    return player;
  };

  if (options.mediaDevices) {
    Object.defineProperty(win.navigator, 'mediaDevices', {
      value: options.mediaDevices,
      configurable: true,
    });
  }

  win.__VOICE__ = Object.assign({mode: 'webview'}, options.voiceCfg);
  win.localStorage.setItem('kissVoiceEnabled', options.startEnabled || '0');
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'voice.js'), 'utf8'));
  return {win, posted, audios};
}

// The full fallback wiring the host now embeds in VOICE_CONFIG.
const FALLBACK_CFG = {
  voskSrc: '/media/vosk.js',
  modelUrl: 'https://models.example.com/model.tar.gz',
  ackAudioUrl: '/media/working-on-it.mp3',
  nonce: 'test-nonce',
};

/** A getUserMedia stub that grants a stream with stoppable tracks. */
function grantingMediaDevices(log) {
  return {
    getUserMedia: constraints => {
      log.push(constraints);
      return Promise.resolve({
        getTracks: () => [{stop: () => log.push('stopped')}],
        getAudioTracks: () => [],
      });
    },
  };
}

/** A getUserMedia stub that refuses like a VS Code webview. */
function refusingMediaDevices(log) {
  return {
    getUserMedia: () => {
      log.push('asked');
      const err = new Error('Permission denied');
      err.name = 'NotAllowedError';
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

function tick() {
  // Two macrotask turns: probeBrowserCapture chains two promise ticks
  // (getUserMedia settlement, then the .then handler).
  return new Promise(resolve => setTimeout(resolve, 20));
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
    console.log(`      ${e.message}`);
  }
}

async function main() {
  await test(
    'granted in-page capture takes over: no voiceToggle, vosk loads with the CSP nonce',
    async () => {
      const log = [];
      const {win, posted} = makeWebview({
        voiceCfg: FALLBACK_CFG,
        mediaDevices: grantingMediaDevices(log),
      });
      micBtn(win).click();
      await tick();
      assert.strictEqual(
        posted.filter(m => m.type === 'voiceToggle').length,
        0,
        `host listener must not be started: ${JSON.stringify(posted)}`,
      );
      assert.ok(log.includes('stopped'), 'the probe stream must be released');
      const vosk = Array.from(
        win.document.querySelectorAll('script'),
      ).find(s => (s.src || '').includes('/media/vosk.js'));
      assert.ok(vosk, 'the vosk bundle must be injected for in-page capture');
      assert.strictEqual(
        vosk.nonce,
        'test-nonce',
        'the injected vosk script must carry the page CSP nonce',
      );
    },
  );

  await test(
    'refused in-page capture falls back to the host listener toggle',
    async () => {
      const log = [];
      const {win, posted} = makeWebview({
        voiceCfg: FALLBACK_CFG,
        mediaDevices: refusingMediaDevices(log),
      });
      micBtn(win).click();
      await tick();
      const toggles = posted.filter(m => m.type === 'voiceToggle');
      assert.strictEqual(toggles.length, 1, JSON.stringify(posted));
      assert.strictEqual(toggles[0].enabled, true);
      assert.deepStrictEqual(log, ['asked']);
    },
  );

  await test(
    'hostMicUnavailable shows the calm unavailable state, never the red error',
    async () => {
      const {win} = makeWebview({
        voiceCfg: FALLBACK_CFG,
        mediaDevices: refusingMediaDevices([]),
      });
      micBtn(win).click();
      await tick();
      send(win, {
        type: 'voiceState',
        listening: false,
        hostMicUnavailable: true,
      });
      const btn = micBtn(win);
      assert.ok(
        btn.classList.contains('voice-unavailable'),
        `expected voice-unavailable, got: ${btn.className}`,
      );
      assert.ok(!btn.classList.contains('voice-error'));
      const tip = btn.getAttribute('data-tooltip') || '';
      assert.ok(
        !/error|OSError/i.test(tip),
        `tooltip must not read as an error: ${tip}`,
      );
      assert.ok(
        /remote web app/i.test(tip),
        `tooltip must point at the remote web app: ${tip}`,
      );
      assert.strictEqual(
        win.localStorage.getItem('kissVoiceEnabled'),
        '0',
        'the persisted intent must be cleared',
      );
    },
  );

  await test(
    'a real listener error after READY still shows the red error state',
    async () => {
      const {win} = makeWebview({
        voiceCfg: FALLBACK_CFG,
        mediaDevices: refusingMediaDevices([]),
      });
      micBtn(win).click();
      await tick();
      send(win, {type: 'voiceState', listening: true});
      send(win, {
        type: 'voiceState',
        listening: false,
        error: 'voice listener exited (code 1): mic watchdog gave up',
      });
      const btn = micBtn(win);
      assert.ok(btn.classList.contains('voice-error'));
      assert.ok(!btn.classList.contains('voice-unavailable'));
    },
  );

  await test(
    'a page without the fallback wiring posts the host toggle synchronously',
    async () => {
      const {win, posted} = makeWebview({voiceCfg: {}});
      micBtn(win).click();
      const toggles = posted.filter(m => m.type === 'voiceToggle');
      assert.strictEqual(
        toggles.length,
        1,
        'legacy pages must keep the synchronous host toggle',
      );
      assert.strictEqual(toggles[0].enabled, true);
    },
  );

  await test(
    'a wired page whose browser lacks mediaDevices also keeps the sync toggle',
    async () => {
      // voskSrc/modelUrl present but no navigator.mediaDevices at all
      // (insecure contexts, very old embedders): capture is impossible,
      // so the probe must be skipped, not attempted.
      const {win, posted} = makeWebview({voiceCfg: FALLBACK_CFG});
      micBtn(win).click();
      const toggles = posted.filter(m => m.type === 'voiceToggle');
      assert.strictEqual(toggles.length, 1, JSON.stringify(posted));
      assert.strictEqual(toggles[0].enabled, true);
    },
  );

  await test(
    'in-page pipeline plays the ack itself; host mode still posts voiceAck',
    async () => {
      // Host-listener mode: a transcript arriving via voiceSpeech acks
      // through the host (it owns the speakers there).
      const hostMode = makeWebview({
        voiceCfg: FALLBACK_CFG,
        mediaDevices: refusingMediaDevices([]),
      });
      hostMode.win.document.getElementById('task-input').value = '';
      send(hostMode.win, {type: 'voiceSpeech', text: 'host words'});
      assert.strictEqual(
        hostMode.posted.filter(m => m.type === 'voiceAck').length,
        1,
        'host mode must ack through the extension host',
      );
      assert.strictEqual(hostMode.audios.length, 0);

      // In-page pipeline: the fallback exists because the host machine
      // has no working audio hardware, so the ack plays in the page.
      const log = [];
      const inPage = makeWebview({
        voiceCfg: FALLBACK_CFG,
        mediaDevices: grantingMediaDevices(log),
      });
      micBtn(inPage.win).click();
      await tick();
      send(inPage.win, {type: 'voiceSpeech', text: 'browser words'});
      assert.strictEqual(
        inPage.posted.filter(m => m.type === 'voiceAck').length,
        0,
        'the in-page pipeline must not ack through the host',
      );
      assert.strictEqual(inPage.audios.length, 1);
      assert.ok(inPage.audios[0].src.includes('working-on-it.mp3'));
    },
  );

  await test(
    'sensitivity changes reach the host only while the host listener is in use',
    async () => {
      const log = [];
      const {win, posted} = makeWebview({
        voiceCfg: FALLBACK_CFG,
        mediaDevices: grantingMediaDevices(log),
      });
      const slider = win.document.getElementById('cfg-voice-sensitivity');
      if (!slider) return; // settings panel absent from this surface
      micBtn(win).click();
      await tick(); // in-page pipeline active
      slider.value = '55';
      slider.dispatchEvent(new win.Event('input', {bubbles: true}));
      assert.strictEqual(
        posted.filter(m => m.type === 'voiceSensitivity').length,
        0,
        'the in-page pipeline reads sensitivity live; nothing to ship',
      );
    },
  );

  await test(
    'toggling off during the capture probe never starts anything',
    async () => {
      // The user clicks the mic ON and immediately OFF while the
      // getUserMedia probe is still pending: the resolved probe must not
      // start the in-page pipeline against the user's OFF.
      let resolveProbe;
      const md = {
        getUserMedia: () =>
          new Promise(resolve => {
            resolveProbe = resolve;
          }),
      };
      const {win, posted} = makeWebview({
        voiceCfg: FALLBACK_CFG,
        mediaDevices: md,
      });
      micBtn(win).click(); // on: probe pending
      micBtn(win).click(); // off
      resolveProbe({
        getTracks: () => [{stop: () => {}}],
        getAudioTracks: () => [],
      });
      await tick();
      const btn = micBtn(win);
      assert.ok(
        btn.classList.contains('voice-off'),
        `mic must stay off: ${btn.className}`,
      );
      const vosk = Array.from(
        win.document.querySelectorAll('script'),
      ).find(s => (s.src || '').includes('/media/vosk.js'));
      assert.ok(!vosk, 'the pipeline must not start after the off click');
      // The off click posts voiceToggle(false) (harmless: no listener
      // runs); no voiceToggle(true) may ever have been posted.
      const onToggles = posted.filter(
        m => m.type === 'voiceToggle' && m.enabled,
      );
      assert.strictEqual(onToggles.length, 0, JSON.stringify(posted));
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
