// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test: in the remote webapp (browser mode)
// the "Sorcar" wake word was recognized but the speech that followed
// was NEVER transcribed to text.
//
// Bug being reproduced: ``media/voice.js`` only implemented post-wake
// speech handling for webview mode (the VS Code extension host runs a
// local listener that captures + translates and posts voiceSpeech
// back).  In browser mode ``triggerWake()`` merely flashed the mic
// button red for 600ms — no audio was captured after the wake word,
// nothing was sent to the server, and no text ever appeared in the
// task input.
//
// The fix makes browser mode capture the utterance that follows the
// wake word (RMS endpointing mirroring the Python listener's
// SpeechCapture), downsample it to 16kHz s16le PCM and post
// ``{type: 'voiceTranscribe', audio: <base64 pcm>}`` to the server
// (via the 'kiss-voice-post' bridge -> WS shim), which replies with
// the ``{type: 'voiceSpeech', text, speaker, language}`` message
// voice.js already consumes.
//
// These tests run the REAL production ``media/voice.js`` in jsdom
// with a browser-mode config, a stub audio pipeline (jsdom has no
// microphone) and a stub Vosk wake recognizer, and lock in:
//
//  1. Wake followed by speech then silence posts a voiceTranscribe
//     message whose base64 audio decodes to the captured 16kHz PCM
//     (FAILS before the fix: nothing is ever posted).
//  2. The wake recognizer does not hear capture audio (blocks are
//     routed to the capture, mirroring the Python listener).
//  3. Wake followed by only silence times out without posting
//     anything and clears the red wake flash (NO_SPEECH).
//  4. The server's voiceSpeech reply inserts the (speaker-prefixed)
//     text into the task input and submits it via kiss-voice-submit.
//  5. A late voiceSpeech from an older round does not clear a newer
//     browser capture's red flash.
//  6. Audio at a 48kHz mic rate is downsampled to 16kHz PCM.
//  7. ACTUAL VOICE (macOS only): real spoken audio synthesized with
//     `say` flows through the real capture path and the posted PCM
//     round-trips with speech-level energy.
//
// Run directly with ``node test/voiceBrowserTranscribe.test.js``.

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawnSync} = require('child_process');
const {JSDOM} = require('jsdom');

const VOICE_JS_PATH = path.join(__dirname, '..', 'media', 'voice.js');
const BLOCK = 4096; // frames per ScriptProcessor block

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

function flush() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

/**
 * Build a jsdom window running the REAL media/voice.js in browser
 * mode on top of a stub audio pipeline: getUserMedia hands out a fake
 * stream, AudioContext exposes a ScriptProcessor whose onaudioprocess
 * the test drives with synthetic (or real, TTS-derived) sample
 * blocks, and window.Vosk provides a recognizer stub whose
 * result/partialresult handlers the test can fire (jsdom cannot run
 * the vosk-browser WASM worker).  The code under test is unmodified.
 */
async function makeBrowserVoice({sampleRate = 16000} = {}) {
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body>' +
      '<button id="voice-btn" class="toggle-btn"></button>' +
      '<textarea id="task-input"></textarea>' +
      '</body></html>',
    {runScripts: 'dangerously', url: 'https://localhost/'},
  );
  const win = dom.window;
  win.__VOICE__ = {
    mode: 'browser',
    voskSrc: 'vosk.js',
    modelUrl: '/voice-model.tar.gz',
  };
  let now = 1000000;
  win.Date.now = () => now;

  const recognizers = [];
  win.Vosk = {
    createModel: () =>
      Promise.resolve({
        KaldiRecognizer: class {
          constructor() {
            this.handlers = {};
            this.acceptedBlocks = 0;
            recognizers.push(this);
          }
          setWords() {}
          on(event, cb) {
            this.handlers[event] = cb;
          }
          acceptWaveform() {
            this.acceptedBlocks++;
          }
          remove() {}
        },
      }),
  };

  const processors = [];
  win.AudioContext = class {
    constructor() {
      this.sampleRate = sampleRate;
      this.state = 'running';
      this.destination = {};
    }
    resume() {
      return Promise.resolve();
    }
    close() {}
    createMediaStreamSource() {
      return {connect() {}, disconnect() {}};
    }
    createScriptProcessor() {
      const node = {
        onaudioprocess: null,
        connect() {},
        disconnect() {},
      };
      processors.push(node);
      return node;
    }
  };

  Object.defineProperty(win.navigator, 'mediaDevices', {
    value: {
      getUserMedia: () =>
        Promise.resolve({
          getTracks: () => [{stop() {}}],
          getAudioTracks: () => [],
        }),
      enumerateDevices: () => Promise.resolve([]),
    },
  });

  const posted = [];
  win.addEventListener('kiss-voice-post', ev => posted.push(ev.detail));
  const submits = {count: 0};
  win.addEventListener('kiss-voice-submit', () => submits.count++);

  win.eval(fs.readFileSync(VOICE_JS_PATH, 'utf8'));

  // Enable listening (browser mode never auto-starts).
  win.document.getElementById('voice-btn').click();
  for (let i = 0; i < 10 && processors.length === 0; i++) await flush();
  assert.strictEqual(processors.length, 1, 'audio pipeline did not start');
  assert.strictEqual(recognizers.length, 1, 'wake recognizer not created');

  const recognizer = recognizers[0];
  return {
    win,
    posted,
    submits,
    recognizer,
    btn: win.document.getElementById('voice-btn'),
    inp: win.document.getElementById('task-input'),
    advance(ms) {
      now += ms;
    },
    feed(samples) {
      processors[0].onaudioprocess({
        inputBuffer: {
          getChannelData: () => samples,
          sampleRate,
        },
      });
    },
    wake() {
      // 300ms of quiet audio satisfies the 200ms post-alias pause
      // gate, then the recognizer's partial result fires the wake.
      this.feed(new win.Float32Array(Math.ceil(0.3 * sampleRate)));
      recognizer.handlers.partialresult({result: {partial: 'sir car'}});
    },
  };
}

function loudBlock(win, amplitude = 0.1, frames = BLOCK) {
  const samples = new win.Float32Array(frames);
  samples.fill(amplitude);
  return samples;
}

function decodePcm(audioB64) {
  const bytes = Buffer.from(audioB64, 'base64');
  const pcm = new Int16Array(bytes.length / 2);
  for (let i = 0; i < pcm.length; i++) pcm[i] = bytes.readInt16LE(2 * i);
  return pcm;
}

async function main() {
  await test(
    'wake then speech then silence posts voiceTranscribe with the PCM',
    async () => {
      const v = await makeBrowserVoice();
      v.wake();
      assert.ok(
        v.btn.classList.contains('voice-triggered'),
        'wake must flash the mic button red',
      );
      // Two loud speech blocks, then 2s (8 x 256ms) of silence.
      v.feed(loudBlock(v.win));
      v.feed(loudBlock(v.win));
      for (let i = 0; i < 8; i++) v.feed(new v.win.Float32Array(BLOCK));

      const msgs = v.posted.filter(m => m.type === 'voiceTranscribe');
      assert.strictEqual(
        msgs.length,
        1,
        'browser mode must post the captured speech for transcription ' +
          '(the reproduced bug: nothing is ever posted)',
      );
      const pcm = decodePcm(msgs[0].audio);
      // 2 loud + 8 trailing-silence blocks, 1:1 at a 16kHz mic rate.
      assert.strictEqual(pcm.length, 10 * BLOCK);
      assert.ok(
        Math.abs(pcm[0] - Math.round(0.1 * 0x7fff)) <= 1,
        `first sample ${pcm[0]} must encode the 0.1 amplitude`,
      );
      assert.ok(
        v.btn.classList.contains('voice-transcribing'),
        'mic button must turn yellow while the server transcribes',
      );
    },
  );

  await test('the wake recognizer does not hear capture audio', async () => {
    const v = await makeBrowserVoice();
    v.wake();
    const before = v.recognizer.acceptedBlocks;
    v.feed(loudBlock(v.win));
    for (let i = 0; i < 8; i++) v.feed(new v.win.Float32Array(BLOCK));
    assert.strictEqual(
      v.recognizer.acceptedBlocks,
      before,
      'capture blocks must be routed away from the wake recognizer',
    );
    // Once the capture ended, the recognizer hears audio again.
    v.feed(new v.win.Float32Array(BLOCK));
    assert.strictEqual(v.recognizer.acceptedBlocks, before + 1);
  });

  await test(
    'wake followed by only silence posts nothing and clears the flash',
    async () => {
      const v = await makeBrowserVoice();
      v.wake();
      // 5s (20 x 256ms) of silence: the no-speech timeout.
      for (let i = 0; i < 20; i++) v.feed(new v.win.Float32Array(BLOCK));
      assert.strictEqual(
        v.posted.filter(m => m.type === 'voiceTranscribe').length,
        0,
        'silence must not be sent for transcription',
      );
      assert.ok(
        !v.btn.classList.contains('voice-triggered') &&
          !v.btn.classList.contains('voice-transcribing'),
        'the wake flash must be cleared after the no-speech timeout',
      );
    },
  );

  await test(
    'the voiceSpeech reply inserts the text and submits it',
    async () => {
      const v = await makeBrowserVoice();
      v.wake();
      v.feed(loudBlock(v.win));
      for (let i = 0; i < 8; i++) v.feed(new v.win.Float32Array(BLOCK));
      assert.strictEqual(
        v.posted.filter(m => m.type === 'voiceTranscribe').length,
        1,
      );
      v.win.dispatchEvent(
        new v.win.MessageEvent('message', {
          data: {type: 'voiceSpeech', text: 'open the readme', speaker: 1},
        }),
      );
      assert.strictEqual(
        v.inp.value,
        'Speaker #1 says that: open the readme',
      );
      assert.strictEqual(v.submits.count, 1, 'the text must be submitted');
      assert.ok(
        !v.btn.classList.contains('voice-transcribing'),
        'the transcribing flash must be cleared by the reply',
      );
    },
  );

  await test(
    'a voiceSpeech reply with a language inserts the language prefix',
    async () => {
      const v = await makeBrowserVoice();
      v.wake();
      v.feed(loudBlock(v.win));
      for (let i = 0; i < 8; i++) v.feed(new v.win.Float32Array(BLOCK));
      v.win.dispatchEvent(
        new v.win.MessageEvent('message', {
          data: {
            type: 'voiceSpeech',
            text: 'open the readme',
            speaker: 1,
            language: 'fr',
          },
        }),
      );
      assert.strictEqual(
        v.inp.value,
        'Speaker #1 says in the language fr that: open the readme',
      );
      assert.strictEqual(v.submits.count, 1, 'the text must be submitted');
    },
  );

  await test(
    'an older voiceSpeech reply does not clear a newer capture flash',
    async () => {
      const v = await makeBrowserVoice();
      v.wake();
      v.feed(loudBlock(v.win));
      for (let i = 0; i < 8; i++) v.feed(new v.win.Float32Array(BLOCK));
      assert.strictEqual(
        v.posted.filter(m => m.type === 'voiceTranscribe').length,
        1,
      );
      assert.ok(v.btn.classList.contains('voice-transcribing'));

      // Start a second round while the first transcription is still
      // pending.  The first voiceSpeech reply is now stale relative to
      // the second round's red capture flash and must not clear it.
      v.advance(2500);
      v.wake();
      assert.ok(v.btn.classList.contains('voice-triggered'));
      v.win.dispatchEvent(
        new v.win.MessageEvent('message', {
          data: {type: 'voiceSpeech', text: 'first task'},
        }),
      );
      assert.ok(
        v.btn.classList.contains('voice-triggered'),
        'late reply from an older round must preserve the newer red flash',
      );

      v.feed(loudBlock(v.win));
      for (let i = 0; i < 8; i++) v.feed(new v.win.Float32Array(BLOCK));
      assert.strictEqual(
        v.posted.filter(m => m.type === 'voiceTranscribe').length,
        2,
      );
      assert.ok(v.btn.classList.contains('voice-transcribing'));
      v.win.dispatchEvent(
        new v.win.MessageEvent('message', {
          data: {type: 'voiceSpeech', text: 'second task'},
        }),
      );
      assert.ok(
        !v.btn.classList.contains('voice-triggered') &&
          !v.btn.classList.contains('voice-transcribing'),
        'the final reply clears the flash after all rounds finish',
      );
      assert.strictEqual(v.inp.value, 'first task second task');
    },
  );

  await test('48kHz mic audio is downsampled to 16kHz PCM', async () => {
    const v = await makeBrowserVoice({sampleRate: 48000});
    v.wake();
    v.feed(loudBlock(v.win, 0.2, 48000)); // 1s of speech at 48kHz
    v.feed(new v.win.Float32Array(96000)); // 2s silence ends the capture
    const msgs = v.posted.filter(m => m.type === 'voiceTranscribe');
    assert.strictEqual(msgs.length, 1);
    const pcm = decodePcm(msgs[0].audio);
    // 1s speech + 2s trailing silence, each block downsampled 3:1,
    // so the PCM must be 16k samples per second of captured audio.
    assert.strictEqual(pcm.length, (48000 + 96000) / 3);
    assert.ok(
      Math.abs(pcm[0] - Math.round(0.2 * 0x7fff)) <= 1,
      `first sample ${pcm[0]} must encode the 0.2 amplitude`,
    );
  });

  await test('actual voice: real TTS speech is captured and posted', async () => {
    const hasSay =
      spawnSync('which', ['say']).status === 0 &&
      spawnSync('which', ['afconvert']).status === 0;
    if (!hasSay) {
      console.log('      (skipped: requires macOS say/afconvert)');
      return;
    }
    const tmpdir = fs.mkdtempSync(
      path.join(os.tmpdir(), 'kiss-voice-browser-'),
    );
    try {
      const aiff = path.join(tmpdir, 'speech.aiff');
      const wav = path.join(tmpdir, 'speech.wav');
      spawnSync('say', ['open the readme file', '-o', aiff]);
      spawnSync('afconvert', [
        '-f', 'WAVE', '-d', 'LEI16@16000', '-c', '1', aiff, wav,
      ]);
      const buf = fs.readFileSync(wav);
      const dataAt = buf.indexOf(Buffer.from('data')) + 8;
      const speech = new Float32Array((buf.length - dataAt) >> 1);
      for (let i = 0; i < speech.length; i++) {
        speech[i] = buf.readInt16LE(dataAt + 2 * i) / 0x8000;
      }

      const v = await makeBrowserVoice();
      v.wake();
      for (let off = 0; off < speech.length; off += BLOCK) {
        const block = new v.win.Float32Array(BLOCK);
        block.set(speech.subarray(off, off + BLOCK));
        v.feed(block);
      }
      for (let i = 0; i < 10; i++) v.feed(new v.win.Float32Array(BLOCK));

      const msgs = v.posted.filter(m => m.type === 'voiceTranscribe');
      assert.strictEqual(msgs.length, 1, 'real speech must be posted');
      const pcm = decodePcm(msgs[0].audio);
      assert.ok(pcm.length > 16000, 'at least 1s of speech captured');
      let sumSquares = 0;
      for (let i = 0; i < pcm.length; i++) {
        sumSquares += (pcm[i] / 0x8000) ** 2;
      }
      const rms = Math.sqrt(sumSquares / pcm.length);
      assert.ok(rms >= 0.01, `captured PCM must carry speech (rms=${rms})`);
    } finally {
      fs.rmSync(tmpdir, {recursive: true, force: true});
    }
  });

  console.log(`\n${passed} passed, ${failures.length} failed`);
  if (failures.length) process.exit(1);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
