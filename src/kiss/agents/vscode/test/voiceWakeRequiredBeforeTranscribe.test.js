// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test: in the remote webapp (browser mode) the
// "Sorcar" wake word must ALWAYS be heard before any speech is
// captured and sent for transcription.
//
// Bug being reproduced: browser-mode ``media/voice.js`` never reset
// its Vosk recognizer after a wake.  The Python listener does exactly
// that (voice_wake.py: "Reset so leftover partial text cannot
// re-trigger after cooldown") but the browser port missed it.  The
// decoded "sorcar" utterance therefore stayed pending inside the
// recognizer while the post-wake capture owned the audio; as soon as
// the capture ended and the recognizer heard audio again, Vosk hit an
// endpoint and emitted a stale final result whose text was the wake
// word itself.  That stale result re-fired the wake (the 2s cooldown
// had long expired during the capture) and the user's NEXT utterance
// was captured and transcribed even though nobody said "Sorcar"
// again.
//
// The fix mirrors the Python listener: on a wake, browser mode
// flushes the recognizer with ``retrieveFinalResult()`` (the only
// reset vosk-browser exposes; the flushed utterance arrives as one
// 'result' event that is consumed and ignored) and drops recognizer
// events while a capture owns the audio.
//
// Two tests:
//
//  1. Deterministic repro with a vosk-faithful recognizer stub (an
//     un-finalized utterance stays pending while the capture starves
//     the recognizer and is finalized at the next silence endpoint,
//     exactly like real Vosk).  FAILS before the fix: the stale
//     "sorcar" final re-triggers and a second utterance is posted for
//     transcription without a wake word.
//
//  2. REAL voice, REAL speaker, REAL microphone (macOS only, skipped
//     when the audio environment is unavailable): `say` synthesizes
//     "Sorcar ... open the read me file please ... this speech has no
//     wake word at all", `afplay` plays it through the machine's
//     speakers, the microphone records it (python sounddevice), and
//     the recording is streamed through the REAL production
//     media/voice.js pipeline backed by the REAL Vosk
//     small-English model (python vosk bridge configured with the
//     exact grammar/word-confidence semantics of the vosk-browser
//     worker).  Exactly one voiceTranscribe may be posted: the
//     sentence spoken without the wake word must never be transcribed.
//
// Run directly with ``node test/voiceWakeRequiredBeforeTranscribe.test.js``.

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const {spawn, spawnSync} = require('child_process');
const {JSDOM} = require('jsdom');

const VOICE_JS_PATH = path.join(__dirname, '..', 'media', 'voice.js');
const REPO_ROOT = path.join(__dirname, '..', '..', '..', '..', '..');
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
 * mode on top of a stub audio pipeline.  The recognizer is supplied
 * by the caller through `makeRecognizer` so the deterministic test
 * can use a vosk-faithful state machine and the real-audio test can
 * proxy to an actual Vosk process.  The code under test is
 * unmodified production code.
 */
async function makeBrowserVoice({sampleRate = 16000, makeRecognizer} = {}) {
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
          constructor(rate, grammar) {
            const rec = makeRecognizer(rate, grammar);
            recognizers.push(rec);
            return rec;
          }
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

  win.eval(fs.readFileSync(VOICE_JS_PATH, 'utf8'));

  // Enable listening (browser mode never auto-starts).
  win.document.getElementById('voice-btn').click();
  for (let i = 0; i < 10 && processors.length === 0; i++) await flush();
  assert.strictEqual(processors.length, 1, 'audio pipeline did not start');
  assert.strictEqual(recognizers.length, 1, 'wake recognizer not created');

  return {
    win,
    posted,
    recognizer: recognizers[0],
    btn: win.document.getElementById('voice-btn'),
    inp: win.document.getElementById('task-input'),
    advance(ms) {
      now += ms;
    },
    feed(samples) {
      // Wall time advances with the audio so the wake cooldown clock
      // behaves like production, where blocks arrive in real time.
      now += (samples.length / sampleRate) * 1000;
      processors[0].onaudioprocess({
        inputBuffer: {
          getChannelData: () => samples,
          sampleRate,
        },
      });
    },
  };
}

function loudBlock(win, amplitude = 0.1, frames = BLOCK) {
  const samples = new win.Float32Array(frames);
  samples.fill(amplitude);
  return samples;
}

/**
 * A recognizer stub faithful to the Vosk semantics this bug depends
 * on: a decoded utterance stays pending until a silence endpoint
 * finalizes it as a 'result' event, and retrieveFinalResult() (the
 * flush the real vosk-browser worker exposes) finalizes it
 * immediately.  While a capture owns the audio the recognizer gets no
 * blocks, so the pending utterance survives the whole capture — the
 * exact state real Vosk is in after voice.js forgot to reset it.
 */
class FaithfulRecognizer {
  constructor() {
    this.handlers = {};
    this.pending = null; // un-finalized decoded utterance text
    this.silentBlocks = 0;
    this.flushes = 0;
  }
  setWords() {}
  on(event, cb) {
    this.handlers[event] = cb;
  }
  /** The test "speaks": the model decodes text into a partial. */
  hear(text) {
    this.pending = text;
    this.silentBlocks = 0;
    this.handlers.partialresult({result: {partial: text}});
  }
  acceptWaveform(inputBuffer) {
    const samples = inputBuffer.getChannelData(0);
    let sumSquares = 0;
    for (let i = 0; i < samples.length; i++) {
      sumSquares += samples[i] * samples[i];
    }
    const loud = Math.sqrt(sumSquares / samples.length) >= 0.01;
    if (this.pending === null) return;
    this.silentBlocks = loud ? 0 : this.silentBlocks + 1;
    if (this.silentBlocks >= 3) {
      // Endpoint on trailing silence: Vosk finalizes the pending
      // utterance as a final result.
      const text = this.pending;
      this.pending = null;
      this.handlers.result({
        result: {
          text,
          result: text
            .split(' ')
            .map(word => ({word, conf: 1.0})),
        },
      });
    }
  }
  retrieveFinalResult() {
    // The vosk-browser worker's FinalResult(): flush the pending
    // utterance and deliver it as a normal 'result' event.
    this.flushes++;
    const text = this.pending || '';
    this.pending = null;
    this.handlers.result({result: {text}});
  }
  remove() {}
}

/** Feed n silent blocks through the page's audio pipeline. */
function silence(v, n) {
  for (let i = 0; i < n; i++) v.feed(new v.win.Float32Array(BLOCK));
}

// ---------------------------------------------------------------------------
// Real-audio helpers (macOS: say -> speaker -> microphone -> real Vosk).
// ---------------------------------------------------------------------------

const RECORDER_PY = `
import sys, wave
import sounddevice as sd

RATE = 48000
frames = []

def cb(indata, n, t, status):
    frames.append(bytes(indata))

stream = sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16",
                           callback=cb)
stream.start()
sys.stdout.write("ready\\n")
sys.stdout.flush()
sys.stdin.read()  # record until stdin closes
stream.stop()
stream.close()
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(RATE)
    w.writeframes(b"".join(frames))
`;

// A JSON-lines bridge exposing a REAL Vosk KaldiRecognizer with the
// exact request handling of the vosk-browser worker: an audio chunk
// yields either a final 'result' (AcceptWaveform hit an endpoint) or
// a 'partialresult'; a "final" request flushes via FinalResult() and
// yields it as a 'result' event.
const VOSK_BRIDGE_PY = `
import base64, json, sys
from vosk import KaldiRecognizer, Model, SetLogLevel

SetLogLevel(-1)
model = Model(sys.argv[1])
rate = float(sys.argv[2])
grammar = sys.argv[3]
rec = KaldiRecognizer(model, rate, grammar)
rec.SetWords(True)
for line in sys.stdin:
    req = json.loads(line)
    if req.get("final"):
        out = {"event": "result", "result": json.loads(rec.FinalResult())}
    else:
        pcm = base64.b64decode(req["pcm"])
        if rec.AcceptWaveform(pcm):
            out = {"event": "result", "result": json.loads(rec.Result())}
        else:
            out = {"event": "partialresult",
                   "result": json.loads(rec.PartialResult())}
    sys.stdout.write(json.dumps(out) + "\\n")
    sys.stdout.flush()
`;

/** Spawn a line-oriented python helper via the repo's uv environment. */
function spawnPython(scriptPath, args) {
  return spawn('uv', ['run', 'python', scriptPath].concat(args), {
    cwd: REPO_ROOT,
    stdio: ['pipe', 'pipe', 'inherit'],
  });
}

/** Read one line from a child's stdout. */
function readLine(child, pending) {
  return new Promise((resolve, reject) => {
    let onData = null;
    const onExit = () => {
      if (onData) child.stdout.removeListener('data', onData);
      reject(new Error('python helper exited'));
    };
    const tryResolve = () => {
      const idx = pending.buf.indexOf('\n');
      if (idx === -1) return false;
      const line = pending.buf.slice(0, idx);
      pending.buf = pending.buf.slice(idx + 1);
      child.removeListener('exit', onExit);
      if (onData) child.stdout.removeListener('data', onData);
      resolve(line);
      return true;
    };
    child.once('exit', onExit);
    if (tryResolve()) return;
    onData = chunk => {
      pending.buf += chunk.toString();
      tryResolve();
    };
    child.stdout.on('data', onData);
  });
}

/**
 * Recognizer proxy that forwards the page's audio to the real Vosk
 * bridge process and replays the bridge's events through the
 * vosk-browser handler interface.  `idle()` awaits every in-flight
 * request so the test can feed audio deterministically.
 */
class RealVoskRecognizer {
  constructor(bridge) {
    this.bridge = bridge;
    this.pendingOut = {buf: ''};
    this.handlers = {};
    this.chain = Promise.resolve();
  }
  setWords() {}
  on(event, cb) {
    this.handlers[event] = cb;
  }
  _request(req) {
    this.chain = this.chain.then(async () => {
      this.bridge.stdin.write(JSON.stringify(req) + '\n');
      const line = await readLine(this.bridge, this.pendingOut);
      const msg = JSON.parse(line);
      const handler = this.handlers[msg.event];
      if (handler) handler({result: msg.result});
    });
  }
  acceptWaveform(inputBuffer) {
    const samples = inputBuffer.getChannelData(0);
    const pcm = Buffer.alloc(samples.length * 2);
    for (let i = 0; i < samples.length; i++) {
      let v = samples[i];
      if (v > 1) v = 1;
      else if (v < -1) v = -1;
      pcm.writeInt16LE(Math.round(v < 0 ? v * 0x8000 : v * 0x7fff), 2 * i);
    }
    this._request({pcm: pcm.toString('base64')});
  }
  retrieveFinalResult() {
    this._request({final: true});
  }
  remove() {}
  idle() {
    return this.chain;
  }
}

function which(cmd) {
  return spawnSync('which', [cmd]).status === 0;
}

function findVoskModel() {
  const candidates = [
    path.join(os.homedir(), '.kiss', 'models', 'vosk-model-small-en-us-0.15'),
    path.join(os.homedir(), '.kiss', 'vosk-model-small-en-us-0.15'),
  ];
  for (const p of candidates) {
    if (fs.existsSync(path.join(p, 'am'))) return p;
  }
  return null;
}

function readWavSamples(wavPath) {
  const buf = fs.readFileSync(wavPath);
  const dataAt = buf.indexOf(Buffer.from('data')) + 8;
  const samples = new Float32Array((buf.length - dataAt) >> 1);
  for (let i = 0; i < samples.length; i++) {
    samples[i] = buf.readInt16LE(dataAt + 2 * i) / 0x8000;
  }
  return samples;
}

async function main() {
  await test(
    'a stale wake-word final after a capture must not start a new ' +
      'transcription (no speech without "Sorcar" may be transcribed)',
    async () => {
      const v = await makeBrowserVoice({
        makeRecognizer: () => new FaithfulRecognizer(),
      });
      const rec = v.recognizer;

      // Round 1: a genuine wake.  300ms of quiet satisfies the
      // post-alias pause gate, then the spoken "sorcar" arrives as a
      // partial and fires the wake.
      silence(v, 2);
      rec.hear('sorcar');
      assert.ok(
        v.btn.classList.contains('voice-triggered'),
        'the real wake must flash the mic button red',
      );
      // The dictated task: speech, then 2s of trailing silence ends
      // the capture and posts it for transcription.
      v.feed(loudBlock(v.win));
      v.feed(loudBlock(v.win));
      silence(v, 8);
      assert.strictEqual(
        v.posted.filter(m => m.type === 'voiceTranscribe').length,
        1,
        'the utterance after the wake word must be transcribed',
      );

      // The recognizer now hears audio again.  Nobody says "Sorcar":
      // only silence and then a sentence.  Real Vosk still holds the
      // pre-capture "sorcar" utterance (voice.js never reset it) and
      // finalizes it at the first silence endpoint as a stale final
      // result — which must NOT re-trigger the wake.
      silence(v, 3);
      assert.ok(
        !v.btn.classList.contains('voice-triggered'),
        'a stale wake-word final must not re-fire the wake ' +
          '(the reproduced bug: it does)',
      );
      v.feed(loudBlock(v.win));
      v.feed(loudBlock(v.win));
      silence(v, 8);
      assert.strictEqual(
        v.posted.filter(m => m.type === 'voiceTranscribe').length,
        1,
        'speech spoken WITHOUT the wake word must never be ' +
          'transcribed (the reproduced bug: it is)',
      );

      // The fix flushes the recognizer at wake time, exactly like the
      // Python listener's Reset().
      assert.ok(
        rec.flushes >= 1,
        'the wake must flush/reset the recognizer so the pending ' +
          'wake-word utterance cannot re-trigger later',
      );

      // A genuine second wake still works after the flush.
      silence(v, 2);
      rec.hear('sore car');
      assert.ok(
        v.btn.classList.contains('voice-triggered'),
        'a genuine second "Sorcar" must still fire the wake',
      );
      v.feed(loudBlock(v.win));
      silence(v, 8);
      assert.strictEqual(
        v.posted.filter(m => m.type === 'voiceTranscribe').length,
        2,
        'the utterance after the second wake word must be transcribed',
      );
    },
  );

  await test(
    'REAL voice/speaker/mic: only wake-word-prefixed speech is ' +
      'transcribed end to end',
    async () => {
      if (process.platform !== 'darwin') {
        console.log('      (skipped: requires macOS audio tooling)');
        return;
      }
      for (const tool of ['say', 'afplay', 'osascript', 'uv']) {
        if (!which(tool)) {
          console.log(`      (skipped: ${tool} not available)`);
          return;
        }
      }
      const modelPath = findVoskModel();
      if (!modelPath) {
        console.log('      (skipped: vosk small-English model not found)');
        return;
      }

      const tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wake-req-'));
      const savedVolume = spawnSync('osascript', [
        '-e',
        'output volume of (get volume settings)',
      ])
        .stdout.toString()
        .trim();
      let recorder = null;
      let bridge = null;
      try {
        // One spoken session: the wake word, a pause, the dictated
        // task, a long pause (the capture ends and the recognizer
        // resumes listening), then a sentence WITHOUT any wake word.
        const aiff = path.join(tmpdir, 'session.aiff');
        spawnSync('say', [
          'Sorcar [[slnc 1500]] open the read me file please ' +
            '[[slnc 5000]] this speech has no wake word at all ' +
            '[[slnc 3500]]',
          '-o',
          aiff,
        ]);

        // Record the REAL microphone while the REAL speakers play it.
        const recorderPy = path.join(tmpdir, 'recorder.py');
        fs.writeFileSync(recorderPy, RECORDER_PY);
        const wav = path.join(tmpdir, 'mic.wav');
        recorder = spawnPython(recorderPy, [wav]);
        const recorderOut = {buf: ''};
        const ready = await readLine(recorder, recorderOut);
        assert.strictEqual(ready, 'ready', 'mic recorder must start');
        spawnSync('osascript', ['-e', 'set volume output volume 55']);
        const play = spawnSync('afplay', [aiff]);
        assert.strictEqual(play.status, 0, 'speaker playback must work');
        await new Promise(r => setTimeout(r, 500));
        recorder.stdin.end();
        await new Promise(r => recorder.once('exit', r));
        recorder = null;
        const recorded = readWavSamples(wav);
        assert.ok(
          recorded.length > 48000 * 5,
          'the microphone must have recorded the played session',
        );

        // Stream the microphone recording through the REAL production
        // voice.js pipeline backed by the REAL Vosk model with the
        // grammar voice.js itself configures.
        const bridgePy = path.join(tmpdir, 'bridge.py');
        fs.writeFileSync(bridgePy, VOSK_BRIDGE_PY);
        let grammarArg = null;
        const v = await makeBrowserVoice({
          sampleRate: 48000,
          makeRecognizer: (rate, grammar) => {
            grammarArg = grammar;
            bridge = spawnPython(bridgePy, [
              modelPath,
              String(rate),
              grammar,
            ]);
            return new RealVoskRecognizer(bridge);
          },
        });
        assert.ok(
          grammarArg && grammarArg.indexOf('sorcar') !== -1,
          'voice.js must configure its wake grammar',
        );
        let wakes = 0;
        const observer = new v.win.MutationObserver(() => {
          if (v.btn.classList.contains('voice-triggered')) wakes++;
        });
        observer.observe(v.btn, {
          attributes: true,
          attributeFilter: ['class'],
        });

        for (let off = 0; off < recorded.length; off += BLOCK) {
          const block = new v.win.Float32Array(BLOCK);
          block.set(recorded.subarray(off, off + BLOCK));
          v.feed(block);
          await v.recognizer.idle();
        }
        // Trailing silence so any open capture finishes.
        silence(v, 40);
        await v.recognizer.idle();
        await flush();

        const msgs = v.posted.filter(m => m.type === 'voiceTranscribe');
        if (wakes === 0 && msgs.length === 0) {
          console.log(
            '      (skipped: the microphone did not pick up the wake ' +
              'word — quiet/muted audio environment)',
          );
          return;
        }
        assert.ok(
          msgs.length >= 1,
          'the spoken task after the real wake word must be transcribed',
        );
        assert.strictEqual(
          msgs.length,
          1,
          'the sentence spoken WITHOUT the wake word must never be ' +
            'transcribed (the reproduced bug: a stale wake-word ' +
            'result re-arms the capture)',
        );
      } finally {
        if (recorder) recorder.kill();
        if (bridge) bridge.kill();
        if (savedVolume) {
          spawnSync('osascript', [
            '-e',
            `set volume output volume ${savedVolume}`,
          ]);
        }
        fs.rmSync(tmpdir, {recursive: true, force: true});
      }
    },
  );

  console.log(`\n${passed} passed, ${failures.length} failed`);
  if (failures.length) process.exit(1);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
