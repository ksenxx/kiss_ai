// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const VOICE_JS_PATH = path.join(__dirname, '..', 'media', 'voice.js');
const BLOCK = 4096;

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

const PAGE_HTML =
  '<!DOCTYPE html><html><body>' +
  '<button id="voice-btn" class="toggle-btn"></button>' +
  '<textarea id="task-input"></textarea>' +
  '<div id="settings-panel"><label class="config-label">' +
  'Wake word sensitivity' +
  '<input type="range" id="cfg-voice-sensitivity" min="0" max="100"' +
  ' step="5" value="80">' +
  '<span id="cfg-voice-sensitivity-value">80</span>' +
  '</label></div>' +
  '</body></html>';

async function makeVoice({mode = 'browser', storedSensitivity} = {}) {
  const dom = new JSDOM(PAGE_HTML, {
    runScripts: 'dangerously',
    url: 'https://localhost/',
  });
  const win = dom.window;
  if (storedSensitivity !== undefined) {
    win.localStorage.setItem('kissVoiceSensitivity', storedSensitivity);
  }
  win.localStorage.setItem('kissVoiceEnabled', '1');
  win.__VOICE__ =
    mode === 'browser'
      ? {mode: 'browser', voskSrc: 'vosk.js', modelUrl: '/m.tar.gz'}
      : {mode: 'webview'};
  let now = 1000000;
  win.Date.now = () => now;

  const recognizers = [];
  win.Vosk = {
    createModel: () =>
      Promise.resolve({
        KaldiRecognizer: class {
          constructor(rate, grammar) {
            this.grammar = grammar;
            this.handlers = {};
            recognizers.push(this);
          }
          setWords() {}
          on(event, cb) {
            this.handlers[event] = cb;
          }
          acceptWaveform() {}
          remove() {}
        },
      }),
  };

  const processors = [];
  win.AudioContext = class {
    constructor() {
      this.sampleRate = 16000;
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
      const node = {onaudioprocess: null, connect() {}, disconnect() {}};
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

  if (mode === 'browser') {
    for (let i = 0; i < 10 && processors.length === 0; i++) await flush();
    assert.strictEqual(processors.length, 1, 'audio pipeline did not start');
    assert.strictEqual(recognizers.length, 1, 'wake recognizer not created');
  }

  const slider = win.document.getElementById('cfg-voice-sensitivity');
  const label = win.document.getElementById('cfg-voice-sensitivity-value');
  return {
    win,
    posted,
    slider,
    label,
    recognizer: recognizers[0],
    btn: win.document.getElementById('voice-btn'),
    advance(ms) {
      now += ms;
    },
    feed(samples) {
      processors[0].onaudioprocess({
        inputBuffer: {getChannelData: () => samples, sampleRate: 16000},
      });
    },
    quiet(ms) {
      this.feed(new win.Float32Array(Math.ceil((ms / 1000) * 16000)));
    },
    setSlider(value) {
      this.slider.value = String(value);
      this.slider.dispatchEvent(
        new win.Event('input', {bubbles: true}),
      );
    },
  };
}

function words(...pairs) {
  return pairs.map(([word, conf]) => ({word, conf}));
}

async function main() {
  await test('slider defaults to 80 and reflects a stored value', async () => {
    const fresh = await makeVoice({mode: 'browser'});
    assert.strictEqual(fresh.slider.value, '80');
    assert.strictEqual(fresh.label.textContent, '80');
    const stored = await makeVoice({mode: 'browser', storedSensitivity: '30'});
    assert.strictEqual(stored.slider.value, '30');
    assert.strictEqual(stored.label.textContent, '30');
    const garbage = await makeVoice({
      mode: 'browser',
      storedSensitivity: 'garbage',
    });
    assert.strictEqual(garbage.slider.value, '80');
  });

  await test('moving the slider persists and updates the label', async () => {
    const v = await makeVoice({mode: 'browser'});
    v.setSlider(60);
    assert.strictEqual(
      v.win.localStorage.getItem('kissVoiceSensitivity'),
      '60',
    );
    assert.strictEqual(v.label.textContent, '60');
  });

  await test(
    'low sensitivity rejects a low-confidence force-fit that the ' +
      'default accepts',
    async () => {
      const v = await makeVoice({mode: 'browser'});
      v.recognizer.handlers.result({
        result: {
          text: 'sar car',
          result: words(['sar', 0.55], ['car', 1.0]),
        },
      });
      assert.ok(
        v.btn.classList.contains('voice-triggered'),
        'default sensitivity must accept conf 0.55',
      );

      const low = await makeVoice({mode: 'browser', storedSensitivity: '10'});
      low.recognizer.handlers.result({
        result: {
          text: 'sar car',
          result: words(['sar', 0.55], ['car', 1.0]),
        },
      });
      assert.ok(
        !low.btn.classList.contains('voice-triggered'),
        'sensitivity 10 must reject conf 0.55',
      );
    },
  );

  await test(
    'trailing alias: the default accepts what low sensitivity rejects',
    async () => {
      const finalResult = {
        result: {
          text: '[unk] sore car',
          result: words(['[unk]', 1.0], ['sore', 0.5], ['car', 0.5]),
        },
      };
      const v = await makeVoice({mode: 'browser', storedSensitivity: '50'});
      v.recognizer.handlers.result(finalResult);
      assert.ok(
        !v.btn.classList.contains('voice-triggered'),
        'sensitivity 50 must reject an alias in [unk] context',
      );
      v.setSlider(85);
      v.recognizer.handlers.result(finalResult);
      assert.ok(
        v.btn.classList.contains('voice-triggered'),
        'sensitivity 85 must accept a trailing alias',
      );
      const fresh = await makeVoice({mode: 'browser'});
      fresh.recognizer.handlers.result(finalResult);
      assert.ok(
        fresh.btn.classList.contains('voice-triggered'),
        'the default sensitivity (80) must accept a trailing alias',
      );
    },
  );

  await test(
    'trailing-alias partial results honor the pause gate at high ' +
      'sensitivity',
    async () => {
      const v = await makeVoice({mode: 'browser', storedSensitivity: '85'});
      v.recognizer.handlers.partialresult({
        result: {partial: '[unk] sore car'},
      });
      assert.ok(
        !v.btn.classList.contains('voice-triggered'),
        'no pause yet: a trailing-alias partial must not fire',
      );
      v.quiet(300);
      v.recognizer.handlers.partialresult({
        result: {partial: '[unk] sore car'},
      });
      assert.ok(
        v.btn.classList.contains('voice-triggered'),
        'after a pause the trailing-alias partial must fire at 85',
      );
    },
  );

  await test(
    'utterances ending in [unk] never wake even at maximum sensitivity',
    async () => {
      const v = await makeVoice({mode: 'browser', storedSensitivity: '100'});
      v.recognizer.handlers.result({
        result: {
          text: '[unk] sir car [unk]',
          result: words(['[unk]', 1.0], ['sir', 1.0], ['car', 1.0],
                        ['[unk]', 1.0]),
        },
      });
      assert.ok(
        !v.btn.classList.contains('voice-triggered'),
        'mid-utterance aliases must never wake',
      );
    },
  );

  await test('webview mode posts the sensitivity to the host', async () => {
    const v = await makeVoice({mode: 'webview', storedSensitivity: '30'});
    const toggles = v.posted.filter(m => m.type === 'voiceToggle');
    assert.strictEqual(toggles.length, 1, 'mic must auto-enable on load');
    assert.strictEqual(toggles[0].enabled, true);
    assert.strictEqual(
      toggles[0].sensitivity,
      30,
      'voiceToggle must carry the stored sensitivity',
    );
    v.setSlider(85);
    const changes = v.posted.filter(m => m.type === 'voiceSensitivity');
    assert.strictEqual(
      changes.length,
      1,
      'a slider move must post exactly one voiceSensitivity message',
    );
    assert.strictEqual(changes[0].value, 85);
  });

  console.log(`\n${passed} passed, ${failures.length} failed`);
  if (failures.length > 0) process.exit(1);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
