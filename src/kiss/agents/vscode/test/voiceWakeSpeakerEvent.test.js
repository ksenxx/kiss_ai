// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {execSync, spawnSync} = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const OUT_VOICEWAKE = path.join(__dirname, '..', 'out', 'voiceWake.js');
const PROJECT_ROOT = path.resolve(__dirname, '..', '..', '..', '..', '..');

function which(cmd) {
  try {
    execSync(`which ${cmd}`, {stdio: 'ignore'});
    return true;
  } catch {
    return false;
  }
}

if (process.platform !== 'darwin' || !which('say') || !which('afconvert')) {
  console.log('SKIP: requires macOS `say` and `afconvert`');
  process.exit(0);
}
if (!which('uv')) {
  console.log('SKIP: requires `uv`');
  process.exit(0);
}
if (!fs.existsSync(OUT_VOICEWAKE)) {
  console.log('SKIP: out/voiceWake.js missing — run `npm run compile`');
  process.exit(0);
}
if (!process.env.OPENAI_API_KEY) {
  console.log('SKIP: requires OPENAI_API_KEY (real gpt-audio translation)');
  process.exit(0);
}

global.__kissVscodeStub = {
  workspace: {
    isTrusted: true,
    getConfiguration: () => ({get: () => undefined}),
  },
};
const realResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return realResolve.call(this, request, ...rest);
};

const tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-voice-spk-'));
const aiff = path.join(tmpdir, 'task.aiff');
const wav = path.join(tmpdir, 'task.wav');
spawnSync(
  'say',
  ['Sorcar [[slnc 1200]] fix the parser bug [[slnc 4000]]', '-o', aiff],
  {stdio: 'inherit'},
);
spawnSync(
  'afconvert',
  ['-f', 'WAVE', '-d', 'LEI16@16000', '-c', '1', aiff, wav],
  {stdio: 'inherit'},
);
assert.ok(fs.existsSync(wav), 'TTS wav was not created');

process.env.KISS_PROJECT_PATH = PROJECT_ROOT;
process.env.KISS_VOICE_WAKE_ARGS = JSON.stringify(['--wav', wav]);

const {VoiceWakeService} = require(OUT_VOICEWAKE);

const wakes = [];
const speeches = [];
const states = [];
const service = new VoiceWakeService(
  roundId => wakes.push(roundId),
  (listening, error) => states.push({listening, error}),
  (roundId, text, speaker) => speeches.push({roundId, text, speaker}),
  () => {},
);
service.start();

const DEADLINE_MS = 600000;
const startedAt = Date.now();

function finish() {
  fs.rmSync(tmpdir, {recursive: true, force: true});
  try {
    assert.ok(
      wakes.length >= 1,
      `expected a WAKE event, states=${JSON.stringify(states)}`,
    );
    assert.ok(
      speeches.length >= 1,
      `expected an onSpeech callback, states=${JSON.stringify(states)}`,
    );
    const {roundId, text, speaker} = speeches[0];
    assert.ok(
      /parser/i.test(text),
      `expected the translated task text, got ${JSON.stringify(text)}`,
    );
    // The words must come back labelled with the wake that produced them.
    assert.strictEqual(
      roundId,
      wakes[0],
      `the transcript must answer the wake that opened its round, got ${JSON.stringify(roundId)}`,
    );
    assert.strictEqual(
      speaker,
      1,
      `first voice must be speaker 1, got ${JSON.stringify(speaker)}`,
    );
    console.log('  \u2713 VoiceWakeService reports the text and speaker');
    console.log('\n1 passed, 0 failed');
    process.exit(0);
  } catch (e) {
    console.log(`  \u2717 ${e.message}`);
    console.log('\n0 passed, 1 failed');
    process.exit(1);
  }
}

const timer = setInterval(() => {
  const exited = !service.running && states.length > 0;
  if ((wakes.length > 0 && speeches.length > 0) || exited) {
    clearInterval(timer);
    finish();
  } else if (Date.now() - startedAt > DEADLINE_MS) {
    clearInterval(timer);
    console.log('  \u2717 timed out waiting for wake/speech events');
    process.exit(1);
  }
}, 200);
