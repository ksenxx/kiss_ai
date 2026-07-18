// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end REAL-VOICE test for wake-word sensitivity in the
// host-side VoiceWakeService: the service must forward a sensitivity
// value to the REAL Python listener (--sensitivity N), and that value
// must actually change what spoken audio wakes it.
//
// Real audio, real speech models, real subprocess tree — only the
// 'vscode' module is stubbed (it does not exist under plain Node).
// The spoken phrase "hey there Sorcar" decodes to "[unk] sore car"
// (alias at the END of the utterance, measured live):
//
//   1. start(50): low sensitivity keeps strict whole-utterance
//      matching -> the audio must NOT wake (fails if start() never
//      passes --sensitivity, because the eager default would wake).
//   2. start() with no sensitivity: the default (80) accepts a
//      trailing alias -> the same audio must WAKE the listener.
//
// Run directly with ``node test/voiceWakeSensitivityService.test.js``
// after ``npm run compile``.

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

// The compiled service imports 'vscode' (through kissPaths); provide
// the shared stub used by the other extension-host tests.
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

// Synthesize "hey there Sorcar" (trailing alias) at 16kHz mono 16-bit.
const tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-voice-sens-'));
const aiff = path.join(tmpdir, 'hey.aiff');
const wav = path.join(tmpdir, 'hey.wav');
// The short pause before "Sorcar" keeps the TTS from slurring the
// phrase into a single [unk] (observed with some prosody): this
// decodes to "[unk] sore car" with conf 1.0 reliably.
spawnSync(
  'say',
  ['hey there [[slnc 300]] Sorcar [[slnc 1500]] ' +
   'hey there [[slnc 300]] Sorcar [[slnc 1500]]',
   '-o', aiff],
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

const DEADLINE_MS = 600000;

/**
 * Run one listener pass over the WAV through a fresh service.
 * Resolves with {wakes, states} once the listener process exits (the
 * WAV is finite, so every pass terminates).
 */
function runService(sensitivity) {
  return new Promise((resolve, reject) => {
    const wakes = [];
    const states = [];
    let sawReady = false;
    const timer = setTimeout(() => {
      service.stop();
      reject(new Error(`listener did not finish within ${DEADLINE_MS}ms`));
    }, DEADLINE_MS);
    const service = new VoiceWakeService(
      () => wakes.push(Date.now()),
      (listening, error) => {
        states.push({listening, error});
        if (listening) sawReady = true;
        // The exit of a finite WAV pass surfaces as a final
        // onState(false, ...) after READY was seen.
        if (sawReady && !listening) {
          clearTimeout(timer);
          resolve({wakes, states});
        }
      },
      () => {},
      () => {},
    );
    service.start(sensitivity);
  });
}

async function main() {
  let failed = 0;

  // 1. Low sensitivity: the service passes --sensitivity 50 and the
  //    trailing-alias audio stays rejected (the eager default of 80
  //    would wake, so this proves --sensitivity forwarding).
  const strict = await runService(50);
  try {
    assert.strictEqual(
      strict.wakes.length,
      0,
      'start(50) must pass --sensitivity 50 so a trailing alias is ' +
        `rejected; states=${JSON.stringify(strict.states)}`,
    );
    assert.ok(
      strict.states.some(s => s.listening === true),
      'READY must surface as onState(true)',
    );
    console.log('  \u2713 start(50) stays strict (real voice)');
  } catch (e) {
    failed++;
    console.log(`  \u2717 ${e.message}`);
  }

  // 2. Default sensitivity (80): the same audio must wake.
  const eager = await runService(undefined);
  try {
    assert.ok(
      eager.wakes.length >= 1,
      'the default sensitivity (80) must accept "hey there Sorcar"; ' +
        `states=${JSON.stringify(eager.states)}`,
    );
    console.log(
      '  \u2713 default start() wakes on a trailing alias (real voice)',
    );
  } catch (e) {
    failed++;
    console.log(`  \u2717 ${e.message}`);
  }

  fs.rmSync(tmpdir, {recursive: true, force: true});
  console.log(`\n${2 - failed} passed, ${failed} failed`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch(e => {
  fs.rmSync(tmpdir, {recursive: true, force: true});
  console.error(e);
  process.exit(1);
});
