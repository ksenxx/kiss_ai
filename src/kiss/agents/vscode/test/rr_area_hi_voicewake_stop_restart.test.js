// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// H-RC5: restarting the voice listener (the voiceSensitivity handler
// does stop() then start()) must wait for the old process to EXIT
// before spawning the new one. On exclusive-capture audio backends a
// listener spawned while its predecessor still holds the microphone
// fails to open it. The fake listener here models the exclusive
// resource with a lock directory: a second instance that starts while
// the first is still dying records an overlap and exits.

/* global require, process, console, __dirname, global, setTimeout, clearTimeout */

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const OUT_VOICEWAKE = path.join(__dirname, '..', 'out', 'voiceWake.js');
const OUT_SIDEBAR = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
const PROJECT_ROOT = path.resolve(__dirname, '..', '..', '..', '..', '..');

if (process.platform === 'win32') {
  console.log('SKIP: POSIX process groups required');
  process.exit(0);
}
for (const compiled of [OUT_VOICEWAKE, OUT_SIDEBAR]) {
  if (!fs.existsSync(compiled)) {
    console.log(`SKIP: ${compiled} missing — run \`npm run compile\``);
    process.exit(0);
  }
}

class StubEventEmitter {
  constructor() {
    this._listeners = [];
    this.event = cb => {
      this._listeners.push(cb);
      return {
        dispose: () => {
          const idx = this._listeners.indexOf(cb);
          if (idx >= 0) this._listeners.splice(idx, 1);
        },
      };
    };
  }
  fire(arg) {
    for (const cb of this._listeners.slice()) cb(arg);
  }
  dispose() {
    this._listeners = [];
  }
}

global.__kissVscodeStub = {
  workspace: {
    isTrusted: true,
    workspaceFolders: [],
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    asRelativePath: p => p,
  },
  EventEmitter: StubEventEmitter,
  Uri: {
    file: p => ({fsPath: p, scheme: 'file'}),
    joinPath: (uri, ...parts) => ({
      fsPath: path.join(uri.fsPath, ...parts),
      scheme: uri.scheme || 'file',
    }),
  },
  ProgressLocation: {Notification: 15},
  window: {
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
  },
  commands: {executeCommand: () => Promise.resolve()},
};
const realResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return realResolve.call(this, request, ...rest);
};

const {VoiceWakeService} = require(OUT_VOICEWAKE);
const {SorcarSidebarView} = require(OUT_SIDEBAR);

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-rrhi-vw-'));
const binDir = path.join(tmpHome, '.local', 'bin');
fs.mkdirSync(binDir, {recursive: true});
const lockDir = path.join(tmpHome, 'mic.lock');
const overlapFile = path.join(tmpHome, 'overlap.txt');
const pidFile = path.join(tmpHome, 'pids.txt');

// The fake listener: grabs the exclusive "microphone", releases it only
// 300ms AFTER receiving SIGTERM (a dying real listener does not release
// the capture device instantly), and records any instance that found
// the microphone already taken.
fs.writeFileSync(
  path.join(binDir, 'uv'),
  '#!/bin/sh\n' +
    `if ! mkdir "${lockDir}" 2>/dev/null; then\n` +
    `  echo overlap >> "${overlapFile}"\n` +
    '  exit 1\n' +
    'fi\n' +
    `echo "$$" >> "${pidFile}"\n` +
    `on_term() { sleep 0.3; rmdir "${lockDir}" 2>/dev/null; exit 0; }\n` +
    'trap on_term TERM\n' +
    'echo READY\n' +
    'while :; do sleep 0.1; done\n',
  {mode: 0o755},
);

process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
process.env.KISS_PROJECT_PATH = PROJECT_ROOT;
process.env.KISS_SORCAR_SOCK = path.join(tmpHome, 'no-daemon.sock');
delete process.env.KISS_VOICE_WAKE_ARGS;

async function waitFor(predicate, message, timeoutMs = 10000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt <= timeoutMs) {
    const value = predicate();
    if (value) return value;
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  throw new Error(message);
}

function spawnedCount() {
  try {
    return fs.readFileSync(pidFile, 'utf-8').trim().split(/\s+/).length;
  } catch {
    return 0;
  }
}

async function testStopResolvesOnExit() {
  const service = new VoiceWakeService(
    () => {},
    () => {},
    () => {},
    () => {},
  );
  service.start();
  await waitFor(() => fs.existsSync(lockDir), 'listener never started');

  const stopped = service.stop();
  assert.ok(
    stopped && typeof stopped.then === 'function',
    'stop() must return a promise',
  );
  await stopped;
  assert.strictEqual(
    fs.existsSync(lockDir),
    false,
    'stop() resolved while the old listener still held the microphone',
  );
  console.log('  ✓ stop() resolves only after the listener has exited');
}

async function testServiceRestartDoesNotOverlap() {
  const service = new VoiceWakeService(
    () => {},
    () => {},
    () => {},
    () => {},
  );
  service.start();
  await waitFor(() => fs.existsSync(lockDir), 'listener never started');

  await service.stop();
  service.start();

  await waitFor(() => fs.existsSync(lockDir), 'restart never started');
  assert.strictEqual(
    fs.existsSync(overlapFile),
    false,
    'restarted listener overlapped the dying one',
  );
  await service.stop();
  console.log('  ✓ await stop() then start() never overlaps listeners');
}

async function testFireAndForgetStopThenStartDoesNotOverlap() {
  // The hide -> show race: SorcarSidebarView's visibility handler does a
  // fire-and-forget stop() on hide and start() on an immediately
  // following show. The start must be queued behind the in-flight stop,
  // never spawned while the old child still holds the microphone.
  const service = new VoiceWakeService(
    () => {},
    () => {},
    () => {},
    () => {},
  );
  const before = spawnedCount();
  service.start();
  await waitFor(() => fs.existsSync(lockDir), 'listener never started');

  void service.stop();
  service.start();

  await waitFor(
    () => spawnedCount() === before + 2 && fs.existsSync(lockDir),
    'queued start never spawned the replacement listener',
  );
  assert.strictEqual(
    fs.existsSync(overlapFile),
    false,
    'start() during an in-flight stop overlapped the dying listener',
  );
  await service.stop();
  assert.strictEqual(
    fs.existsSync(lockDir),
    false,
    'final stop left the listener running',
  );
  console.log(
    '  ✓ fire-and-forget stop() then start() queues instead of overlapping',
  );
}

async function testStopCancelsQueuedStart() {
  // hide -> show -> hide: the second stop() must cancel the start queued
  // by the show, so the listener ends (and stays) stopped.
  const service = new VoiceWakeService(
    () => {},
    () => {},
    () => {},
    () => {},
  );
  const before = spawnedCount();
  service.start();
  await waitFor(() => fs.existsSync(lockDir), 'listener never started');

  void service.stop();
  service.start();
  await service.stop();

  await new Promise(resolve => setTimeout(resolve, 400));
  assert.strictEqual(
    spawnedCount(),
    before + 1,
    'a stop() issued after a queued start still spawned a listener',
  );
  assert.strictEqual(
    fs.existsSync(lockDir),
    false,
    'listener still holds the microphone after the final stop',
  );
  console.log('  ✓ stop() cancels a start queued behind an in-flight stop');
}

async function testSidebarSensitivityRestart() {
  const view = new SorcarSidebarView({
    fsPath: path.resolve(__dirname, '..'),
    scheme: 'file',
  });
  const messageListeners = [];
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-webview://stub',
    asWebviewUri: uri => ({toString: () => `vscode-webview://${uri.fsPath}`}),
    postMessage: () => Promise.resolve(true),
    onDidReceiveMessage: cb => {
      messageListeners.push(cb);
      return {dispose: () => {}};
    },
  };
  view.resolveWebviewView(
    {
      webview,
      visible: true,
      show: () => {},
      onDidDispose: () => ({dispose: () => {}}),
      onDidChangeVisibility: () => ({dispose: () => {}}),
    },
    {state: undefined},
    {
      isCancellationRequested: false,
      onCancellationRequested: () => ({dispose: () => {}}),
    },
  );
  const fire = m => {
    for (const cb of messageListeners.slice()) cb(m);
  };

  const before = spawnedCount();
  fire({type: 'voiceToggle', enabled: true, sensitivity: 30});
  await waitFor(
    () => spawnedCount() === before + 1 && fs.existsSync(lockDir),
    'sidebar never started the listener',
  );

  // The regression under test: this handler used to fire-and-forget
  // stop() and spawn the replacement immediately.
  fire({type: 'voiceSensitivity', value: 70});
  await waitFor(
    () => spawnedCount() === before + 2 && fs.existsSync(lockDir),
    'sidebar never restarted the listener after voiceSensitivity',
  );
  assert.strictEqual(
    fs.existsSync(overlapFile),
    false,
    'voiceSensitivity restart overlapped the dying listener',
  );
  fire({type: 'voiceToggle', enabled: false});
  await waitFor(() => !fs.existsSync(lockDir), 'listener never stopped');
  console.log('  ✓ voiceSensitivity restart waits for the old listener');
}

async function main() {
  try {
    await testStopResolvesOnExit();
    await testServiceRestartDoesNotOverlap();
    await testFireAndForgetStopThenStartDoesNotOverlap();
    await testStopCancelsQueuedStart();
    await testSidebarSensitivityRestart();
  } finally {
    fs.rmSync(tmpHome, {recursive: true, force: true});
  }
  console.log('rr_area_hi_voicewake_stop_restart: all tests passed');
  process.exit(0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
