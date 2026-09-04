// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-09-02 (vscode-ext partition): voice wake listener lifecycle.
//
// 1. RACE: the sidebar's `voiceSensitivity` handler restarted the listener
//    with `await stop(); if (!running) start()`.  A `voiceToggle
//    {enabled:false}` that arrived while the stop was in flight shared the
//    same stop promise, so when it settled the handler saw "not running" and
//    started a listener the user had just switched OFF.
// 2. HANG: VoiceWakeService.stop() resolved only from the child's `exit`
//    event.  A child whose spawn FAILS (EACCES, ENOENT) emits `error` and
//    `close` but never `exit`, so a stop() issued before the failure
//    surfaced left `_stopping` pending for ever and every later start()
//    was queued behind it: voice could not be re-enabled without a window
//    reload.
// 3. REVIEW FOLLOW-UP (review-vscode.md #6): the pid-less child's `error`
//    handler ignored a requested stop: `start(); await stop()` on a
//    non-executable uv reported "voice listener error: ... EACCES" as if
//    the listener had died on its own, and `_stopRequestedFor` kept the
//    dead child for ever (only the never-fired `exit` handler cleared it).
//
// The fake listener models the exclusive microphone with a lock directory
// (released 300ms after SIGTERM) and records every spawn and its argv.

/* global require, process, console, __dirname, global, setTimeout */

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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-audit-vw-'));
const binDir = path.join(tmpHome, '.local', 'bin');
fs.mkdirSync(binDir, {recursive: true});
const uvPath = path.join(binDir, 'uv');
const lockDir = path.join(tmpHome, 'mic.lock');
const overlapFile = path.join(tmpHome, 'overlap.txt');
const pidFile = path.join(tmpHome, 'pids.txt');
const argsFile = path.join(tmpHome, 'args.txt');

fs.writeFileSync(
  uvPath,
  '#!/bin/sh\n' +
    `if ! mkdir "${lockDir}" 2>/dev/null; then\n` +
    `  echo overlap >> "${overlapFile}"\n` +
    '  exit 1\n' +
    'fi\n' +
    `echo "$$" >> "${pidFile}"\n` +
    `echo "$@" >> "${argsFile}"\n` +
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

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitFor(predicate, message, timeoutMs = 10000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt <= timeoutMs) {
    const value = predicate();
    if (value) return value;
    await sleep(25);
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

function lastSpawnArgs() {
  try {
    const lines = fs.readFileSync(argsFile, 'utf-8').trim().split('\n');
    return lines[lines.length - 1];
  } catch {
    return '';
  }
}

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise.then(() => 'settled'),
    sleep(ms).then(() => {
      throw new Error(`${label} did not settle within ${ms}ms`);
    }),
  ]);
}

function makeSidebar() {
  const view = new SorcarSidebarView({
    fsPath: path.resolve(__dirname, '..'),
    scheme: 'file',
  });
  const messageListeners = [];
  const posted = [];
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-webview://stub',
    asWebviewUri: uri => ({toString: () => `vscode-webview://${uri.fsPath}`}),
    postMessage: m => {
      posted.push(m);
      return Promise.resolve(true);
    },
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
  return {view, fire, posted};
}

async function testSensitivityThenToggleOffStaysOff() {
  const {view, fire} = makeSidebar();
  const before = spawnedCount();
  fire({type: 'voiceToggle', enabled: true, sensitivity: 30});
  await waitFor(
    () => spawnedCount() === before + 1 && fs.existsSync(lockDir),
    'sidebar never started the listener',
  );

  // The race: the slider moves, and while the old listener is still
  // dying the user clicks the mic OFF.  The off switch must win.
  fire({type: 'voiceSensitivity', value: 70});
  fire({type: 'voiceToggle', enabled: false});

  await waitFor(() => !fs.existsSync(lockDir), 'listener never stopped');
  // Give a wrongly queued/awaited restart every chance to happen.
  await sleep(700);
  assert.strictEqual(
    spawnedCount(),
    before + 1,
    'voiceSensitivity restarted a listener the user had switched off',
  );
  assert.strictEqual(
    fs.existsSync(lockDir),
    false,
    'listener holds the microphone after the user switched voice off',
  );
  view.dispose();
  console.log('  ✓ voiceToggle(off) during a sensitivity restart wins');
}

async function testSensitivityRestartUsesNewValue() {
  const {view, fire} = makeSidebar();
  const before = spawnedCount();
  fire({type: 'voiceToggle', enabled: true, sensitivity: 30});
  await waitFor(
    () => spawnedCount() === before + 1 && fs.existsSync(lockDir),
    'sidebar never started the listener',
  );
  assert.ok(
    lastSpawnArgs().includes('--sensitivity 30'),
    `initial listener argv lacks --sensitivity 30: ${lastSpawnArgs()}`,
  );

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
  assert.ok(
    lastSpawnArgs().includes('--sensitivity 70'),
    `restarted listener argv lacks --sensitivity 70: ${lastSpawnArgs()}`,
  );

  // A slider drag emits a burst of values: exactly one more listener
  // must come up, carrying the LAST value.
  fire({type: 'voiceSensitivity', value: 40});
  fire({type: 'voiceSensitivity', value: 50});
  fire({type: 'voiceSensitivity', value: 60});
  await waitFor(
    () => spawnedCount() === before + 3 && fs.existsSync(lockDir),
    'burst of voiceSensitivity did not restart the listener once',
  );
  await sleep(700);
  assert.strictEqual(
    spawnedCount(),
    before + 3,
    'a burst of voiceSensitivity values spawned more than one listener',
  );
  assert.ok(
    lastSpawnArgs().includes('--sensitivity 60'),
    `listener argv lacks the last value --sensitivity 60: ${lastSpawnArgs()}`,
  );
  assert.strictEqual(fs.existsSync(overlapFile), false, 'listeners overlapped');

  fire({type: 'voiceToggle', enabled: false});
  await waitFor(() => !fs.existsSync(lockDir), 'listener never stopped');
  view.dispose();
  console.log('  ✓ voiceSensitivity restarts once with the newest value');
}

async function testSensitivityWhileOffDoesNotStart() {
  const {view, fire} = makeSidebar();
  const before = spawnedCount();
  // Create the service without starting it, then move the slider.
  fire({type: 'voiceToggle', enabled: false});
  fire({type: 'voiceSensitivity', value: 55});
  fire({type: 'voiceSensitivity', value: 'not-a-number'});
  await sleep(400);
  assert.strictEqual(
    spawnedCount(),
    before,
    'voiceSensitivity started a listener while voice was off',
  );
  // The stored value is used by the next start.
  fire({type: 'voiceToggle', enabled: true});
  await waitFor(
    () => spawnedCount() === before + 1 && fs.existsSync(lockDir),
    'sidebar never started the listener',
  );
  assert.ok(
    lastSpawnArgs().includes('--sensitivity 55'),
    `listener argv lacks the stored --sensitivity 55: ${lastSpawnArgs()}`,
  );
  fire({type: 'voiceToggle', enabled: false});
  await waitFor(() => !fs.existsSync(lockDir), 'listener never stopped');
  view.dispose();
  console.log('  ✓ voiceSensitivity while off only stores the value');
}

async function testStopSettlesWhenSpawnFails() {
  const states = [];
  const service = new VoiceWakeService(
    () => {},
    (listening, error) => states.push({listening, error}),
    () => {},
    () => {},
  );
  // A listener binary that exists but cannot be executed: spawn() emits
  // `error` (EACCES) asynchronously and never `exit`.
  fs.chmodSync(uvPath, 0o644);
  try {
    const before = spawnedCount();
    service.start(20);
    assert.strictEqual(service.running, true, 'start() did not track child');
    // stop() issued before the spawn failure surfaces.
    const stopped = service.stop();
    await withTimeout(stopped, 3000, 'stop() after a failed spawn');
    assert.strictEqual(
      service.running,
      false,
      'service still counts as running after the failed child was stopped',
    );
    // A stop the user asked for is not an error, even when the child it
    // stopped never spawned: every reported state is a clean "off".
    assert.ok(states.length >= 1, 'stop() reported no state at all');
    assert.deepStrictEqual(
      states.filter(s => s.listening || s.error),
      [],
      `requested stop reported an error: ${JSON.stringify(states)}`,
    );
    assert.strictEqual(
      service._stopRequestedFor,
      undefined,
      'the stopped child is still referenced after the stop settled',
    );

    // The service must be usable again: a start() after the failed round
    // must actually spawn (not be queued behind a stop that never ends).
    fs.chmodSync(uvPath, 0o755);
    service.start(20);
    await waitFor(
      () => spawnedCount() === before + 1 && fs.existsSync(lockDir),
      'start() after a failed spawn never launched the listener',
      5000,
    );
    await service.stop();
    assert.strictEqual(
      fs.existsSync(lockDir),
      false,
      'final stop left the listener running',
    );
  } finally {
    fs.chmodSync(uvPath, 0o755);
  }
  console.log('  ✓ stop() settles when the child failed to spawn');
}

async function testErrorAfterStopReportsCleanState() {
  // Same failure, but observed through the state callback: the failed
  // child must not leave the UI believing a listener is up.
  const states = [];
  const service = new VoiceWakeService(
    () => {},
    (listening, error) => states.push({listening, error}),
    () => {},
    () => {},
  );
  fs.chmodSync(uvPath, 0o644);
  try {
    service.start();
    await waitFor(
      () => states.some(s => s.listening === false && s.error),
      'spawn failure never reported through onState',
      3000,
    );
    assert.strictEqual(service.running, false, 'failed child still tracked');
    assert.strictEqual(
      states.filter(s => s.error).length,
      1,
      `an unsolicited spawn failure is reported exactly once: ${JSON.stringify(states)}`,
    );
    assert.match(states.find(s => s.error).error, /voice listener error/);
    // stop() on an already-failed child settles immediately.
    await withTimeout(service.stop(), 1000, 'stop() after error');
    assert.strictEqual(service._stopRequestedFor, undefined);
  } finally {
    fs.chmodSync(uvPath, 0o755);
  }
  console.log('  ✓ spawn failure without stop() reports an error state');
}

async function main() {
  try {
    await testSensitivityThenToggleOffStaysOff();
    await testSensitivityRestartUsesNewValue();
    await testSensitivityWhileOffDoesNotStart();
    await testStopSettlesWhenSpawnFails();
    await testErrorAfterStopReportsCleanState();
  } finally {
    fs.rmSync(tmpHome, {recursive: true, force: true});
  }
  console.log('audit0902_vscode_ext_voice_lifecycle: all tests passed');
  process.exit(0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
