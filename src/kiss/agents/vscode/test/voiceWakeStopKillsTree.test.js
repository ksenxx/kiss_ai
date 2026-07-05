// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the host-side VoiceWakeService process
// lifecycle.  They run the REAL compiled extension code and REAL
// subprocess trees; only the VS Code API module is stubbed because it
// does not exist under plain Node.
//
// Regressions locked in:
//   1. stop() kills the WHOLE `uv run` -> Python tree, not just the
//      wrapper (the old wrapper-only kill orphaned Python mic holders).
//   2. stdout produced by a stopped/superseded listener is ignored, so
//      a late READY/WAKE cannot turn the mic UI back on after stop().
//   3. SorcarSidebarView.dispose() tears down its VoiceWakeService, so
//      extension deactivation/reload cannot leave a detached listener
//      holding the microphone.
//
// Run directly with `node test/voiceWakeStopKillsTree.test.js` after
// `npm run compile`.

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

// The compiled service imports 'vscode' (through kissPaths, and the
// sidebar imports it directly); provide the shared stub used by the
// other extension-host tests.
global.__kissVscodeStub = {
  workspace: {
    isTrusted: true,
    workspaceFolders: [],
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    asRelativePath: p => p,
  },
  EventEmitter: StubEventEmitter,
  Uri: {file: p => ({fsPath: p, scheme: 'file'})},
  ProgressLocation: {Notification: 15},
  window: {
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showTextDocument: () => Promise.resolve(),
    tabGroups: {all: [], close: () => Promise.resolve()},
  },
  commands: {executeCommand: () => Promise.resolve()},
  ViewColumn: {Beside: 2},
};
const realResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return realResolve.call(this, request, ...rest);
};

const {VoiceWakeService} = require(OUT_VOICEWAKE);
const {SorcarSidebarView} = require(OUT_SIDEBAR);

function alive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function killGroup(pid, signal = 'SIGKILL') {
  if (!pid) return;
  try {
    process.kill(-pid, signal);
    return;
  } catch {
    // The process may not be a group leader (or may already be dead).
  }
  try {
    process.kill(pid, signal);
  } catch {
    // Already dead.
  }
}

function makeFakeHome(scriptBody) {
  const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-voice-home-'));
  const binDir = path.join(tmpHome, '.local', 'bin');
  fs.mkdirSync(binDir, {recursive: true});
  const pidFile = path.join(tmpHome, 'pids.txt');
  fs.writeFileSync(path.join(binDir, 'uv'), scriptBody, {mode: 0o755});
  return {tmpHome, pidFile};
}

function useFakeHome(tmpHome, pidFile) {
  process.env.HOME = tmpHome;
  process.env.USERPROFILE = tmpHome;
  process.env.KISS_PROJECT_PATH = PROJECT_ROOT;
  process.env.KISS_TEST_PID_FILE = pidFile;
  delete process.env.KISS_VOICE_WAKE_ARGS;
}

function cleanupHome(tmpHome, pidFile) {
  try {
    const pids = fs
      .readFileSync(pidFile, 'utf-8')
      .trim()
      .split(/\s+/)
      .map(Number)
      .filter(Boolean);
    for (const pid of pids) killGroup(pid);
  } catch {
    // No pid file — nothing was spawned.
  }
  fs.rmSync(tmpHome, {recursive: true, force: true});
}

async function waitFor(predicate, message, timeoutMs = 5000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt <= timeoutMs) {
    const value = predicate();
    if (value) return value;
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  throw new Error(message);
}

function readPids(pidFile) {
  return fs
    .readFileSync(pidFile, 'utf-8')
    .trim()
    .split(/\s+/)
    .map(Number);
}

function makeTreeScript() {
  return (
    '#!/bin/sh\n' +
    'sleep 300 &\n' +
    'child=$!\n' +
    'echo "$$ $child" > "$KISS_TEST_PID_FILE"\n' +
    'wait "$child"\n'
  );
}

async function testStopKillsTree() {
  const {tmpHome, pidFile} = makeFakeHome(makeTreeScript());
  useFakeHome(tmpHome, pidFile);
  const states = [];
  const service = new VoiceWakeService(
    () => {},
    (listening, error) => states.push({listening, error}),
    () => {},
    () => {},
  );
  try {
    service.start();
    await waitFor(
      () => fs.existsSync(pidFile),
      `listener tree never started, states=${JSON.stringify(states)}`,
      10000,
    );
    const [wrapperPid, childPid] = readPids(pidFile);
    assert.ok(wrapperPid > 0 && childPid > 0, 'pid file must hold two pids');
    assert.ok(alive(wrapperPid), 'wrapper must be alive after start()');
    assert.ok(alive(childPid), 'child must be alive after start()');

    service.stop();

    await waitFor(
      () => !alive(wrapperPid) && !alive(childPid),
      `after stop(): wrapper dead=${!alive(wrapperPid)}, ` +
        `child dead=${!alive(childPid)} — process tree leaked`,
    );
    console.log('  ✓ stop() kills the wrapper AND its child');
  } finally {
    service.stop();
    cleanupHome(tmpHome, pidFile);
  }
}

async function testLateStdoutIgnoredAfterStop() {
  const script =
    '#!/bin/sh\n' +
    'echo "$$" > "$KISS_TEST_PID_FILE"\n' +
    "trap 'echo READY; sleep 300' TERM\n" +
    'while :; do sleep 1; done\n';
  const {tmpHome, pidFile} = makeFakeHome(script);
  useFakeHome(tmpHome, pidFile);
  const states = [];
  const service = new VoiceWakeService(
    () => {},
    (listening, error) => states.push({listening, error}),
    () => {},
    () => {},
  );
  try {
    service.start();
    await waitFor(() => fs.existsSync(pidFile), 'listener never started');
    service.stop();
    await new Promise(resolve => setTimeout(resolve, 500));
    assert.deepStrictEqual(
      states.filter(s => s.listening),
      [],
      `late READY from stopped listener changed state: ${JSON.stringify(
        states,
      )}`,
    );
    console.log('  ✓ late stdout from a stopped listener is ignored');
  } finally {
    service.stop();
    cleanupHome(tmpHome, pidFile);
  }
}

async function testSidebarDisposeKillsVoiceTree() {
  const {tmpHome, pidFile} = makeFakeHome(makeTreeScript());
  useFakeHome(tmpHome, pidFile);
  const view = new SorcarSidebarView({
    fsPath: path.join(__dirname, '..'),
    scheme: 'file',
  });
  try {
    view._handleMessage({type: 'voiceToggle', enabled: true});
    await waitFor(() => fs.existsSync(pidFile), 'voice listener never started');
    const [wrapperPid, childPid] = readPids(pidFile);
    assert.ok(alive(wrapperPid), 'wrapper must be alive after voiceToggle');
    assert.ok(alive(childPid), 'child must be alive after voiceToggle');

    view.dispose();

    await waitFor(
      () => !alive(wrapperPid) && !alive(childPid),
      `after SorcarSidebarView.dispose(): wrapper dead=${!alive(
        wrapperPid,
      )}, child dead=${!alive(childPid)} — process tree leaked`,
    );
    console.log('  ✓ SorcarSidebarView.dispose() kills the voice listener tree');
  } finally {
    view.dispose();
    cleanupHome(tmpHome, pidFile);
  }
}

async function run() {
  const tests = [
    testStopKillsTree,
    testLateStdoutIgnoredAfterStop,
    testSidebarDisposeKillsVoiceTree,
  ];
  let passed = 0;
  for (const test of tests) {
    try {
      await test();
      passed += 1;
    } catch (err) {
      console.log(`  ✗ ${err && err.message ? err.message : err}`);
      console.log(`\n${passed} passed, ${tests.length - passed} failed`);
      process.exit(1);
    }
  }
  console.log(`\n${passed} passed, 0 failed`);
}

run().catch(err => {
  console.log(`  ✗ ${err && err.message ? err.message : err}`);
  console.log('\n0 passed, 1 failed');
  process.exit(1);
});
