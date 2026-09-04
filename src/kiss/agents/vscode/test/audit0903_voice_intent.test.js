// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-09-03 (vscode-main partition): voice listener intent races.
//
// The sidebar used `this._voiceWake?.running` as "the user wants voice on".
// But `running` deliberately stays true while a stopped listener is still
// DYING (it holds the exclusive microphone until its exit event fires), so
// "running" conflates the user's intent with the physical process state.
// Three interleavings turned the microphone back on against the user's
// wishes:
//
// 1. voiceToggle {enabled:false} → voiceSensitivity while the old process
//    is still dying: the handler saw running=true and queued a restart, so
//    the mic came back ON after the user had switched it off.
// 2. voiceToggle {enabled:false} → view hidden while the process is still
//    dying: the hide handler saw running=true, latched
//    _voiceWakeSuspendedByHide, and the next show restarted the listener
//    the user had switched off.
// 3. view hidden (listener suspended) → voiceSensitivity while the dying
//    process still counted as running: the handler queued a restart, so
//    the mic went live while the view was HIDDEN — exactly what the
//    hide-suspension exists to prevent.
//
// Fix: the sidebar tracks the user's last voiceToggle choice in
// _voiceEnabled and restart decisions follow that intent (and the
// hide-suspension flag), never the physical `running` state.
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

const OUT_SIDEBAR = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
const PROJECT_ROOT = path.resolve(__dirname, '..', '..', '..', '..', '..');

if (process.platform === 'win32') {
  console.log('SKIP: POSIX process groups required');
  process.exit(0);
}
if (!fs.existsSync(OUT_SIDEBAR)) {
  console.log(`SKIP: ${OUT_SIDEBAR} missing — run \`npm run compile\``);
  process.exit(0);
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

const {SorcarSidebarView} = require(OUT_SIDEBAR);

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-audit-vi-'));
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

// Like the voice-lifecycle suite's makeSidebar, plus a controllable
// visibility flag whose changes fire the real onDidChangeVisibility
// subscription — the hide/show suspension logic under test lives there.
function makeSidebar() {
  const view = new SorcarSidebarView({
    fsPath: path.resolve(__dirname, '..'),
    scheme: 'file',
  });
  const state = {messageListeners: [], visibilityListeners: [], dispose: []};
  const resolve = () => {
    state.messageListeners = [];
    state.visibilityListeners = [];
    state.dispose = [];
    const webview = {
      options: {},
      html: '',
      cspSource: 'vscode-webview://stub',
      asWebviewUri: uri => ({
        toString: () => `vscode-webview://${uri.fsPath}`,
      }),
      postMessage: () => Promise.resolve(true),
      onDidReceiveMessage: cb => {
        state.messageListeners.push(cb);
        return {dispose: () => {}};
      },
    };
    const webviewView = {
      webview,
      visible: true,
      show: () => {},
      onDidDispose: cb => {
        state.dispose.push(cb);
        return {dispose: () => {}};
      },
      onDidChangeVisibility: cb => {
        state.visibilityListeners.push(cb);
        return {dispose: () => {}};
      },
    };
    view.resolveWebviewView(
      webviewView,
      {state: undefined},
      {
        isCancellationRequested: false,
        onCancellationRequested: () => ({dispose: () => {}}),
      },
    );
    return webviewView;
  };
  let webviewView = resolve();
  const fire = m => {
    for (const cb of state.messageListeners.slice()) cb(m);
  };
  const setVisible = v => {
    webviewView.visible = v;
    for (const cb of state.visibilityListeners.slice()) cb();
  };
  // Simulates VS Code destroying the webview (view closed), then a fresh
  // webview resolving in its place.
  const recreateWebview = () => {
    for (const cb of state.dispose.slice()) cb();
    webviewView = resolve();
  };
  return {view, fire, setVisible, recreateWebview};
}

async function startListener(fire, before, sensitivity) {
  fire({type: 'voiceToggle', enabled: true, sensitivity});
  await waitFor(
    () => spawnedCount() === before + 1 && fs.existsSync(lockDir),
    'sidebar never started the listener',
  );
}

// Interleaving 1: the user switches voice OFF, and while the old process
// is still dying (it holds the mic for ~300ms after SIGTERM) a debounced
// slider value arrives.  The off switch must win: no restart, ever.
async function testSensitivityAfterToggleOffStaysOff() {
  const {view, fire} = makeSidebar();
  const before = spawnedCount();
  await startListener(fire, before, 30);

  fire({type: 'voiceToggle', enabled: false});
  // The old process is still dying here, so a `running` check is true.
  fire({type: 'voiceSensitivity', value: 70});

  await waitFor(() => !fs.existsSync(lockDir), 'listener never stopped');
  // Give a wrongly queued restart every chance to happen.
  await sleep(700);
  assert.strictEqual(
    spawnedCount(),
    before + 1,
    'voiceSensitivity after voiceToggle(off) restarted the listener',
  );
  assert.strictEqual(
    fs.existsSync(lockDir),
    false,
    'listener holds the microphone after the user switched voice off',
  );
  view.dispose();
  console.log('  ✓ voiceSensitivity after voiceToggle(off) stays off');
}

// Interleaving 2: the user switches voice OFF and hides the view while
// the old process is still dying.  The hide handler must not latch the
// suspended-by-hide flag for a listener the user already switched off —
// otherwise the next show restarts it.
async function testHideDuringToggleOffDoesNotResume() {
  const {view, fire, setVisible} = makeSidebar();
  const before = spawnedCount();
  await startListener(fire, before, 30);

  fire({type: 'voiceToggle', enabled: false});
  // Old process still dying: `running` is still true here.
  setVisible(false);
  await waitFor(() => !fs.existsSync(lockDir), 'listener never stopped');
  setVisible(true);
  await sleep(700);
  assert.strictEqual(
    spawnedCount(),
    before + 1,
    'show after hide restarted a listener the user had switched off',
  );
  assert.strictEqual(
    fs.existsSync(lockDir),
    false,
    'listener holds the microphone after the user switched voice off',
  );
  view.dispose();
  console.log('  ✓ hide/show during voiceToggle(off) stays off');
}

// Interleaving 3: the view is hidden (listener suspended) and a slider
// value arrives while the suspended process is still dying.  The mic
// must stay off until the view is shown again; the show must then start
// exactly one listener carrying the newest value.
async function testSensitivityWhileHiddenWaitsForShow() {
  const {view, fire, setVisible} = makeSidebar();
  const before = spawnedCount();
  await startListener(fire, before, 30);

  setVisible(false);
  // Suspended process still dying: `running` is still true here.
  fire({type: 'voiceSensitivity', value: 65});
  await waitFor(() => !fs.existsSync(lockDir), 'listener never stopped');
  await sleep(700);
  assert.strictEqual(
    spawnedCount(),
    before + 1,
    'voiceSensitivity restarted the listener while the view was hidden',
  );
  assert.strictEqual(
    fs.existsSync(lockDir),
    false,
    'microphone is live while the view is hidden',
  );

  setVisible(true);
  await waitFor(
    () => spawnedCount() === before + 2 && fs.existsSync(lockDir),
    'show did not resume the suspended listener',
  );
  assert.ok(
    lastSpawnArgs().includes('--sensitivity 65'),
    `resumed listener argv lacks --sensitivity 65: ${lastSpawnArgs()}`,
  );
  assert.strictEqual(
    fs.existsSync(overlapFile),
    false,
    'resumed listener overlapped the dying one',
  );

  fire({type: 'voiceToggle', enabled: false});
  await waitFor(() => !fs.existsSync(lockDir), 'listener never stopped');
  view.dispose();
  console.log('  ✓ voiceSensitivity while hidden waits for show');
}

// Regression guard: the intent flag must not break the plain flows —
// sensitivity restart while visible-and-on, and hide/show suspension of
// a healthy listener.
async function testIntentKeepsPlainFlowsWorking() {
  const {view, fire, setVisible, recreateWebview} = makeSidebar();
  const before = spawnedCount();
  await startListener(fire, before, 30);

  fire({type: 'voiceSensitivity', value: 45});
  await waitFor(
    () => spawnedCount() === before + 2 && fs.existsSync(lockDir),
    'voiceSensitivity did not restart the visible listener',
  );
  assert.ok(
    lastSpawnArgs().includes('--sensitivity 45'),
    `restarted listener argv lacks --sensitivity 45: ${lastSpawnArgs()}`,
  );

  setVisible(false);
  await waitFor(() => !fs.existsSync(lockDir), 'hide never stopped listener');
  setVisible(true);
  await waitFor(
    () => spawnedCount() === before + 3 && fs.existsSync(lockDir),
    'show did not resume the suspended listener',
  );
  assert.strictEqual(fs.existsSync(overlapFile), false, 'listeners overlapped');

  // A webview dispose stops the listener; a fresh webview that only
  // reports a slider value (no voiceToggle yet) must not start the mic.
  recreateWebview();
  await waitFor(() => !fs.existsSync(lockDir), 'webview dispose never stopped');
  fire({type: 'voiceSensitivity', value: 80});
  await sleep(700);
  assert.strictEqual(
    spawnedCount(),
    before + 3,
    "a fresh webview's voiceSensitivity started the mic before voiceToggle",
  );
  // ... and its voiceToggle brings voice back, with the stored value.
  fire({type: 'voiceToggle', enabled: true});
  await waitFor(
    () => spawnedCount() === before + 4 && fs.existsSync(lockDir),
    'voiceToggle after a webview swap never started the listener',
  );
  assert.ok(
    lastSpawnArgs().includes('--sensitivity 80'),
    `listener argv lacks the stored --sensitivity 80: ${lastSpawnArgs()}`,
  );

  view.dispose();
  await waitFor(() => !fs.existsSync(lockDir), 'dispose never stopped');
  await sleep(400);
  assert.strictEqual(spawnedCount(), before + 4, 'dispose spawned a listener');
  console.log(
    '  ✓ intent tracking keeps sensitivity restart and hide/show working',
  );
}

async function main() {
  try {
    await testSensitivityAfterToggleOffStaysOff();
    await testHideDuringToggleOffDoesNotResume();
    await testSensitivityWhileHiddenWaitsForShow();
    await testIntentKeepsPlainFlowsWorking();
  } finally {
    fs.rmSync(tmpHome, {recursive: true, force: true});
  }
  console.log('audit0903_voice_intent: all tests passed');
  process.exit(0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
