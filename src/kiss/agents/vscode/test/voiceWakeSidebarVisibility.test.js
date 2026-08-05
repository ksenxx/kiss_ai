// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

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

function makeUri(fsPath) {
  return {fsPath, scheme: 'file', toString: () => `file://${fsPath}`};
}

global.__kissVscodeStub = {
  workspace: {
    isTrusted: true,
    workspaceFolders: [],
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    asRelativePath: p => p,
    textDocuments: [],
  },
  EventEmitter: StubEventEmitter,
  Uri: {
    file: p => makeUri(p),
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: s => makeUri(s),
  },
  ProgressLocation: {Notification: 15},
  window: {
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: () => {},
    showErrorMessage: () => {},
    showTextDocument: () => Promise.resolve(),
    activeTextEditor: undefined,
    tabGroups: {all: [], close: () => Promise.resolve()},
  },
  commands: {executeCommand: () => Promise.resolve()},
  ViewColumn: {One: 1, Beside: 2},
};
const realResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return realResolve.call(this, request, ...rest);
};

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
  }
  try {
    process.kill(pid, signal);
  } catch {
  }
}

function makeFakeHome() {
  const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-voice-vis-'));
  const binDir = path.join(tmpHome, '.local', 'bin');
  fs.mkdirSync(binDir, {recursive: true});
  const pidFile = path.join(tmpHome, 'pids.txt');
  const script =
    '#!/bin/sh\n' +
    'sleep 300 &\n' +
    'child=$!\n' +
    'echo "$$ $child" >> "$KISS_TEST_PID_FILE"\n' +
    'wait "$child"\n';
  fs.writeFileSync(path.join(binDir, 'uv'), script, {mode: 0o755});
  process.env.HOME = tmpHome;
  process.env.USERPROFILE = tmpHome;
  process.env.KISS_PROJECT_PATH = PROJECT_ROOT;
  process.env.KISS_TEST_PID_FILE = pidFile;
  delete process.env.KISS_VOICE_WAKE_ARGS;
  return {tmpHome, pidFile};
}

function readPidLines(pidFile) {
  if (!fs.existsSync(pidFile)) return [];
  return fs
    .readFileSync(pidFile, 'utf-8')
    .trim()
    .split('\n')
    .filter(Boolean)
    .map(line => line.trim().split(/\s+/).map(Number));
}

function cleanupHome(tmpHome, pidFile) {
  for (const pids of readPidLines(pidFile)) {
    for (const pid of pids) killGroup(pid);
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

function makeWebviewView() {
  const posted = [];
  const recvEmitter = new StubEventEmitter();
  const disposeEmitter = new StubEventEmitter();
  const visEmitter = new StubEventEmitter();
  const webviewView = {
    webview: {
      options: {},
      html: '',
      cspSource: 'vscode-resource:',
      asWebviewUri: uri => makeUri(uri.fsPath),
      postMessage: msg => {
        posted.push(msg);
        return Promise.resolve(true);
      },
      onDidReceiveMessage: cb => recvEmitter.event(cb),
    },
    visible: true,
    show: () => {},
    onDidChangeVisibility: cb => visEmitter.event(cb),
    onDidDispose: cb => disposeEmitter.event(cb),
  };
  return {
    webviewView,
    posted,
    fireMessage: m => recvEmitter.fire(m),
    setVisible(visible) {
      webviewView.visible = visible;
      visEmitter.fire();
    },
  };
}

function makeView() {
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();
  view.resolveWebviewView(wv.webviewView, {}, {});
  return {view, wv};
}

async function testHideStopsListener() {
  const {tmpHome, pidFile} = makeFakeHome();
  const {view, wv} = makeView();
  try {
    wv.fireMessage({type: 'voiceToggle', enabled: true});
    await waitFor(
      () => readPidLines(pidFile).length >= 1,
      'voice listener never started',
      10000,
    );
    const [wrapperPid, childPid] = readPidLines(pidFile)[0];
    assert.ok(alive(wrapperPid), 'wrapper must be alive after voiceToggle');
    assert.ok(alive(childPid), 'child must be alive after voiceToggle');

    wv.setVisible(false);

    await waitFor(
      () => !alive(wrapperPid) && !alive(childPid),
      'BUG: closing the secondary side bar (view hidden, not disposed) ' +
        `left the wake-word mic listener running — wrapper alive=${alive(
          wrapperPid,
        )}, child alive=${alive(childPid)}`,
    );
    console.log('  ✓ hiding the sidebar kills the wake-word listener tree');
  } finally {
    view.dispose();
    cleanupHome(tmpHome, pidFile);
  }
}

async function testShowRestartsListener() {
  const {tmpHome, pidFile} = makeFakeHome();
  const {view, wv} = makeView();
  try {
    wv.fireMessage({type: 'voiceToggle', enabled: true});
    await waitFor(
      () => readPidLines(pidFile).length >= 1,
      'voice listener never started',
      10000,
    );
    const [wrapperPid, childPid] = readPidLines(pidFile)[0];

    wv.setVisible(false);
    await waitFor(
      () => !alive(wrapperPid) && !alive(childPid),
      'listener must stop when the sidebar is hidden',
    );

    wv.setVisible(true);

    await waitFor(
      () => readPidLines(pidFile).length >= 2,
      'BUG: reopening the secondary side bar did not restart the ' +
        'wake-word listener that was on before the bar was closed',
      10000,
    );
    const [newWrapper, newChild] = readPidLines(pidFile)[1];
    await waitFor(
      () => alive(newWrapper) && alive(newChild),
      'restarted listener tree must be alive after the sidebar is shown',
    );
    console.log('  ✓ re-showing the sidebar restarts the wake-word listener');
  } finally {
    view.dispose();
    cleanupHome(tmpHome, pidFile);
  }
}

async function testStaleHiddenEventDoesNotStopFreshViewListener() {
  const {tmpHome, pidFile} = makeFakeHome();
  const {view, wv} = makeView();
  try {
    wv.fireMessage({type: 'voiceToggle', enabled: true});
    await waitFor(
      () => readPidLines(pidFile).length >= 1,
      'voice listener never started',
      10000,
    );
    const [wrapperPid, childPid] = readPidLines(pidFile)[0];

    const fresh = makeWebviewView();
    view.resolveWebviewView(fresh.webviewView, {}, {});
    wv.setVisible(false);

    await new Promise(resolve => setTimeout(resolve, 700));
    assert.ok(
      alive(wrapperPid) && alive(childPid),
      'a stale hidden event from a superseded webview must not stop ' +
        'the listener for the fresh visible sidebar view',
    );
    assert.strictEqual(
      readPidLines(pidFile).length,
      1,
      'stale hidden events must not trigger an unnecessary restart',
    );
    console.log('  ✓ stale visibility events do not stop the fresh view mic');
  } finally {
    view.dispose();
    cleanupHome(tmpHome, pidFile);
  }
}

async function testShowDoesNotStartWhenMicWasOff() {
  const {tmpHome, pidFile} = makeFakeHome();
  const {view, wv} = makeView();
  try {
    wv.setVisible(false);
    wv.setVisible(true);
    await new Promise(resolve => setTimeout(resolve, 700));
    assert.strictEqual(
      readPidLines(pidFile).length,
      0,
      'hide/show of a sidebar whose mic was never enabled must not ' +
        'start the wake-word listener',
    );
    console.log('  ✓ hide/show with mic off never starts the listener');
  } finally {
    view.dispose();
    cleanupHome(tmpHome, pidFile);
  }
}

async function testExplicitOffIsNotResumed() {
  const {tmpHome, pidFile} = makeFakeHome();
  const {view, wv} = makeView();
  try {
    wv.fireMessage({type: 'voiceToggle', enabled: true});
    await waitFor(
      () => readPidLines(pidFile).length >= 1,
      'voice listener never started',
      10000,
    );
    const [wrapperPid, childPid] = readPidLines(pidFile)[0];

    wv.fireMessage({type: 'voiceToggle', enabled: false});
    await waitFor(
      () => !alive(wrapperPid) && !alive(childPid),
      'listener must stop on explicit voiceToggle off',
    );
    wv.setVisible(false);
    wv.setVisible(true);

    await new Promise(resolve => setTimeout(resolve, 700));
    assert.strictEqual(
      readPidLines(pidFile).length,
      1,
      'a mic the user explicitly turned OFF must not auto-start when ' +
        'the secondary side bar is closed and reopened',
    );
    console.log('  ✓ explicitly-disabled mic stays off across hide/show');
  } finally {
    view.dispose();
    cleanupHome(tmpHome, pidFile);
  }
}

async function run() {
  const tests = [
    testHideStopsListener,
    testShowRestartsListener,
    testStaleHiddenEventDoesNotStopFreshViewListener,
    testShowDoesNotStartWhenMicWasOff,
    testExplicitOffIsNotResumed,
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
