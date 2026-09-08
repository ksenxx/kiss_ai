// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests for the mic-less-host voice fallback (real child processes,
// real UDS server, compiled extension code — no mocks).
//
// Bug being reproduced: the KISS daemon/extension host runs on a machine
// without a microphone (e.g. a headless cloud VM reached via Remote-SSH
// or code-server). Clicking the mic button sent `voiceToggle` to the
// host, which spawned `kiss.server.voice_wake`; the child died instantly
// with `OSError: PortAudio library not found` and the webview painted the
// mic button with a red "Voice trigger error: voice listener exited
// (code 1): OSError: PortAudio library not found".
//
// Fixed behavior under test:
//  1. A listener that dies BEFORE ever printing READY makes the host send
//     `voiceState {listening:false, hostMicUnavailable:true}` — no error
//     text — and clears the host's voice intent so a later sensitivity
//     change never respawns the doomed listener. An explicit new
//     voiceToggle(on) may retry.
//  2. A listener that dies AFTER READY (a real runtime failure) still
//     reports the descriptive error exactly as before.
//  3. A spawn failure (uv present but not executable) is also a
//     hostMicUnavailable condition, not an error.
//  4. The in-page capture fallback's `voiceTranscribe` webview message is
//     forwarded verbatim to the daemon over the UDS, and the daemon's
//     `voiceSpeech` reply is relayed back to the webview — the round trip
//     that lets the BROWSER's microphone do the recording.
//
// The "KISS project or uv binary not found" branch also reports
// hostMicUnavailable, but it is not reproducible here without stubbing
// the machine's own /usr/local/bin and `which` lookup (findUvPath checks
// fixed system paths), so it is intentionally untested.

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');

const OUT_SIDEBAR = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
const PROJECT_ROOT = path.resolve(__dirname, '..', '..', '..', '..', '..');

if (process.platform === 'win32') {
  console.log('SKIP: POSIX shell scripts required');
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-micless-'));
const binDir = path.join(tmpHome, '.local', 'bin');
fs.mkdirSync(binDir, {recursive: true});
const uvPath = path.join(binDir, 'uv');
const pidFile = path.join(tmpHome, 'pids.txt');
const sockPath = path.join(tmpHome, 'daemon.sock');

process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
process.env.KISS_PROJECT_PATH = PROJECT_ROOT;
process.env.KISS_SORCAR_SOCK = sockPath;
delete process.env.KISS_VOICE_WAKE_ARGS;

// The exact stderr tail the real listener leaves on a machine without
// PortAudio (reproduced by running `python -m kiss.server.voice_wake` on
// a mic-less VM).
const OSERROR_LINE = 'OSError: PortAudio library not found';

/** Install a fake `uv` that dies before READY, like a mic-less host. */
function installFailingUv() {
  fs.writeFileSync(
    uvPath,
    '#!/bin/sh\n' +
      `echo "$$" >> "${pidFile}"\n` +
      'echo "Traceback (most recent call last):" >&2\n' +
      `echo "${OSERROR_LINE}" >&2\n` +
      'exit 1\n',
    {mode: 0o755},
  );
}

/** Install a fake `uv` that reaches READY and then dies (runtime error). */
function installDyingAfterReadyUv() {
  fs.writeFileSync(
    uvPath,
    '#!/bin/sh\n' +
      `echo "$$" >> "${pidFile}"\n` +
      'echo READY\n' +
      'sleep 0.2\n' +
      'echo "mic watchdog: input stream still silent; giving up" >&2\n' +
      'exit 1\n',
    {mode: 0o755},
  );
}

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

// A REAL daemon socket: records every JSON frame the extension sends and
// lets tests push frames back, exactly like kiss-web's UDS transport
// (newline-delimited JSON, no handshake).
const daemonFrames = [];
const daemonConns = [];
const daemonServer = net.createServer(conn => {
  daemonConns.push(conn);
  let buf = '';
  conn.on('data', chunk => {
    buf += chunk.toString('utf-8');
    let idx = buf.indexOf('\n');
    while (idx >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (line) {
        try {
          daemonFrames.push(JSON.parse(line));
        } catch {}
      }
      idx = buf.indexOf('\n');
    }
  });
  conn.on('error', () => {});
});

function daemonSend(msg) {
  const line = JSON.stringify(msg) + '\n';
  for (const conn of daemonConns) {
    try {
      conn.write(line);
    } catch {}
  }
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
    asWebviewUri: uri => ({
      toString: () => `vscode-webview://${uri.fsPath}`,
    }),
    postMessage: msg => {
      posted.push(msg);
      return Promise.resolve(true);
    },
    onDidReceiveMessage: cb => {
      messageListeners.push(cb);
      return {dispose: () => {}};
    },
  };
  const webviewView = {
    webview,
    visible: true,
    show: () => {},
    onDidDispose: () => ({dispose: () => {}}),
    onDidChangeVisibility: () => ({dispose: () => {}}),
  };
  view.resolveWebviewView(
    webviewView,
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

function voiceStates(posted) {
  return posted.filter(m => m.type === 'voiceState');
}

// -- Test 1: pre-READY death => calm hostMicUnavailable, never an error --
async function testPreReadyDeathIsHostMicUnavailable() {
  installFailingUv();
  const {view, fire, posted} = makeSidebar();
  const before = spawnedCount();

  fire({type: 'voiceToggle', enabled: true, sensitivity: 40});
  const state = await waitFor(
    () => voiceStates(posted).find(m => m.hostMicUnavailable),
    'host never reported hostMicUnavailable for a pre-READY death',
  );
  assert.strictEqual(state.listening, false);
  assert.strictEqual(
    state.error,
    undefined,
    `hostMicUnavailable must carry no error text: ${JSON.stringify(state)}`,
  );
  const errored = voiceStates(posted).filter(
    m => typeof m.error === 'string' && m.error.includes('OSError'),
  );
  assert.strictEqual(
    errored.length,
    0,
    `the raw OSError leaked to the webview: ${JSON.stringify(errored)}`,
  );

  // The host cleared its voice intent: a sensitivity change (which
  // restarts a wanted listener) must NOT respawn the doomed child.
  const afterFirst = spawnedCount();
  assert.ok(afterFirst >= before + 1, 'the first attempt never spawned');
  fire({type: 'voiceSensitivity', value: 75});
  await sleep(500);
  assert.strictEqual(
    spawnedCount(),
    afterFirst,
    'a sensitivity change respawned the listener on a mic-less host',
  );

  // An explicit new toggle may retry (the user plugged in a mic?), and
  // on this host fails the same calm way.
  posted.length = 0;
  fire({type: 'voiceToggle', enabled: true, sensitivity: 40});
  await waitFor(
    () => spawnedCount() === afterFirst + 1,
    'an explicit voiceToggle(on) retry never respawned the listener',
  );
  await waitFor(
    () => voiceStates(posted).find(m => m.hostMicUnavailable),
    'the retry did not report hostMicUnavailable again',
  );
  fire({type: 'voiceToggle', enabled: false});
  view.dispose();
  console.log('ok - pre-READY death reports hostMicUnavailable, no OSError');
}

// -- Test 2: post-READY death keeps the descriptive error --------------
async function testPostReadyDeathStillReportsError() {
  installDyingAfterReadyUv();
  const {view, fire, posted} = makeSidebar();

  fire({type: 'voiceToggle', enabled: true});
  await waitFor(
    () => voiceStates(posted).find(m => m.listening === true),
    'listener never reported READY',
  );
  const state = await waitFor(
    () =>
      voiceStates(posted).find(
        m => typeof m.error === 'string' && m.error.includes('exited'),
      ),
    'a post-READY death lost its descriptive error',
  );
  assert.ok(
    !state.hostMicUnavailable,
    'a runtime death after READY must not be classified hostMicUnavailable',
  );
  fire({type: 'voiceToggle', enabled: false});
  view.dispose();
  console.log('ok - post-READY death still reports the descriptive error');
}

// -- Test 2b: an 'error' event AFTER READY is not hostMicUnavailable ----
//
// Node emits 'error' on an already-spawned child only for kernel-level
// failures (e.g. a kill() that fails with EPERM), which cannot be
// produced deterministically in a test. The event is therefore injected
// on the REAL child process object after the REAL listener has reached
// READY: the production handler, service state, and callbacks all run
// unchanged — only the triggering syscall failure is synthesized.
async function testPostReadyErrorEventKeepsError() {
  fs.rmSync(uvPath, {force: true});
  fs.writeFileSync(
    uvPath,
    '#!/bin/sh\n' + `echo "$$" >> "${pidFile}"\n` + 'echo READY\nsleep 30\n',
    {mode: 0o755},
  );
  const {VoiceWakeService} = require(
    path.join(__dirname, '..', 'out', 'voiceWake.js'),
  );
  const states = [];
  const svc = new VoiceWakeService(
    () => {},
    (listening, error, hostMicUnavailable) =>
      states.push({listening, error, hostMicUnavailable}),
    () => {},
    () => {},
  );
  svc.start(40);
  await waitFor(
    () => states.find(s => s.listening === true),
    'listener never reported READY',
  );
  const child = svc._proc;
  assert.ok(child, 'the service must be holding its child');
  child.emit('error', new Error('kill EPERM'));
  const state = await waitFor(
    () => states.find(s => typeof s.error === 'string'),
    "the injected 'error' was never reported",
  );
  assert.ok(state.error.includes('voice listener error'), state.error);
  assert.ok(
    !state.hostMicUnavailable,
    "an 'error' after READY must not be classified hostMicUnavailable",
  );
  // The service dropped the child on 'error'; reap the real process
  // (spawned detached, so it leads its own group).
  try {
    process.kill(-child.pid, 'SIGKILL');
  } catch {
    try {
      child.kill('SIGKILL');
    } catch {}
  }
  console.log("ok - a post-READY 'error' keeps the descriptive error");
}

// -- Test 3: spawn failure is a hostMicUnavailable condition -----------
async function testSpawnFailureIsHostMicUnavailable() {
  // uv exists but is not executable: spawn fails with EACCES via the
  // child's 'error' event (the only end such a child has). The file must
  // be recreated: writeFileSync applies `mode` only to NEW files, and the
  // earlier fakes were executable.
  fs.rmSync(uvPath);
  fs.writeFileSync(uvPath, '#!/bin/sh\nexit 0\n', {mode: 0o644});
  const {view, fire, posted} = makeSidebar();
  fire({type: 'voiceToggle', enabled: true});
  const state = await waitFor(
    () => voiceStates(posted).find(m => m.hostMicUnavailable),
    'a spawn failure was not reported as hostMicUnavailable',
  );
  assert.strictEqual(state.error, undefined);
  fire({type: 'voiceToggle', enabled: false});
  view.dispose();
  console.log('ok - spawn failure reports hostMicUnavailable');
}

// -- Test 4: voiceTranscribe forwards to the daemon; voiceSpeech relays --
async function testVoiceTranscribeRoundTrip() {
  const {view, fire, posted} = makeSidebar();

  // The AgentClient is created lazily by the first forward and queues
  // the command until its UDS connect completes, so the frame's arrival
  // at the server is itself the connect signal.
  fire({
    type: 'voiceTranscribe',
    audio: 'QUJDRA==',
    wakePrefixed: true,
    wakeSamples: 320,
  });
  const frame = await waitFor(
    () => daemonFrames.find(f => f.type === 'voiceTranscribe'),
    'voiceTranscribe was never forwarded to the daemon',
  );
  assert.strictEqual(frame.audio, 'QUJDRA==');
  assert.strictEqual(frame.wakePrefixed, true);
  assert.strictEqual(frame.wakeSamples, 320);

  daemonSend({
    type: 'voiceSpeech',
    text: 'hello from the daemon',
    speaker: null,
    language: 'en-US',
  });
  const relayed = await waitFor(
    () => posted.find(m => m.type === 'voiceSpeech'),
    "the daemon's voiceSpeech reply was not relayed to the webview",
  );
  assert.strictEqual(relayed.text, 'hello from the daemon');
  assert.strictEqual(relayed.language, 'en-US');
  view.dispose();
  console.log('ok - voiceTranscribe forwards and voiceSpeech relays back');
}

async function main() {
  await new Promise(resolve => daemonServer.listen(sockPath, resolve));
  try {
    await testPreReadyDeathIsHostMicUnavailable();
    await testPostReadyDeathStillReportsError();
    await testPostReadyErrorEventKeepsError();
    await testSpawnFailureIsHostMicUnavailable();
    await testVoiceTranscribeRoundTrip();
  } finally {
    for (const conn of daemonConns) conn.destroy();
    await new Promise(resolve => daemonServer.close(resolve));
  }
  console.log('All voiceHostMicUnavailable tests passed');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  },
);
