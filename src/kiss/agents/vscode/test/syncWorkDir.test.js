// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for ``SorcarSidebarView.syncWorkDir()`` — the
// activation-time helper that pushes the current VS Code workspace
// folder to the kiss-web daemon as its ``work_dir``.
//
// Regression locked in:
//
//   The daemon caches ``self.work_dir`` once at process start from
//   ``KISS_WORKDIR`` / ``os.getcwd()``.  When VS Code launches and the
//   dependency installer hits the *fast path* (all deps installed,
//   daemon already running, no extension-update marker), the daemon
//   is NOT restarted, so it retains the previous session's folder.
//   Until the user first interacts with the sidebar — which lazily
//   triggers ``_getClient()`` — every backend call that doesn't carry
//   an explicit ``workDir`` (autocomplete file list, commit-message
//   generation, worktree actions, etc.) keeps using the stale folder.
//
//   The fix: ``extension.ts`` activate() calls
//   ``sidebarView.syncWorkDir()`` immediately after creating the
//   sidebar view, which forces ``_getClient()`` to open the UDS
//   connection and send ``setWorkDir`` with the current workspace
//   folder.  ``AgentClient`` queues the command if the socket isn't
//   ready yet and flushes it on connect.
//
// This test exercises the real compiled ``SorcarSidebarView.js`` and
// ``AgentClient.js`` (no mocks of project code) — only the ``vscode``
// module is stubbed (it has no Node-side npm package) and the daemon
// UDS endpoint is replaced with a real in-process socket server that
// captures the JSON line.
//
// Run directly with ``node`` — no VS Code extension host required:
//
//     node src/kiss/agents/vscode/test/syncWorkDir.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');

// ---------------------------------------------------------------------------
// Stub the ``vscode`` module before any project source loads it.
// ---------------------------------------------------------------------------

class StubEventEmitter {
  constructor() {
    this._listeners = [];
    this.event = (cb) => {
      this._listeners.push(cb);
      return {dispose: () => {
        const i = this._listeners.indexOf(cb);
        if (i >= 0) this._listeners.splice(i, 1);
      }};
    };
  }
  fire(arg) {
    for (const cb of this._listeners.slice()) cb(arg);
  }
  dispose() {
    this._listeners = [];
  }
}

let workspaceFolders = [];
const folderChangeListeners = [];

const vscodeStub = {
  workspace: {
    get workspaceFolders() {
      return workspaceFolders;
    },
    getConfiguration: () => ({
      // Returning a non-empty string short-circuits the
      // ``|| getDefaultModel()`` fallback so the constructor never
      // shells out to ``uv`` (which would be very slow and depend on
      // the dev box's tooling).
      get: () => 'stub-default-model',
    }),
    onDidChangeWorkspaceFolders: (cb) => {
      folderChangeListeners.push(cb);
      return {dispose: () => {
        const i = folderChangeListeners.indexOf(cb);
        if (i >= 0) folderChangeListeners.splice(i, 1);
      }};
    },
  },
  EventEmitter: StubEventEmitter,
  Uri: {file: (p) => ({fsPath: p, scheme: 'file'})},
  // Unused by syncWorkDir() but referenced by other class fields.
  ProgressLocation: {Notification: 15},
  window: {
    withProgress: (_opts, task) => task({report: () => {}}, {onCancellationRequested: () => ({dispose: () => {}})}),
  },
  commands: {executeCommand: () => Promise.resolve()},
};

// Splice the stub into Node's resolver so any ``require('vscode')`` in
// the compiled project code returns ``vscodeStub`` instead of failing
// with MODULE_NOT_FOUND.
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};

// Write the stub to disk in the same dir as this test so the resolver
// can locate it via the override above.  Keeps the test self-contained.
const stubPath = path.join(__dirname, '_vscode-stub.js');
fs.writeFileSync(
  stubPath,
  // Re-export the in-memory object so the resolver-redirected require
  // returns the same instance the test inspects.
  `'use strict';\nmodule.exports = global.__kissVscodeStub;\n`,
);
global.__kissVscodeStub = vscodeStub;

// ---------------------------------------------------------------------------
// Run a real UDS server at the path AgentClient.defaultSocketPath()
// expects (``~/.kiss/sorcar.sock``).  Redirect HOME to a tempdir so
// the test never touches the real ~/.kiss/sorcar.sock.
// ---------------------------------------------------------------------------

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-syncwd-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

// On Windows, AbstractListen on a filesystem path under HOME would
// fail; this test is POSIX-only (the extension's UDS path is too).
if (process.platform === 'win32') {
  console.log('  skipped on win32 (UDS test)');
  fs.rmSync(tmpHome, {recursive: true, force: true});
  fs.rmSync(stubPath, {force: true});
  process.exit(0);
}

const received = [];
let serverResolveLine = null;
const linePromise = new Promise((resolve) => {
  serverResolveLine = resolve;
});

let lastServerSock = null;
const server = net.createServer((sock) => {
  lastServerSock = sock;
  let buf = '';
  sock.on('data', (chunk) => {
    buf += chunk.toString();
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const msg = JSON.parse(line);
        received.push(msg);
        if (serverResolveLine) {
          const r = serverResolveLine;
          serverResolveLine = null;
          r(msg);
        }
      } catch (err) {
        console.error('bad json:', line, err);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

async function runTests() {
  await new Promise((res, rej) => server.listen(sockPath, (err) => (err ? rej(err) : res())));

  // Load the compiled SorcarSidebarView ONLY after HOME / module
  // resolver are in place — AgentClient's ``defaultSocketPath()`` is
  // evaluated lazily (in the constructor), so as long as HOME is set
  // before ``new AgentClient()`` runs we're fine.
  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath} — run \`tsc -p .\` first`,
  );
  // Bust any cached copy from prior test runs.
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);

  // ------------------------------------------------------------------
  // Test 1 — syncWorkDir() pushes the current workspace folder.
  // ------------------------------------------------------------------
  const wsA = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-ws-a-'));
  workspaceFolders = [{uri: {fsPath: wsA, scheme: 'file'}}];

  const view = new SorcarSidebarView({fsPath: '/fake/ext', scheme: 'file'});

  view.syncWorkDir();

  const firstMsg = await Promise.race([
    linePromise,
    new Promise((_, rej) => setTimeout(() => rej(new Error('timeout waiting for setWorkDir')), 5000)),
  ]);
  assert.strictEqual(firstMsg.type, 'setWorkDir',
    `expected setWorkDir, got ${JSON.stringify(firstMsg)}`);
  assert.strictEqual(firstMsg.workDir, wsA,
    `expected workDir=${wsA}, got ${firstMsg.workDir}`);
  console.log('  ok - syncWorkDir() sends setWorkDir with current workspace folder');

  // ------------------------------------------------------------------
  // Test 2 — onDidChangeWorkspaceFolders push must follow.
  // syncWorkDir() also subscribes to workspace-folder changes; firing
  // a change must result in a second setWorkDir command.
  // ------------------------------------------------------------------
  // Re-arm the line capturer.
  const secondPromise = new Promise((resolve) => {
    serverResolveLine = resolve;
  });

  const wsB = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-ws-b-'));
  workspaceFolders = [{uri: {fsPath: wsB, scheme: 'file'}}];
  // Fire the listeners registered via onDidChangeWorkspaceFolders.
  for (const cb of folderChangeListeners.slice()) cb({added: [], removed: []});

  const secondMsg = await Promise.race([
    secondPromise,
    new Promise((_, rej) => setTimeout(() => rej(new Error('timeout waiting for follow-up setWorkDir')), 5000)),
  ]);
  assert.strictEqual(secondMsg.type, 'setWorkDir');
  assert.strictEqual(secondMsg.workDir, wsB,
    `expected workDir=${wsB} after folder change, got ${secondMsg.workDir}`);
  console.log('  ok - workspace-folder change pushes follow-up setWorkDir');

  // ------------------------------------------------------------------
  // Test 3 — syncWorkDir() must be idempotent (no duplicate clients).
  // Calling it twice must NOT open a second connection or emit a
  // duplicate setWorkDir.
  // ------------------------------------------------------------------
  const beforeCount = received.length;
  view.syncWorkDir();
  // Give the event loop a few ticks to flush any spurious sends.
  await new Promise((res) => setTimeout(res, 200));
  assert.strictEqual(received.length, beforeCount,
    `idempotent syncWorkDir must not re-send; received ${received.length - beforeCount} extra messages: ${JSON.stringify(received.slice(beforeCount))}`);
  console.log('  ok - repeated syncWorkDir() is idempotent (no duplicate setWorkDir)');

  // ------------------------------------------------------------------
  // Test 4 — reconnect must re-send setWorkDir.
  // The daemon tracks work_dir PER CONNECTION, so its state for this
  // window starts empty on every new socket.  When the connection
  // drops (daemon restart, transient error) and AgentClient
  // auto-reconnects, the 'connect' preamble must push setWorkDir
  // again — otherwise the window would fall back to the daemon-global
  // work_dir, which another window may have pointed elsewhere.
  // ------------------------------------------------------------------
  const reconnectPromise = new Promise((resolve) => {
    serverResolveLine = resolve;
  });
  assert.ok(lastServerSock, 'server never accepted a connection');
  lastServerSock.destroy(); // simulate a daemon restart dropping the UDS

  const reconnectMsg = await Promise.race([
    reconnectPromise,
    new Promise((_, rej) => setTimeout(() => rej(new Error('timeout waiting for setWorkDir after reconnect')), 5000)),
  ]);
  assert.strictEqual(reconnectMsg.type, 'setWorkDir',
    `expected setWorkDir after reconnect, got ${JSON.stringify(reconnectMsg)}`);
  assert.strictEqual(reconnectMsg.workDir, wsB,
    `expected workDir=${wsB} after reconnect, got ${reconnectMsg.workDir}`);
  console.log('  ok - reconnect re-sends setWorkDir (per-connection daemon state)');

  // Cleanup
  if (typeof view.dispose === 'function') view.dispose();
  fs.rmSync(wsA, {recursive: true, force: true});
  fs.rmSync(wsB, {recursive: true, force: true});
}

runTests().then(
  () => {
    server.close(() => {
      try {fs.unlinkSync(sockPath);} catch {}
      fs.rmSync(tmpHome, {recursive: true, force: true});
      fs.rmSync(stubPath, {force: true});
      console.log('\n4 passed, 0 failed');
      process.exit(0);
    });
  },
  (err) => {
    console.error('FAIL:', err);
    server.close(() => {
      try {fs.unlinkSync(sockPath);} catch {}
      fs.rmSync(tmpHome, {recursive: true, force: true});
      fs.rmSync(stubPath, {force: true});
      process.exit(1);
    });
  },
);
