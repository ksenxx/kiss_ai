// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');

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
  ProgressLocation: {Notification: 15},
  window: {
    withProgress: (_opts, task) => task({report: () => {}}, {onCancellationRequested: () => ({dispose: () => {}})}),
  },
  commands: {executeCommand: () => Promise.resolve()},
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};

global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-syncwd-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

if (process.platform === 'win32') {
  console.log('  skipped on win32 (UDS test)');
  fs.rmSync(tmpHome, {recursive: true, force: true});
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

async function runTests() {
  await new Promise((res, rej) => server.listen(sockPath, (err) => (err ? rej(err) : res())));

  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath} — run \`tsc -p .\` first`,
  );
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);

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

  const secondPromise = new Promise((resolve) => {
    serverResolveLine = resolve;
  });

  const wsB = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-ws-b-'));
  workspaceFolders = [{uri: {fsPath: wsB, scheme: 'file'}}];
  for (const cb of folderChangeListeners.slice()) cb({added: [], removed: []});

  const secondMsg = await Promise.race([
    secondPromise,
    new Promise((_, rej) => setTimeout(() => rej(new Error('timeout waiting for follow-up setWorkDir')), 5000)),
  ]);
  assert.strictEqual(secondMsg.type, 'setWorkDir');
  assert.strictEqual(secondMsg.workDir, wsB,
    `expected workDir=${wsB} after folder change, got ${secondMsg.workDir}`);
  console.log('  ok - workspace-folder change pushes follow-up setWorkDir');

  const beforeCount = received.length;
  view.syncWorkDir();
  await new Promise((res) => setTimeout(res, 200));
  assert.strictEqual(received.length, beforeCount,
    `idempotent syncWorkDir must not re-send; received ${received.length - beforeCount} extra messages: ${JSON.stringify(received.slice(beforeCount))}`);
  console.log('  ok - repeated syncWorkDir() is idempotent (no duplicate setWorkDir)');

  const reconnectPromise = new Promise((resolve) => {
    serverResolveLine = resolve;
  });
  assert.ok(lastServerSock, 'server never accepted a connection');
  lastServerSock.destroy();

  const reconnectMsg = await Promise.race([
    reconnectPromise,
    new Promise((_, rej) => setTimeout(() => rej(new Error('timeout waiting for setWorkDir after reconnect')), 5000)),
  ]);
  assert.strictEqual(reconnectMsg.type, 'setWorkDir',
    `expected setWorkDir after reconnect, got ${JSON.stringify(reconnectMsg)}`);
  assert.strictEqual(reconnectMsg.workDir, wsB,
    `expected workDir=${wsB} after reconnect, got ${reconnectMsg.workDir}`);
  console.log('  ok - reconnect re-sends setWorkDir (per-connection daemon state)');

  if (typeof view.dispose === 'function') view.dispose();
  fs.rmSync(wsA, {recursive: true, force: true});
  fs.rmSync(wsB, {recursive: true, force: true});
}

runTests().then(
  () => {
    server.close(() => {
      try {fs.unlinkSync(sockPath);} catch {}
      fs.rmSync(tmpHome, {recursive: true, force: true});
      console.log('\n4 passed, 0 failed');
      process.exit(0);
    });
  },
  (err) => {
    console.error('FAIL:', err);
    server.close(() => {
      try {fs.unlinkSync(sockPath);} catch {}
      fs.rmSync(tmpHome, {recursive: true, force: true});
      process.exit(1);
    });
  },
);
