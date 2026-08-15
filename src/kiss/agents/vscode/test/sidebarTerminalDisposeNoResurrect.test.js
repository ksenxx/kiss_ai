// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Regression test: terminal SorcarSidebarView.dispose() must be final.
//
// dispose() nulls the cached SorcarApi wrapper, but before the fix the
// webview onDidReceiveMessage registration was never disposed and _view was
// never cleared, so one late queued webview message (e.g. 'getConfig')
// reached _handleMessage(), whose _getApi()/_getClient() then built and
// connected a brand-new AgentClient and registered a fresh
// workspace-folders listener AFTER teardown: daemon connections went 1 -> 2
// and workspace subscriptions 0 -> 1.
//
// This test reproduces that scenario against a real UDS daemon stub and
// asserts that after dispose() no new daemon connection and no new
// workspace-folders subscription can ever appear.

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
      return {
        dispose: () => {
          const i = this._listeners.indexOf(cb);
          if (i >= 0) this._listeners.splice(i, 1);
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
  return {
    fsPath,
    scheme: 'file',
    toString: () => `file://${fsPath}`,
  };
}

let workspaceFolders = [];
// Live workspace-folder subscriptions: +1 on subscribe, -1 on dispose.
let activeWorkspaceSubs = 0;

const vscodeStub = {
  workspace: {
    get workspaceFolders() {
      return workspaceFolders;
    },
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => {
      activeWorkspaceSubs++;
      let disposed = false;
      return {
        dispose: () => {
          if (!disposed) {
            disposed = true;
            activeWorkspaceSubs--;
          }
        },
      };
    },
    openTextDocument: () =>
      Promise.resolve({uri: makeUri('/x'), getText: () => ''}),
    textDocuments: [],
  },
  EventEmitter: StubEventEmitter,
  Uri: {
    file: (p) => makeUri(p),
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: (s) => makeUri(s),
  },
  ProgressLocation: {Notification: 15},
  ViewColumn: {One: 1},
  window: {
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: () => {},
    showErrorMessage: () => {},
    showTextDocument: () => Promise.resolve({}),
    activeTextEditor: undefined,
    tabGroups: {all: []},
  },
  commands: {executeCommand: () => Promise.resolve()},
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};

global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-term-disp-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

if (process.platform === 'win32') {
  console.log('  skipped on win32 (UDS test)');
  fs.rmSync(tmpHome, {recursive: true, force: true});
  process.exit(0);
}

let connectionCount = 0;
let lastServerSock = null;
const server = net.createServer((sock) => {
  connectionCount++;
  lastServerSock = sock;
  sock.on('data', () => {});
  sock.on('error', () => {});
});

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitForClient() {
  for (let i = 0; i < 100 && connectionCount === 0; i++) {
    await sleep(20);
  }
  assert.ok(connectionCount > 0, 'client never connected to daemon');
}

function makeWebviewView() {
  const posted = [];
  const recvEmitter = new StubEventEmitter();
  const disposeEmitter = new StubEventEmitter();
  const visEmitter = new StubEventEmitter();
  // Raw message callbacks, kept even after the registration is disposed,
  // to model a message VS Code already dequeued when dispose() ran.
  const rawReceiveCallbacks = [];
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: (uri) => makeUri(uri.fsPath),
    postMessage: (msg) => {
      posted.push(msg);
      return Promise.resolve(true);
    },
    onDidReceiveMessage: (cb) => {
      rawReceiveCallbacks.push(cb);
      return recvEmitter.event(cb);
    },
  };
  const webviewView = {
    webview,
    visible: true,
    show: () => {},
    onDidChangeVisibility: (cb) => visEmitter.event(cb),
    onDidDispose: (cb) => disposeEmitter.event(cb),
  };
  return {
    webviewView,
    posted,
    rawReceiveCallbacks,
    fireMessage: (m) => recvEmitter.fire(m),
    recvListenerCount: () => recvEmitter._listeners.length,
  };
}

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, (err) => (err ? rej(err) : res())),
  );

  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath} — run \`tsc -p .\` first`,
  );
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-term-disp-ws-'));
  workspaceFolders = [{uri: makeUri(ws)}];

  const extUri = makeUri(path.join(__dirname, '..'));
  const view = new SorcarSidebarView(extUri);

  const wv = makeWebviewView();
  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: 'tab-A', restoredTabs: []});
  await waitForClient();
  await sleep(100);

  assert.strictEqual(
    connectionCount,
    1,
    'sanity: exactly one daemon connection before dispose',
  );
  assert.strictEqual(
    activeWorkspaceSubs,
    1,
    'sanity: one live workspace-folders subscription before dispose',
  );

  // --- Terminal teardown ---
  view.dispose();
  await sleep(100);

  assert.strictEqual(
    activeWorkspaceSubs,
    0,
    'dispose() must release the workspace-folders subscription',
  );
  assert.strictEqual(
    wv.recvListenerCount(),
    0,
    'dispose() must detach the webview onDidReceiveMessage registration',
  );

  // A message VS Code had already dequeued when dispose() ran: deliver it
  // straight to the registered callback, bypassing the (now disposed)
  // registration, exactly like a late in-flight webview message.
  for (const cb of wv.rawReceiveCallbacks) cb({type: 'getConfig'});
  // And a message arriving through the (disposed) registration.
  wv.fireMessage({type: 'getConfig'});
  await sleep(300);

  assert.strictEqual(
    connectionCount,
    1,
    'BUG: a late webview message after terminal dispose() must not ' +
      'connect a new AgentClient to the daemon (connections went 1 -> ' +
      connectionCount +
      ')',
  );
  assert.strictEqual(
    activeWorkspaceSubs,
    0,
    'BUG: a late webview message after terminal dispose() must not ' +
      'register a new workspace-folders subscription (subs went 0 -> ' +
      activeWorkspaceSubs +
      ')',
  );

  // A whole new webview resolution after terminal dispose() must also be
  // inert: the provider is dead for good.
  const wv2 = makeWebviewView();
  view.resolveWebviewView(wv2.webviewView, {}, {});
  wv2.fireMessage({type: 'ready', tabId: 'tab-B', restoredTabs: []});
  await sleep(300);

  assert.strictEqual(
    connectionCount,
    1,
    'resolveWebviewView after terminal dispose() must not reconnect',
  );
  assert.strictEqual(
    activeWorkspaceSubs,
    0,
    'resolveWebviewView after terminal dispose() must not resubscribe',
  );

  fs.rmSync(ws, {recursive: true, force: true});
}

function cleanup() {
  try {
    if (lastServerSock) lastServerSock.destroy();
  } catch {}
  try {
    server.close();
  } catch {}
  try {
    fs.unlinkSync(sockPath);
  } catch {}
  fs.rmSync(tmpHome, {recursive: true, force: true});
}

runTests().then(
  () => {
    cleanup();
    console.log('\nAll tests passed');
    process.exit(0);
  },
  (err) => {
    console.error('FAIL:', err);
    cleanup();
    process.exit(1);
  },
);
