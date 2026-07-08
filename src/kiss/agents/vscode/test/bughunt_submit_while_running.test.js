// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test: the compiled extension (``SorcarSidebarView.js`` +
// ``AgentClient.js``) must NEVER silently drop a ``submit`` that arrives
// for a tab it already considers running.  Such a ``submit`` happens
// when a re-opened webview has not yet re-learned the running state and
// therefore sends ``submit`` instead of ``appendUserMessage``.  Rather
// than discard the user's text, the extension must forward it to the
// daemon as an ``appendUserMessage`` so it is injected into the live
// agent — exactly like a tab that loaded the task.
//
// Drives the REAL compiled extension against a real UDS stub daemon;
// only ``vscode`` is stubbed.
//
//     node src/kiss/agents/vscode/test/bughunt_submit_while_running.test.js

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
    this.event = cb => {
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
  return {fsPath, scheme: 'file', toString: () => `file://${fsPath}`};
}

let workspaceFolders = [];
const vscodeStub = {
  workspace: {
    get workspaceFolders() {
      return workspaceFolders;
    },
    getConfiguration: () => ({get: () => 'stub-default-model'}),
    onDidChangeWorkspaceFolders: () => ({dispose: () => {}}),
    openTextDocument: () =>
      Promise.resolve({uri: makeUri('/x'), getText: () => ''}),
    textDocuments: [],
  },
  EventEmitter: StubEventEmitter,
  Uri: {
    file: p => makeUri(p),
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: s => makeUri(s),
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
// ``_vscode-stub.js`` is a git-tracked fixture shared by several tests
// that run in parallel; it already contains
// ``module.exports = global.__kissVscodeStub;`` — never write or delete
// it here or concurrent tests lose their ``vscode`` module mid-run.
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-sub-run-'));
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
let lastServerSock = null;
const server = net.createServer(sock => {
  lastServerSock = sock;
  let buf = '';
  sock.on('data', chunk => {
    buf += chunk.toString();
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        received.push(JSON.parse(line));
      } catch {
        /* ignore */
      }
    }
  });
});

function daemonSend(msg) {
  assert.ok(lastServerSock, 'daemon has no connected client socket');
  lastServerSock.write(JSON.stringify(msg) + '\n');
  return new Promise(r => setTimeout(r, 60));
}

async function waitForClient() {
  for (let i = 0; i < 100 && !lastServerSock; i++) {
    await new Promise(r => setTimeout(r, 20));
  }
  assert.ok(lastServerSock, 'client never connected to daemon');
}

function makeWebviewView() {
  const recv = new StubEventEmitter();
  const dispose = new StubEventEmitter();
  const vis = new StubEventEmitter();
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: uri => makeUri(uri.fsPath),
    postMessage: () => Promise.resolve(true),
    onDidReceiveMessage: cb => recv.event(cb),
  };
  const webviewView = {
    webview,
    visible: true,
    show: () => {},
    onDidChangeVisibility: cb => vis.event(cb),
    onDidDispose: cb => dispose.event(cb),
  };
  return {webviewView, fireMessage: m => recv.fire(m)};
}

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );

  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  assert.ok(fs.existsSync(sourcePath), `compiled extension missing: ${sourcePath}`);
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-sub-run-ws-'));
  workspaceFolders = [{uri: makeUri(ws)}];
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));

  const TAB = 'tab-A';
  const wv = makeWebviewView();
  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: TAB, restoredTabs: []});
  await waitForClient();

  // Start a task and have the daemon confirm it's running so the
  // extension records TAB in its _runningTabs set.
  wv.fireMessage({type: 'submit', prompt: 'long task', model: 'm', tabId: TAB});
  await daemonSend({type: 'status', running: true, tabId: TAB});

  // Now the (desynced) webview sends a plain ``submit`` while the task
  // is still running — the user's text must NOT be dropped.
  received.length = 0;
  wv.fireMessage({
    type: 'submit',
    prompt: 'inject me into the running agent',
    model: 'm',
    tabId: TAB,
  });
  await new Promise(r => setTimeout(r, 80));

  const appended = received.filter(c => c.type === 'appendUserMessage');
  assert.strictEqual(
    appended.length,
    1,
    'BUG: a submit for an already-running tab must be forwarded to the ' +
      'daemon as an appendUserMessage, not silently dropped (got types: ' +
      JSON.stringify(received.map(c => c.type)) +
      ')',
  );
  assert.strictEqual(appended[0].prompt, 'inject me into the running agent');
  assert.strictEqual(appended[0].tabId, TAB);
  // It must NOT have started a second run.
  assert.ok(
    !received.some(c => c.type === 'run'),
    'a submit for a running tab must not start a second run',
  );

  if (typeof view.dispose === 'function') view.dispose();
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
  err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    cleanup();
    process.exit(1);
  },
);
