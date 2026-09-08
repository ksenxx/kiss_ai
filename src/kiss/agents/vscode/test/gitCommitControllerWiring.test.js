// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end test for SorcarSidebarView.gitCommit() (compiled
// out/SorcarSidebarView.js against the shared vscode stub): the
// editor-title git-commit button's controller hop. It must reveal the
// chat webview and post {type: 'gitCommit'} to it — the webview side
// (media/main.js) is covered by editorTitleGitCommit.test.js and the
// command registration by editorTabsExtensionWiring.test.js.

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

global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-gitcommit-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

if (process.platform === 'win32') {
  console.log('  skipped on win32 (UDS test)');
  fs.rmSync(tmpHome, {recursive: true, force: true});
  process.exit(0);
}

let lastServerSock = null;
const server = net.createServer(sock => {
  lastServerSock = sock;
  sock.on('data', () => {});
});

function makeWebviewView() {
  const posted = [];
  const shows = [];
  const recvEmitter = new StubEventEmitter();
  const disposeEmitter = new StubEventEmitter();
  const visEmitter = new StubEventEmitter();
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: uri => makeUri(uri.fsPath),
    postMessage: msg => {
      posted.push(msg);
      return Promise.resolve(true);
    },
    onDidReceiveMessage: cb => recvEmitter.event(cb),
  };
  const webviewView = {
    webview,
    visible: true,
    show: preserveFocus => shows.push(preserveFocus),
    onDidChangeVisibility: cb => visEmitter.event(cb),
    onDidDispose: cb => disposeEmitter.event(cb),
  };
  return {
    webviewView,
    posted,
    shows,
    fireMessage: m => recvEmitter.fire(m),
  };
}

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );

  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath} — run \`tsc -p .\` first`,
  );
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-gitcommit-ws-'));
  workspaceFolders = [{uri: makeUri(ws)}];

  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();
  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: 'tab-root', restoredTabs: []});

  const showsBefore = wv.shows.length;
  await view.gitCommit();

  assert.ok(
    wv.posted.some(m => m.type === 'gitCommit'),
    'gitCommit() posts {type: gitCommit} to the webview',
  );
  assert.ok(
    wv.shows.length > showsBefore,
    'gitCommit() reveals the chat webview first',
  );
  // The reveal precedes the message: the commit progress lines land in
  // a visible transcript.
  const gitCommitIndex = wv.posted.findIndex(m => m.type === 'gitCommit');
  const focusIndex = wv.posted.findIndex(m => m.type === 'focusInput');
  assert.ok(
    focusIndex >= 0 && focusIndex < gitCommitIndex,
    'the reveal/focus precedes the gitCommit message',
  );

  if (typeof view.dispose === 'function') view.dispose();
  fs.rmSync(ws, {recursive: true, force: true});
  console.log('  ok - gitCommit() reveals the chat and posts gitCommit');
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
    console.error('FAIL:', err);
    cleanup();
    process.exit(1);
  },
);
