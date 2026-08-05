// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of the extension host's 'checkPaths' handler: the chat
// webview asks which file paths exist, and SorcarSidebarView must reply
// with a 'pathsExist' message where a path is true ONLY when clicking it
// would actually open a file (inside the workspace, exists, is a file).

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

class StubCancellationTokenSource {
  constructor() {
    this.token = {onCancellationRequested: () => ({dispose: () => {}})};
  }
  dispose() {}
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
    openTextDocument: () => Promise.resolve({getText: () => ''}),
    textDocuments: [],
  },
  EventEmitter: StubEventEmitter,
  CancellationTokenSource: StubCancellationTokenSource,
  Uri: {
    file: p => makeUri(p),
    joinPath: (base, ...parts) => makeUri(path.join(base.fsPath, ...parts)),
    parse: s => makeUri(s),
  },
  Position: class {},
  Range: class {},
  Selection: class {},
  TextEditorRevealType: {InCenter: 2, AtTop: 3},
  ProgressLocation: {Notification: 15},
  ViewColumn: {One: 1},
  window: {
    withProgress: (_opts, task) =>
      task(
        {report: () => {}},
        {onCancellationRequested: () => ({dispose: () => {}})},
      ),
    showInformationMessage: () => Promise.resolve(undefined),
    showWarningMessage: () => Promise.resolve(undefined),
    showErrorMessage: () => Promise.resolve(undefined),
    showTextDocument: () => Promise.resolve({}),
    activeTextEditor: undefined,
    tabGroups: {all: []},
  },
  commands: {
    executeCommand: () => Promise.resolve(),
  },
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-checkpaths-'));
const tmpDirs = [tmpHome];
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

if (process.platform === 'win32') {
  console.log('  skipped on win32 (UDS test)');
  for (const dir of tmpDirs.slice().reverse()) {
    fs.rmSync(dir, {recursive: true, force: true});
  }
  process.exit(0);
}

let lastServerSock = null;
const server = net.createServer(sock => {
  lastServerSock = sock;
  sock.on('data', () => {});
});

async function waitForClient() {
  for (let i = 0; i < 100 && !lastServerSock; i++) {
    await new Promise(r => setTimeout(r, 20));
  }
  assert.ok(lastServerSock, 'client never connected to daemon');
}

function makeWebviewView() {
  const recv = new StubEventEmitter();
  const posted = [];
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: uri => makeUri(uri.fsPath),
    postMessage: msg => {
      posted.push(msg);
      return Promise.resolve(true);
    },
    onDidReceiveMessage: cb => recv.event(cb),
  };
  const webviewView = {
    webview,
    visible: true,
    show: () => {},
    onDidChangeVisibility: () => ({dispose: () => {}}),
    onDidDispose: () => ({dispose: () => {}}),
  };
  return {webviewView, posted, fireMessage: m => recv.fire(m)};
}

async function waitFor(predicate, message, timeoutMs = 1500) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const value = predicate();
    if (value) return value;
    await new Promise(r => setTimeout(r, 10));
  }
  throw new Error(message || 'waitFor timed out');
}

async function runTests() {
  await new Promise((res, rej) =>
    server.listen(sockPath, err => (err ? rej(err) : res())),
  );

  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarSidebarView.js');
  assert.ok(
    fs.existsSync(sourcePath),
    `compiled extension missing: ${sourcePath}`,
  );
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-checkpaths-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];

  const realFile = path.join(ws, 'src', 'main.py');
  fs.mkdirSync(path.dirname(realFile), {recursive: true});
  fs.writeFileSync(realFile, 'print("hello")\n');
  const outsideFile = path.join(tmpHome, 'outside.txt');
  fs.writeFileSync(outsideFile, 'outside the workspace\n');

  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();
  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: 'tab1', restoredTabs: []});
  await waitForClient();

  // One batched query mixing every case the linkifier can produce.
  wv.fireMessage({
    type: 'checkPaths',
    paths: [
      'src/main.py', // exists (relative)
      realFile, // exists (absolute)
      'src/missing.py', // does not exist
      'src', // a directory, not a file
      '../escape.txt', // resolves outside the workspace
      outsideFile, // absolute path outside the workspace
      '', // degenerate: empty
      42, // degenerate: not a string
    ],
    workDir: ws,
    tabId: 'tab1',
  });
  const reply = await waitFor(
    () => wv.posted.find(m => m.type === 'pathsExist'),
    'a pathsExist reply must be posted to the webview',
  );
  assert.strictEqual(reply.tabId, 'tab1', 'tabId must be echoed');
  assert.strictEqual(reply.workDir, ws, 'workDir must be echoed');
  assert.strictEqual(
    reply.results['src/main.py'],
    true,
    'relative path to an existing file must be true',
  );
  assert.strictEqual(
    reply.results[realFile],
    true,
    'absolute path to an existing file must be true',
  );
  assert.strictEqual(
    reply.results['src/missing.py'],
    false,
    'missing file must be false',
  );
  assert.strictEqual(reply.results['src'], false, 'directory must be false');
  assert.strictEqual(
    reply.results['../escape.txt'],
    false,
    'path escaping the workspace must be false',
  );
  assert.strictEqual(
    reply.results[outsideFile],
    false,
    'absolute path outside the workspace must be false (openFile refuses it)',
  );
  assert.ok(!('' in reply.results), 'empty path must be skipped, not reported');
  assert.ok(!(42 in reply.results), 'non-string entries must be skipped');
  console.log('  ok - pathsExist reports existence exactly like openFile');

  // Malformed queries must not crash and must still reply.
  wv.fireMessage({type: 'checkPaths', paths: 'not-an-array', tabId: 'tab1'});
  const reply2 = await waitFor(
    () =>
      wv.posted.filter(m => m.type === 'pathsExist').length === 2 &&
      wv.posted.filter(m => m.type === 'pathsExist')[1],
    'a pathsExist reply must be posted even for a malformed query',
  );
  assert.deepStrictEqual(
    reply2.results,
    {},
    'malformed paths field must yield empty results',
  );
  console.log('  ok - malformed checkPaths yields an empty pathsExist');

  view.dispose();
}

runTests()
  .then(() => {
    console.log('checkPathsExistence.test.js: all tests passed');
  })
  .catch(err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    process.exitCode = 1;
  })
  .finally(() => {
    server.close();
    if (lastServerSock) lastServerSock.destroy();
    for (const dir of tmpDirs.slice().reverse()) {
      fs.rmSync(dir, {recursive: true, force: true});
    }
  });
