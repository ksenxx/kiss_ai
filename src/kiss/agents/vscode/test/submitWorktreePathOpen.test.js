// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of the path-only submit shortcut against a pending
// worktree: typing just `reports/analysis.html` and pressing Send must
// open the tab's worktree copy of the file — not fall through to
// _startTask and launch an unintended agent run.

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
  extensions: {
    getExtension: () => undefined,
  },
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wtsubmit-'));
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
const daemonReceived = [];
let daemonBuffer = '';
const server = net.createServer(sock => {
  lastServerSock = sock;
  sock.on('data', chunk => {
    daemonBuffer += chunk.toString();
    let idx;
    while ((idx = daemonBuffer.indexOf('\n')) >= 0) {
      const line = daemonBuffer.slice(0, idx).trim();
      daemonBuffer = daemonBuffer.slice(idx + 1);
      if (!line) continue;
      try {
        daemonReceived.push(JSON.parse(line));
      } catch {}
    }
  });
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

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wtsubmit-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];

  // The report exists only in the tab's pending worktree.
  const wt = path.join(ws, '.kiss-worktrees', 'kiss_wt-1');
  fs.mkdirSync(path.join(wt, 'reports'), {recursive: true});
  const wtReport = path.join(wt, 'reports', 'analysis.html');
  fs.writeFileSync(wtReport, '<h1>report</h1>\n');

  // reports/analysis.html is an HTML file, so a path-only submit renders
  // it in a webview panel tab (exactly like a clicked file link — both
  // routes share _openResolvedFile). The panel's asWebviewUri call
  // receives the opened file's directory and the panel title is its
  // basename, which together identify the file that was opened.
  const opened = [];
  vscodeStub.window.createWebviewPanel = (viewType, title) => ({
    viewType,
    title,
    webview: {
      html: '',
      asWebviewUri: uri => {
        opened.push(path.join(uri.fsPath, title));
        return makeUri(uri.fsPath);
      },
    },
    reveal: () => {},
    onDidDispose: () => ({dispose: () => {}}),
    dispose: () => {},
  });

  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();
  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: 'tab1', restoredTabs: []});
  await waitForClient();

  function sendDaemon(msg) {
    lastServerSock.write(JSON.stringify(msg) + '\n');
  }
  function runCommands() {
    return daemonReceived.filter(m => m.type === 'run');
  }

  sendDaemon({
    type: 'worktree_created',
    worktreeDir: wt,
    branch: 'kiss/wt-1',
    tabId: 'tab1',
  });
  await waitFor(
    () => wv.posted.find(m => m.type === 'worktree_created'),
    'worktree_created must reach the webview',
  );

  // 1. A path-only submit opens the pending worktree copy — no run.
  wv.fireMessage({
    type: 'submit',
    prompt: 'reports/analysis.html',
    workDir: ws,
    tabId: 'tab1',
  });
  await waitFor(() => opened.length === 1, 'submit must open the file');
  assert.strictEqual(
    opened[0],
    wtReport,
    'a path-only submit must open the pending worktree copy',
  );
  assert.strictEqual(
    runCommands().length,
    0,
    'a path-only submit that resolves must not start an agent task',
  );
  console.log('  ok - path-only submit opens the worktree copy, no run');

  // 2. A non-path prompt still starts a task.
  wv.fireMessage({
    type: 'submit',
    prompt: 'summarize the repo',
    workDir: ws,
    tabId: 'tab1',
  });
  await waitFor(
    () => runCommands().length === 1,
    'a non-path prompt must start an agent task',
  );
  assert.strictEqual(opened.length, 1, 'no extra file must be opened');
  console.log('  ok - a non-path prompt still starts a task');

  view.dispose();
}

runTests()
  .then(() => {
    console.log('submitWorktreePathOpen.test.js: all tests passed');
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
