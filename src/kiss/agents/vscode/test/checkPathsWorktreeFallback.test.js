// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of the extension host's pending-worktree fallback: a
// path that exists only in a tab's un-merged worktree must be reported
// clickable for that tab (checkPaths), open the worktree copy (openFile),
// and stop resolving once worktree_result reports the merge finished.

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
  // worktree_created triggers _openWorktreeInScm, which asks the git
  // extension to open the worktree repository; report it absent.
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wtfallback-'));
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

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wtfallback-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];

  // The task's pending worktree holds the committed artifact; the
  // workspace root does not have it yet (the branch is not merged).
  const wt = path.join(ws, '.kiss-worktrees', 'kiss_wt-1');
  fs.mkdirSync(path.join(wt, 'reports'), {recursive: true});
  const wtReport = path.join(wt, 'reports', 'analysis.html');
  fs.writeFileSync(wtReport, '<h1>report</h1>\n');

  // reports/analysis.html is an HTML file, so openFile renders it in a
  // webview panel tab (not the text editor). The panel's asWebviewUri
  // call receives the opened file's directory and the panel title is
  // its basename, which together identify the file that was opened.
  const opened = [];
  vscodeStub.window.createWebviewPanel = (viewType, title) => {
    const panel = {
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
    };
    return panel;
  };

  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();
  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: 'tab1', restoredTabs: []});
  await waitForClient();

  function sendDaemon(msg) {
    lastServerSock.write(JSON.stringify(msg) + '\n');
  }
  function pathsExistReplies() {
    return wv.posted.filter(m => m.type === 'pathsExist');
  }

  // The daemon announces the tab's pending worktree.
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

  // 1. checkPaths resolves via the pending worktree — for tab1 only.
  const query = {
    type: 'checkPaths',
    paths: ['reports/analysis.html'],
    workDir: ws,
    tabId: 'tab1',
  };
  wv.fireMessage(query);
  let reply = await waitFor(
    () => pathsExistReplies()[0],
    'a pathsExist reply must be posted',
  );
  assert.strictEqual(
    reply.results['reports/analysis.html'],
    true,
    'a path existing only in the tab pending worktree must be true',
  );
  console.log('  ok - worktree-only path is reported existing for its tab');

  wv.fireMessage({...query, tabId: 'tab2'});
  reply = await waitFor(
    () => pathsExistReplies()[1],
    'a second pathsExist reply must be posted',
  );
  assert.strictEqual(
    reply.results['reports/analysis.html'],
    false,
    'another tab must not see this tab worktree',
  );
  console.log('  ok - the fallback is scoped to the owning tab');

  // 2. openFile opens the worktree copy while the merge is pending.
  wv.fireMessage({
    type: 'openFile',
    path: 'reports/analysis.html',
    workDir: ws,
    tabId: 'tab1',
  });
  await waitFor(() => opened.length === 1, 'openFile must open the file');
  assert.strictEqual(
    opened[0],
    wtReport,
    'openFile must open the pending worktree copy',
  );
  console.log('  ok - clicking the link opens the worktree copy');

  // 3. A successful worktree_result drops the fallback ...
  sendDaemon({
    type: 'worktree_result',
    success: true,
    message: "Merged branch 'kiss/wt-1'.",
    tabId: 'tab1',
  });
  await waitFor(
    () => wv.posted.find(m => m.type === 'worktree_result'),
    'worktree_result must reach the webview',
  );
  wv.fireMessage(query);
  reply = await waitFor(
    () => pathsExistReplies()[2],
    'a third pathsExist reply must be posted',
  );
  assert.strictEqual(
    reply.results['reports/analysis.html'],
    false,
    'a finished worktree_result must drop the fallback',
  );
  console.log('  ok - the fallback is dropped once the merge finishes');

  // 4. ... and the merged copy under the workspace root wins.
  const wsReport = path.join(ws, 'reports', 'analysis.html');
  fs.mkdirSync(path.dirname(wsReport), {recursive: true});
  fs.writeFileSync(wsReport, '<h1>report</h1>\n');
  wv.fireMessage(query);
  reply = await waitFor(
    () => pathsExistReplies()[3],
    'a fourth pathsExist reply must be posted',
  );
  assert.strictEqual(
    reply.results['reports/analysis.html'],
    true,
    'the merged workspace copy must resolve after the merge',
  );
  wv.fireMessage({
    type: 'openFile',
    path: 'reports/analysis.html',
    workDir: ws,
    tabId: 'tab1',
  });
  await waitFor(() => opened.length === 2, 'openFile must open the file');
  assert.strictEqual(
    opened[1],
    wsReport,
    'openFile must prefer the merged workspace copy',
  );
  console.log('  ok - the merged workspace copy wins after the merge');

  view.dispose();
}

runTests()
  .then(() => {
    console.log('checkPathsWorktreeFallback.test.js: all tests passed');
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
