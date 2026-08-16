// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of the extension host's worktree-fallback restore from
// a replayed transcript: a host that (re)connects while a worktree task
// is running never saw the live worktree_created — the daemon replays it
// nested inside a task_events envelope, and checkPaths must still
// resolve the worktree-only artifact.  Also verifies that the fallback
// prefers worktreeWorkDir (the task's cwd inside the worktree) over the
// worktree root, and that a replayed successful worktree_result nets the
// fallback out.

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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wtreplay-'));
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

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-wtreplay-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];

  // The tab's workDir is a NESTED project dir; the running task's
  // artifact lives at the same offset inside the worktree.
  const nestedWd = path.join(ws, 'packages', 'app');
  fs.mkdirSync(nestedWd, {recursive: true});
  const wt = path.join(ws, '.kiss-worktrees', 'kiss_wt-live');
  const wtWorkDir = path.join(wt, 'packages', 'app');
  fs.mkdirSync(path.join(wtWorkDir, 'reports'), {recursive: true});
  const wtReport = path.join(wtWorkDir, 'reports', 'live.html');
  fs.writeFileSync(wtReport, '<h1>live</h1>\n');

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

  // This host never saw a live worktree_created: the daemon replays it
  // nested inside the running task's transcript envelope.
  sendDaemon({
    type: 'task_events',
    events: [
      {type: 'prompt', text: 'do work'},
      {
        type: 'worktree_created',
        worktreeDir: wt,
        worktreeWorkDir: wtWorkDir,
        branch: 'kiss/wt-live',
      },
    ],
    task: 'do work',
    tabId: 'tab1',
  });
  await waitFor(
    () => wv.posted.find(m => m.type === 'task_events'),
    'task_events must reach the webview',
  );

  // 1. The replayed nested worktree_created restored the fallback, and
  //    the nested worktreeWorkDir (not the worktree root) resolves the
  //    relative artifact path.
  const query = {
    type: 'checkPaths',
    paths: ['reports/live.html'],
    workDir: nestedWd,
    tabId: 'tab1',
  };
  wv.fireMessage(query);
  let reply = await waitFor(
    () => pathsExistReplies()[0],
    'a pathsExist reply must be posted',
  );
  assert.strictEqual(
    reply.results['reports/live.html'],
    true,
    'a replayed worktree_created must restore the nested fallback',
  );
  console.log('  ok - replayed worktree_created restores the fallback');

  // 2. openFile serves the worktree copy from the nested cwd.
  // reports/live.html is an HTML file, so openFile renders it in a
  // webview panel tab (not the text editor). The panel's asWebviewUri
  // call receives the opened file's directory and the panel title is
  // its basename, which together identify the file that was opened.
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
  wv.fireMessage({
    type: 'openFile',
    path: 'reports/live.html',
    workDir: nestedWd,
    tabId: 'tab1',
  });
  await waitFor(() => opened.length === 1, 'openFile must open the file');
  assert.strictEqual(
    opened[0],
    wtReport,
    'openFile must serve the nested worktree copy',
  );
  console.log('  ok - the nested worktreeWorkDir wins over the root');

  // 3. A replay of an already-merged task nets out to no fallback.
  sendDaemon({
    type: 'task_events',
    events: [
      {
        type: 'worktree_created',
        worktreeDir: wt,
        worktreeWorkDir: wtWorkDir,
        branch: 'kiss/wt-live',
      },
      {
        type: 'worktree_result',
        success: true,
        message: "Merged branch 'kiss/wt-live'.",
      },
    ],
    task: 'do work',
    tabId: 'tab1',
  });
  await waitFor(
    () => wv.posted.filter(m => m.type === 'task_events').length === 2,
    'the second task_events must reach the webview',
  );
  wv.fireMessage(query);
  reply = await waitFor(
    () => pathsExistReplies()[1],
    'a second pathsExist reply must be posted',
  );
  assert.strictEqual(
    reply.results['reports/live.html'],
    false,
    'a replayed successful worktree_result must retire the fallback',
  );
  console.log('  ok - a replayed merged transcript nets out the fallback');

  view.dispose();
}

runTests()
  .then(() => {
    console.log('worktreeReplayFallbackRestore.test.js: all tests passed');
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
