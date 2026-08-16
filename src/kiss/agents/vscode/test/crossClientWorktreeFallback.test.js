// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// End-to-end test of the extension host's worktree fallback for tabs it
// never claimed: a canonical tab created by another client is adopted by
// the webview from `tabs_state` without any message that would register
// it in the host's _ownTabs, yet its worktree_created/worktree_done must
// still record the pending-worktree dir so the mirrored transcript's
// links resolve. A tab that drops out of a later `tabs_state` snapshot
// (closed remotely) must release that fallback instead of leaking it.

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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-xclient-'));
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

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-xclient-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];

  // Two pending worktrees: one for a tab another client created
  // ('remote'), one for this window's own tab ('local').
  const wtRemote = path.join(ws, '.kiss-worktrees', 'kiss_wt-remote');
  fs.mkdirSync(path.join(wtRemote, 'reports'), {recursive: true});
  fs.writeFileSync(
    path.join(wtRemote, 'reports', 'remote.html'),
    '<h1>remote</h1>\n',
  );
  const wtLocal = path.join(ws, '.kiss-worktrees', 'kiss_wt-local');
  fs.mkdirSync(path.join(wtLocal, 'reports'), {recursive: true});
  fs.writeFileSync(
    path.join(wtLocal, 'reports', 'local.html'),
    '<h1>local</h1>\n',
  );

  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();
  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: 'local', restoredTabs: []});
  await waitForClient();

  function sendDaemon(msg) {
    lastServerSock.write(JSON.stringify(msg) + '\n');
  }
  function pathsExistReplies() {
    return wv.posted.filter(m => m.type === 'pathsExist');
  }
  async function checkPath(p, tabId) {
    const before = pathsExistReplies().length;
    wv.fireMessage({type: 'checkPaths', paths: [p], workDir: ws, tabId});
    const reply = await waitFor(
      () => pathsExistReplies()[before],
      'a pathsExist reply must be posted',
    );
    return reply.results[p];
  }

  // Another client created tab 'remote'; this webview adopts it from
  // the canonical snapshot without sending the host any message about
  // it. The daemon then announces the tab's pending worktree.
  const remoteEntry = {tabId: 'remote', chatId: 'c2', title: 'r', workDir: ws};
  const localEntry = {tabId: 'local', chatId: 'c1', title: 'l', workDir: ws};
  sendDaemon({type: 'tabs_state', tabs: [localEntry, remoteEntry], tabId: ''});
  sendDaemon({
    type: 'worktree_created',
    worktreeDir: wtRemote,
    branch: 'kiss/wt-remote',
    tabId: 'remote',
  });
  sendDaemon({
    type: 'worktree_created',
    worktreeDir: wtLocal,
    branch: 'kiss/wt-local',
    tabId: 'local',
  });
  await waitFor(
    () =>
      wv.posted.filter(m => m.type === 'worktree_created').length === 2,
    'both worktree_created events must reach the webview',
  );

  // 1. The unclaimed tab's worktree-only path must resolve.
  assert.strictEqual(
    await checkPath('reports/remote.html', 'remote'),
    true,
    "a cross-client tab's worktree-only path must be clickable",
  );
  console.log("  ok - another client's tab gets the worktree fallback");

  // 2. A merge finished by the other client retires the fallback.
  sendDaemon({
    type: 'worktree_result',
    success: true,
    message: "Merged branch 'kiss/wt-remote'.",
    tabId: 'remote',
  });
  await waitFor(
    () => wv.posted.find(m => m.type === 'worktree_result'),
    'worktree_result must reach the webview',
  );
  assert.strictEqual(
    await checkPath('reports/remote.html', 'remote'),
    false,
    "another client's merge must drop the fallback",
  );
  console.log("  ok - another client's merge retires the fallback");

  // 3. A remote close (the tab vanishes from the snapshot) releases the
  // fallback of a still-pending worktree; other tabs keep theirs.
  sendDaemon({
    type: 'worktree_created',
    worktreeDir: wtRemote,
    branch: 'kiss/wt-remote',
    tabId: 'remote',
  });
  await waitFor(
    () =>
      wv.posted.filter(m => m.type === 'worktree_created').length === 3,
    'the re-announced worktree_created must reach the webview',
  );
  assert.strictEqual(await checkPath('reports/remote.html', 'remote'), true);
  sendDaemon({type: 'tabs_state', tabs: [localEntry], tabId: ''});
  await waitFor(
    () => wv.posted.filter(m => m.type === 'tabs_state').length === 2,
    'the pruning tabs_state must reach the webview',
  );
  assert.strictEqual(
    await checkPath('reports/remote.html', 'remote'),
    false,
    'a remotely closed tab must not keep its worktree fallback',
  );
  assert.strictEqual(
    await checkPath('reports/local.html', 'local'),
    true,
    "pruning a closed tab must not touch other tabs' fallbacks",
  );
  console.log('  ok - a remote close releases the fallback, others survive');

  view.dispose();
}

runTests()
  .then(() => {
    console.log('crossClientWorktreeFallback.test.js: all tests passed');
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
