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
    openTextDocument: () =>
      Promise.resolve({uri: makeUri('/x'), getText: () => ''}),
    textDocuments: [],
  },
  EventEmitter: StubEventEmitter,
  CancellationTokenSource: StubCancellationTokenSource,
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
    showInformationMessage: () => Promise.resolve(undefined),
    showWarningMessage: () => Promise.resolve(undefined),
    showErrorMessage: () => Promise.resolve(undefined),
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

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-autoc-ntf-'));
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
const server = net.createServer(sock => {
  lastServerSock = sock;
  let buf = '';
  sock.on('data', chunk => {
    buf += chunk.toString('utf8');
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      if (!line.trim()) continue;
      try {
        daemonReceived.push(JSON.parse(line));
      } catch {
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

async function waitFor(predicate, message, timeoutMs = 2000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const value = predicate();
    if (value) return value;
    await new Promise(r => setTimeout(r, 20));
  }
  throw new Error(message || 'waitFor timed out');
}

function notificationsByType(posts) {
  return posts.filter(m => m && m.type === 'notification');
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
  const notificationPath = path.join(
    __dirname,
    '..',
    'out',
    'WebviewNotifications.js',
  );
  assert.ok(
    fs.existsSync(notificationPath),
    `compiled notification helper missing: ${notificationPath}`,
  );
  delete require.cache[require.resolve(notificationPath)];
  delete require.cache[require.resolve(sourcePath)];
  const {SorcarSidebarView} = require(sourcePath);

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-autoc-ntf-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();
  const TAB = 'tab-autocommit';

  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: TAB, restoredTabs: []});
  await waitForClient();

  view._autocommitProgressTimeoutMs = 200;
  const buggyTab = 'tab-autocommit-buggy';
  view._ownTabs.add(buggyTab);
  const buggyBeforeCount = notificationsByType(wv.posted).length;
  wv.fireMessage({
    type: 'autocommitAction',
    action: 'commit',
    tabId: buggyTab,
    workDir: ws,
  });
  const buggyInitial = await waitFor(
    () =>
      notificationsByType(wv.posted)
        .slice(buggyBeforeCount)
        .find(
          m =>
            m.message === 'Auto-committing…' &&
            m.progress === true &&
            m.sticky === true &&
            !m.close,
        ),
    'sticky "Auto-committing…" toast was not posted for buggy run',
  );
  const buggyId = buggyInitial.id;
  await daemonSend({
    type: 'autocommit_progress',
    message: 'Generating commit message…',
    tabId: buggyTab,
  });
  await waitFor(
    () =>
      notificationsByType(wv.posted).some(
        m =>
          m.id === buggyId &&
          m.progressMessage === 'Generating commit message…',
      ),
    '"Generating commit message…" update missing on buggy run',
  );
  await waitFor(
    () =>
      notificationsByType(wv.posted).some(
        m => m.id === buggyId && m.close === true,
      ),
    'BUG REPRO: progress notification was NOT prematurely closed by the safety timer — the timeout machinery is broken',
    800,
  );

  view._autocommitProgressTimeoutMs = undefined;
  const fixBeforeCount = notificationsByType(wv.posted).length;
  wv.fireMessage({
    type: 'autocommitAction',
    action: 'commit',
    tabId: TAB,
    workDir: ws,
  });

  const initial = await waitFor(
    () =>
      notificationsByType(wv.posted)
        .slice(fixBeforeCount)
        .find(
          m =>
            m.message === 'Auto-committing…' &&
            m.progress === true &&
            m.sticky === true &&
            !m.close,
        ),
    'sticky "Auto-committing…" progress notification was not posted',
  );
  assert.ok(initial.id, 'notification must have an id');
  const NOTIF_ID = initial.id;

  await waitFor(
    () =>
      daemonReceived.some(
        m =>
          m &&
          m.type === 'autocommitAction' &&
          m.action === 'commit' &&
          m.tabId === TAB,
      ),
    'autocommitAction command was not forwarded to the daemon',
  );

  await daemonSend({
    type: 'autocommit_progress',
    message: 'Staging…',
    tabId: TAB,
  });
  await daemonSend({
    type: 'autocommit_progress',
    message: 'Generating commit message…',
    tabId: TAB,
  });

  await waitFor(
    () =>
      notificationsByType(wv.posted).some(
        m =>
          m.id === NOTIF_ID &&
          m.progress === true &&
          m.progressMessage === 'Generating commit message…',
      ),
    '"Generating commit message…" progress update was not posted to webview',
  );

  const QUIET_MS = 400;
  await new Promise(r => setTimeout(r, QUIET_MS));
  const prematureClose = notificationsByType(wv.posted).find(
    m => m.id === NOTIF_ID && m.close === true,
  );
  assert.strictEqual(
    prematureClose,
    undefined,
    `progress notification ${NOTIF_ID} was closed before autocommit_done arrived`,
  );

  await daemonSend({
    type: 'autocommit_done',
    success: true,
    message: 'Committed: feat: test',
    tabId: TAB,
  });
  await waitFor(
    () =>
      notificationsByType(wv.posted).some(
        m => m.id === NOTIF_ID && m.close === true,
      ),
    'progress notification was not closed after autocommit_done',
  );

  const fastTab = 'tab-fast-timeout';
  const progressMap = new Map();
  const resolveMap = new Map();
  const beforeCount = notificationsByType(wv.posted).length;
  view._showActionProgress(
    'Fast timeout…',
    fastTab,
    progressMap,
    resolveMap,
    150,
  );
  const fastInitial = await waitFor(
    () =>
      notificationsByType(wv.posted)
        .slice(beforeCount)
        .find(m => m.message === 'Fast timeout…' && !m.close),
    'fast-timeout progress notification was not posted',
  );
  const fastId = fastInitial.id;
  await waitFor(
    () =>
      notificationsByType(wv.posted).some(
        m => m.id === fastId && m.close === true,
      ),
    'fast-timeout notification was not auto-dismissed (timeout machinery broken)',
    1000,
  );

  view._autocommitProgressTimeoutMs = undefined;
  const disconnectTab = 'tab-autocommit-disconnect';
  view._ownTabs.add(disconnectTab);
  const disconnectBeforeCount = notificationsByType(wv.posted).length;
  wv.fireMessage({
    type: 'autocommitAction',
    action: 'commit',
    tabId: disconnectTab,
    workDir: ws,
  });
  const disconnectInitial = await waitFor(
    () =>
      notificationsByType(wv.posted)
        .slice(disconnectBeforeCount)
        .find(m => m.message === 'Auto-committing…' && !m.close),
    'auto-commit toast not posted before simulated disconnect',
  );
  const disconnectId = disconnectInitial.id;
  if (lastServerSock) lastServerSock.destroy();
  lastServerSock = null;
  await waitFor(
    () =>
      notificationsByType(wv.posted).some(
        m => m.id === disconnectId && m.close === true,
      ),
    'autocommit progress toast was not closed after daemon disconnect',
    2000,
  );
  await waitForClient();

  const tab2 = 'tab-autocommit-2';
  view._ownTabs.add(tab2);
  wv.fireMessage({
    type: 'autocommitAction',
    action: 'commit',
    tabId: tab2,
    workDir: ws,
  });
  await waitFor(
    () => view._autocommitActionResolves.has(tab2),
    'autocommit resolver was not registered for tab2',
  );
  view.dispose();
  assert.strictEqual(
    view._autocommitActionResolves.size,
    0,
    'dispose must drain _autocommitActionResolves',
  );
  assert.strictEqual(
    view._autocommitProgresses.size,
    0,
    'dispose must drain _autocommitProgresses',
  );
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
  for (const dir of tmpDirs.slice().reverse()) {
    fs.rmSync(dir, {recursive: true, force: true});
  }
}

runTests().then(
  () => {
    cleanup();
    console.log('\nAll tests passed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    cleanup();
    process.exit(1);
  },
);
