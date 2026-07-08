// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test: a Sorcar webview tab that STARTED a running task,
// then was closed and re-opened while the task is still running, MUST
// keep receiving the daemon's tab-stamped events (status, task_events,
// …) — even when VS Code fires the OLD webview's ``onDidDispose``
// AFTER it has already re-resolved the view with the NEW webview.
//
// Why the earlier fix (resetting ``_disposed`` in resolveWebviewView)
// is NOT enough:
//
//   VS Code does not guarantee dispose-then-resolve ordering when a
//   sidebar webview view is re-shown / moved.  The fresh webview is
//   commonly resolved FIRST (``resolveWebviewView`` runs, ``_disposed``
//   is cleared, ``_view`` points at the new webview) and only THEN does
//   the stale OLD webview's ``onDidDispose`` fire.  The previous fix's
//   ``onDidDispose`` set ``this._disposed = true`` UNCONDITIONALLY, so
//   the late dispose of the OLD webview clobbered the flag and the NEW
//   webview stopped receiving every daemon->webview message again — the
//   reopened tab never learns the task is still running, leaves
//   ``isRunning`` false, and the user's next message is dropped exactly
//   as in the original bug.
//
// This test drives the REAL compiled ``SorcarSidebarView.js`` and
// ``AgentClient.js`` (only ``vscode`` is stubbed) against a real UDS
// daemon socket.  It starts a task, RE-RESOLVES the view with a fresh
// webview (reopen) and only THEN fires the OLD webview's onDidDispose
// (the real VS Code ordering), then has the daemon stream live events
// and asserts the NEW webview receives them.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/bughunt_reopen_late_dispose.test.js

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

// ``_vscode-stub.js`` is a git-tracked fixture shared by tests running
// in parallel; it already re-exports ``global.__kissVscodeStub`` — never
// rewrite or delete it here (writeFileSync truncates first, racing a
// concurrent ``require('vscode')`` in sibling test processes).
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-reopen-ld-'));
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
        received.push(JSON.parse(line));
      } catch (err) {
        console.error('bad json:', line, err);
      }
    }
  });
});

function daemonSend(msg) {
  assert.ok(lastServerSock, 'daemon has no connected client socket');
  lastServerSock.write(JSON.stringify(msg) + '\n');
  return new Promise((r) => setTimeout(r, 60));
}

async function waitForClient() {
  for (let i = 0; i < 100 && !lastServerSock; i++) {
    await new Promise((r) => setTimeout(r, 20));
  }
  assert.ok(lastServerSock, 'client never connected to daemon');
}

function makeWebviewView() {
  const posted = [];
  const recvEmitter = new StubEventEmitter();
  const disposeEmitter = new StubEventEmitter();
  const visEmitter = new StubEventEmitter();
  const webview = {
    options: {},
    html: '',
    cspSource: 'vscode-resource:',
    asWebviewUri: (uri) => makeUri(uri.fsPath),
    postMessage: (msg) => {
      posted.push(msg);
      return Promise.resolve(true);
    },
    onDidReceiveMessage: (cb) => recvEmitter.event(cb),
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
    fireMessage: (m) => recvEmitter.fire(m),
    fireDispose: () => disposeEmitter.fire(),
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

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-reopen-ld-ws-'));
  workspaceFolders = [{uri: makeUri(ws)}];

  const extUri = makeUri(path.join(__dirname, '..'));
  const view = new SorcarSidebarView(extUri);

  const TAB = 'tab-A';

  // --- Open the tab and start a task ---------------------------------
  const wv1 = makeWebviewView();
  view.resolveWebviewView(wv1.webviewView, {}, {});
  wv1.fireMessage({type: 'ready', tabId: TAB, restoredTabs: []});
  await waitForClient();

  wv1.fireMessage({
    type: 'submit',
    prompt: 'do a long task',
    model: 'm',
    tabId: TAB,
  });
  await daemonSend({
    type: 'status',
    running: true,
    tabId: TAB,
    startTs: Date.now(),
  });

  const wv1GotRunning = wv1.posted.some(
    (m) => m.type === 'status' && m.running === true && m.tabId === TAB,
  );
  assert.ok(
    wv1GotRunning,
    'sanity: the launching tab receives status running:true',
  );

  // --- Re-open the tab while the task is still running ----------------
  // CRITICAL: VS Code re-resolves the view with the NEW webview FIRST
  // (this is the common ordering when a sidebar view is re-shown), and
  // only THEN fires the OLD webview's onDidDispose.
  const wv2 = makeWebviewView();
  view.resolveWebviewView(wv2.webviewView, {}, {});

  // The stale OLD webview's dispose fires LATE — AFTER the new webview
  // is already live.  It must NOT clobber forwarding to the new view.
  wv1.fireDispose();

  wv2.fireMessage({
    type: 'ready',
    tabId: TAB,
    restoredTabs: [{tabId: TAB, chatId: 'chat-1'}],
  });
  await waitForClient();

  // The daemon re-broadcasts the running status on resume and streams
  // live events (exactly what _replay_session does on resumeSession).
  await daemonSend({
    type: 'status',
    running: true,
    tabId: TAB,
    startTs: Date.now(),
  });
  await daemonSend({
    type: 'task_events',
    events: [],
    task: 'do a long task',
    tabId: TAB,
    chat_id: 'chat-1',
  });

  const wv2GotRunning = wv2.posted.some(
    (m) => m.type === 'status' && m.running === true && m.tabId === TAB,
  );
  assert.ok(
    wv2GotRunning,
    'BUG: the re-opened tab must receive daemon status events even when ' +
      "the OLD webview's onDidDispose fires AFTER resolveWebviewView — a " +
      'late dispose must not set _disposed=true and silence the new webview',
  );

  const wv2GotEvents = wv2.posted.some((m) => m.type === 'task_events');
  assert.ok(
    wv2GotEvents,
    'the re-opened tab must continue to receive the live task event stream',
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
  (err) => {
    console.error('FAIL:', err);
    cleanup();
    process.exit(1);
  },
);
