// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for three SorcarSidebarView bugs, driven through the
// REAL compiled ``SorcarSidebarView.js`` + ``AgentClient.js`` against a
// real UDS daemon socket (only ``vscode`` is stubbed):
//
//   1. remote_url after reopen — ``_lastSentUrl`` deduping survived
//      webview disposal, but the webview DOM does not: after closing and
//      re-opening the sidebar, ``ready`` → ``_sendRemoteUrl`` early-
//      returned on the dedup key and the welcome-page remote-URL panel
//      stayed permanently blank.  The dedup key must be reset whenever a
//      fresh webview is resolved.
//
//   2. Spurious "Process stopped" — the ``status running:false`` handler
//      fired ``onCommitMessage {error:'Process stopped'}`` whenever ANY
//      own tab stopped while ``_commitPendingTabs.size > 0``, even though
//      the stopping tab was unrelated to the tab awaiting a commit
//      message.  It must check ``.has(statusTabId)`` instead of
//      ``.size > 0``.
//
//   3. commitMessage with tabId '' — ``generateCommitMessage`` always
//      sends tabId ''; the daemon stamps the reply with the requester's
//      tabId (''), but ``_isOwnTab('')`` was false (only ``undefined``
//      was treated as global), so the reply was dropped and the SCM
//      input box never received the generated message.
//
// Run directly with ``node`` (after ``tsc -p .``):
//
//     node src/kiss/agents/vscode/test/sidebarReopenUrlCommitPending.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const Module = require('module');

// ---------------------------------------------------------------------------
// Minimal ``vscode`` stub — only the surface SorcarSidebarView touches.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Real UDS daemon at ~/.kiss/sorcar.sock (HOME redirected to a tempdir).
// ---------------------------------------------------------------------------

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-sbv-'));
process.env.HOME = tmpHome;
process.env.USERPROFILE = tmpHome;
fs.mkdirSync(path.join(tmpHome, '.kiss'), {recursive: true});
const sockPath = path.join(tmpHome, '.kiss', 'sorcar.sock');

// Remote tunnel URL the daemon-side helper wrote to disk.
fs.writeFileSync(
  path.join(tmpHome, '.kiss', 'remote-url.json'),
  JSON.stringify({tunnel: 'https://tunnel.example.dev', local: 'http://localhost:8787'}),
);

if (process.platform === 'win32') {
  console.log('  skipped on win32 (UDS test)');
  fs.rmSync(tmpHome, {recursive: true, force: true});
  process.exit(0);
}

let lastServerSock = null;
const server = net.createServer((sock) => {
  lastServerSock = sock;
  sock.on('data', () => {});
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

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
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

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-sbv-ws-'));
  workspaceFolders = [{uri: makeUri(ws)}];

  const extUri = makeUri(path.join(__dirname, '..'));
  const view = new SorcarSidebarView(extUri);

  const TAB = 'tab-A';

  // --- Open the sidebar; the welcome page needs the remote URL --------
  const wv1 = makeWebviewView();
  view.resolveWebviewView(wv1.webviewView, {}, {});
  wv1.fireMessage({type: 'ready', tabId: TAB, restoredTabs: []});
  await waitForClient();
  await sleep(120);

  const url1 = wv1.posted.filter((m) => m.type === 'remote_url');
  assert.ok(
    url1.some((m) => m.url === 'https://tunnel.example.dev' && m.tunnelActive),
    'sanity: the first webview receives the remote_url event',
  );

  // --- Test 1: close + reopen must re-send remote_url -----------------
  wv1.fireDispose();
  const wv2 = makeWebviewView();
  view.resolveWebviewView(wv2.webviewView, {}, {});
  wv2.fireMessage({type: 'ready', tabId: TAB, restoredTabs: []});
  await sleep(120);

  assert.ok(
    wv2.posted.some(
      (m) =>
        m.type === 'remote_url' &&
        m.url === 'https://tunnel.example.dev' &&
        m.tunnelActive,
    ),
    'BUG 1: the re-opened webview must receive remote_url again — the ' +
      '_lastSentUrl dedup key must be reset when a fresh webview is ' +
      'resolved, otherwise the welcome-page remote panel stays blank',
  );

  // --- Test 2: unrelated tab stopping must NOT abort a pending
  //     commit-message generation with "Process stopped" ----------------
  const commitEvents = [];
  const sub = view.onCommitMessage((e) => commitEvents.push(e));

  // Tab TAB is a running chat tab owned by this window.
  await daemonSend({type: 'status', running: true, tabId: TAB});

  // Kick off SCM commit-message generation (tabId '').
  let commitResolved = false;
  const commitPromise = view
    .generateCommitMessage(undefined, '')
    .then(() => (commitResolved = true));

  // The unrelated chat tab finishes.
  await daemonSend({type: 'status', running: false, tabId: TAB});
  await sleep(150);

  assert.strictEqual(
    commitEvents.filter((e) => e.error === 'Process stopped').length,
    0,
    'BUG 2: an unrelated own tab stopping must not fire ' +
      "onCommitMessage {error: 'Process stopped'} while a commit-message " +
      'generation for a DIFFERENT tab is pending',
  );
  assert.strictEqual(
    commitResolved,
    false,
    'BUG 2: the pending commit-message generation must not resolve early ' +
      'when an unrelated tab stops',
  );

  // --- Test 3: the daemon reply stamped with the requester tabId ''
  //     must be delivered (and resolve the pending generation) ----------
  await daemonSend({type: 'commitMessage', message: 'feat: add tests', tabId: ''});
  await sleep(150);

  assert.ok(
    commitEvents.some((e) => e.message === 'feat: add tests'),
    "BUG 3: a commitMessage event stamped with tabId '' (the requester's " +
      'own id) must pass the _isOwnTab gate and reach onCommitMessage',
  );
  await commitPromise;
  assert.strictEqual(commitResolved, true, 'commit generation resolves');

  // --- The intended stop path still works: a stop for the commit
  //     requester's own tabId ('') aborts the pending generation --------
  const p2 = view.generateCommitMessage(undefined, '');
  await daemonSend({type: 'status', running: false, tabId: ''});
  await sleep(150);
  assert.ok(
    commitEvents.some((e) => e.error === 'Process stopped'),
    "a status running:false stamped with the requester's tabId must still " +
      'abort the pending commit-message generation',
  );
  await p2;

  sub.dispose();
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
