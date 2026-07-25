// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for the SorcarSidebarView side of the B1
// tabId-routing fix (multi-model review, see tabIdRoutingFixes.test.js
// for the webview side).
//
// Bug reproduced: ``openSubagentTab`` ownership adoption used
// ``(!parent_tab_id || this._ownTabs.has(parent_tab_id))`` — a BLANK
// ``parent_tab_id`` (the direct history-open of a sub-agent row, where
// ``_resolve_parent_tab_id_for_sub`` validly returns "") made EVERY
// VS Code window adopt the broadcast tab id into ``_ownTabs``.  Foreign
// tab-stamped events (worktree_result, merge_data, askUser, ...) then
// passed the adopting window's ``_isOwnTab`` gate and triggered native
// UI side effects for another window's task.
//
// Fixed behaviour (all four branches of the adoption conditional):
//   1. blank parent + target tab ALREADY owned  -> stays owned
//      (the window whose webview created the tab and posted
//      resumeSession keeps the conversion; its tab-stamped events
//      still pass _isOwnTab).
//   2. blank parent + target tab NOT owned      -> NOT adopted
//      (another window's broadcast; its tab-stamped worktree_result
//      must not render a notification here).
//   3. parent owned                              -> sub tab adopted
//      (live run_parallel spawn under a parent this window owns).
//   4. parent NOT owned                          -> NOT adopted
//      (another window's fan-out).
//
// This test drives the real compiled SorcarSidebarView + AgentClient
// against a real Unix-domain socket daemon stub.  Only the `vscode`
// module is stubbed (shared _vscode-stub.js fixture).
//
// Run directly with ``node`` (after ``npm run compile``):
//
//     node src/kiss/agents/vscode/test/sidebarSubagentOwnership.test.js

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
  CancellationTokenSource: class {
    constructor() {
      this.token = {onCancellationRequested: () => ({dispose: () => {}})};
    }
    dispose() {}
  },
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
// ``_vscode-stub.js`` is a git-tracked fixture shared by several tests
// that run in parallel; it already contains
// ``module.exports = global.__kissVscodeStub;`` — never write or delete
// it here or concurrent tests lose their ``vscode`` module mid-run.
global.__kissVscodeStub = vscodeStub;

const tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-sub-own-'));
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

function daemonSend(msg) {
  assert.ok(lastServerSock, 'daemon has no connected client socket');
  lastServerSock.write(JSON.stringify(msg) + '\n');
  return new Promise(r => setTimeout(r, 80));
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

function notificationsIn(posted) {
  return posted.filter(m => m.type === 'notification');
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

  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-sub-own-ws-'));
  tmpDirs.push(ws);
  workspaceFolders = [{uri: makeUri(ws)}];
  const view = new SorcarSidebarView(makeUri(path.join(__dirname, '..')));
  const wv = makeWebviewView();

  const OWNED_PARENT = 'tab-owned-parent';
  view.resolveWebviewView(wv.webviewView, {}, {});
  wv.fireMessage({type: 'ready', tabId: OWNED_PARENT, restoredTabs: []});
  await waitForClient();
  assert.ok(
    view._ownTabs.has(OWNED_PARENT),
    'ready must register the webview tab as owned',
  );

  // ---- Branch 1: blank parent + target ALREADY owned -> stays owned.
  // The direct history-open flow: the webview created the tab itself
  // and posted resumeSession (ownership learned from webview traffic)
  // BEFORE the daemon broadcast the blank-parent conversion.
  const HISTORY_SUB = 'tab-history-sub';
  wv.fireMessage({type: 'resumeSession', taskId: 'sub-task-1', tabId: HISTORY_SUB});
  assert.ok(
    view._ownTabs.has(HISTORY_SUB),
    'resumeSession must register the created tab as owned',
  );
  await daemonSend({
    type: 'openSubagentTab',
    tab_id: HISTORY_SUB,
    parent_tab_id: '',
    description: 'history-opened sub-agent',
    task_id: 'sub-task-1',
    isSubagentTab: true,
    isDone: true,
  });
  assert.ok(
    view._ownTabs.has(HISTORY_SUB),
    'blank-parent conversion of an owned tab must keep ownership',
  );
  // Its tab-stamped events still pass the _isOwnTab gate end-to-end.
  await daemonSend({
    type: 'worktree_result',
    success: true,
    message: 'own-tab worktree merged.',
    tabId: HISTORY_SUB,
  });
  assert.strictEqual(
    notificationsIn(wv.posted).length,
    1,
    'an owned (converted) tab\'s worktree_result must reach the webview',
  );
  console.log('  ok - blank-parent conversion keeps an owned tab owned');

  // ---- Branch 2: blank parent + target NOT owned -> NOT adopted.
  // Another window's direct history-open broadcast reaches this
  // window too; adopting it re-introduced the cross-window leak.
  const FOREIGN_SUB = 'tab-foreign-sub';
  await daemonSend({
    type: 'openSubagentTab',
    tab_id: FOREIGN_SUB,
    parent_tab_id: '',
    description: 'another window\'s sub-agent',
    task_id: 'sub-task-2',
    isSubagentTab: true,
    isDone: false,
  });
  assert.ok(
    !view._ownTabs.has(FOREIGN_SUB),
    'REGRESSION: blank-parent openSubagentTab for a foreign tab was ' +
      'adopted into _ownTabs (cross-window leak)',
  );
  const before = notificationsIn(wv.posted).length;
  await daemonSend({
    type: 'worktree_result',
    success: true,
    message: 'foreign worktree merged.',
    tabId: FOREIGN_SUB,
  });
  assert.strictEqual(
    notificationsIn(wv.posted).length,
    before,
    'a foreign tab\'s worktree_result must not render a notification here',
  );
  console.log('  ok - foreign blank-parent conversion is not adopted');

  // ---- Branch 3: parent owned -> sub tab adopted (live spawn).
  const LIVE_SUB = 'tab-owned-parent__sub_0';
  await daemonSend({
    type: 'openSubagentTab',
    tab_id: LIVE_SUB,
    parent_tab_id: OWNED_PARENT,
    description: 'live fan-out sub-agent',
    task_id: 'sub-task-3',
  });
  assert.ok(
    view._ownTabs.has(LIVE_SUB),
    'a sub-agent spawned under an owned parent must be adopted',
  );
  console.log('  ok - owned-parent sub-agent tab adopted');

  // ---- Branch 4: parent NOT owned -> NOT adopted.
  const FOREIGN_PARENT_SUB = 'tab-foreign-parent__sub_0';
  await daemonSend({
    type: 'openSubagentTab',
    tab_id: FOREIGN_PARENT_SUB,
    parent_tab_id: 'tab-foreign-parent',
    description: 'another window\'s fan-out',
    task_id: 'sub-task-4',
  });
  assert.ok(
    !view._ownTabs.has(FOREIGN_PARENT_SUB),
    'a sub-agent of a foreign parent must not be adopted',
  );
  console.log('  ok - foreign-parent sub-agent tab not adopted');

  // ---- Guard: missing tab_id -> no adoption, no crash.
  const ownedBefore = view._ownTabs.size;
  await daemonSend({
    type: 'openSubagentTab',
    parent_tab_id: OWNED_PARENT,
    description: 'malformed (no tab_id)',
  });
  assert.strictEqual(
    view._ownTabs.size,
    ownedBefore,
    'an openSubagentTab without tab_id must not change ownership',
  );
  console.log('  ok - missing tab_id changes nothing');

  view.dispose();
}

runTests()
  .then(() => {
    console.log('sidebarSubagentOwnership: all tests passed');
    server.close();
    for (const dir of tmpDirs.slice().reverse()) {
      fs.rmSync(dir, {recursive: true, force: true});
    }
    process.exit(0);
  })
  .catch(err => {
    console.error(err && err.stack ? err.stack : err);
    process.exit(1);
  });
