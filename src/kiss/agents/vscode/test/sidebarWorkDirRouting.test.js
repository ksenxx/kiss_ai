// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// E2E tests for SorcarSidebarView routing/lifecycle findings:
// VS-011: openFile/checkPaths carry the tab's workDir to the daemon, which
//         resolves them (web_server.py _resolve_tab_file) and answers this
//         window with openResolvedFile — opened by the host as sent.
// VS-016: closeTab must release per-tab host resources.
// VS-018: stopTask without a resolved webview must stop via the API.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const opened = [];

class EventEmitterLite {
  constructor() {
    this._subs = [];
  }
  event = cb => {
    this._subs.push(cb);
    return {dispose() {}};
  };
  fire(v) {
    for (const cb of this._subs) cb(v);
  }
  dispose() {}
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-sidebar-route-'));
const wsRoot = path.join(tmp, 'workspace');
const tabRepo = path.join(tmp, 'tab-repo');
fs.mkdirSync(wsRoot, {recursive: true});
fs.mkdirSync(tabRepo, {recursive: true});
fs.writeFileSync(path.join(tabRepo, 'inrepo.txt'), 'tab repo file\n');
fs.writeFileSync(path.join(wsRoot, 'inws.txt'), 'workspace file\n');

const vscodeStub = {
  Uri: {
    file: p => ({fsPath: p, scheme: 'file'}),
    joinPath: (base, ...parts) => ({
      fsPath: path.join(base.fsPath, ...parts),
      scheme: 'file',
    }),
    parse: s => ({fsPath: s.replace(/^file:\/\//, ''), scheme: 'file'}),
  },
  Position: class {
    constructor(line, character) {
      this.line = line;
      this.character = character;
    }
  },
  Range: class {
    constructor(a, b, c, d) {
      this.start = a && a.line !== undefined ? a : {line: a, character: b};
      this.end = b && b.line !== undefined ? b : {line: c, character: d};
    }
  },
  Selection: class {
    constructor(a, b) {
      this.anchor = a;
      this.active = b;
    }
  },
  TextEditorRevealType: {InCenter: 2},
  ViewColumn: {One: 1},
  ProgressLocation: {Notification: 15},
  EventEmitter: EventEmitterLite,
  TabInputText: class {},
  window: {
    visibleTextEditors: [],
    activeTextEditor: undefined,
    tabGroups: {all: [], close: () => Promise.resolve(true)},
    createTextEditorDecorationType: () => ({dispose() {}}),
    onDidChangeVisibleTextEditors: () => ({dispose() {}}),
    showTextDocument: doc => {
      opened.push(doc.uri.fsPath);
      return Promise.resolve({
        document: doc,
        selection: null,
        revealRange() {},
        setDecorations() {},
        edit: () => Promise.resolve(true),
      });
    },
    showInformationMessage: () => undefined,
    showWarningMessage: () => undefined,
    showErrorMessage: () => undefined,
    withProgress: (_opts, task) =>
      task({report() {}}, {onCancellationRequested: () => ({dispose() {}})}),
    createTerminal: () => ({show() {}, sendText() {}}),
  },
  workspace: {
    isTrusted: true,
    workspaceFolders: [{uri: {fsPath: wsRoot, scheme: 'file'}}],
    getConfiguration: () => ({get: () => undefined}),
    onWillSaveTextDocument: () => ({dispose() {}}),
    onDidSaveTextDocument: () => ({dispose() {}}),
    onDidChangeWorkspaceFolders: () => ({dispose() {}}),
    textDocuments: [],
    openTextDocument: uri =>
      Promise.resolve({
        uri,
        getText: () => '',
        lineCount: 1,
        lineAt: () => ({text: '', range: {end: {line: 0, character: 0}}}),
        isDirty: false,
      }),
    saveAll: () => Promise.resolve(true),
    applyEdit: () => Promise.resolve(true),
  },
  commands: {executeCommand: () => Promise.resolve()},
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};
global.__kissVscodeStub = vscodeStub;

const outDir = path.join(__dirname, '..', 'out');
assert.ok(
  fs.existsSync(path.join(outDir, 'SorcarSidebarView.js')),
  'compiled extension missing — run `npm run compile` first',
);
const {SorcarSidebarView} = require(path.join(outDir, 'SorcarSidebarView.js'));

function makeView() {
  const view = new SorcarSidebarView({fsPath: path.join(tmp, 'ext')});
  const sent = [];
  // Replace the daemon client with a recorder (no real daemon in tests).
  view._api = {
    _sent: sent,
    forward: c => sent.push(c),
    run: c => sent.push({type: 'run', ...c}),
    stop: tabId => sent.push({type: 'stop', tabId}),
    complete: c => sent.push({type: 'complete', ...c}),
    closeTab: tabId => sent.push({type: 'closeTab', tabId}),
    getModels: () => {},
    getInputHistory: () => {},
    getConfig: () => {},
    setWorkDir: () => {},
    recordFileUsage: () => {},
    selectModel: () => {},
    userAnswer: () => {},
    resumeSession: () => {},
    worktreeAction: () => {},
    generateCommitMessage: () => {},
    serverReset: () => {},
    appendUserMessage: () => {},
  };
  return {view, sent};
}

function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// VS-011: openFile is forwarded with the tab's workDir, and the daemon's
// resolved answer is what the host opens.
async function testOpenFileHonorsTabWorkDir() {
  const {view, sent} = makeView();
  opened.length = 0;
  await view._handleMessage({
    type: 'openFile',
    path: 'inrepo.txt',
    workDir: tabRepo,
    tabId: 't1',
  });
  assert.deepStrictEqual(
    sent.filter(c => c.type === 'openFile'),
    [{type: 'openFile', path: 'inrepo.txt', line: undefined,
      workDir: tabRepo, tabId: 't1'}],
    'VS-011: openFile must reach the daemon with the tab workDir, unresolved',
  );
  assert.deepStrictEqual(opened, [], 'the host does not resolve paths itself');
  const handlers = {};
  view._installClientListener({
    on: (evt, cb) => {
      handlers[evt] = cb;
    },
  });
  handlers.message({
    type: 'openResolvedFile',
    path: path.join(tabRepo, 'inrepo.txt'),
    tabId: 't1',
  });
  await delay(10);
  assert.deepStrictEqual(
    opened,
    [path.join(tabRepo, 'inrepo.txt')],
    'VS-011: the daemon-resolved path is what opens',
  );
  handlers.message({
    type: 'openResolvedFile',
    path: 'inws.txt',
    tabId: 't1',
    error: 'File not found: inws.txt',
  });
  await delay(10);
  assert.strictEqual(opened.length, 1, 'an error reply opens nothing');
  view.dispose();
  console.log('ok - VS-011 openFile carries the tab workDir to the daemon');
}

// VS-011: checkPaths is forwarded with the supplied workDir; the daemon
// answers pathsExist on this connection and the host relays it.
async function testCheckPathsHonorsTabWorkDir() {
  const {view, sent} = makeView();
  await view._handleMessage({
    type: 'checkPaths',
    paths: ['inrepo.txt', 'inws.txt'],
    workDir: tabRepo,
    tabId: 't1',
  });
  assert.deepStrictEqual(
    sent.filter(c => c.type === 'checkPaths'),
    [{type: 'checkPaths', paths: ['inrepo.txt', 'inws.txt'],
      workDir: tabRepo, tabId: 't1'}],
    'VS-011: checkPaths must reach the daemon with the tab workDir',
  );
  view.dispose();
  console.log('ok - VS-011 checkPaths carries the tab workDir to the daemon');
}

// VS-016: closeTab releases per-tab resources.
async function testCloseTabCleansResources() {
  const {view, sent} = makeView();
  view._ownTabs.add('tabY');
  view._runningTabs.add('tabY');
  view._commitPendingTabs.add('tabY');
  view._worktreeDirs.set('tabY', tabRepo);
  let resolved = false;
  view._worktreeActionResolves.set('tabY', () => (resolved = true));
  view._worktreeProgresses.set('tabY', {report() {}});
  await view._handleMessage({type: 'closeTab', tabId: 'tabY'});
  assert.ok(!view._runningTabs.has('tabY'), 'VS-016: runningTabs leaked');
  assert.ok(
    !view._commitPendingTabs.has('tabY'),
    'VS-016: commitPendingTabs leaked',
  );
  assert.ok(!view._worktreeDirs.has('tabY'), 'VS-016: worktreeDirs leaked');
  assert.ok(resolved, 'VS-016: worktree progress resolver not resolved');
  assert.ok(
    !view._worktreeActionResolves.has('tabY') &&
      !view._worktreeProgresses.has('tabY'),
    'VS-016: worktree progress maps leaked',
  );
  assert.ok(
    sent.some(m => m.type === 'closeTab' && m.tabId === 'tabY'),
    'closeTab still forwarded to the daemon',
  );
  view.dispose();
  console.log('ok - VS-016 closeTab releases per-tab resources');
}

// VS-018: stopTask without a resolved webview stops via the API.
async function testStopTaskWithoutWebview() {
  const {view, sent} = makeView();
  view._runningTabs.add('run1');
  view._runningTabs.add('run2');
  view.stopTask(); // no view resolved — must fall back to direct stops
  const stops = sent.filter(m => m.type === 'stop').map(m => m.tabId);
  assert.deepStrictEqual(
    stops.sort(),
    ['run1', 'run2'],
    'VS-018: stopTask must stop running tabs when no webview is resolved',
  );
  view.dispose();
  console.log('ok - VS-018 stopTask stops via API without a webview');
}

// VS-013: ghost completion must use the webview-supplied tabId, not the
// host's stale notion of the active tab.
async function testCompleteUsesMessageTabId() {
  const {view, sent} = makeView();
  view._activeTabId = 'stale-tab';
  await view._handleMessage({type: 'complete', query: 'fix bug', tabId: 'fresh-tab'});
  const msg = sent.find(m => m.type === 'complete');
  assert.ok(msg, 'complete forwarded');
  assert.strictEqual(
    msg.tabId,
    'fresh-tab',
    'VS-013: completion routed to the stale host-side active tab',
  );
  view.dispose();
  console.log('ok - VS-013 completion uses the message tabId');
}

async function run() {
  await testOpenFileHonorsTabWorkDir();
  await testCompleteUsesMessageTabId();
  await testCheckPathsHonorsTabWorkDir();
  await testCloseTabCleansResources();
  await testStopTaskWithoutWebview();
  await delay(10);
  console.log('\nAll sidebar workDir-routing tests passed');
}

run().then(
  () => {
    fs.rmSync(tmp, {recursive: true, force: true});
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err);
    fs.rmSync(tmp, {recursive: true, force: true});
    process.exit(1);
  },
);
